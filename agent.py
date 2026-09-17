"""
파이프라인 — `판정 → 근거 조립 → 답변 → 검증` (PLAN 슬라이스 5·6).

네 노드짜리 LangGraph 다. 검증 노드(`verify.py`, PLAN D4·D11)가 답변의 수치·조문 번호·
식품유형명을 주입된 근거와 대조하고, 걸리면 위반 목록을 프롬프트에 넣어 **재생성을 한 번**
시킨다. 두 번째도 걸리면 위반을 그대로 들고 끝낸다 — 거기서 넘기기로 보내는 것은 슬라이스 7이다.

**넘기기 두 갈래는 아직 없다.** 판정 노드가 카테고리를 하나도 고르지 않으면 근거 없이
답변 노드로 가고, 답변 노드는 지어내지 말고 넘기라고만 지시받는다.
`out_of_scope` 와 `no_evidence` 가 **구분되지 않는 상태**가 이 슬라이스의 정상이다.

노드 사이로 흐르는 것은 `State` 하나뿐이고 노드는 부분 딕셔너리를 돌려준다
(agentteam-m01 관행). 그래프는 `build_graph()` 가 매번 `StateGraph` 부터 새로 만든다 —
`compile()` 은 재정의된 함수를 다시 잡지 않는다 (PLAN 검증절 함정 1).

    uv run python -m agent "유자즙 28% 음료를 유자주스로 팔아도 되나요?"
    uv run python -m agent --goldenset          # 평가셋 전 문항 관통 (완료 기준 b·c)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import tools
import verify
from tools import ToolResult

load_dotenv()

HERE = Path(__file__).parent
GOLDENSET = HERE / "data" / "goldenset.json"
PROMPT_EXAMPLES = HERE / "data" / "prompt-examples.json"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 답변 본문에서 인용 표기를 뽑는 규칙. 청크 ID 는 `FDC-016` · `LBL-T6-001` 꼴이다.
CITE = re.compile(r"\[([A-Z]{3}(?:-[A-Z0-9]+)+)\]")


# ─────────────────────────────────────────────────────────────── 상태


def _extend(a: list, b: list) -> list:
    return [*a, *b]


class State(TypedDict, total=False):
    question: str
    tools: list[str]  # 판정 노드가 고른 카테고리 = 호출할 도구
    routing_note: str
    results: list[ToolResult]
    evidence_ids: list[str]
    answer: str
    citations: list[str]
    attempts: int  # 답변 노드가 몇 번 돌았나. 재생성은 한 번까지 (PLAN 슬라이스 6)
    answers: Annotated[list[str], _extend]  # 재생성 전후를 둘 다 남긴다 (완료 기준 d)
    violations: list[dict]  # 마지막 검증에서 근거 밖으로 잡힌 표기
    verify_ok: bool
    verify_checked: int
    trace: Annotated[list[str], _extend]


# ────────────────────────────────────────────────────────── 1. 판정


class Routing(BaseModel):
    """카테고리 판정 결과. 과제 지표가 도구 **집합** 일치라서 목록으로 받는다 (PLAN D2)."""

    tools: list[
        Literal[
            "lookup_food_type",
            "lookup_labeling",
            "lookup_additive",
            "lookup_ad_claims",
        ]
    ] = Field(description="근거가 필요한 카테고리. 넷 중 어느 것도 아니면 빈 목록")
    reason: str = Field(description="왜 그렇게 골랐는지 한 문장")


ROUTER_PROMPT = """너는 **음료류 식품 표시·규격** 질문을 카테고리로 나누는 라우터다.
답을 만들지 않는다 — 어느 근거를 읽어야 하는지만 고른다.

카테고리 넷:
- `lookup_food_type` — 내 제품이 어느 식품유형인지, 그 유형의 규격·함량 기준·원료 기준
  (식품공전 음료류 유형 정의, 과·채주스/과·채음료/혼합음료/액상차, 식품원료로 쓸 수 있는지)
- `lookup_labeling` — 라벨에 반드시 들어가야 하는 표시사항, 표시 방법·활자 크기·함량 표시
  (식품등의 표시기준)
- `lookup_additive` — 첨가물을 쓸 수 있는지·얼마까지인지·원재료명에 뭐라 적는지
  (식품첨가물공전 사용기준, 표시기준의 첨가물 표시 별표)
- `lookup_ad_claims` — 패키지·광고 **문구**가 부당한 표시·광고에 걸리는지
  (질병 효능 표방, 건강기능식품 오인, 「무(無)」·「천연」·「100%」 표시)

규칙:
1. 근거가 정말 필요한 것만 고른다. 한 질문이 두 카테고리에 걸치면 둘 다 고른다 (최대 2개).
2. 아래는 **소관이 아니다** — 넷 중 무엇도 고르지 말고 빈 목록을 돌려준다.
   영업등록·HACCP·시설기준·수입신고·통관, 건강기능식품 기능성 인정,
   **위반 시 처분 수위(과태료·영업정지) 안내**, 음료류가 아닌 식품유형.
3. 「과태료가 얼마인가」처럼 표시·광고법 **안**의 말이 섞여 있어도 묻는 것이 처분 수위면
   소관이 아니다.

질문: {question}"""


def _router_llm():
    return ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Routing)


def classify(state: State) -> dict:
    question = state["question"]
    routing: Routing = _router_llm().invoke(ROUTER_PROMPT.format(question=question))
    picked = list(dict.fromkeys(routing.tools))  # 같은 도구를 두 번 고르는 일이 있다
    return {
        "tools": picked,
        "routing_note": routing.reason,
        "trace": [f"판정: {picked or '(없음)'} — {routing.reason}"],
    }


# ─────────────────────────────────────────────────────── 2. 근거 조립


def gather(state: State) -> dict:
    results: list[ToolResult] = []
    ids: list[str] = []
    trace: list[str] = []
    for name in state.get("tools", []):
        result = tools.CALL[name](state["question"])
        results.append(result)
        ids.extend(result.chunk_ids)
        trace.append(f"조회: {name} → 청크 {len(result.chunk_ids)}개 ({len(result.text):,}자)")
    if not results:
        trace.append("조회: 없음 — 판정 노드가 카테고리를 고르지 않았다")
    return {"results": results, "evidence_ids": ids, "trace": trace}


# ────────────────────────────────────────────────────────── 3. 답변

ANSWER_SYSTEM = """너는 음료 브랜드를 준비하는 **비전문가**에게 식품 표시·규격 법령을 안내한다.
사용자가 가장 중요하게 여기는 것은 답이 맞는지보다 **어디에 그렇게 쓰여 있는지**다.

지켜야 할 것:
1. **주어진 근거 안에 있는 것만 쓴다.** 수치·조문 번호·식품유형명은 근거에서 그대로 옮긴다.
   근거에 없는 숫자는 알고 있어도 쓰지 않는다.
2. 사실을 말한 문장 끝마다 출처를 `[청크ID]` 로 붙인다 (예: `과·채즙 95% 이상이다 [FDC-016]`).
   근거 블록 머리의 대괄호 안 ID 를 그대로 쓴다. 없는 ID 를 만들지 않는다.
3. 판정에 필요한 정보가 질문에 없으면 **되묻지 않는다.** 갈래를 다 보여주고
   무엇이 더 필요한지 명시한다 (예: "판정하려면 과즙 함량이 필요하다").
4. **법적 판단과 처분 수위를 말하지 않는다.** 「위법입니다」·「과태료 얼마입니다」라고 쓰지 않는다.
   조문을 보여주고 판단은 사용자와 전문가에게 남긴다.
5. 근거에 답이 없거나 근거가 하나도 주어지지 않았으면 **모른다고 말하고 넘긴다.**
   지어내지 말고, 어디에 물어야 하는지 알려준다.

형식: 한국어 평문. 결론 한 줄 → 근거가 되는 조문 내용 → (필요하면) 더 필요한 정보.
400자 안팎으로 짧게."""

ANSWER_USER = """질문: {question}

{evidence}

위 근거만으로 답해라."""

NO_EVIDENCE = "(조회된 근거가 없다. 이 질문은 네 카테고리 어디에도 들어가지 않는다고 판정되었다.)"


def _few_shot() -> list[dict]:
    """프롬프트 예시는 채점용과 분리된 파일에서 읽는다 (PRD 4절 평가셋 요건)."""
    data = json.loads(PROMPT_EXAMPLES.read_text(encoding="utf-8"))
    out = []
    for ex in data["items"]:
        cites = "".join(f" [{c}]" for c in ex["citations"])
        out.append({"q": ex["question"], "a": ex["answerSketch"] + cites})
    return out


def _answer_llm():
    return ChatOpenAI(model=MODEL, temperature=0)


def answer(state: State) -> dict:
    results = state.get("results", [])
    evidence = verify.evidence_text(results) if results else NO_EVIDENCE
    attempt = state.get("attempts", 0) + 1

    messages: list[tuple[str, str]] = [("system", ANSWER_SYSTEM)]
    for ex in _few_shot():
        messages.append(("human", f"질문: {ex['q']}\n\n(예시 — 근거는 생략)"))
        messages.append(("ai", ex["a"]))
    messages.append(("human", ANSWER_USER.format(question=state["question"], evidence=evidence)))

    # 재생성 — 검증 노드가 잡은 것을 그대로 보여준다. 무엇이 왜 걸렸는지 모르면
    # 같은 말을 다시 쓴다 (m01 의 「근거를 함께 넘긴다」와 같은 발상).
    previous = state.get("answer", "")
    flagged = state.get("violations", [])
    if previous and flagged:
        report = verify.Report(
            violations=[verify.Violation(v["kind"], v["token"], verify.canon(v["token"])) for v in flagged],
            checked=state.get("verify_checked", 0),
        )
        messages.append(("ai", previous))
        messages.append(("human", verify.regen_note(report)))

    text = _answer_llm().invoke(messages).content
    if isinstance(text, list):  # 일부 모델은 조각 목록으로 돌려준다
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    text = text.strip()
    cited = list(dict.fromkeys(CITE.findall(text)))
    head = "재생성" if attempt > 1 else "답변"
    return {
        "answer": text,
        "citations": cited,
        "attempts": attempt,
        "answers": [text],
        "trace": [f"{head}: {len(text):,}자, 인용 {len(cited)}건"],
    }


# ────────────────────────────────────────────────────────── 4. 검증

# 재생성은 한 번까지. 두 번째도 걸리면 위반을 들고 끝낸다 — 넘기기는 슬라이스 7이 받는다.
MAX_ATTEMPTS = 2


def verify_node(state: State) -> dict:
    results = state.get("results", [])
    report = verify.check(
        state.get("answer", ""),
        verify.evidence_text(results),
        state["question"],
    )
    return {
        "violations": report.as_json(),
        "verify_ok": report.ok,
        "verify_checked": report.checked,
        "trace": [f"검증({state.get('attempts', 1)}차): {report.summary()}"],
    }


def after_verify(state: State) -> str:
    """통과했거나 재생성을 이미 썼으면 끝낸다."""
    if state.get("verify_ok") or state.get("attempts", 1) >= MAX_ATTEMPTS:
        return "end"
    return "answer"


# ───────────────────────────────────────────────────────────── 그래프


def build_graph():
    """매번 `StateGraph` 부터 새로 만든다 — `compile()` 은 재정의된 함수를 다시 잡지 않는다."""
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("gather", gather)
    builder.add_node("answer", answer)
    builder.add_node("verify", verify_node)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "gather")
    builder.add_edge("gather", "answer")
    builder.add_edge("answer", "verify")
    builder.add_conditional_edges("verify", after_verify, {"answer": "answer", "end": END})
    return builder.compile()


@dataclass
class Run:
    question: str
    tools: list[str] = field(default_factory=list)
    routing_note: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    attempts: int = 1
    answers: list[str] = field(default_factory=list)  # 재생성 전후 (완료 기준 d)
    violations: list[dict] = field(default_factory=list)
    verify_ok: bool = True
    verify_checked: int = 0
    trace: list[str] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def regenerated(self) -> bool:
        return self.attempts > 1

    @property
    def valid_citations(self) -> list[str]:
        """실제로 주입된 근거를 가리키는 인용만. 지어낸 ID 는 여기서 빠진다."""
        injected = set(self.evidence_ids)
        return [c for c in self.citations if c in injected]

    @property
    def invented_citations(self) -> list[str]:
        injected = set(self.evidence_ids)
        return [c for c in self.citations if c not in injected]


def ask(question: str, graph=None) -> Run:
    graph = graph or build_graph()
    started = time.monotonic()
    final = graph.invoke({"question": question})
    return Run(
        question=question,
        tools=final.get("tools", []),
        routing_note=final.get("routing_note", ""),
        evidence_ids=final.get("evidence_ids", []),
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        attempts=final.get("attempts", 1),
        answers=final.get("answers", []),
        violations=final.get("violations", []),
        verify_ok=final.get("verify_ok", True),
        verify_checked=final.get("verify_checked", 0),
        trace=final.get("trace", []),
        results=final.get("results", []),
        seconds=time.monotonic() - started,
    )


# ────────────────────────────────────────────────────────────── 출력


def show(run: Run) -> None:
    print(f"\n질문: {run.question}\n")
    print("─" * 70)
    print("호출 도구:", ", ".join(run.tools) if run.tools else "(없음)")
    print("판정 근거:", run.routing_note)
    print("─" * 70)
    print(run.answer)
    print("─" * 70)
    print("인용 근거:")
    by_id = {}
    for result in run.results:
        for ev in result.evidence:
            by_id[ev.chunk.id] = ev.chunk
    if not run.valid_citations:
        print("  (없음)")
    for cid in run.valid_citations:
        c = by_id[cid]
        print(f"  [{cid}] {c.law_name} · {c.path} (시행 {c.effective_date})")
        print(f"        {c.head[:90]}")
        print(f"        {c.url}")
    if run.invented_citations:
        print("  ! 근거에 없는 ID 를 인용했다:", ", ".join(run.invented_citations))
    print("─" * 70)
    print("환각 검증:", end=" ")
    if run.verify_ok:
        print(f"통과 — 수치·조문·유형명 {run.verify_checked}건을 근거와 대조했다")
    else:
        print(f"위반 {len(run.violations)}건 (재생성 뒤에도 남았다)")
        for v in run.violations:
            print(f"  ! {v['kind']}「{v['token']}」— 근거에 없다")
    if run.regenerated:
        print(f"  재생성 1회. 첫 답변: {run.answers[0][:80]}…")
    print(f"\n조립한 근거 {len(run.evidence_ids)}청크 · {run.seconds:.1f}초")
    for line in run.trace:
        print(" ", line)


# ────────────────────────────────────────── 완료 기준 점검 (평가셋 관통)


def run_goldenset(out: Path | None = None, limit: int | None = None) -> int:
    """평가셋 전 문항을 흘린다.

    슬라이스 5의 완료 기준(에러 0 · 근거가 있으면 인용 있음)에 슬라이스 6이 얹는 것 —
    문항별 **위반 목록과 재생성 여부**를 기록에 남기고(b·d), 남은 위반을 눈으로 가릴 수 있게
    출력한다(c). 남은 위반 자체는 실패로 치지 않는다 — 그 문항을 넘기기로 보내는 것은
    슬라이스 7이다. 채점(S1·S2)은 슬라이스 8이다.
    """
    data = json.loads(GOLDENSET.read_text(encoding="utf-8"))
    items = data["items"][: limit or None]
    graph = build_graph()

    errors: list[str] = []
    no_cite: list[str] = []
    regenerated: list[str] = []
    remaining: list[str] = []
    rows = []
    for i, item in enumerate(items, 1):
        head = f"[{i:2}/{len(items)}] {item['id']:6}"
        try:
            run = ask(item["question"], graph)
        except Exception as exc:  # noqa: BLE001 — 관통 확인이 목적이라 전부 잡아 보고한다
            errors.append(f"{item['id']}: {type(exc).__name__}: {exc}")
            print(f"{head} 에러 — {type(exc).__name__}: {exc}")
            continue

        # (c) 는 근거가 주입된 문항에만 건다. 소관 밖 문항은 인용할 근거 자체가 없다.
        expects_evidence = bool(run.evidence_ids)
        cited = run.valid_citations
        if expects_evidence and not cited:
            no_cite.append(item["id"])
        if run.invented_citations:
            errors.append(f"{item['id']}: 없는 청크 ID 인용 {run.invented_citations}")

        if run.regenerated:
            regenerated.append(item["id"])
        if not run.verify_ok:
            remaining.append(item["id"])

        mark = "ok      " if (cited or not expects_evidence) else "인용없음"
        verdict = "검증ok" if run.verify_ok else f"위반{len(run.violations)}"
        print(
            f"{head} {mark} 도구={','.join(run.tools) or '-':38} "
            f"근거={len(run.evidence_ids):3} 인용={len(cited)} "
            f"{verdict:6}{'/재생성' if run.regenerated else '      '} {run.seconds:4.1f}s"
        )
        for v in run.violations:
            print(f"{'':14}   ! {v['kind']}「{v['token']}」")
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expectedTools": item["expectedTools"],
                "tools": run.tools,
                "routingNote": run.routing_note,
                "evidenceIds": run.evidence_ids,
                "answer": run.answer,
                "citations": cited,
                "inventedCitations": run.invented_citations,
                # 슬라이스 6 — 검증 결과 (b)(d)
                "attempts": run.attempts,
                "regenerated": run.regenerated,
                "answerHistory": run.answers,
                "verifyOk": run.verify_ok,
                "verifyChecked": run.verify_checked,
                "violations": run.violations,
                "seconds": round(run.seconds, 2),
            }
        )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"model": MODEL, "at": time.strftime("%Y-%m-%d %H:%M"), "runs": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n기록: {out}")

    print(f"\n문항 {len(items)} · 에러 {len(errors)} · 근거는 있는데 인용 없음 {len(no_cite)}")
    for line in errors:
        print("  에러:", line)
    if no_cite:
        print("  인용 없음:", ", ".join(no_cite))
    print(f"검증: 재생성 {len(regenerated)}건 {regenerated or ''} · 재생성 뒤에도 위반 "
          f"{len(remaining)}건 {remaining or ''}")
    if remaining:
        print("  → 남은 위반이 실제 위반인지 오탐인지 가른다 (완료 기준 c, docs/VERIFY-NOTES.md)")
        print("     넘기기로 보내는 것은 슬라이스 7이다 — 여기서 실패로 치지 않는다")
    ok = not errors and not no_cite
    print("완료 기준:", "통과" if ok else "실패")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="agent", description="식품 표시·규격 라우팅 에이전트")
    ap.add_argument("question", nargs="?", help="질문 한 줄")
    ap.add_argument("--goldenset", action="store_true", help="평가셋 전 문항 관통")
    ap.add_argument("--out", type=Path, help="관통 결과를 JSON 으로 저장")
    ap.add_argument("--limit", type=int, help="관통할 문항 수 (확인용)")
    args = ap.parse_args(argv)

    if args.goldenset:
        return run_goldenset(args.out, args.limit)
    if not args.question:
        ap.error("질문을 주거나 --goldenset 을 써라")
    show(ask(args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
