"""
파이프라인 — `판정 → 근거 조립 → 답변 → 검증 → (넘기기)` (PLAN 슬라이스 5·6·7).

다섯 노드짜리 LangGraph 다. 검증 노드(`verify.py`, PLAN D4·D11)가 답변의 수치·조문 번호·
식품유형명을 주입된 근거와 대조하고, 걸리면 위반 목록을 프롬프트에 넣어 **재생성을 한 번**
시킨다. 두 번째도 걸리면 넘기기 노드로 보낸다.

**넘기기는 두 갈래이고 서로 다른 노드에서 정해진다** (PLAN D3·D10):

- `out_of_scope` — 소관이 아님. **판정 노드**가 카테고리를 하나도 고르지 않으면 그 자리에서
  결정된다. `gather` 를 건너뛰므로 호출 도구는 **0개**이고 LLM 답변 호출도 없다
- `no_evidence` — 카테고리는 맞는데 발췌 범위에 답이 없음. **답변 노드의 구조화 출력**
  (`Answer.verdict`)이 정한다. 근거를 실제로 읽은 주체가 답변자라서다. 검증이 재생성 뒤에도
  실패하면 같은 갈래로 보낸다 — 근거 밖 표기가 남은 답변은 내보내지 않는다

두 갈래가 다른 노드에서 정해져야 기대 도구 집합(공집합 vs 1개)이 정의된다. 최종 본문은
`escalate` 노드가 **결정적으로** 조립한다 — 머리줄·문의처 줄이 고정이라 넘겼는지를 규칙으로 잰다.

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
    escalation: str | None  # None | "out_of_scope" | "no_evidence" (PLAN 슬라이스 7)
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
    # 카테고리가 하나도 없으면 소관 밖이다. 여기서 결정적으로 정한다 (PLAN D3·D10) —
    # 근거를 조회하지 않으므로 이 갈래의 기대 도구 집합은 공집합이 된다.
    escalation = None if picked else "out_of_scope"
    head = "판정" if picked else "판정(소관 밖)"
    return {
        "tools": picked,
        "routing_note": routing.reason,
        "escalation": escalation,
        "trace": [f"{head}: {picked or '(없음)'} — {routing.reason}"],
    }


def after_classify(state: State) -> str:
    """소관 밖이면 근거 조립도 답변 생성도 건너뛴다."""
    return "escalate" if state.get("escalation") == "out_of_scope" else "gather"


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
5. **근거에 답이 없으면 지어내지 않는다.** 질문이 지목한 것(그 원료·그 첨가물·그 낱말)이
   근거에 나오지 않거나, 근거가 「…목록은 [별표 N]과 같다」처럼 **주어지지 않은 다른 문서**를
   가리키면, 근거가 어디까지 말하는지만 적고 가부를 단정하지 않는다
   (「쓸 수 있습니다」·「쓸 수 없습니다」를 쓰지 않는다). 문의처 안내는 쓰지 않는다 —
   그 줄은 시스템이 붙인다.

6. `verdict` 는 **다 쓰고 나서 방금 쓴 답변에 붙이는 딱지**다. 근거가 넉넉한지에 대한 느낌이
   아니라, 쓴 것이 무엇인지 보고 고른다.
   - `answered` — 근거 조문을 인용해 질문에 답을 줬다. 규칙 3의 조건부 답변도, 규칙 4처럼
     판단을 사용자에게 남긴 답변도, 「수치로 된 한도가 따로 없다」도 전부 답을 준 것이다.
     함량·용도처럼 **질문이 안 알려 준 정보** 때문에 갈래를 보여준 것도 `answered` 다.
     문구를 묻는 질문은 그 낱말(「천연」·「무(無)」·「100%」·질병 효능)을 다룬 조문을 인용했으면
     답을 준 것이다 — 제품명 전체가 근거에 그대로 있을 필요는 없다.
   - `no_evidence` — 규칙 5에 걸려 **모른다고 말하고 넘긴** 답변이다.

형식: 한국어 평문. 결론 한 줄 → 근거가 되는 조문 내용 → (필요하면) 더 필요한 정보.
400자 안팎으로 짧게."""

class Answer(BaseModel):
    """답변 노드의 구조화 출력. `no_evidence` 판정을 답변자가 직접 낸다 (PLAN D10).

    **`text` 가 먼저다.** 모델은 필드 순서대로 쓰고, 뒤 필드는 앞 필드를 보고 쓴다.
    `verdict` 를 앞에 두면 근거가 넉넉한지를 미리 점치게 되어 판정이 흔들린다 —
    실측에서 제대로 답해 놓고 `no_evidence` 를 붙이는 문항이 나왔다. 뒤에 두면 방금 쓴
    답변에 딱지를 붙이는 일이 된다.
    """

    text: str = Field(description="답변 본문. 사실마다 [청크ID] 인용을 붙인다")
    verdict: Literal["answered", "no_evidence"] = Field(
        description="방금 쓴 text 가 근거를 인용해 답을 준 것이면 answered, "
        "근거에 답이 없어 모른다고 말하고 넘긴 것이면 no_evidence"
    )


ANSWER_USER = """질문: {question}

{evidence}

위 근거만으로 답해라."""

NO_EVIDENCE = "(조회된 근거가 없다. 이 질문은 네 카테고리 어디에도 들어가지 않는다고 판정되었다.)"


def _few_shot() -> list[dict]:
    """프롬프트 예시는 채점용과 분리된 파일에서 읽는다 (PRD 4절 평가셋 요건).

    소관 밖 예시는 뺀다 — 슬라이스 7부터 그런 질문은 판정 노드에서 끊겨 답변 노드에 오지 않는다.
    남는 예시에는 `verdict` 를 붙여 보여준다. 붙이지 않으면 모델이 「단정할 수 없는 질문」을
    전부 `no_evidence` 로 넘긴다 (실측: 조건부 답변이 정답인 문항 셋이 넘기기로 샜다).
    """
    data = json.loads(PROMPT_EXAMPLES.read_text(encoding="utf-8"))
    out = []
    for ex in data["items"]:
        if ex.get("expectedEscalation") == "out_of_scope":
            continue
        cites = "".join(f" [{c}]" for c in ex["citations"])
        out.append(
            {
                "q": ex["question"],
                "verdict": ex.get("expectedEscalation") or "answered",
                "a": ex["answerSketch"] + cites,
            }
        )
    return out


def _answer_llm():
    # json_schema — 스키마를 모델 쪽에서 강제한다. 앞의 예시가 평문이라 함수 호출 방식은
    # 평문으로 새는 일이 있다.
    return ChatOpenAI(model=MODEL, temperature=0).with_structured_output(
        Answer, method="json_schema"
    )


def answer(state: State) -> dict:
    results = state.get("results", [])
    evidence = verify.evidence_text(results) if results else NO_EVIDENCE
    attempt = state.get("attempts", 0) + 1

    messages: list[tuple[str, str]] = [("system", ANSWER_SYSTEM)]
    for ex in _few_shot():
        messages.append(("human", f"질문: {ex['q']}\n\n(예시 — 근거는 생략)"))
        messages.append(
            ("ai", json.dumps({"text": ex["a"], "verdict": ex["verdict"]}, ensure_ascii=False))
        )
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

    result: Answer = _answer_llm().invoke(messages)
    text = result.text.strip()
    cited = list(dict.fromkeys(CITE.findall(text)))
    # 재생성이면 직전 판정을 덮어쓴다 — 두 번째 답변이 근거를 다시 읽고 낸 판정이 최신이다.
    escalation = "no_evidence" if result.verdict == "no_evidence" else None
    head = "재생성" if attempt > 1 else "답변"
    tail = "" if escalation is None else " · 판정=no_evidence"
    return {
        "answer": text,
        "citations": cited,
        "escalation": escalation,
        "attempts": attempt,
        "answers": [text],
        "trace": [f"{head}: {len(text):,}자, 인용 {len(cited)}건{tail}"],
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
    """걸렸고 기회가 남았으면 재생성, 넘기기로 정해졌거나 재생성 뒤에도 걸렸으면 넘긴다."""
    if not state.get("verify_ok") and state.get("attempts", 1) < MAX_ATTEMPTS:
        return "answer"
    if state.get("escalation") or not state.get("verify_ok"):
        return "escalate"
    return "end"


# ──────────────────────────────────────────────────────── 5. 넘기기
#
# 두 갈래의 최종 본문을 **결정적으로** 조립한다 (PLAN D10). 머리줄과 문의처 줄이 고정이라
# 「넘겼는가」와 「어디에 물어야 하는지 알려줬는가」(완료 기준 d)를 규칙으로 잴 수 있다.
# `no_evidence` 는 답변자가 쓴 「근거가 어디까지 말하는지」를 가운데 그대로 끼운다 —
# 이걸 버리면 사용자가 무엇이 확인되고 무엇이 안 됐는지 알 수 없다.

ESCALATION_HEAD = {
    "out_of_scope": "이 질문은 제가 다루는 범위(음료류 식품의 표시·규격) 밖입니다. 답하지 않고 넘깁니다.",
    "no_evidence": "조회한 근거 범위 안에는 이 질문의 답이 없습니다. 지어내지 않고 넘깁니다.",
}

ESCALATION_WHERE = {
    "out_of_scope": (
        "문의처: 영업등록·HACCP·시설기준·수입신고·위반 시 처분 수위는 관할 시·군·구 위생부서가 "
        "소관입니다. 식약처 식품안전상담 1399, 식품안전나라(foodsafetykorea.go.kr)에서도 확인할 수 있습니다."
    ),
    "no_evidence": (
        "문의처: 조회 범위 밖의 원문(공전·고시의 별표 전문)은 식품안전나라"
        "(foodsafetykorea.go.kr)에서 확인하고, 개별 제품에 대한 판단은 식약처 식품안전상담 1399 "
        "또는 관할 시·군·구 위생부서에 문의하세요."
    ),
}

HANDOFF_MARK = "문의처:"  # 완료 기준 (d) 를 규칙으로 확인할 때 쓰는 표지

ESCALATION_LABEL = {
    "out_of_scope": "소관 밖 (UC-5)",
    "no_evidence": "소관이지만 근거에 답 없음 (UC-6)",
}


def escalate(state: State) -> dict:
    reason = state.get("escalation") or "no_evidence"
    verified = state.get("verify_ok", True)
    # 검증에 걸린 채 남은 답변은 버린다 — 근거 밖 표기가 든 문장을 내보내지 않는다.
    kept = state.get("answer", "").strip() if reason == "no_evidence" and verified else ""
    text = "\n\n".join(x for x in (ESCALATION_HEAD[reason], kept, ESCALATION_WHERE[reason]) if x)
    note = f"넘기기: {reason}"
    if not verified:
        note += " (재생성 뒤에도 검증 실패 — 본문을 버렸다)"
    return {
        "escalation": reason,
        "answer": text,
        "citations": list(dict.fromkeys(CITE.findall(text))),
        "answers": [text],
        "trace": [note],
    }


# ───────────────────────────────────────────────────────────── 그래프


def build_graph():
    """매번 `StateGraph` 부터 새로 만든다 — `compile()` 은 재정의된 함수를 다시 잡지 않는다."""
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("gather", gather)
    builder.add_node("answer", answer)
    builder.add_node("verify", verify_node)
    builder.add_node("escalate", escalate)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", after_classify, {"gather": "gather", "escalate": "escalate"}
    )
    builder.add_edge("gather", "answer")
    builder.add_edge("answer", "verify")
    builder.add_conditional_edges(
        "verify", after_verify, {"answer": "answer", "escalate": "escalate", "end": END}
    )
    builder.add_edge("escalate", END)
    return builder.compile()


@dataclass
class Run:
    question: str
    tools: list[str] = field(default_factory=list)
    routing_note: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    escalation: str | None = None  # 슬라이스 7
    attempts: int = 0  # 답변 노드가 돈 횟수. 소관 밖이면 0 (LLM 답변 호출이 없다)
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
        escalation=final.get("escalation"),
        attempts=final.get("attempts", 0),
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
    if run.escalation:
        print(f"넘기기: {run.escalation} — {ESCALATION_LABEL[run.escalation]}")
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
    if run.attempts == 0:
        print("생략 — 소관 밖이라 답변을 생성하지 않았다")
    elif run.verify_ok:
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

    슬라이스 5의 완료 기준(에러 0 · 근거가 있으면 인용 있음)에 슬라이스 6(문항별 위반 목록과
    재생성 여부)과 슬라이스 7이 얹힌다. 슬라이스 7이 여기서 실행으로 확인하는 것:

    - (a) OS-1·OS-2 가 `out_of_scope` 이고 호출 도구 0개
    - (b) FT-4 가 `no_evidence` 이고 호출 도구가 기대 집합(`lookup_food_type`) 그대로
    - (c) 나머지 15문항의 `escalation` 이 전부 `None` — 멀쩡한 문항이 넘기기로 새면 실패
    - (d) 넘긴 문항의 본문에 문의처 줄이 들어간다

    셋 다 평가셋의 `expectedEscalation` 으로 일반화해 확인한다. 채점(S1·S2)은 슬라이스 8이다.
    """
    data = json.loads(GOLDENSET.read_text(encoding="utf-8"))
    items = data["items"][: limit or None]
    graph = build_graph()

    errors: list[str] = []
    escalation_fail: list[str] = []  # 슬라이스 7 완료 기준 (a)(b)(c)(d)
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

        # 슬라이스 7 — 넘기기 두 갈래
        expected_escalation = item.get("expectedEscalation")
        if run.escalation != expected_escalation:
            escalation_fail.append(
                f"{item['id']}: 넘기기 {run.escalation or 'None'} (기대 {expected_escalation or 'None'})"
            )
        elif expected_escalation:
            if run.tools != item["expectedTools"]:  # (a) 공집합 · (b) 도구 1개
                escalation_fail.append(
                    f"{item['id']}: 넘기기는 맞는데 도구 {run.tools or '[]'} "
                    f"(기대 {item['expectedTools'] or '[]'})"
                )
            if HANDOFF_MARK not in run.answer:  # (d)
                escalation_fail.append(f"{item['id']}: 넘기기 문구에 문의처가 없다")

        if run.regenerated:
            regenerated.append(item["id"])
        if not run.verify_ok:
            remaining.append(item["id"])

        mark = "ok      " if (cited or not expects_evidence) else "인용없음"
        verdict = "검증ok" if run.verify_ok else f"위반{len(run.violations)}"
        esc = {"out_of_scope": "넘김:소관밖", "no_evidence": "넘김:근거없음", None: ""}[run.escalation]
        print(
            f"{head} {mark} 도구={','.join(run.tools) or '-':38} "
            f"근거={len(run.evidence_ids):3} 인용={len(cited)} "
            f"{verdict:6}{'/재생성' if run.regenerated else '      '} {run.seconds:4.1f}s {esc}"
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
                # 슬라이스 7 — 넘기기 두 갈래
                "expectedEscalation": item.get("expectedEscalation"),
                "escalation": run.escalation,
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
        print("  → 재생성 뒤에도 걸린 문항은 넘기기(no_evidence)로 보냈다 (슬라이스 7)")

    escalated = [r["id"] for r in rows if r["escalation"]]
    print(f"넘기기: {len(escalated)}건 {escalated or ''} · 기대와 다른 문항 {len(escalation_fail)}건")
    for line in escalation_fail:
        print("  넘기기 불일치:", line)

    ok = not errors and not no_cite and not escalation_fail
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
