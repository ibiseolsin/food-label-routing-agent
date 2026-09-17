"""
채점기 — S1(도구 집합) · S2(답변 적절성) · S3(근거 밖 진술) · S4(넘기기) (PLAN 슬라이스 8).

지표마다 채점 방법이 다르다 (PLAN D5·D12). **LLM 을 태우는 것은 S2 의 필수 사실 하나뿐이다.**

| 지표 | 방법 | 1점 조건 |
|---|---|---|
| S1 | 규칙 — 집합 비교 | 호출 도구 **집합**이 기대 집합과 같다 (PLAN D2) |
| S2 | 두 층 — 필수 사실은 LLM, 금지 표현은 규칙 | 필수 사실 전부 충족 **AND** 금지 표현 0건 |
| S3 | 규칙 — `verify.check` 재사용 | 답변의 수치·조문·유형명이 전부 근거 안에 있다 |
| S4 | 규칙 — 라벨 일치 | 넘기기 갈래가 기대와 같고, 넘긴 답변에는 문의처가 있다 |

**금지 표현은 `tools.norm` 으로 정규화한 뒤 부분 문자열로 본다.** 「과·채즙 50% 이상」과
「과채즙 50 % 이상」은 같은 말이고, 부호·띄어쓰기를 바꿔 피해 가는 것을 답으로 쳐 주면
S2 가 표현 채점이 된다. `--selftest` 가 이걸 확인한다 (완료 기준 c).

세 갈래로 돈다:

    uv run python evaluate.py --selftest                  # 규칙 층 자체 확인 (API 키 불필요)
    uv run python evaluate.py --reference                 # 모범 답안 채점 = S5 역검증
    uv run python evaluate.py --negative                  # 일부러 망친 답변이 0점인지 (역대조)
    uv run python evaluate.py --score data/runs/slice7.json   # 기록된 실행을 채점 (LLM 1회/문항)
    uv run python evaluate.py --out data/runs/baseline.json   # 파이프라인을 돌려 채점

**채점기를 모범 답안에 맞춰 고치면 S5 가 무의미해진다** (PLAN 검증절 함정 3).
`--reference` 가 실패하면 고칠 곳은 채점기이지 모범 답안이 아니다.

그리고 **전부 통과시키는 채점기도 S5 를 통과한다.** `--reference` 하나로는 채점기가 무엇을
가르는지 알 수 없어서 `--negative` 를 함께 둔다 — 모범 답안을 한 군데씩 망가뜨려 넣고
의도한 지표가 실제로 0점이 되는지 본다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import tools
import verify
from tools import norm

load_dotenv()

HERE = Path(__file__).parent
GOLDENSET = HERE / "data" / "goldenset.json"
REFERENCE = HERE / "data" / "reference-answers.json"

# 채점기 모델은 파이프라인 모델과 **따로** 둔다. 파이프라인 `model` 은 개선 축이라(PLAN D13)
# 축을 바꿀 때마다 채점 잣대까지 같이 바뀌면 회차 비교가 무의미해진다.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")


# ─────────────────────────────────────────────── S2 필수 사실 — LLM 층 (D12)


class FactVerdict(BaseModel):
    index: int = Field(description="필수 사실의 번호 (1부터)")
    present: bool = Field(description="답변이 그 사실을 담고 있으면 true")
    why: str = Field(description="판단 근거 한 문장. 담지 않았다면 무엇이 빠졌는지")


class FactJudgment(BaseModel):
    facts: list[FactVerdict]


JUDGE_SYSTEM = """너는 식품 법령 안내 답변을 채점한다. 판단할 것은 **하나뿐**이다 —
아래 필수 사실 각각을 답변이 **사실로** 담고 있는가.

지켜야 할 것:
1. **표현이 아니라 사실로 본다.** 문장이 달라도, 순서가 달라도, 말을 풀어 썼어도
   같은 사실을 말하면 담은 것이다. 인용 표기(`[FDC-016]`)의 유무는 보지 않는다.
2. 사실에 **수치·명칭이 들어 있으면 그 값이 같아야** 한다. 값이 다르면 담지 않은 것이다.
3. 답변이 그 사실을 **명시적으로 말해야** 한다. 답변에서 추론할 수 있을 뿐이면 담지 않은 것이다.
4. 답변이 더 많은 말을 한 것은 감점 사유가 아니다. 필수 사실이 들어 있는지만 본다.
5. 사실이 「…를 명시한다」·「…를 함께 보여준다」처럼 **말하기를 요구하는 형태**면,
   답변이 실제로 그 말을 했는지 본다.

답변이 옳은지, 법적으로 맞는지는 판단하지 않는다. 담았는지만 본다."""

JUDGE_USER = """질문: {question}

필수 사실:
{facts}

채점할 답변:
{answer}

필수 사실 {n}건 각각에 대해 판단해라."""


def _judge_llm():
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
        FactJudgment, method="json_schema"
    )


def judge_facts(question: str, answer: str, facts: list[dict], llm=None) -> list[dict]:
    """필수 사실이 답변에 들어 있는지 LLM 이 판단한다. 한 문항에 호출 1회."""
    if not facts:
        return []
    listed = "\n".join(f"{i}. {f['fact']}" for i, f in enumerate(facts, 1))
    llm = llm or _judge_llm()
    result: FactJudgment = llm.invoke(
        [
            ("system", JUDGE_SYSTEM),
            (
                "human",
                JUDGE_USER.format(
                    question=question, facts=listed, answer=answer, n=len(facts)
                ),
            ),
        ]
    )
    by_index = {v.index: v for v in result.facts}
    out = []
    for i, f in enumerate(facts, 1):
        v = by_index.get(i)
        out.append(
            {
                "fact": f["fact"],
                "present": bool(v and v.present),
                "why": v.why if v else "채점기가 이 사실을 판단하지 않았다",
            }
        )
    return out


# ────────────────────────────────────────────── S2 금지 표현 — 규칙 층 (D12)


def forbidden_hits(answer: str, phrases: list[str]) -> list[str]:
    """정규화 후 부분 문자열 대조. 부호·띄어쓰기를 바꾼 같은 말도 잡는다 (완료 기준 c)."""
    body = norm(answer)
    return [p for p in phrases if norm(p) and norm(p) in body]


# ─────────────────────────────────────────────────────────── 채점 대상 한 줄


@dataclass
class Row:
    """채점기가 받는 한 문항. 출처(파이프라인 실행·기록·모범 답안)가 달라도 모양은 같다."""

    id: str
    question: str
    tools: list[str]
    answer: str
    escalation: str | None
    evidence: str  # S3 대조용 근거 전체
    evidence_ids: list[str]
    citations: list[str]


def _evidence_for(question: str, names: list[str]) -> tuple[str, list[str]]:
    """근거 조립은 결정적이다 — 질문과 도구 목록만 있으면 그대로 다시 만들 수 있다."""
    results = [tools.CALL[n](question) for n in names]
    ids = [cid for r in results for cid in r.chunk_ids]
    return verify.evidence_text(results), ids


# ───────────────────────────────────────────────────────────────── 채점


@dataclass
class Score:
    id: str
    s1: bool = False
    s2: bool = False
    s3: bool = False
    s4: bool = False
    reasons: list[str] = field(default_factory=list)  # 왜 0점인지 (완료 기준 d)
    expected_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    checked: int = 0
    expected_escalation: str | None = None
    escalation: str | None = None
    citations: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)  # gold 밖 인용 (완료 기준 b)

    @property
    def total(self) -> int:
        return sum((self.s1, self.s2, self.s3, self.s4))

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "S1": int(self.s1),
            "S2": int(self.s2),
            "S3": int(self.s3),
            "S4": int(self.s4),
            "why": self.reasons,
            "expectedTools": self.expected_tools,
            "tools": self.tools,
            "facts": self.facts,
            "forbiddenHits": self.forbidden,
            "violations": self.violations,
            "verifyChecked": self.checked,
            "expectedEscalation": self.expected_escalation,
            "escalation": self.escalation,
            "citations": self.citations,
            "citationsOutsideGold": self.bad_citations,
        }


def score_row(item: dict, row: Row, llm=None) -> Score:
    s = Score(
        id=item["id"],
        expected_tools=list(item["expectedTools"]),
        tools=list(row.tools),
        expected_escalation=item.get("expectedEscalation"),
        escalation=row.escalation,
        citations=list(row.citations),
    )

    # S1 — 집합 비교 (PLAN D2). 순서와 중복은 보지 않는다.
    s.s1 = set(row.tools) == set(item["expectedTools"])
    if not s.s1:
        s.reasons.append(
            f"S1: 호출 {sorted(set(row.tools)) or '[]'} ≠ 기대 {sorted(set(item['expectedTools'])) or '[]'}"
        )

    # S2 — 필수 사실(LLM) AND 금지 표현(규칙)
    s.facts = judge_facts(row.question, row.answer, item.get("requiredFacts", []), llm)
    s.forbidden = forbidden_hits(row.answer, item.get("forbiddenPhrases", []))
    missing = [f for f in s.facts if not f["present"]]
    s.s2 = not missing and not s.forbidden
    for f in missing:
        s.reasons.append(f"S2 필수 사실 누락: {f['fact']} — {f['why']}")
    for p in s.forbidden:
        s.reasons.append(f"S2 금지 표현: 「{p}」")

    # S3 — 슬라이스 6의 검증기를 그대로 재사용한다
    report = verify.check(row.answer, row.evidence, row.question)
    s.violations = report.as_json()
    s.checked = report.checked
    s.s3 = report.ok
    for v in s.violations:
        s.reasons.append(f"S3 근거 밖 {v['kind']}: 「{v['token']}」")

    # S4 — 넘기기 갈래 일치. 넘긴 문항은 문의처까지 봐야 「넘겼다」로 친다
    expected = item.get("expectedEscalation")
    s.s4 = row.escalation == expected
    if not s.s4:
        s.reasons.append(f"S4: 넘기기 {row.escalation or 'None'} (기대 {expected or 'None'})")
    elif expected and "문의처:" not in row.answer:
        s.s4 = False
        s.reasons.append("S4: 넘기기는 맞는데 어디에 물어야 하는지가 없다")

    # 모범 답안 전용 확인 (완료 기준 b) — 채점에는 넣지 않고 따로 보고한다
    gold = set(item.get("gold", []))
    s.bad_citations = [c for c in row.citations if c not in gold]
    return s


# ─────────────────────────────────────────── 채점 대상 만들기 — 세 갈래


def load_items(limit: int | None = None) -> list[dict]:
    data = json.loads(GOLDENSET.read_text(encoding="utf-8"))
    return data["items"][:limit] if limit else data["items"]


def reference_rows(items: list[dict]) -> list[Row]:
    """모범 답안을 파이프라인 출력과 **같은 모양**으로 만든다 (S5).

    넘기기 문항은 최종 본문을 `agent.escalate` 와 똑같이 조립한다 — 머리줄·문의처 줄을
    모범 답안 파일에 베껴 두면 둘이 갈라지고, 그러면 S4 채점이 파이프라인이 아니라
    베낀 문자열을 재게 된다. 파일에는 가운데 본문만 둔다.
    """
    import agent  # 넘기기 문구를 여기서만 쓴다. 채점 자체는 agent 없이 돈다

    refs = {r["id"]: r for r in json.loads(REFERENCE.read_text(encoding="utf-8"))["items"]}
    rows = []
    for item in items:
        ref = refs.get(item["id"])
        if ref is None:
            raise SystemExit(f"모범 답안이 없다: {item['id']}")
        body = ref["text"].strip()
        escalation = item.get("expectedEscalation")
        if escalation:
            text = "\n\n".join(
                x
                for x in (
                    agent.ESCALATION_HEAD[escalation],
                    body,
                    agent.ESCALATION_WHERE[escalation],
                )
                if x
            )
        else:
            text = body
        names = list(item["expectedTools"])
        evidence, ids = _evidence_for(item["question"], names)
        rows.append(
            Row(
                id=item["id"],
                question=item["question"],
                tools=names,
                answer=text,
                escalation=escalation,
                evidence=evidence,
                evidence_ids=ids,
                citations=list(dict.fromkeys(agent.CITE.findall(text))),
            )
        )
    return rows


def record_rows(items: list[dict], path: Path) -> list[Row]:
    """기록된 실행(`agent --goldenset --out`)을 채점한다. 근거는 도구로 다시 만든다."""
    import agent

    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in data["runs"]}
    rows = []
    for item in items:
        rec = by_id.get(item["id"])
        if rec is None:
            raise SystemExit(f"기록에 문항이 없다: {item['id']} ({path})")
        evidence, ids = _evidence_for(rec["question"], rec["tools"])
        if ids != rec["evidenceIds"]:
            print(f"  ! {item['id']}: 근거가 기록과 다르다 — 코퍼스가 바뀌었다")
        rows.append(
            Row(
                id=item["id"],
                question=rec["question"],
                tools=rec["tools"],
                answer=rec["answer"],
                escalation=rec.get("escalation"),
                evidence=evidence,
                evidence_ids=ids,
                citations=list(dict.fromkeys(agent.CITE.findall(rec["answer"]))),
            )
        )
    return rows


def pipeline_rows(items: list[dict]) -> list[Row]:
    """파이프라인을 실제로 돌려 채점 대상을 만든다 (슬라이스 9의 기준 측정이 쓴다)."""
    import agent

    graph = agent.build_graph()
    rows = []
    for i, item in enumerate(items, 1):
        run = agent.ask(item["question"], graph)
        print(
            f"[{i:2}/{len(items)}] {item['id']:6} 도구={','.join(run.tools) or '-':38} "
            f"근거={len(run.evidence_ids):3} {run.seconds:4.1f}s"
        )
        rows.append(
            Row(
                id=item["id"],
                question=item["question"],
                tools=run.tools,
                answer=run.answer,
                escalation=run.escalation,
                evidence=verify.evidence_text(run.results),
                evidence_ids=run.evidence_ids,
                citations=run.citations,
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────── 출력


def report(scores: list[Score], mode: str, out: Path | None) -> dict:
    n = len(scores)
    s1 = sum(s.s1 for s in scores) / n
    s2 = sum(s.s2 for s in scores) / n
    s3_bad = sum(len(s.violations) for s in scores)
    s4_bad = sum(not s.s4 for s in scores)

    print()
    print("─" * 78)
    for s in scores:
        marks = "".join("o" if x else "X" for x in (s.s1, s.s2, s.s3, s.s4))
        print(f"  {s.id:6} S1234={marks}  대조 {s.checked:2}건" + ("" if s.total == 4 else "  ←"))
        for line in s.reasons:
            print(f"         {line}")
        if s.bad_citations:
            print(f"         인용이 gold 밖: {', '.join(s.bad_citations)}")
    print("─" * 78)
    print(f"S1 도구 집합 일치   {s1:.2f}  ({sum(s.s1 for s in scores)}/{n})")
    print(f"S2 답변 적절성      {s2:.2f}  ({sum(s.s2 for s in scores)}/{n})")
    print(f"S3 근거 밖 진술     {s3_bad}건  (문항 {sum(not s.s3 for s in scores)}개)")
    print(f"S4 넘기기           위반 {s4_bad}건")

    summary = {
        "mode": mode,
        "at": time.strftime("%Y-%m-%d %H:%M"),
        "judgeModel": JUDGE_MODEL,
        "items": n,
        "S1": round(s1, 4),
        "S2": round(s2, 4),
        "S3violations": s3_bad,
        "S4violations": s4_bad,
    }
    payload = {**summary, "scores": [s.as_json() for s in scores]}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n기록: {out}")
    return payload


# ───────────────────────────────────────── 규칙 층 자체 확인 (완료 기준 c)
#
# API 키 없이 도는 부분만. 금지 표현 대조가 **정규화 대조**임을 부호·띄어쓰기를 일부러
# 바꾼 표현으로 못박는다 — 이게 깨지면 S2 가 사실 채점이 아니라 표현 채점이 된다.

_SELFTEST_FORBIDDEN: tuple[tuple[str, str, str, bool], ...] = (
    (
        "가운뎃점과 공백을 지운 같은 말",
        "과·채즙 50% 이상",
        "이 제품은 과채즙 50 % 이상이므로 과·채주스입니다.",
        True,
    ),
    (
        "금지 표현 쪽에만 부호가 있는 경우",
        "10포인트 이상",
        "함량을 주표시면에 10 포인트 이상의 글씨로 적으면 됩니다.",
        True,
    ),
    (
        "괄호·따옴표를 섞어 쓴 같은 말",
        "무방부제 표시는 허용됩니다",
        "'무방부제' 표시는 허용됩니다.",
        True,
    ),
    (
        "값이 다르면 걸리지 않는다",
        "과·채즙 50% 이상",
        "과·채주스는 과·채즙 95% 이상이어야 합니다.",
        False,
    ),
    (
        "다른 수치의 포인트 표기는 걸리지 않는다",
        "10포인트 이상",
        "주표시면에 14포인트 이상의 글씨로 표시하여야 합니다.",
        False,
    ),
)

_SELFTEST_SET: tuple[tuple[str, list[str], list[str], bool], ...] = (
    ("순서만 다른 집합", ["lookup_labeling", "lookup_additive"], ["lookup_additive", "lookup_labeling"], True),
    ("하나를 더 고른 과선택", ["lookup_food_type", "lookup_labeling"], ["lookup_food_type"], False),
    ("둘 다 공집합", [], [], True),
    ("빈 집합인데 도구를 불렀다", ["lookup_ad_claims"], [], False),
)


def selftest() -> int:
    failed = 0
    print("금지 표현 — 정규화 대조 (완료 기준 c)")
    for name, phrase, answer, want in _SELFTEST_FORBIDDEN:
        got = bool(forbidden_hits(answer, [phrase]))
        bad = got != want
        print(f"  {'실패' if bad else 'ok  '} [{'걸려야' if want else '통과해야'}] {name:24} 「{phrase}」")
        if bad:
            print(f"        → {'못 잡았다' if want else '엉뚱한 것을 잡았다'}")
        failed += bad

    print("\nS1 — 집합 비교 (PLAN D2)")
    for name, called, expected, want in _SELFTEST_SET:
        got = set(called) == set(expected)
        bad = got != want
        print(f"  {'실패' if bad else 'ok  '} [{'같아야' if want else '달라야'}] {name}")
        failed += bad

    print("\nS4 — 넘기기 라벨과 문의처")
    cases = (
        ("갈래가 다르면 0점", "no_evidence", "out_of_scope", "문의처: …", False),
        ("문의처가 없으면 0점", "out_of_scope", "out_of_scope", "답하지 않습니다.", False),
        ("갈래도 문의처도 맞다", "out_of_scope", "out_of_scope", "넘깁니다.\n\n문의처: …", True),
        ("멀쩡한 문항은 None 이어야 한다", None, None, "과·채음료입니다.", True),
    )
    for name, got_esc, want_esc, answer, want in cases:
        item = {"id": "X", "expectedTools": [], "expectedEscalation": want_esc,
                "requiredFacts": [], "forbiddenPhrases": [], "gold": []}
        row = Row("X", "q", [], answer, got_esc, "", [], [])
        s = score_row(item, row, llm=None)
        bad = s.s4 != want
        print(f"  {'실패' if bad else 'ok  '} [{'1점' if want else '0점'}] {name}")
        failed += bad

    print(f"\n자체 확인 실패 {failed}")
    return 0 if not failed else 1


# ──────────────────────────────────────────────── 역대조 — 망친 답변 (S5 의 짝)
#
# 모범 답안을 **한 군데씩** 망가뜨려 넣고 의도한 지표가 0점이 되는지 본다. 전부 1점을 주는
# 채점기도 `--reference` 는 통과하므로, 가르는 힘은 여기서만 확인된다.
#
# `LB-2` 가 S3 가 아니라 S2 에서만 걸리는 것이 이 확인의 값이다 — 「14포인트」를 「12포인트」로
# 바꿔도 12포인트가 표시기준의 **다른 조문**(별지 1 바. 글씨크기)에 실재해서 근거 대조를 통과한다.
# `docs/VERIFY-NOTES.md` 가 적어 둔 규칙의 한계가 채점기에서도 같은 모양으로 나타난다.
# S3 는 「근거 밖에서 왔는가」만 재고, 「엉뚱한 조문에서 왔는가」는 S2 의 필수 사실이 잡는다.

_NEGATIVE: tuple[tuple[str, str, dict, str], ...] = (
    (
        "FT-1",
        "필수 사실 하나를 통째로 뺐다 (95% 경계 없음)",
        {
            "answer": "유자즙 28%라면 식품유형은 과·채음료입니다 [FDC-016]. "
            "과일즙·채소즙 또는 과·채즙 10% 이상이면 과·채음료 정의에 들어갑니다 [FDC-016]."
        },
        "S2",
    ),
    (
        "LB-2",
        "수치를 바꿨다 (14포인트 → 12포인트). S3 는 못 잡는다 — 12포인트는 다른 조문에 실재한다",
        {
            "answer": "원재료명을 제품명의 일부로 쓰면 그 함량을 주표시면에 12포인트 이상의 "
            "글씨로 표시하여야 합니다 [LBL-F1-003]. 제품명 글씨크기가 22포인트 미만이면 "
            "7포인트 이상으로 표시할 수 있습니다 [LBL-F1-003]."
        },
        "S2",
    ),
    (
        "AD-1",
        "없는 한도를 지어냈다 (금지 표현 + 근거 밖 수치)",
        {
            "answer": "비타민C는 음료류에 1.0 g/kg 이하로 사용할 수 있습니다 [FAC-167]. "
            "식품첨가물은 필요한 최소량으로 사용하여야 합니다 [FAC-001]."
        },
        "S2+S3",
    ),
    (
        "FT-3",
        "조건부로 답하지 않고 되물었다 (PLAN D9)",
        {"answer": "제품 정보를 알려주시면 다시 답변드리겠습니다."},
        "S2",
    ),
    ("MX-2", "두 도구 중 하나만 불렀다", {"tools": ["lookup_food_type"]}, "S1"),
    (
        "FT-4",
        "넘기지 않고 가부를 단정했다",
        {
            "answer": "유자 발효추출물은 식품원료로 사용할 수 있습니다 [FDC-060].",
            "escalation": None,
        },
        "S2+S4",
    ),
    (
        "OS-1",
        "소관 밖인데 도구를 부르고 답했다",
        {
            "answer": "HACCP 인증은 선행요건 관리기준을 작성하여 신청합니다.",
            "escalation": None,
            "tools": ["lookup_labeling"],
        },
        "S1+S2+S4",
    ),
)


def negative() -> int:
    items = {it["id"]: it for it in load_items()}
    rows = {r.id: r for r in reference_rows(list(items.values()))}
    llm = _judge_llm()

    failed = 0
    for item_id, what, patch, want_zero in _NEGATIVE:
        row = rows[item_id]
        broken = Row(
            id=row.id,
            question=row.question,
            tools=patch.get("tools", row.tools),
            answer=patch.get("answer", row.answer),
            escalation=patch["escalation"] if "escalation" in patch else row.escalation,
            evidence=row.evidence,
            evidence_ids=row.evidence_ids,
            citations=row.citations,
        )
        s = score_row(items[item_id], broken, llm)
        zeros = [n for n, ok in (("S1", s.s1), ("S2", s.s2), ("S3", s.s3), ("S4", s.s4)) if not ok]
        bad = not set(want_zero.split("+")) <= set(zeros)
        failed += bad
        print(f"  {'실패' if bad else 'ok  '} {item_id:6} {what}")
        print(f"         0점: {'+'.join(zeros) or '(없음)'}  기대 {want_zero}")
        for line in s.reasons:
            print(f"         · {line}")

    print(f"\n역대조 {len(_NEGATIVE)}건 · 의도한 지표를 놓친 것 {failed}")
    return 0 if not failed else 1


# ────────────────────────────────────────────────────────────────── 실행


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="evaluate", description="S1~S4 채점기")
    ap.add_argument("--selftest", action="store_true", help="규칙 층 자체 확인 (API 키 불필요)")
    ap.add_argument("--reference", action="store_true", help="모범 답안 채점 = S5 역검증")
    ap.add_argument("--negative", action="store_true", help="망친 답변이 0점인지 역대조")
    ap.add_argument("--score", type=Path, help="기록된 실행 JSON 을 채점")
    ap.add_argument("--out", type=Path, help="채점 결과를 JSON 으로 저장")
    ap.add_argument("--limit", type=int, help="앞에서 N 문항만")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.negative:
        return negative()

    items = load_items(args.limit)
    if args.reference:
        mode, rows = "reference", reference_rows(items)
    elif args.score:
        mode, rows = f"record:{args.score.name}", record_rows(items, args.score)
    else:
        mode, rows = "pipeline", pipeline_rows(items)

    llm = _judge_llm()
    scores = [score_row(item, row, llm) for item, row in zip(items, rows, strict=True)]
    report(scores, mode, args.out)

    if mode != "reference":
        return 0

    # S5 — 모범 답안은 전 문항 4점이어야 하고 gold 밖을 인용하면 안 된다 (완료 기준 a·b)
    short = [s for s in scores if s.total < 4]
    strayed = [s for s in scores if s.bad_citations]
    print()
    if not short and not strayed:
        print(f"S5 통과 — 모범 답안 {len(scores)}문항이 전부 S1~S4 1점, gold 밖 인용 0건")
        return 0
    if short:
        print(f"S5 실패 — 1점이 아닌 문항 {len(short)}건: {', '.join(s.id for s in short)}")
    if strayed:
        print(f"S5 실패 — gold 밖을 인용한 문항 {len(strayed)}건: {', '.join(s.id for s in strayed)}")
    print("  → 고칠 곳은 채점기다. 모범 답안을 채점기에 맞춰 고치면 역검증이 무의미해진다.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
