"""
환각 검증 — 답변에 나온 **수치 · 조문 번호 · 식품유형명**이 주입된 근거에 실제로 있는지
문자열로 대조한다 (PLAN D4·D11).

LLM 판정기를 쓰지 않는 이유는 S3(「근거 밖 진술 0건」)가 **측정 가능해야** 하기 때문이다.
같은 입력에 다른 답이 나오면 잴 수 없다. 그래서 여기 있는 것은 전부 정규식과 문자열 대조다.

대조는 `tools.norm` 으로 부호·공백을 지운 뒤 **주입된 근거 전체**(`ToolResult.text` 를 이은 것)에
건다. 청크 하나가 아니라 전체에 거는 이유는 답변이 두 청크를 합쳐 한 문장을 쓰는 것이 정상이라서다.

    uv run python verify.py --selftest                     # 규칙 자체 확인 (API 키 불필요)
    uv run python verify.py --replay data/runs/slice5.json # 기록된 답변에 규칙을 걸어 본다
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import tools
from tools import ToolResult, norm

HERE = Path(__file__).parent


# ────────────────────────────────────────────────────── 대조용 정규화
#
# `tools.norm` 은 부호와 공백을 지운다. 여기서는 그 위에 **단위 표기 흔들림**만 더 없앤다 —
# 공전은 `㎎`·`㎖`(합자)를 쓰고 답변은 `mg`·`mL` 로 옮겨 적는다. 같은 값이 다른 글자가 되면
# 있는 근거를 없다고 잡는 오탐이 된다. 양쪽에 **똑같이** 건다.

_UNIT_ALIAS = [
    ("퍼센트", "%"),
    ("㎎", "mg"),
    ("㎍", "ug"),
    ("㎖", "ml"),
    ("㎏", "kg"),
    ("㏄", "cc"),
    ("포인트", "pt"),
]


def canon(s: str) -> str:
    out = norm(s)
    for src, dst in _UNIT_ALIAS:
        out = out.replace(src, dst)
    return out


# ──────────────────────────────────────────────────── 뽑기 전에 가리는 것
#
# PLAN D11 의 면책 목록. 여기 걸리는 자리는 뽑지 않는다 — 뽑으면 전부 오탐이다.

# (1) 인용 표기 자체. `[LBL-T6-003]` 의 숫자는 근거 내용이 아니라 청크 이름이다.
_CITE = re.compile(r"\[[A-Z]{3}(?:-[A-Z0-9]+)+\]")

# (2) 근거 블록 머리줄의 시행일. 답변이 `(시행 2026-01-01)`·`2026년 1월 1일 시행` 으로 옮겨 적으면
#     「1일」이 수치로 잡힌다. 시행일은 근거의 내용이 아니라 꼬리표다.
_EFFECTIVE = re.compile(
    r"\(?\s*시행\s*\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]\s*\d{1,2}\s*[일.]?\s*\)?"
    r"|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\s*시행"
)

_MASK = "　"  # 가린 자리. 낱말이 이어 붙지 않게 공백류 한 글자로 바꾼다


def _mask(text: str) -> str:
    text = _CITE.sub(_MASK, text)
    return _EFFECTIVE.sub(_MASK, text)


# ───────────────────────────────────────────────────────────── 뽑는 규칙

# 수치 — 단위가 붙은 것만. 단위 없는 맨 숫자는 항·호 번호와 구분되지 않는다.
_NUMBER = re.compile(
    r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|퍼센트|㎎|㎍|㎖|㎏|㏄|mg|ug|mL|ml|kg|g|L|배|일|개월|pt|포인트)"
    r"(?![A-Za-z가-힣])"
)

# 조문 번호 — 「제12조의2제1항제3호」까지 한 덩어리로. 별표·별지·별도도 같은 층위다.
_ARTICLE = re.compile(
    r"제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
    r"|별[표지도]\s*\d+(?:\s*의\s*\d+)?"
)

# 식품유형명 — 식품공전 제5. 9. 음료류의 고정 사전. 코퍼스에서 읽지 않고 여기 박아 둔다.
# 근거가 바뀌어도 「없는 유형명을 지어냈다」의 기준은 흔들리면 안 된다.
FOOD_TYPES: tuple[str, ...] = (
    # 식품군
    "다류",
    "커피",
    "과일·채소류음료",
    "탄산음료류",
    "두유류",
    "발효음료류",
    "인삼·홍삼음료",
    "기타음료",
    # 식품유형
    "침출차",
    "액상차",
    "고형차",
    "농축과·채즙",
    "과·채분",
    "과·채주스",
    "과·채음료",
    "탄산음료",
    "탄산수",
    "원액두유",
    "가공두유",
    "유산균음료",
    "효모음료",
    "기타발효음료",
    "혼합음료",
    "음료베이스",
)

# 긴 이름부터 본다 — 「탄산음료류」를 「탄산음료」로 잘라 세면 안 된다.
_TYPES_SORTED = tuple(sorted(FOOD_TYPES, key=lambda t: -len(canon(t))))


@dataclass(frozen=True)
class Violation:
    kind: str  # 수치 | 조문 | 유형
    token: str  # 답변에 나온 표기 그대로
    key: str  # 대조에 쓴 정규화 형태

    def __str__(self) -> str:
        return f"{self.kind}「{self.token}」"


@dataclass
class Report:
    violations: list[Violation]
    checked: int  # 뽑아서 대조한 토큰 수

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.ok:
            return f"통과 (대조 {self.checked}건)"
        return f"위반 {len(self.violations)}건 / 대조 {self.checked}건 — " + ", ".join(
            str(v) for v in self.violations
        )

    def as_json(self) -> list[dict]:
        return [{"kind": v.kind, "token": v.token} for v in self.violations]


def _tokens(answer: str) -> list[tuple[str, str]]:
    """답변에서 대조할 것을 (종류, 표기) 로 뽑는다. 가릴 자리는 이미 가려진 문자열을 받는다."""
    found: list[tuple[str, str]] = []
    for m in _NUMBER.finditer(answer):
        found.append(("수치", m.group(0).strip()))
    for m in _ARTICLE.finditer(answer):
        found.append(("조문", m.group(0).strip()))

    # 유형명은 겹쳐 세지 않는다. 찾은 자리를 지워 가며 긴 것부터 훑는다.
    canon_answer = canon(answer)
    for name in _TYPES_SORTED:
        key = canon(name)
        if key and key in canon_answer:
            found.append(("유형", name))
            canon_answer = canon_answer.replace(key, _MASK)
    return found


def evidence_text(results: list[ToolResult]) -> str:
    return "\n\n".join(r.text for r in results)


def check(answer: str, evidence: str, question: str = "") -> Report:
    """답변의 수치·조문·유형명이 근거에 있는지 본다. 질문에 이미 있던 말은 면책한다."""
    haystack = canon(evidence)
    asked = canon(question)
    masked = _mask(answer)

    violations: list[Violation] = []
    seen: set[str] = set()
    checked = 0
    for kind, token in _tokens(masked):
        key = canon(token)
        if not key or key in seen:
            continue
        seen.add(key)
        checked += 1
        if key in haystack:
            continue
        if key in asked:  # 「유자즙 28%」를 되풀이한 것 — 지어낸 것이 아니다
            continue
        violations.append(Violation(kind=kind, token=token, key=key))
    return Report(violations=violations, checked=checked)


# ─────────────────────────────────────────── 재생성 프롬프트 (슬라이스 6 b·d)

REGEN_NOTE = """방금 쓴 답변이 근거 대조에서 걸렸다. 아래 표기는 **주어진 근거에 없다**.

{items}

걸린 표기만 고친다 — 근거에 있는 표기로 바꾸거나, 근거가 그 사실을 담고 있지 않으면 그 문장을 지운다.
**근거에 있는 나머지 내용은 그대로 살려라.** 지울 것을 지우고 남는 것이 없을 때만 모른다고 말하고 넘긴다.
새로 지어내지 마라. 고쳐 쓴 답변 전체를 다시 내라."""


def regen_note(report: Report) -> str:
    items = "\n".join(f"- {v.kind}: {v.token}" for v in report.violations)
    return REGEN_NOTE.format(items=items)


# ───────────────────────────────────────────────────────── 자체 확인 (a)
#
# 규칙이 **무엇을 잡고 무엇을 통과시켜야 하는지**를 자작 답변 6종으로 못박는다.
# 근거는 실제 코퍼스에서 도구로 뽑는다 — 규칙만 맞고 실제 근거에는 안 맞는 일이 없게.

_FT_QUESTION = "유자즙 28% 들어간 음료인데 제품명을 '유자주스'로 해도 되나요?"


@dataclass(frozen=True)
class Case:
    name: str
    should_catch: bool
    question: str
    tools: tuple[str, ...]
    answer: str
    expect: tuple[str, ...] = ()  # 잡아야 하는 표기 (should_catch 일 때)


SELFTEST: tuple[Case, ...] = (
    Case(
        name="함량 기준값 바꾸기",
        should_catch=True,
        question=_FT_QUESTION,
        tools=("lookup_food_type",),
        # 실제 경계는 95%(과·채주스)와 10%(과·채음료) 둘뿐이다.
        answer="과·채주스는 과·채즙 90% 이상이어야 한다 [FDC-016].",
        expect=("90%",),
    ),
    Case(
        name="근거에 없는 조문 번호",
        should_catch=True,
        question=_FT_QUESTION,
        tools=("lookup_food_type",),
        answer="식품공전 제37조제2항에 따라 과·채음료로 표시한다 [FDC-016].",
        expect=("제37조제2항",),
    ),
    Case(
        name="근거에 없는 식품유형명",
        should_catch=True,
        question="구연산을 넣었는데 원재료명에는 뭐라고 적나요?",
        tools=("lookup_additive",),
        # 유형명 자체는 실재하지만 이 근거(첨가물 사용기준·표시 별표)에는 없다.
        # 유형 판정은 `lookup_food_type` 이 근거를 가져와야 할 수 있는 말이다.
        answer="구연산이 든 음료는 과·채주스로 분류되므로 그대로 적는다 [LBL-T4-001].",
        expect=("과·채주스",),
    ),
    Case(
        name="사용량 한도 바꾸기",
        should_catch=True,
        question="비타민C를 산화방지 목적으로 넣었는데 최대 얼마까지 쓸 수 있나요?",
        tools=("lookup_additive",),
        answer="비타민C는 음료류에 최대 2.4g/kg 까지 쓸 수 있다 [FAC-001].",
        expect=("2.4g",),
    ),
    Case(
        name="질문의 숫자를 되풀이한 답변",
        should_catch=False,
        question=_FT_QUESTION,
        tools=("lookup_food_type",),
        # 「28%」는 근거에 없다. 사용자가 말한 배합 비율이 법령 본문에 있을 리가 없다.
        answer="유자즙 28% 이면 과·채즙 10% 이상이므로 과·채음료다 [FDC-016].",
    ),
    Case(
        name="근거를 그대로 옮긴 답변",
        should_catch=False,
        question=_FT_QUESTION,
        tools=("lookup_food_type",),
        answer=(
            "과·채주스는 과·채즙 95% 이상인 것을 말하고, 과·채음료는 과·채즙 10% 이상인 것을 "
            "말한다 [FDC-016]. (시행 2026-01-01)"
        ),
    ),
)


def _evidence_for(case: Case) -> str:
    return evidence_text([tools.CALL[name](case.question) for name in case.tools])


def selftest() -> int:
    failed = 0
    for case in SELFTEST:
        report = check(case.answer, _evidence_for(case), case.question)
        want = "잡아야" if case.should_catch else "통과해야"
        bad = []
        if case.should_catch:
            if report.ok:
                bad.append("아무것도 못 잡았다")
            else:
                got = {v.key for v in report.violations}
                for token in case.expect:
                    if canon(token) not in got:
                        bad.append(f"「{token}」을 못 잡았다")
        elif not report.ok:
            bad.append("있는 근거를 없다고 잡았다 (오탐)")

        mark = "실패" if bad else "ok  "
        print(f"  {mark} [{want}] {case.name:22} {report.summary()}")
        for line in bad:
            print(f"        → {line}")
        failed += bool(bad)

    print(f"\n자체 확인 {len(SELFTEST)}건 · 실패 {failed}")
    return 0 if not failed else 1


# ─────────────────────────────────────────── 기록된 답변에 규칙 걸어 보기 (c)
#
# 완료 기준 (c)「18문항에서 오탐 0건」을 LLM 호출 없이 재는 자리. 근거 조립은 결정적이라
# 질문과 도구 목록만 있으면 그대로 다시 만들 수 있다.


def replay(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["runs"]
    total = 0
    flagged = []
    for row in rows:
        results = [tools.CALL[name](row["question"]) for name in row["tools"]]
        ids = [cid for r in results for cid in r.chunk_ids]
        drift = "" if ids == row["evidenceIds"] else "  ! 근거가 기록과 다르다"
        report = check(row["answer"], evidence_text(results), row["question"])
        total += report.checked
        print(f"  {row['id']:6} {report.summary()}{drift}")
        if not report.ok:
            flagged.append(row["id"])
    print(f"\n문항 {len(rows)} · 대조 {total}건 · 위반이 잡힌 문항 {len(flagged)}")
    if flagged:
        print("  ", ", ".join(flagged))
        print("  → 각 건이 실제 위반인지 오탐인지 눈으로 가른다 (docs/VERIFY-NOTES.md)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verify", description="환각 검증 규칙")
    ap.add_argument("--selftest", action="store_true", help="자작 답변 6종으로 규칙 확인")
    ap.add_argument("--replay", type=Path, help="기록된 실행(JSON)의 답변에 규칙을 건다")
    args = ap.parse_args(argv)

    if args.replay:
        return replay(args.replay)
    if args.selftest:
        return selftest()
    ap.error("--selftest 나 --replay 를 써라")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
