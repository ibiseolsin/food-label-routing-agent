"""
평가셋 검증기. `uv run python validate_goldenset.py` — 실패하면 종료 코드 1.

슬라이스 1·2·3 과 같은 규약이다: 완료 기준을 사람이 눈으로 세지 않고 스크립트가 확인한다.
여기서 보는 것은 다섯 가지.

  스키마    필드가 다 있고 타입이 맞는가, 기대 도구 이름이 실재하는가
  분포      문항 수 · 카테고리별 균형 · 넘기기 · 복수 카테고리 · 조건부 답변 문항
  중복      id · 질문 중복, 그리고 **프롬프트 예시와 goldenset 이 겹치지 않는가**
  goldCheck 정규식이 gold 청크 본문에 실제로 맞는가 (코퍼스를 다시 수집하면 여기서 깨진다)
  금지 표현 그럴듯한 오답이 **근거 안에 없는가** — 근거에 있으면 그건 오답이 아니라 정답이다

마지막 하나가 이 검증기의 핵심이다. 금지 표현을 눈대중으로 쓰면 원문에 실제로 있는 문장을
금지해 버린다. 부호·공백을 지우고(`tools.norm`) 대조하므로 「과･채즙」·「과·채즙」 표기 차이는
넘어가고 숫자만 다른 것은 걸린다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from corpus import load
from tools import TOOLS, norm

HERE = Path(__file__).parent
GOLDENSET = HERE / "data" / "goldenset.json"
EXAMPLES = HERE / "data" / "prompt-examples.json"

TOOL_NAMES = set(TOOLS)
ESCALATIONS = {None, "out_of_scope", "no_evidence"}

MIN_ITEMS = 12
MIN_ESCALATION = 3
MAX_MULTI = 2  # PLAN D2 — 복수 카테고리 문항은 2건까지
MAX_CATEGORY_GAP = 1

ITEM_FIELDS = {
    "id": str,
    "category": str,
    "usecase": str,
    "question": str,
    "expectedTools": list,
    "requiredFacts": list,
    "forbiddenPhrases": list,
    "gold": list,
    "goldCheck": list,
    "note": str,
}


def check_schema(items: list[dict], fail) -> None:
    for item in items:
        where = item.get("id", "<id 없음>")
        for field, typ in ITEM_FIELDS.items():
            if field not in item:
                fail(f"{where}: 필드 `{field}` 가 없다")
            elif not isinstance(item[field], typ):
                fail(f"{where}: `{field}` 가 {typ.__name__} 이 아니다")

        if item.get("expectedEscalation", "<없음>") not in ESCALATIONS:
            fail(f"{where}: expectedEscalation 이 {sorted(str(e) for e in ESCALATIONS)} 밖이다")

        for name in item.get("expectedTools", []):
            if name not in TOOL_NAMES:
                fail(f"{where}: 기대 도구 `{name}` 은 tools.py 에 없다")

        for fact in item.get("requiredFacts", []):
            for field in ("fact", "source", "where"):
                if field not in fact:
                    fail(f"{where}: requiredFacts 항목에 `{field}` 가 없다")
            src = fact.get("source")
            if src is not None and src not in item.get("gold", []):
                fail(f"{where}: 필수 사실의 근거 `{src}` 가 gold 목록에 없다")

        if not item.get("requiredFacts"):
            fail(f"{where}: 필수 사실이 비었다")
        if not item.get("forbiddenPhrases"):
            fail(f"{where}: 금지 표현이 비었다")


def check_escalation_shape(items: list[dict], fail) -> None:
    """PLAN D3 — 넘기기 두 갈래는 기대 도구 집합의 모양이 다르다."""
    for item in items:
        kind, tools = item.get("expectedEscalation"), item.get("expectedTools", [])
        if kind == "out_of_scope":
            if tools:
                fail(f"{item['id']}: out_of_scope 인데 기대 도구가 있다 — 공집합이어야 한다")
            if item.get("gold"):
                fail(f"{item['id']}: out_of_scope 인데 gold 청크가 있다")
        elif kind == "no_evidence":
            if len(tools) != 1:
                fail(f"{item['id']}: no_evidence 는 기대 도구가 1개여야 한다 (지금 {len(tools)}개)")
        else:
            if not tools:
                fail(f"{item['id']}: 넘기기가 아닌데 기대 도구가 비었다")


def check_distribution(items: list[dict], fail) -> dict[str, int]:
    if len(items) < MIN_ITEMS:
        fail(f"문항이 {len(items)}건 — {MIN_ITEMS}건 이상이어야 한다")

    # 카테고리 균형은 **도구별 등장 횟수**로 센다. 복수 카테고리 문항은 두 도구에 모두 센다 —
    # 채점이 집합 비교라 어느 도구든 그 문항으로 시험되기 때문이다.
    per_tool = {name: 0 for name in TOOL_NAMES}
    for item in items:
        for name in item.get("expectedTools", []):
            per_tool[name] += 1

    if min(per_tool.values()) == 0:
        fail(f"문항이 하나도 없는 도구가 있다: {[k for k, v in per_tool.items() if v == 0]}")
    gap = max(per_tool.values()) - min(per_tool.values())
    if gap > MAX_CATEGORY_GAP:
        fail(f"카테고리별 개수 차이가 {gap}건 — {MAX_CATEGORY_GAP}건 이내여야 한다 ({per_tool})")

    kinds = [i.get("expectedEscalation") for i in items]
    n_esc = sum(1 for k in kinds if k)
    if n_esc < MIN_ESCALATION:
        fail(f"넘기기 정답이 {n_esc}건 — {MIN_ESCALATION}건 이상이어야 한다")
    for kind in ("out_of_scope", "no_evidence"):
        if kinds.count(kind) < 1:
            fail(f"`{kind}` 문항이 없다 — 각각 1건 이상이어야 한다")

    n_multi = sum(1 for i in items if len(i.get("expectedTools", [])) > 1)
    if n_multi != MAX_MULTI:
        fail(f"복수 카테고리 문항이 {n_multi}건 — {MAX_MULTI}건이어야 한다 (PLAN D2)")

    n_cond = sum(1 for i in items if i.get("conditionalAnswer"))
    if n_cond < 1:
        fail("판정 정보가 빠진 조건부 답변 문항이 없다 (PLAN D9 · 완료 기준 b-2)")

    return per_tool


def check_duplicates(items: list[dict], examples: list[dict], fail) -> None:
    def dupes(values):
        seen, out = set(), []
        for v in values:
            if v in seen:
                out.append(v)
            seen.add(v)
        return out

    for dup in dupes([i.get("id") for i in items]):
        fail(f"id 중복: {dup}")
    for dup in dupes([norm(i.get("question", "")) for i in items]):
        fail(f"질문 중복: {dup}")

    gold_q = {norm(i.get("question", "")) for i in items}
    for ex in examples:
        if norm(ex.get("question", "")) in gold_q:
            fail(f"프롬프트 예시 {ex.get('id')} 의 질문이 goldenset 과 겹친다")
    for dup in dupes([i.get("id") for i in examples]):
        fail(f"프롬프트 예시 id 중복: {dup}")


def check_gold(items: list[dict], fail) -> None:
    by_id = load().by_id
    for item in items:
        where = item["id"]
        golds = []
        for cid in item.get("gold", []):
            chunk = by_id.get(cid)
            if chunk is None:
                fail(f"{where}: gold 청크 `{cid}` 가 코퍼스에 없다")
            else:
                golds.append(chunk)

        if item.get("gold") and not item.get("goldCheck"):
            fail(f"{where}: gold 가 있는데 goldCheck 정규식이 없다")

        # **위치도 근거의 일부다.** `tools._render` 가 청크마다 `path` 를 본문과 함께 넘기고,
        # 별표는 규칙이 표 제목에 있다 — 「별표 6 명칭, 간략명 또는 주용도를 표시하여야 하는
        # 식품첨가물」. 본문만 대조하면 표 제목에 실린 규칙을 근거로 인정하지 못한다.
        body = "\n".join(f"{c.path}\n{c.text}" for c in golds)
        for pattern in item.get("goldCheck", []):
            try:
                rx = re.compile(pattern)
            except re.error as e:
                fail(f"{where}: goldCheck 정규식 오류 `{pattern}` — {e}")
                continue
            if not rx.search(body):
                fail(f"{where}: goldCheck `{pattern}` 가 gold 본문에 맞지 않는다")

        normed = norm(body)
        for phrase in item.get("forbiddenPhrases", []):
            if normed and norm(phrase) in normed:
                fail(f"{where}: 금지 표현 `{phrase}` 가 gold 근거 안에 실제로 있다")


def report_reachability(items: list[dict]) -> None:
    """기대 도구가 gold 청크를 실제로 물어 오는가 — **보고만 하고 실패시키지 않는다.**

    이건 평가셋의 결함이 아니라 발췌 규칙의 성능이고, 고치는 자리는 슬라이스 9(개선 3회)다.
    여기서 막으면 검색 튜닝이 평가셋 커밋을 잡아 두게 된다. 대신 지금 수치를 남겨 두어
    슬라이스 9 가 비교할 기준선으로 쓴다.
    """
    from tools import CALL

    total = hit = 0
    misses: list[str] = []
    for item in items:
        if not item["gold"]:
            continue
        got: set[str] = set()
        for name in item["expectedTools"]:
            got |= set(CALL[name](item["question"]).chunk_ids)
        missed = [g for g in item["gold"] if g not in got]
        total += len(item["gold"])
        hit += len(item["gold"]) - len(missed)
        if missed:
            misses.append(f"{item['id']}: {', '.join(missed)}")

    print(f"\n■ gold 도달률 (참고 — 슬라이스 9 기준선) {hit}/{total}")
    for line in misses:
        print(f"  못 물어 온 것 — {line}")


def main() -> int:
    failures: list[str] = []
    fail = failures.append

    data = json.loads(GOLDENSET.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))["items"]
    items = data["items"]

    check_schema(items, fail)
    check_escalation_shape(items, fail)
    per_tool = check_distribution(items, fail)
    check_duplicates(items, examples, fail)
    check_gold(items, fail)

    print(f"■ 평가셋 {GOLDENSET.relative_to(HERE)} — 문항 {len(items)}건")
    for name in sorted(TOOL_NAMES):
        print(f"  {name:20} {per_tool[name]:>3}건")
    kinds = [i.get("expectedEscalation") for i in items]
    print(
        f"  {'넘기기':20} {sum(1 for k in kinds if k):>3}건 "
        f"(out_of_scope {kinds.count('out_of_scope')} · no_evidence {kinds.count('no_evidence')})"
    )
    print(f"  {'복수 카테고리':17} {sum(1 for i in items if len(i['expectedTools']) > 1):>3}건")
    print(f"  {'조건부 답변':18} {sum(1 for i in items if i.get('conditionalAnswer')):>3}건")
    print(f"  {'필수 사실':19} {sum(len(i['requiredFacts']) for i in items):>3}개")
    print(f"  {'금지 표현':19} {sum(len(i['forbiddenPhrases']) for i in items):>3}개")
    print(f"  {'gold 청크':19} {len({c for i in items for c in i['gold']}):>3}개")
    print(f"■ 프롬프트 예시 {EXAMPLES.relative_to(HERE)} — {len(examples)}건 (채점 제외)")

    report_reachability(items)

    if failures:
        print(f"\n✗ 실패 {len(failures)}건")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n✓ 스키마 · 분포 · 중복 · goldCheck · 금지 표현 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
