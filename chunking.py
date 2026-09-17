"""
법령·고시 본문을 **사실 단위 청크**로 자르는 규칙.

`m05-rag-chatbot-service/app/scripts/fetch-corpus.mjs` 의 파싱 규칙을 Python 으로 옮긴 것이다
(PLAN.md 슬라이스 1). 코드가 아니라 규칙을 옮겼다 — 언어가 다르고, 여기서는 벡터 검색이
아니라 섹션 인덱스 + 키워드 매칭으로 쓴다 (PLAN D1).

슬라이스 2(식품공전·첨가물공전)도 이 모듈을 그대로 쓴다.
"""

from __future__ import annotations

import re

# ────────────────────────────────────────────────────────────── 공통

def squash(s: object) -> str:
    """연속 공백을 하나로 접는다."""
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip()


def as_list(v: object) -> list:
    """API 는 원소가 하나면 배열 대신 객체를 준다."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def iso_date(yyyymmdd: object) -> str | None:
    s = str(yyyymmdd if yyyymmdd is not None else "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if re.fullmatch(r"\d{8}", s) else None


def url_name(name: object) -> str:
    """law.go.kr 인용 URL 은 법령명에서 공백을 뺀 형태여야 200 이 온다 (m05 실측)."""
    return re.sub(r"\s+", "", str(name if name is not None else ""))


# 순번 글자. 마커 판별은 공백이 아니라 **순번**으로 한다 —
# 고시 원문은 마커가 앞 단어에 붙어서 온다(`식품가.` `떡류1)`). 공백을 요구하면 계층을
# 통째로 놓치고, 조건을 풀면 「열량 표시 제외)」의 `외)` 가 마커가 된다.
KO_ORD = list(
    "가나다라마바사아자차카타파하"
    "거너더러머버서어저처커터퍼허"
    "고노도로모보소오조초코토포호"
)


def ko_ord(ch: str) -> int:
    """순번 글자가 아니면 0."""
    return KO_ORD.index(ch) + 1 if ch in KO_ORD else 0


# ────────────────────────────────────────────────── 긴 본문 재분할

MAX_CHARS = 1200
MIN_CHARS = 250   # 이보다 짧은 조각은 형제와 붙인다
HEAD_MAX = 300    # 이보다 긴 머리말은 맥락이 아니라 그 자체로 하나의 청크다

# 하위 표기 계층. 여는 괄호는 배제해야 한다 — `(3)` 안의 `3)` 이 형제로 잡히면 위치가 한 칸 틀린다.
LEVELS: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"(?<![제항호조\d])(\d+)\.\s"), lambda m: int(m.group(1))),
    (re.compile(r"([가-힣])\.\s"), lambda m: ko_ord(m.group(1))),
    (re.compile(r"(?<!\()(\d+)\)\s?"), lambda m: int(m.group(1))),
    (re.compile(r"(?<!\()([가-힣])\)\s?"), lambda m: ko_ord(m.group(1))),
    (re.compile(r"\((\d+)\)\s?"), lambda m: int(m.group(1))),
]


def marker_cuts(text: str, level) -> list[int]:
    """이 계층에서 1 부터 순서대로 오르는 마커의 위치만 돌려준다.

    `외` 는 순번 글자가 아니고, 날짜 「2025. 8. 29.」의 `8.` 은 1 로 시작하지 않는다.
    항목 2 안에 든 `1)` 도 기대값과 달라 형제로 오인되지 않는다.
    본문 중간에 낀 인용(「[별표 2] Ⅰ. 1. 가.」의 `1.`)도 순번이 안 맞아 걸러진다.
    """
    pattern, ord_of = level
    cuts: list[int] = []
    expect = 1
    for m in pattern.finditer(text):
        if ord_of(m) != expect:
            continue
        cuts.append(m.start())
        expect += 1
    return cuts


def _cut_at(text: str, level):
    """마커 계층 하나로 자른다. 항목이 둘 미만이면 나눌 의미가 없으므로 None."""
    at = marker_cuts(text, level)
    if len(at) < 2:
        return None
    items = []
    for i, start in enumerate(at):
        stop = at[i + 1] if i + 1 < len(at) else len(text)
        piece = text[start:stop].strip()
        if piece:
            items.append(piece)
    return text[: at[0]].strip(), items


def _pack(items: list[str], budget: int) -> list[str]:
    """연속 조각을 MIN_CHARS 이상 budget 이하로 묶는다. 목록 항목들은 원래 같이 읽는 글이다."""
    out: list[str] = []
    for it in items:
        if out and len(out[-1]) < MIN_CHARS and len(out[-1]) + 1 + len(it) <= budget:
            out[-1] = f"{out[-1]} {it}"
        else:
            out.append(it)
    # 꼬리 조각이 홀로 남으면 앞 묶음에 붙인다
    if len(out) > 1 and len(out[-1]) < MIN_CHARS:
        tail = out.pop()
        if len(out[-1]) + 1 + len(tail) <= budget:
            out[-1] += f" {tail}"
        else:
            out.append(tail)
    return out


def split_long(text: str, prefix: str = "", depth: int = 0) -> list[str]:
    """너무 긴 본문을 하위 표기 계층으로 더 자른다.

    한 청크가 수천 자면 검색은 걸려도 "어느 문장이 근거인가"를 짚을 수 없다.
    반대로 계층 끝까지 쪼개면 「아) 용기ㆍ포장 재질」 같은 11자 조각이 나와 근거가 되지 못한다.
    그래서 머리말은 조각마다 맥락으로 붙이되 다시 쪼개지 않고, 형제는 MIN_CHARS 까지 묶는다.
    자를 수 없으면 자르지 않는다 — 글자 수로 끊으면 문장이 반토막 난다.
    """
    def glue(s: str) -> str:
        return f"{prefix} {s}" if prefix else s

    if len(glue(text)) <= MAX_CHARS:
        return [glue(text)]

    for d in range(depth, len(LEVELS)):
        cut = _cut_at(text, LEVELS[d])
        if not cut:
            continue
        head, items = cut

        # 짧은 머리말은 모든 조각이 물고 간다. 긴 머리말은 따로 떼어 그것도 재귀로 자른다
        short = bool(head) and len(head) <= HEAD_MAX
        carry = glue(head) if short else prefix
        lead = split_long(head, prefix, d + 1) if head and not short else []

        budget = MAX_CHARS - (len(carry) + 1 if carry else 0)
        if budget < MIN_CHARS:
            continue  # 머리말이 너무 커서 이 계층으로는 못 나눈다

        groups = _pack(items, budget)
        if len(groups) < 2 and not lead:
            continue

        out = list(lead)
        for g in groups:
            if len(g) > budget:
                out.extend(split_long(g, carry, d + 1))
            else:
                out.append(f"{carry} {g}" if carry else g)
        return out

    # 계층으로 못 자를 때의 마지막 수단 — **문장 경계**로 묶는다.
    # 한국어 법령문은 거의 모두 「…다.」로 끝나므로 이 하나로 문장이 갈린다.
    sents = [x.strip() for x in re.split(r"(?<=다\.)\s*", text) if x.strip()]
    if len(sents) > 1:
        budget = MAX_CHARS - (len(prefix) + 1 if prefix else 0)
        if budget >= MIN_CHARS:
            groups = _pack(sents, budget)
            if len(groups) > 1:
                return [f"{prefix} {g}" if prefix else g for g in groups]
    return [glue(text)]


# ─────────────────────────────────────────── 조 번호 없는 고시의 위치 표기

PATH_MAX = 44
HEADING_MAX = 30  # 이보다 긴 조각은 제목이 아니라 본문이다

# 계층 표기와 그 깊이. 한글 표기는 순번 글자만 인정한다 —
# 그러지 않으면 「…하여야 함.」의 `함.` 이 표기로 잡혀 위치가 「함.」 으로 나온다.
#
# 로마숫자(최상위 장)는 **청크 맨 앞에 있을 때만** 표기로 인정한다. 본문 중간의 것은
# 장 제목이 아니라 참조다 — 「…표시방법은 Ⅱ. 공통표시기준에 따른다」의 `Ⅱ.` 를 조상으로
# 읽으면 음료류(Ⅲ. 개별표시사항) 조항의 출처가 Ⅱ 장으로 찍힌다 (실측).
SECTION_LEVELS = [
    (0, re.compile(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\.\s?"), lambda m: m.start() == 0),
    (1, re.compile(r"(?<![제항호조\d])\d+\.\s"), lambda m: True),
    (2, re.compile(r"(?<!\()([가-힣])\.\s"), lambda m: ko_ord(m.group(1)) > 0),
    (3, re.compile(r"(?<!\()\d+\)\s?"), lambda m: True),
    (4, re.compile(r"(?<!\()([가-힣])\)\s?"), lambda m: ko_ord(m.group(1)) > 0),
    (5, re.compile(r"\(\d+\)\s?"), lambda m: True),
]


def section_path(text: str, fallback: str = "본문") -> str:
    """청크가 고시 본문의 어디인지를 **계층 표기에서 읽어** 위치로 쓴다.

    규칙은 하나다 — **애매해지는 깊이에서 멈춘다.** 어떤 깊이의 표기가 청크 안에 둘 이상
    있으면 그 청크는 그 계층의 형제 여럿에 걸쳐 있다는 뜻이므로, 그중 하나를 위치로 적으면
    거짓이 된다. 못 읽으면 「본문」으로 남긴다 — 없는 위치를 지어내지 않는 것이 이 함수의
    유일한 목적이다 (`본문제1호` 같은 표기는 없는 호 번호를 만들어 내는 것이다).
    """
    t = squash(text)
    hits = []
    for depth, pattern, ok in SECTION_LEVELS:
        for m in pattern.finditer(t):
            if not ok(m):
                continue
            hits.append((m.start(), depth, len(m.group(0))))
    if not hits:
        return fallback
    hits.sort(key=lambda h: (h[0], h[1]))

    seen: dict[int, int] = {}
    for _, d, _len in hits:
        seen[d] = seen.get(d, 0) + 1

    segs: list[str] = []
    for depth, _pattern, _ok in SECTION_LEVELS:
        n = seen.get(depth, 0)
        if n == 0:
            continue  # 이 계층을 쓰지 않는 고시도 있다
        if n > 1:
            break     # 형제 여럿에 걸쳐 있다 — 여기부터는 단정할 수 없다
        k = next(i for i, h in enumerate(hits) if h[1] == depth)
        stop = hits[k + 1][0] if k + 1 < len(hits) else len(t)
        seg = squash(t[hits[k][0]: stop])
        # 라벨이 길면 본문이 이어진 것이다. 번호만 남긴다
        segs.append(seg if len(seg) <= HEADING_MAX else squash(t[hits[k][0]: hits[k][0] + hits[k][2]]))

    # 길면 위쪽 조상을 조각째로 접는다. 남는 것은 항상 실제 사슬의 뒷부분이다
    keep = segs
    while len(keep) > 1 and len(" ".join(keep)) > PATH_MAX:
        keep = keep[1:]
    path = " ".join(keep)
    if not path or len(path) > PATH_MAX:
        return fallback
    return f"… {path}" if len(keep) < len(segs) else path
