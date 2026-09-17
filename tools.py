"""
카테고리별 근거 조회 도구 4종 (PLAN 슬라이스 3 · 설계 결정 D1).

**카테고리 = 도구** 다. 과제 지표가 「호출한 도구 집합이 기대 집합과 정확히 일치하면 1점」
이므로 도구 경계가 흐릿하면 그 지표를 잴 수 없다. 도구 하나가 자료의 어느 부분을 보는지는
`docs/CATEGORY-MAP.md` 의 매핑표가 원본이고, 이 파일의 `SCOPES`·`TOPICS` 가 그 구현이다.

발췌 방법은 **섹션 인덱스 + 키워드 매칭**이다 (D1 — 벡터 검색을 쓰지 않는다). 셋으로 겹쳐 쓴다.

1. **주제 색인** (`TOPICS`) — 질의에 특정 말이 있으면 미리 정해 둔 섹션을 확정 근거로 넣는다.
   「면역력」처럼 법령 본문에 그 낱말이 **한 번도 안 나오는** 질의가 있기 때문이다.
   질병 예방·치료 효능 조항은 「면역」이라는 말을 쓰지 않는다 — 키워드 매칭만으로는 못 간다.
2. **키워드 점수** — 남은 자리를 질의어가 실제로 든 청크로 채운다. 흔한 말이 순위를 뒤집지
   않도록 문서빈도로 눌러 준다.
3. **별칭 색인** (첨가물) — 「비타민C」와 「L-아스코브산나트륨」이 같은 자리를 가리킨다.

셋 다 결정적이다. 같은 질의는 늘 같은 근거를 돌려준다 — 채점기가 흔들리면 개선 축을
하나씩 바꾼 효과를 잴 수 없다.

직접 써 보려면:
    uv run python tools.py                          # 완료 기준 자체 점검
    uv run python tools.py lookup_food_type "유자즙 28%"
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

from corpus import Chunk, Corpus, load

# 도구 1회 반환 한도 (PLAN 슬라이스 3 완료 기준 d). 문서를 통째로 넣지 않는다.
MAX_RESULT_CHARS = 8000

# 주제 색인이 확정 근거를 다 넣고 나면 남는 자리를 키워드 매칭이 채운다.
MAX_KEYWORD_CHUNKS = 12


# ──────────────────────────────────────────────────────── 문자열 정규화
#
# 공전·고시는 같은 말을 여러 부호로 쓴다 — 「과･채주스」(U+FF65) · 「식품ㆍ의약품」(U+318D) ·
# 「제조·가공」(U+00B7). 게다가 슬라이스 2의 PDF 되붙이기 규칙 때문에 띄어쓰기가 한 군데씩
# 사라진 자리가 있다. 그래서 **부호와 공백을 전부 지우고** 맞춘다.

_DROP = re.compile(r"[\s·ㆍ・･‧∙⋅\-–—_,()\[\]{}「」『』'\"'']")


def norm(s: str) -> str:
    return _DROP.sub("", s).lower()


_NORM_CACHE: dict[str, str] = {}


def _norm_chunk(chunk: Chunk) -> tuple[str, str]:
    key = chunk.id
    if key not in _NORM_CACHE:
        _NORM_CACHE[key] = norm(chunk.text)
        _NORM_CACHE[key + "#p"] = norm(chunk.path)
    return _NORM_CACHE[key], _NORM_CACHE[key + "#p"]


# 질의에서 뽑은 낱말. 조사 목록을 따로 두지 않고 **뒤에서 세 글자까지 잘라 본다** —
# 「유자즙이」→「유자즙」, 「음료수를」→「음료」. 조사 목록은 늘 빠진 것이 생긴다.
_WORD = re.compile(r"[가-힣A-Za-z0-9]+")
_QUERY_STOP = {"무엇", "어떻게", "어디", "해야", "하나요", "되나요", "있나요", "인가요", "때문"}


def query_words(query: str) -> list[str]:
    """질의어 목록.

    두 자리 이하의 **순수 숫자는 버린다.** 법령 본문에서 그런 숫자는 거의 항상 항·호
    번호다 — 「유자즙 28%」의 28 이 제조·가공기준 「28) 냉동수산물을 …」을 끌어왔다(실측).
    사용자가 말한 배합 비율은 애초에 법령 본문에 있을 수가 없다. 함량 질문은
    주제 색인의 `%` 가 받는다.
    """
    return [
        w
        for w in _WORD.findall(query)
        if len(w) >= 2 and w not in _QUERY_STOP and not (w.isdigit() and len(w) < 3)
    ]


def _candidates(word: str) -> list[str]:
    """낱말과 그 앞부분들. 긴 것부터 보고 처음 맞는 것 하나만 센다."""
    return [word[:k] for k in range(len(word), 1, -1)][:4]


# ──────────────────────────────────────────────────────────── 주제 색인


@dataclass(frozen=True)
class Topic:
    """질의에 `terms` 중 하나가 들어 있으면 `sections` 를 확정 근거로 넣는다.

    섹션은 청크 ID 가 아니라 **위치 정규식**으로 적는다. 코퍼스를 다시 수집하면 ID 는
    밀리지만 「제5. 9. 음료류」라는 자리는 그대로다 (PLAN 검증절의 함정 2).
    """

    name: str
    terms: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    why: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    label: str
    scope_desc: str
    scope: tuple[tuple[str, str], ...]
    topics: tuple[Topic, ...]
    anchors: tuple[tuple[str, str], ...]  # 아무것도 안 걸릴 때의 기본 근거


# ── 1. 식품유형 · 규격 ────────────────────────────────────────────────
# 제2. 식품일반 공통기준에서 무엇을 끌어오는지는 PLAN D8 이 정했다 —
# 제2. 1.(식품원료 기준) 과 제2. 2.(제조·가공기준) 만. 위생지표균·오염물질·보존유통은 넣지 않는다.

FOOD_TYPE = Tool(
    name="lookup_food_type",
    label="식품유형·규격",
    scope_desc="식품공전 제5. 9. 음료류 · 제2. 1. 식품원료 기준 · 제2. 2. 제조·가공기준",
    scope=(
        ("FDC", r"^제5\. 9\. 음료류"),
        ("FDC", r"^제2\. 1\. 식품원료 기준"),
        ("FDC", r"^제2\. 2\. 제조·가공기준"),
    ),
    topics=(
        Topic(
            "과채음료",
            ("과즙", "채즙", "과채", "과일", "채소", "주스", "쥬스", "착즙", "농축과", "퓨레"),
            (("FDC", r"^제5\. 9\. 음료류 9-3"),),
            "과·채주스/과·채음료/농축과·채즙의 정의와 함량 기준이 여기 다 있다 (UC-1)",
        ),
        Topic(
            "다류",
            ("액상차", "침출차", "고형차", "다류", "차류", "티백"),
            (("FDC", r"^제5\. 9\. 음료류 9-1"),),
            "유자차처럼 과일 원료를 써도 액상차가 될 수 있다 — 과·채음료와 갈리는 자리",
        ),
        Topic("커피", ("커피", "원두", "에스프레소"), (("FDC", r"^제5\. 9\. 음료류 9-2"),)),
        Topic("탄산음료", ("탄산",), (("FDC", r"^제5\. 9\. 음료류 9-4"),)),
        Topic("두유류", ("두유", "대두", "콩물"), (("FDC", r"^제5\. 9\. 음료류 9-5"),)),
        Topic("발효음료", ("발효음료", "유산균", "발효유"), (("FDC", r"^제5\. 9\. 음료류 9-6"),)),
        Topic("인삼홍삼음료", ("인삼", "홍삼"), (("FDC", r"^제5\. 9\. 음료류 9-7"),)),
        Topic(
            "기타음료",
            ("혼합음료", "음료베이스", "기타음료", "먹는물"),
            (("FDC", r"^제5\. 9\. 음료류 9-8"),),
        ),
        Topic(
            "유형판정",
            ("식품유형", "유형", "분류", "품목제조", "무슨 음료", "어느 유형"),
            (("FDC", r"^제5\. 9\. 음료류 9-\d+ .*4\) 식품유형"), ("FDC", r"^제5\. 9\. 음료류 9\. 음료류")),
            "유형 정의를 나란히 놓고 봐야 겹치는 자리가 보인다",
        ),
        Topic(
            "원료배합",
            ("함량", "%", "퍼센트", "배합", "몇프로", "비율"),
            (
                ("FDC", r"^제2\. 2\..*(원료배합|배합기준)"),
                ("FDC", r"^제5\. 9\. 음료류 9-\d+ .*4\) 식품유형"),
            ),
            "「유자즙 28%」를 어떻게 세는지가 제2. 2. 4)에 있다 (PLAN D8)",
        ),
        Topic(
            "식품원료",
            ("원료", "추출물", "발효추출물", "식품원료", "사용할수", "쓸수", "부원료"),
            (("FDC", r"^제2\. 1\. 식품원료 기준"),),
            "UC-6(근거 없음)이 여기서 갈린다 — 도구는 잡히되 답이 없는 자리",
        ),
        Topic(
            "규격",
            ("규격", "세균", "대장균", "살균", "멸균", "산도", "당도", "brix", "보존료"),
            (("FDC", r"^제5\. 9\. 음료류 9-\d+ .*5\) 규격"),),
        ),
    ),
    anchors=(
        ("FDC", r"^제5\. 9\. 음료류 9\. 음료류"),
        ("FDC", r"^제5\. 9\. 음료류 9-\d+ .*4\) 식품유형"),
    ),
)


# ── 2. 의무 표시사항 ──────────────────────────────────────────────────

LABELING = Tool(
    name="lookup_labeling",
    label="의무 표시사항",
    scope_desc="식품등의 표시기준 본문(Ⅰ~Ⅲ) + 별표 1~6 · 별지 1 · 별도 1~7",
    scope=(("LBL", r""),),
    topics=(
        Topic(
            "음료류표시사항",
            ("음료", "주스", "쥬스", "액상차", "탄산", "다류", "커피"),
            (("LBL", r"자\. 음료류"),),
            "음료류에만 붙는 표시사항. 공통과 구분해서 답해야 한다 (UC-2)",
        ),
        Topic(
            "제품명",
            ("제품명", "이름", "명칭", "네이밍", "브랜드", "상표"),
            (("LBL", r"^별지 1 .*가\. 제품명"),),
            "원재료명을 제품명에 쓰려면 함량표시가 따라붙는다 — UC-1 의 「유자주스」가 이 자리",
        ),
        Topic(
            "원재료명",
            ("원재료", "성분명", "배합", "원료명"),
            (("LBL", r"^별지 1 .*(바\. 원재료명|사\. 성분명)"),),
        ),
        Topic(
            "함량표시",
            ("함량", "%", "퍼센트"),
            (("LBL", r"^별지 1 .*(가\. 제품명|사\. 성분명)"),),
        ),
        Topic(
            "영양성분",
            ("영양", "열량", "칼로리", "나트륨", "당류", "단백질", "지방", "콜레스테롤"),
            (("LBL", r"^별지 1 .*아\. 영양성분"), ("LBL", r"^별표 [23] ")),
        ),
        Topic(
            "소비기한",
            ("소비기한", "유통기한", "제조연월일", "제조일자", "품질유지기한"),
            (("LBL", r"^별지 1 .*(다\. 제조연월일|라\. 소비기한)"),),
        ),
        Topic("내용량", ("내용량", "용량", "중량", "ml", "리터"), (("LBL", r"^별지 1 .*마\. 내용량"),)),
        Topic(
            "글씨크기",
            ("활자", "글씨", "글자", "포인트", "크기"),
            (("LBL", r"^Ⅱ\. 공통표시기준 6\."), ("LBL", r"^Ⅰ\. 총 칙 3\. 용어의 정의")),
        ),
        Topic(
            "표시면",
            ("주표시면", "정보표시면", "표시면", "어디에", "도안", "서식"),
            (("LBL", r"^별도 1 "), ("LBL", r"^Ⅱ\. 공통표시기준 1\. 표시방법")),
        ),
        Topic(
            "알레르기",
            ("알레르기", "알러지", "알러젠"),
            (("LBL", r"^Ⅲ\..*자\. 음료류 2\) 표시사항"), ("LBL", r"^Ⅱ\. 공통표시기준 1\. 표시방법")),
        ),
        Topic(
            "첨가물표시",
            ("첨가물", "감미료", "보존료", "착색료", "산화방지제", "산도조절제", "향료", "비타민"),
            (("LBL", r"^별표 [456] "),),
            "첨가물을 라벨에 어떻게 적는지. lookup_additive 와 같은 별표를 본다 (의도된 겹침)",
        ),
        Topic(
            "의무표시사항",
            ("표시사항", "의무", "반드시", "빠진", "필수", "라벨", "누락"),
            (("LBL", r"^Ⅲ\..*자\. 음료류"), ("LBL", r"^Ⅱ\. 공통표시기준 1\. 표시방법")),
        ),
    ),
    anchors=(("LBL", r"^Ⅲ\..*자\. 음료류"),),
)


# ── 3. 첨가물 사용기준 · 표시 ─────────────────────────────────────────
# 제2. 3. 에서는 「3) 식품첨가물」 한 항목만 가져온다 (PLAN D8).

ADDITIVE = Tool(
    name="lookup_additive",
    label="첨가물 사용기준·표시",
    scope_desc=(
        "식품첨가물공전 II. 2. 일반사용기준 + III. 품목별 사용기준 1~4 · "
        "식품공전 제2. 3. 3) 식품첨가물 · 표시기준 별표 4~6"
    ),
    scope=(
        ("FAC", r""),
        ("FDC", r"^제2\. 3\..*3\) 식품첨가물$"),
        ("LBL", r"^별표 [456] "),
    ),
    topics=(
        Topic(
            "일반사용기준",
            ("사용기준", "사용량", "최대", "한도", "얼마", "제한", "최소량"),
            (("FAC", r"^II\. 2\. 일반사용기준 [1-7]\)"),),
            "품목표가 「II. 2. 1)의 규정에 따라」로만 적힌 품목이 많다 — 그 II. 2 가 없으면 답이 안 된다. "
            "8)(조제유류·영유아용)은 뺐다 — 음료류 범위 밖인데 표 하나가 6,000자다",
        ),
        Topic(
            "표시방법",
            ("표시", "라벨", "원재료명", "적나", "표기", "간략명", "용도"),
            (("LBL", r"^별표 4 "),),
            "별표 5·6 은 품목명 순 목록이라 통째로는 못 넣는다. 품목이 지목되면 그 품목이 실린 "
            "조각만 별칭 색인이 붙인다",
        ),
        Topic(
            "식품첨가물규격",
            ("첨가물", "허용", "사용가능", "써도"),
            (("FDC", r"^제2\. 3\..*3\) 식품첨가물$"), ("FAC", r"^II\. 2\. 일반사용기준 1\)")),
        ),
    ),
    anchors=(
        ("FAC", r"^II\. 2\. 일반사용기준 1\)"),
        ("FDC", r"^제2\. 3\..*3\) 식품첨가물$"),
    ),
)


# ── 4. 표시·광고 금지표현 ─────────────────────────────────────────────

AD_CLAIMS = Tool(
    name="lookup_ad_claims",
    label="표시·광고 금지표현",
    scope_desc="식품표시광고법 제1~10조 · 같은 법 시행령 제1~6조 · 부당한 표시·광고 내용 기준 고시 + 별표 1~2",
    scope=(("FLA", r""), ("FLD", r""), ("UNF", r"")),
    topics=(
        Topic(
            "질병효능",
            (
                "질병", "예방", "치료", "효능", "효과", "의약품", "면역", "항암", "혈압",
                "혈당", "당뇨", "다이어트", "피로", "해독", "항염", "노화", "숙취",
            ),
            (
                ("FLA", r"^제8조①제1호$"),
                ("FLA", r"^제8조①제2호$"),
                ("UNF", r"^제2조제1호$"),
                ("FLD", r"^제2조제1호$"),
            ),
            "「면역력」은 법령 본문에 한 번도 안 나온다 — 주제 색인이 없으면 이 자리를 못 찾는다",
        ),
        Topic(
            "건기식오인",
            (
                "건강기능식품", "기능성", "오인", "건기식",
                # 건강기능식품 고시가 기능성으로 인정한 표현들. 일반식품에 쓰면 제1호(질병 효능)보다
                # **제3호(건기식 오인)** 에 먼저 걸린다. 낱말만 보고 제1·2호로 보내면 답변이
                # 근거 하나를 통째로 빠뜨린다 — 평가셋 AC-1 이 이걸 잡았다
                "면역", "면역력", "기억력", "체지방", "장건강", "혈행", "항산화", "관절",
            ),
            (("FLA", r"^제8조①제3호$"), ("UNF", r"^제2조제2호$")),
            "「면역력」처럼 질병효능과 건기식오인에 동시에 걸리는 말은 두 주제가 같이 켜져야 한다",
        ),
        Topic(
            "무첨가",
            ("무첨가", "무설탕", "무가당", "무색소", "무보존료", "free", "제로", "없다", "안들어"),
            (("UNF", r"^제2조제3호$"), ("FLA", r"^제8조①제5호$")),
            "「무(無)」 표시는 조건부 허용이라 조문을 봐야 한다 (UC-4)",
        ),
        Topic(
            "거짓과장",
            ("거짓", "과장", "최고", "최초", "1위", "유일", "특허", "인증"),
            (("FLA", r"^제8조①제4호$"), ("UNF", r"^제2조제4호$")),
        ),
        Topic(
            "천연100",
            ("천연", "100%", "순수", "자연", "무농약", "유기농"),
            (("UNF", r"^제2조제3호$"), ("UNF", r"^제2조제5호$"), ("UNF", r"^별표 2 ")),
        ),
        Topic(
            "비방비교",
            ("비방", "비교", "타사", "경쟁사"),
            (("FLA", r"^제8조①제6호$"), ("FLA", r"^제8조①제7호$"), ("UNF", r"^제2조제4호$")),
        ),
        Topic(
            "한약처방명",
            ("한약", "처방명", "쌍화", "십전대보"),
            (("UNF", r"^제2조제1호$"), ("UNF", r"^별표 1 ")),
        ),
        Topic(
            "실증",
            ("실증", "입증", "근거자료", "객관적"),
            (("FLA", r"^제9조"),),
        ),
        Topic(
            "자율심의",
            ("심의", "사전심의", "자율심의"),
            (("FLA", r"^제10조"), ("FLD", r"^제4조")),
        ),
        Topic(
            "표시광고범위",
            ("광고", "표시", "패키지", "카피", "문구", "홍보"),
            (("FLA", r"^제8조①제1호$"), ("FLD", r"^제2조"), ("FLA", r"^제2조")),
        ),
    ),
    anchors=(("FLA", r"^제8조①제[1-5]호$"),),
)


TOOLS: dict[str, Tool] = {
    t.name: t for t in (FOOD_TYPE, LABELING, ADDITIVE, AD_CLAIMS)
}


# ──────────────────────────────────────────────────── 첨가물 별칭 색인
#
# 대표는 「비타민C」라고 부르지만 공전의 품목명은 「비타민C」이고 라벨에 쓸 이름은
# 「비타민C」 또는 「L-아스코브산나트륨」처럼 염 이름이다. 셋이 같은 자리를 가리켜야 한다.
#
# 별칭의 출처는 **수집한 원문 안에만** 있다.
#   - 품목 청크의 머리줄: `비타민C (L-Ascorbic Acid 이명 : Ascorbic acid) 주용도: …`
#   - 표시기준 별표 5·6 의 「간략명」 칸: 라벨에 쓸 수 있는 줄임 이름
# 예외는 아래 표기 흔들림 보정뿐이고, 이건 같은 이름의 **철자 차이**다 — 없는 품목을
# 만들어 내지 않는다.

SPELLING_VARIANTS = {
    "아스코르브": "아스코브",   # 공전 표기는 「L-아스코브산」
    "아스콜빈": "아스코브",
    "비타민씨": "비타민c",
    "구연산나트륨": "구연산삼나트륨",  # 별표 6 간략명이 「구연산Na」인 품목
}

_PRODUCT_PATH = re.compile(r"^III\. \d+\. (일반식품첨가물|가공보조제|영양강화제|혼합제제류) (.+)$")
_HEAD = re.compile(r"^(?P<name>.+?)\s*\((?P<paren>[^)]*)\)")
_USES = re.compile(r"주용도\s*:\s*(?P<uses>.+?)(?:\s*/|$)")


@dataclass(frozen=True)
class Additive:
    name: str
    chunk_id: str
    group: str
    english: str = ""
    aliases: tuple[str, ...] = ()
    brief_names: tuple[str, ...] = ()  # 표시기준 별표 5·6 의 간략명
    uses: tuple[str, ...] = ()

    @property
    def all_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for n in (self.name, self.english, *self.aliases, *self.brief_names):
            if n and n not in seen:
                seen.append(n)
        return tuple(seen)


def _brief_name_table(corpus: Corpus) -> dict[str, list[str]]:
    """별표 5·6 에서 품목명 → 간략명 목록을 읽는다. 표 한 행이 한 품목이다."""
    out: dict[str, list[str]] = {}
    for chunk in corpus.chunks:
        if not re.match(r"^별표 [56] ", chunk.path):
            continue
        for line in chunk.text.splitlines()[1:]:  # 첫 줄은 칸 이름
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 2 or not cells[0] or cells[0] == "식품첨가물의 명칭":
                continue
            briefs = [b.strip() for b in cells[1].split(",") if b.strip()]
            if briefs:
                out.setdefault(cells[0], []).extend(briefs)
    return out


def build_additive_index(corpus: Corpus) -> list[Additive]:
    briefs = _brief_name_table(corpus)
    index: list[Additive] = []
    for chunk in corpus.chunks:
        m = _PRODUCT_PATH.match(chunk.path)
        if not m:
            continue
        group, name = m.group(1), m.group(2).strip()
        # 「1) 공통기준」 「2) 품목별 기준」 같은 머리글은 품목이 아니다
        if re.match(r"^(\d+\)|III\.)", name):
            continue

        english, aliases = "", []
        head = _HEAD.match(chunk.text)
        if head and norm(head.group("name")).startswith(norm(name)[:4]):
            paren = head.group("paren")
            left, _, right = paren.partition("이명")
            english = left.strip()
            # 이명 구분자는 세미콜론뿐이다. 쉼표로도 자르면
            # 「2-Hydroxy-1,2,3-propane-tricarboxylic acid」가 세 개로 쪼개진다
            aliases = [a.strip(" :;") for a in right.split(";") if a.strip(" :;")]

        uses_m = _USES.search(chunk.text)
        uses = tuple(uses_m.group("uses").split()) if uses_m else ()

        index.append(
            Additive(
                name=name,
                chunk_id=chunk.id,
                group=group,
                english=english,
                aliases=tuple(aliases),
                brief_names=tuple(briefs.get(name, ())),
                uses=uses,
            )
        )
    return index


def _variant(text: str) -> str:
    out = norm(text)
    for a, b in SPELLING_VARIANTS.items():
        out = out.replace(norm(a), norm(b))
    return out


def match_additives(query: str, index: list[Additive], limit: int = 8) -> list[Additive]:
    """질의어에 품목 이름이 들어 있으면(또는 그 반대면) 그 품목을 고른다.

    **앞 방향**(품목명이 질의에 들어 있다)이 정확한 지목이라 점수가 높다.
    **뒷 방향**(질의어가 품목명 안에 들어 있다)은 「아스코브산」으로 그 염 품목들을 찾는 경로다.
    """
    nq = _variant(query)
    words = [_variant(w) for w in query_words(query)]
    hits: list[tuple[float, int, Additive]] = []
    for order, add in enumerate(index):
        best = 0.0
        for nm in add.all_names:
            n = norm(nm)
            if len(n) < 2:
                continue
            if n in nq:
                best = max(best, len(n) * 2.0)
            elif len(n) >= 4:
                for w in words:
                    if len(w) >= 3 and w in n:
                        best = max(best, float(len(w)))
        if best:
            hits.append((best, order, add))
    hits.sort(key=lambda h: (-h[0], h[1]))
    return [a for _, _, a in hits[:limit]]


# ─────────────────────────────────────────────────────────── 근거 조립


@dataclass
class Evidence:
    chunk: Chunk
    reason: str  # 왜 뽑혔는지 — 주제 이름 · 별칭 · 키워드 · 기본


@dataclass
class ToolResult:
    tool: str
    label: str
    query: str
    scope_desc: str
    topics: list[str] = field(default_factory=list)
    additives: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    text: str = ""
    dropped: int = 0  # 한도 때문에 못 넣은 청크 수

    @property
    def is_empty(self) -> bool:
        return not self.evidence

    @property
    def chunk_ids(self) -> list[str]:
        return [e.chunk.id for e in self.evidence]


def _resolve(corpus: Corpus, sections: tuple[tuple[str, str], ...]) -> list[Chunk]:
    out: list[Chunk] = []
    seen: set[str] = set()
    for code, pattern in sections:
        for chunk in corpus.select(code, pattern):
            if chunk.id not in seen:
                seen.add(chunk.id)
                out.append(chunk)
    return out


def _keyword_ranked(query: str, pool: list[Chunk], skip: set[str]) -> list[tuple[float, Chunk]]:
    """질의어가 실제로 든 청크를 점수순으로. 흔한 말은 문서빈도로 눌러 순위를 못 뒤집게 한다."""
    words = query_words(query)
    if not words or not pool:
        return []
    total = len(pool)
    normed = [(c, *_norm_chunk(c)) for c in pool]

    df: dict[str, int] = {}

    def freq(term: str) -> int:
        if term not in df:
            df[term] = sum(1 for _, t, p in normed if term in t or term in p)
        return df[term]

    scored: list[tuple[float, int, Chunk]] = []
    for order, (chunk, text, path) in enumerate(normed):
        if chunk.id in skip:
            continue
        score = 0.0
        for word in words:
            for cand in _candidates(word):
                term = norm(cand)
                if len(term) < 2:
                    continue
                where = 2.0 if term in path else (1.0 if term in text else 0.0)
                if not where:
                    continue
                idf = math.log(total / max(freq(term), 1)) + 0.1
                score += len(term) ** 2 * where * idf
                break
        if score > 0:
            scored.append((score, order, chunk))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(s, c) for s, _, c in scored]


def _render(result: ToolResult) -> str:
    head = [f"# 근거 — {result.tool} ({result.label})", f"범위: {result.scope_desc}"]
    if result.topics:
        head.append(f"질의에서 잡은 주제: {', '.join(result.topics)}")
    if result.additives:
        head.append(f"질의에서 잡은 첨가물 품목: {', '.join(result.additives)}")

    body = []
    for e in result.evidence:
        c = e.chunk
        body.append(f"[{c.id}] {c.law_name} · {c.path} (시행 {c.effective_date}) <{e.reason}>")
        body.append(c.text)
        body.append("")

    sources = []
    for c in (e.chunk for e in result.evidence):
        line = f"- {c.law_name} (시행 {c.effective_date}) {c.url}"
        if line not in sources:
            sources.append(line)

    tail = ["## 출처", *sources]
    if result.dropped:
        tail.append(f"(한도 {MAX_RESULT_CHARS}자에 맞춰 {result.dropped}개 청크를 뺐다)")
    return "\n".join([*head, "", *body, *tail])


def _assemble(
    tool: Tool,
    query: str,
    corpus: Corpus,
    extra: list[Evidence],
    additives: list[str] | None = None,
) -> ToolResult:
    """확정 근거 → 키워드 근거 → (아무것도 없으면) 기본 근거 순으로 한도까지 채운다.

    머리말은 **채우기 전에** 다 정해 둔다 — 나중에 한 줄이라도 덧붙이면 그만큼 한도를 넘는다.
    """
    result = ToolResult(
        tool=tool.name,
        label=tool.label,
        query=query,
        scope_desc=tool.scope_desc,
        additives=list(additives or []),
    )
    pool = _resolve(corpus, tool.scope)
    in_scope = {c.id for c in pool}

    picked: list[Evidence] = []
    seen: set[str] = set()

    def add(chunk: Chunk, reason: str) -> None:
        if chunk.id in seen or chunk.id not in in_scope:
            return
        seen.add(chunk.id)
        picked.append(Evidence(chunk=chunk, reason=reason))

    # 순서가 곧 우선순위다 — 한도에 걸리면 뒤부터 잘린다.
    # 별칭 색인이 고른 품목이 가장 구체적이고, 주제 색인은 그 다음이다.
    for ev in extra:
        if ev.chunk.id not in seen and ev.chunk.id in in_scope:
            seen.add(ev.chunk.id)
            picked.append(ev)

    nq = norm(query)
    for topic in tool.topics:
        if not any(norm(t) in nq for t in topic.terms):
            continue
        result.topics.append(topic.name)
        for chunk in _resolve(corpus, topic.sections):
            add(chunk, f"주제 {topic.name}")

    for _score, chunk in _keyword_ranked(query, pool, seen)[:MAX_KEYWORD_CHUNKS]:
        add(chunk, "키워드")

    if not picked:
        for chunk in _resolve(corpus, tool.anchors):
            add(chunk, "기본 근거")

    # 한도까지 채운다. 렌더 결과로 재므로 머리말·출처까지 포함해 8,000자를 넘지 않는다.
    for ev in picked:
        result.evidence.append(ev)
        if len(_render(result)) > MAX_RESULT_CHARS:
            result.evidence.pop()
            result.dropped += 1
    result.text = _render(result)
    return result


@lru_cache(maxsize=None)
def scope_ids(name: str) -> set[str]:
    """도구가 **애초에 볼 수 있는** 청크 전체. 발췌된 것이 아니라 범위 그 자체다.

    슬라이스 9의 원인 분류가 쓴다 — gold 청크가 여기 없으면 `도구 범위 밖`이고,
    여기 있는데 발췌에 안 들어왔으면 `발췌 누락`이다. 둘은 고칠 곳이 다르다.
    """
    return {c.id for c in _resolve(load(), TOOLS[name].scope)}


# ────────────────────────────────────────────────────────────── 도구 4종


def lookup_food_type(query: str) -> ToolResult:
    """식품유형 판정과 그 유형의 규격 (식품공전 음료류 + 제2. 식품일반 공통기준 일부)."""
    return _assemble(FOOD_TYPE, query, load(), [])


def lookup_labeling(query: str) -> ToolResult:
    """의무 표시사항과 표시 방법 (식품등의 표시기준 본문 + 별표)."""
    return _assemble(LABELING, query, load(), [])


def lookup_additive(query: str) -> ToolResult:
    """첨가물의 사용 가부·사용량 한도·라벨 표기 (첨가물공전 + 표시기준 별표 4~6)."""
    corpus = load()
    index = additive_index()
    matched = match_additives(query, index)

    extra: list[Evidence] = []
    by_id = corpus.by_id
    for add in matched:
        chunk = by_id.get(add.chunk_id)
        if chunk:
            extra.append(Evidence(chunk=chunk, reason=f"품목 {add.name}"))

    if matched:
        names = [norm(a.name) for a in matched]
        # 라벨에 뭐라 적는지는 별표 4·5·6 에 있다 — 잡힌 품목이 실린 표 조각만 붙인다
        for chunk in corpus.select("LBL", r"^별표 [456] "):
            body = norm(chunk.text)
            if any(n in body for n in names):
                extra.append(Evidence(chunk=chunk, reason="별표(표시 방법)"))
        # 품목표가 「II. 2. 1)의 규정에 따라」로 넘기면 그 일반사용기준이 곧 답이다
        if any("II. 2." in (by_id[a.chunk_id].text if a.chunk_id in by_id else "") for a in matched):
            for chunk in corpus.select("FAC", r"^II\. 2\. 일반사용기준 1\)"):
                extra.append(Evidence(chunk=chunk, reason="일반사용기준(품목표가 가리킴)"))

    names_seen: list[str] = []
    for a in matched:
        if a.name not in names_seen:  # 같은 이름이 일반식품첨가물·영양강화제 양쪽에 있다
            names_seen.append(a.name)
    return _assemble(ADDITIVE, query, corpus, extra, additives=names_seen)


def lookup_ad_claims(query: str) -> ToolResult:
    """표시·광고 문구가 부당한 표시·광고에 걸리는지 (표시광고법·시행령·부당표시광고 고시)."""
    return _assemble(AD_CLAIMS, query, load(), [])


_ADDITIVE_INDEX: list[Additive] | None = None


def additive_index() -> list[Additive]:
    global _ADDITIVE_INDEX
    if _ADDITIVE_INDEX is None:
        _ADDITIVE_INDEX = build_additive_index(load())
    return _ADDITIVE_INDEX


CALL = {
    "lookup_food_type": lookup_food_type,
    "lookup_labeling": lookup_labeling,
    "lookup_additive": lookup_additive,
    "lookup_ad_claims": lookup_ad_claims,
}


# ─────────────────────────────────────────────────────── 완료 기준 점검
#
# 슬라이스 1·2 와 같은 규약 — 스크립트가 끝에서 스스로 확인하고 실패하면 종료 코드 1.

CHECK_QUERIES = [
    ("lookup_food_type", "유자즙 28% 들어간 음료인데 제품명을 '유자주스'로 해도 되나요?"),
    ("lookup_food_type", "유자즙 28%"),
    ("lookup_food_type", "유자 발효추출물을 식품원료로 쓸 수 있나요?"),
    ("lookup_food_type", "액상차와 혼합음료는 뭐가 다른가요"),
    ("lookup_labeling", "과채음료 라벨에 반드시 들어가야 하는 표시사항이 뭔가요"),
    ("lookup_labeling", "제품명에 유자를 쓰면 함량도 적어야 하나요"),
    ("lookup_labeling", "영양성분 표시는 어떻게 하나요"),
    ("lookup_additive", "비타민C"),
    ("lookup_additive", "비타민C를 산화방지 목적으로 넣었는데 최대 얼마까지 되고 원재료명에 뭐라고 적나요"),
    ("lookup_additive", "구연산 사용량 한도"),
    ("lookup_additive", "아스코르브산"),
    ("lookup_ad_claims", "면역력"),
    ("lookup_ad_claims", "'면역력에 좋은 유자'라고 써도 되나요"),
    ("lookup_ad_claims", "'무첨가'는 어떤 조건에서 쓸 수 있나요"),
    ("lookup_ad_claims", "천연 100% 유자주스라고 광고해도 되나요"),
    ("lookup_food_type", "zzz 아무 말도 안 되는 질의"),
    ("lookup_labeling", "zzz 아무 말도 안 되는 질의"),
    ("lookup_additive", "zzz 아무 말도 안 되는 질의"),
    ("lookup_ad_claims", "zzz 아무 말도 안 되는 질의"),
]

# 완료 기준 (b) — 세 질의어가 각각 맞는 섹션을 잡는가
SECTION_CHECKS = [
    ("lookup_food_type", "유자즙 28%", r"9-3 과일.채소류음료 4\) 식품유형"),
    ("lookup_additive", "비타민C", r"^III\. 1\. 일반식품첨가물 비타민C$"),
    ("lookup_ad_claims", "면역력", r"^제8조①제1호$"),
]

MAP_DOC = "docs/CATEGORY-MAP.md"


def self_check() -> int:
    from pathlib import Path

    corpus = load()
    failures: list[str] = []
    print("■ 도구 범위와 주제 색인")
    for tool in TOOLS.values():
        pool = _resolve(corpus, tool.scope)
        chars = sum(len(c.text) for c in pool)
        print(f"  {tool.name:20} 청크 {len(pool):>5}  ({chars:,}자)  주제 {len(tool.topics)}개")
        if not pool:
            failures.append(f"{tool.name}: 범위가 비었다")
        if not _resolve(corpus, tool.anchors):
            failures.append(f"{tool.name}: 기본 근거가 비었다")
        for topic in tool.topics:
            hit = _resolve(corpus, topic.sections)
            if not hit:
                failures.append(f"{tool.name}/{topic.name}: 연결된 섹션이 없다")

    index = additive_index()
    with_alias = [a for a in index if len(a.all_names) > 1]
    print(f"\n■ 첨가물 별칭 색인: 품목 {len(index)}종, 별칭이 있는 품목 {len(with_alias)}종")
    if len(index) < 500:
        failures.append(f"첨가물 색인이 너무 작다: {len(index)}종")

    print("\n■ 질의별 결과 (완료 기준 a·d)")
    print(f"  {'도구':20}{'글자':>7}{'청크':>5}{'뺀것':>5}  질의")
    for name, query in CHECK_QUERIES:
        r = CALL[name](query)
        print(f"  {name:20}{len(r.text):>7}{len(r.evidence):>5}{r.dropped:>5}  {query[:34]}")
        if r.is_empty:
            failures.append(f"{name}('{query[:20]}'): 근거가 비었다")
        if len(r.text) > MAX_RESULT_CHARS:
            failures.append(f"{name}('{query[:20]}'): {len(r.text)}자 — 한도 초과")

    print("\n■ 질의어별 섹션 적중 (완료 기준 b)")
    for name, query, pattern in SECTION_CHECKS:
        r = CALL[name](query)
        hit = [e.chunk for e in r.evidence if re.search(pattern, e.chunk.path)]
        mark = "OK " if hit else "실패"
        where = f"{hit[0].id} {hit[0].path}" if hit else pattern
        print(f"  {mark} {name:20} '{query}' → {where}")
        if not hit:
            failures.append(f"{name}('{query}'): {pattern} 에 맞는 근거가 없다")

    print("\n■ 매핑표 (완료 기준 c)")
    doc = Path(__file__).parent / MAP_DOC
    if not doc.exists():
        failures.append(f"{MAP_DOC} 가 없다")
    else:
        body = doc.read_text(encoding="utf-8")
        missing = [t.name for t in TOOLS.values() if t.name not in body]
        missing += [
            f"{t.name}/{p.name}"
            for t in TOOLS.values()
            for p in t.topics
            if p.name not in body
        ]
        print(f"  {MAP_DOC}: {len(body):,}자, 빠진 항목 {len(missing)}개")
        if missing:
            failures.append(f"{MAP_DOC} 에 빠진 항목: {', '.join(missing[:8])}")

    if failures:
        print(f"\n✗ 실패 {len(failures)}건")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("\n✓ 슬라이스 3 완료 기준 (a)(b)(c)(d) 통과")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] in CALL:
        print(CALL[argv[1]](argv[2]).text)
        return 0
    if len(argv) > 1:
        print(f"사용법: python tools.py [{' | '.join(CALL)}] \"<질의>\"")
        return 2
    return self_check()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
