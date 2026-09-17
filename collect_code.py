"""
식품안전나라 「식품분야 공전 온라인 서비스」 → 사실 단위 청크 JSON.

    uv run python collect_code.py

산출물은 `data/corpus/FDC.json`(식품공전) · `FAC.json`(식품첨가물공전) 과 수집 리포트다.
받은 PDF 는 `.cache/fsd/` 에 두고 커밋하지 않는다 (5MB 짜리가 있다). 커밋되는 것은 청크다.

**왜 브라우저를 안 쓰나.** PLAN 슬라이스 2는 `/fsd/api/tree` 가 CORS 로 막히니 페이지를
렌더해 긁자고 했는데, CORS 는 브라우저 규칙이라 서버 호출에는 걸리지 않는다 — httpx 로
그냥 200 이 온다. 그리고 화면에 뜨는 것 자체가 PDF 뷰어라서 렌더해도 긁을 HTML 이 없다.
공전은 **항목마다 PDF 한 개**이고, 트리 API 가 그 PDF 의 fileId 를 준다.

**왜 pdfplumber 인가.** 첨가물 사용기준은 `품목명 | 정의 | INS/CAS | 사용기준 | 주용도`
5열 표다. 줄 단위로 뽑으면(pypdf) 열이 섞여 옆 품목의 사용량이 내 품목 밑에 붙는다 —
이 제품에서 가장 위험한 종류의 오염이다. 셀 단위로 뽑아 **한 품목 = 한 청크**로 만든다.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pdfplumber

from chunking import split_long, squash
from collect_law import resolve_admrul_version
from sources import FSD_BASE, FSD_CODES

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "corpus"
CACHE_DIR = HERE / ".cache" / "fsd"

TODAY = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")

# 식품안전나라는 Referer 를 안 보내면 500 을 준다
client = httpx.Client(
    timeout=180.0,
    follow_redirects=True,
    headers={"User-Agent": "Mozilla/5.0", "Referer": f"{FSD_BASE}/"},
)


# ──────────────────────────────────────────────────────── 받아오기


def fsd_tree(pubtlgr: str) -> list[dict]:
    res = client.post(f"{FSD_BASE}/api/tree/getPubtlgrTree", json={"pubtlgrCode": pubtlgr})
    res.raise_for_status()
    rows = res.json().get("data", {}).get("default", {}).get("list")
    if not rows:
        raise RuntimeError(f"공전 트리가 비어 있다: {pubtlgr}")
    return rows


def resolve_item(tree: list[dict], name_path: list[str]) -> dict:
    """이름 경로로 항목을 찾는다. itemCode 는 고시가 바뀌면 밀릴 수 있어 키로 쓰지 않는다."""
    parent = None
    node = None
    for name in name_path:
        hits = [
            x
            for x in tree
            if squash(x["name"]) == name and (parent is None or x.get("upperIemCode") == parent)
        ]
        if len(hits) != 1:
            raise RuntimeError(f"항목을 하나로 못 좁혔다: {' > '.join(name_path)} — '{name}' {len(hits)}건")
        node = hits[0]
        parent = node["itemCode"]
    if not node or not node.get("fileId"):
        raise RuntimeError(f"PDF 가 없는 항목이다(폴더): {' > '.join(name_path)}")
    return node


def item_file_url(node: dict) -> str:
    return (
        f"{FSD_BASE}/api/object-storage/download/pubtlgr_iem_pdf/"
        f"{node['pubtlgrCode']}_{node['ntfcNo']}_{node['itemCode']}/{node['fileId']}"
    )


def item_viewer_url(node: dict) -> str:
    """사람이 원문의 그 자리를 여는 주소. 앱이 쓰는 뷰어 URL 그대로다."""
    return f"{FSD_BASE}/vendor/pdfjs/web/viewer.html?file=/fsd/api/object-storage/download/pubtlgr_iem_pdf/{node['pubtlgrCode']}_{node['ntfcNo']}_{node['itemCode']}/{node['fileId']}"


def download(node: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{node['pubtlgrCode']}_{node['ntfcNo']}_{node['itemCode']}.pdf"
    if not path.exists() or path.stat().st_size == 0:
        res = client.get(item_file_url(node))
        res.raise_for_status()
        path.write_bytes(res.content)
    return path


# ───────────────────────────────────────────── 미시행 개정 주석 걷어내기
#
# 온라인 공전 본문은 **오늘 시행 중인 통합본**이고, 아직 시행되지 않은 개정은 본문 안에
#   <제2026-50호, ‘26.7.13.>[시행일 : ‘28.1.1.] 새 문구…
# 형태의 주석으로 덧붙어 있다 (실측: 일반식품첨가물 91곳). 이걸 그대로 두면 한 청크에
# 현행 문구와 2028년 문구가 같이 담겨, 쓰는 쪽에서 어느 쪽이 지금 기준인지 구분할 수 없다.
# PRD 4절이 "현행 시행본을 쓰고 미시행 개정본은 제외한다"고 못박은 자리다.

AMEND = re.compile(r"<\s*(?:식약처\s*고시\s*)?제\s*\d{4}\s*-\s*\d+\s*호[^>]{0,80}>")
EFFECT = re.compile(
    r"\[\s*시행일\s*[:：]?\s*[‘'’\"]?\s*(\d{2,4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*\]"
)


# ───────────────────────────────────────── PDF 가 끊은 줄을 되붙이기
#
# PDF 는 좁은 칸 안에서 **단어 한가운데를 끊는다** — 「사용/하여야」, 「안식향/산의」,
# 「0.05g/kg 이/하」(/ 가 줄바꿈 자리다). 줄을 공백으로 이으면 「사용 하여야」, 「이 하」가 되어 숫자와 낱말이
# 쪼개진다. 이 제품에서 그 숫자가 곧 답이라 그냥 둘 수 없다 (PRD 1절).
#
# 그래서 **한글과 한글 사이는 공백 없이 잇는다.** 줄이 마침 띄어쓰기 자리에서 끊긴
# 경우에만 공백 하나가 사라진다(「아래의/식품에」→「아래의식품에」). 실측 표본 하나에서
# 공백으로 이으면 8군데가 깨졌고, 붙여 이으면 1군데가 붙었다 — 깨진 숫자보다 붙은 낱말이 낫다.
HANGUL = re.compile(r"[가-힣]")


def tidy(s: object) -> str:
    """줄바꿈은 살리고 그 외 공백만 접는다."""
    return re.sub(r"[^\S\n]+", " ", "" if s is None else str(s)).strip()


def join_lines(text: object) -> str:
    lines = [l.strip() for l in str(text or "").split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return ""
    out = lines[0]
    for line in lines[1:]:
        glue = "" if HANGUL.fullmatch(out[-1]) and HANGUL.fullmatch(line[0]) else " "
        out += glue + line
    return squash(out)


def _effect_date(m: re.Match[str]) -> str:
    year = m.group(1)
    year = year if len(year) == 4 else f"20{year}"
    return f"{year}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def split_pending(text: str, keep_lines: bool = False) -> tuple[str, list[dict]]:
    """미시행 개정을 잘라낸다. 돌려주는 것은 (오늘 본문, [잘라낸 미시행 조각]).

    자르는 위치는 `[시행일 …]` 바로 앞의 개정 표지(`<제…호…>`)다 — 표지가 없으면
    `[시행일` 자리에서 자른다. 이미 시행된 날짜(오늘 이하)의 주석은 본문이 이미 그 문구이므로
    표지만 지우고 본문은 그대로 둔다.

    `keep_lines` 는 줄글용이다. 줄을 접으면 뒤의 목차 파싱이 자리를 잃는다.

    **줄글은 자르지 않는다.** 표 칸에서는 주석이 칸 끝까지 이어져 자르는 자리가 분명하지만,
    줄글에서는 개정 상자가 본문 위에 떠 있어 주석 뒤에도 현행 본문이 이어진다 (실측: 제2. 3.
    식품일반의 오염물질 표). 거기서 자르면 시행 중인 기준을 통째로 버린다. 그래서 표지만 지우고
    **어디에 걸쳐 있는지를 기록**해 둔다 — 잘라낸 것과 걸쳐 있는 것을 구분하려고 `kind` 를 붙인다.
    """
    t = tidy(text) if keep_lines else squash(text)
    cut = None
    pendings = []
    for m in EFFECT.finditer(t):
        date = _effect_date(m)
        if date <= TODAY:
            continue
        start = m.start()
        for a in AMEND.finditer(t, max(0, m.start() - 160), m.end() + 160):
            if a.start() < m.start():
                start = min(start, a.start())
        pendings.append({"effectiveDate": date, "at": start})
        cut = start if cut is None else min(cut, start)

    strip = lambda s: EFFECT.sub(" ", AMEND.sub(" ", s))
    for p in pendings:
        at = p.pop("at")
        if keep_lines:
            p["kind"] = "본문에 겹침 (자르지 않음)"
            p["text"] = join_lines(strip(t[at : at + 300]))
        else:
            p["kind"] = "잘라냄"
            p["text"] = join_lines(strip(t[at:]))[:400]
    if cut is not None and not keep_lines:
        t = t[:cut]
    kept = strip(t)
    return (tidy(kept) if keep_lines else join_lines(kept)), pendings


# ───────────────────────────────────────────────── PDF → 블록 읽기


def page_blocks(page) -> list[tuple[str, object]]:
    """한 페이지를 세로 순서대로 (`text` | `table`) 블록으로 쪼갠다.

    표를 따로 떼지 않으면 `extract_text()` 가 열을 가로로 이어 붙여
    「안식향산 0.6 이하(안식향산으로서…」가 품목과 사용량이 뒤섞인 한 줄이 된다.
    """
    x0, top0, x1, bottom0 = page.bbox
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    blocks: list[tuple[str, object]] = []
    top = top0

    def band(a: float, b: float) -> None:
        if b - a < 2:
            return
        txt = page.crop((x0, a, x1, b)).extract_text() or ""
        if txt.strip():
            blocks.append(("text", txt))

    for t in tables:
        band(top, max(top, t.bbox[1]))
        blocks.append(("table", t.extract()))
        top = max(top, t.bbox[3])
    band(top, bottom0)
    return blocks


ITEM_HEADER = "품목명"


def header_map(row: list) -> dict[str, int] | None:
    """품목별 사용기준 표의 헤더면 열 위치를, 아니면 None."""
    cells = [squash(c) for c in row]
    if ITEM_HEADER not in cells:
        return None
    out = {"name": cells.index(ITEM_HEADER)}
    for key, label in (("use", "사용기준"), ("purpose", "주용도"), ("nutrient", "영양소"), ("define", "정의")):
        hit = next((i for i, c in enumerate(cells) if c == label), None)
        if hit is not None:
            out[key] = hit
    return out if "use" in out else None


def render_table(rows: list[list]) -> str:
    """품목 표가 아닌 표(규격표 등)는 행을 줄로 편다."""
    lines = []
    for row in rows:
        cells = [squash(c) for c in row if squash(c)]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def read_pdf(path: Path, known: dict[str, str]) -> tuple[str, list[dict]]:
    """(줄글, 품목 행 목록). 품목 표가 없는 문서는 두 번째가 빈 목록이다."""
    prose: list[str] = []
    items: list[dict] = []
    cols: dict[str, int] | None = None
    nutrient = ""

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for kind, payload in page_blocks(page):
                if kind == "text":
                    prose.append(payload)
                    continue

                rows = payload
                head = header_map(rows[0]) if rows else None
                if head:
                    cols = head
                    rows = rows[1:]
                elif cols is None or len(rows[0]) != max(cols.values()) + 1:
                    # 품목 표가 시작되기 전이거나 열 수가 다른 표 — 줄글로 보낸다
                    prose.append(render_table(rows))
                    continue

                for row in rows:
                    if len(row) <= cols["name"]:
                        continue
                    def take(k, row=row):
                        if k not in cols or cols[k] >= len(row):
                            return ""
                        return (purpose_text if k == "purpose" else join_lines)(row[cols[k]])

                    if "nutrient" in cols and take("nutrient"):
                        nutrient = take("nutrient")
                    name, alias = item_name(row[cols["name"]], known)
                    if not name:
                        # 품목명이 빈 행은 앞 품목이 쪽을 넘어 이어진 것이다
                        if items:
                            for key in ("define", "use", "purpose"):
                                part = take(key)
                                # 셀이 세로로 병합된 표에서는 같은 문장이 다시 온다
                                if part and part not in items[-1][key]:
                                    merged = join_lines(f"{items[-1][key]}\n{part}")
                                    # 이어 붙인 뒤 다시 거른다 — 부스러기만 든 조각은 여기서 떨어진다
                                    items[-1][key] = purpose_text(merged) if key == "purpose" else merged
                        continue
                    items.append(
                        {
                            "name": name,
                            "alias": alias,
                            "nutrient": nutrient if "nutrient" in cols else "",
                            "define": take("define"),
                            "use": take("use"),
                            "purpose": take("purpose"),
                        }
                    )
    return "\n".join(prose), items


# 품목명 칸에는 국문명 아래로 영문명·이명이 줄바꿈으로 딸려 온다 —
#   「비타민C**」 「L-Ascorbic Acid」 「이명 :」 「Ascorbic acid」
# 뒤엣것을 이름에 붙이면 「비타민C」로 찾을 수 없다. 그렇다고 첫 줄만 쓰면 이번엔 국문명 자체가
# 칸 안에서 끊긴 것을 놓친다 — 「디부틸히드록시」 「톨루엔」이 BHT 를 통째로 사라지게 했다(실측).
# 몇 글자에서 끊기는지는 쪽마다 다르다.
#
# 그래서 **공전이 가진 품목명 목록으로 되짚는다.** 트리 API 의 「IV. 품목별 성분규격」이
# 품목 하나에 파일 하나라 이름 사전이 거기 있다. 앞 줄들을 하나씩 이어 붙이며 사전에 있는
# 가장 긴 이름을 고르고, 없으면 첫 줄로 돌아간다.
#
# `*`/`**` 는 "일반식품첨가물/가공보조제 용도로도 쓸 수 있다"는 각주 표시라 이름이 아니다.
# 아래첨자(비타민B₂)는 별도 줄로 떨어져 오므로 숫자만 있는 줄도 이름의 일부로 본다.
STAR = re.compile(r"[*＊]+")


DIGITS = re.compile(r"^\d+$")
LATIN_RUN = re.compile(r"[A-Za-z]+")


def item_name(cell: object, known: dict[str, str]) -> tuple[str, str]:
    lines = [STAR.sub("", l).strip() for l in str(cell or "").split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return "", ""

    head: list[str] = []
    for line in lines:
        if not (HANGUL.search(line) or DIGITS.match(line)):
            break
        head.append(line)
    head = head or lines[:1]

    best, name = 1, ""
    for n in range(1, len(head) + 1):
        for cand in _name_candidates(head[:n]):
            hit = known.get(cand.replace(" ", ""))
            if hit:
                best, name = n, hit  # 사전에 적힌 철자를 쓴다 — 우리가 복원한 것보다 낫다
    if not name:
        name = join_lines(head[0])
    return name, join_lines("\n".join(lines[best:]))


def _name_candidates(head: list[str]) -> list[str]:
    """아래첨자 줄이 이름의 어디에 붙는지 몰라서 두 가지로 놓아 본다.

    비타민B₂ 는 「비타민B」 다음 줄에 「2」로 오지만(→ 뒤에 붙임),
    비타민B₂인산에스테르나트륨은 「비타민B 인산에 / 2 / 스테르나트륨」으로 온다
    (→ 첫 영문자 뒤에 끼움). 사전에 맞는 쪽만 남는다.
    """
    plain = join_lines("\n".join(head)).replace(" ", "")
    digits = [l for l in head if DIGITS.match(l)]
    rest = [l for l in head if not DIGITS.match(l)]
    if not digits:
        return [plain]
    body = join_lines("\n".join(rest)).replace(" ", "")
    m = LATIN_RUN.search(body)
    inserted = body[: m.end()] + "".join(digits) + body[m.end() :] if m else body
    return [plain, inserted]


def known_names(tree: list[dict]) -> dict[str, str]:
    """공전이 가진 품목명 사전. 띄어쓰기를 지운 것을 열쇠로 둔다 — PDF 쪽 띄어쓰기는 못 믿는다."""
    return {squash(x["name"]).replace(" ", ""): squash(x["name"]) for x in tree if x.get("fileId")}


# 주용도 칸은 용도를 한 줄에 하나씩 쌓아 두는데, 칸이 좁아 그 안에서 또 줄이 끊긴다 —
# 「산화방지/제/산도조절/제」. 게다가 미시행 개정을 덧대는 상자가 이 칸 위로 겹쳐서
# 「실주를」 「g)」 「이상을」 같은 옆 칸 부스러기가 섞여 들어온다 (실측 48건).
#
# 용도는 공전이 정한 **닫힌 목록**이라 붙여 놓고 목록으로 되짚으면 둘 다 풀린다.
# 긴 것부터 맞춰 「거품제거제」가 「거품제 + 거제」로 갈라지지 않게 한다.
PURPOSES = (
    "감미료", "거품제거제", "고결방지제", "껌기초제", "밀가루개량제", "발색제", "보존료",
    "분사제", "산도조절제", "산화방지제", "살균제", "습윤제", "안정제", "여과보조제",
    "영양강화제", "유화제", "운반체", "응고제", "이형제", "제조용제",
    "젤형성제", "증점제", "착색료", "청관제", "추출용제", "충전제", "팽창제", "표면처리제",
    "표백제", "피막제", "향료", "향미증진제", "효소제",
)
PURPOSE_RE = re.compile("|".join(sorted(PURPOSES, key=len, reverse=True)))


def purpose_text(cell: object) -> str:
    joined = join_lines(cell)
    hits = list(dict.fromkeys(PURPOSE_RE.findall(joined)))
    # 목록에 없는 값이면 지우지 않고 그대로 넘긴다 — 모르는 용도를 조용히 버리지 않는다
    return " ".join(hits) if hits else joined


# ─────────────────────────────────────────────────────── 청킹
#
# 공전의 계층은 법령의 조/항/호가 아니다 —
#   제5. 9. 음료류  →  9-3 과일･채소류음료  →  4) 식품유형  →  (2) 과･채주스
# 사실 단위는 `N)` 이다. 「4) 식품유형」 하나에 그 유형군의 정의가 다 들어 있어서
# 더 쪼개면 "과･채주스 95%"와 "과･채음료 10%"가 서로 다른 청크로 흩어진다.


def line_cuts(lines: list[str], pattern: re.Pattern[str]) -> list[int]:
    """1부터 순서대로 오르는 줄머리 마커의 줄 번호. 본문 속 참조는 순번이 안 맞아 걸러진다."""
    hits = [(i, int(m.group(1))) for i, l in enumerate(lines) if (m := pattern.match(l))]
    out, expect = [], 1
    for i, n in hits:
        if n == expect:
            out.append(i)
            expect += 1
    return out


GROUP_TITLE_MAX = 30
ITEM_RE = re.compile(r"^\s*(\d+)\)\s*")


def outline_chunks(text: str, section: str, group_no: int | None) -> list[dict]:
    lines = [l.strip() for l in text.split("\n")]
    out: list[dict] = []

    def emit(group: str, item_lines: list[str]) -> None:
        body = join_lines("\n".join(item_lines))
        if len(body) < 10:
            return
        head = squash(item_lines[0])[:GROUP_TITLE_MAX]
        path = " ".join(x for x in (section, group, head) if x)
        # 쪼개진 조각도 단독으로 읽히게 유형군 제목을 앞에 붙인다 (슬라이스 1의 stem 과 같은 이유)
        for piece in split_long(body, group):
            out.append({"path": path, "text": piece})

    def by_item(chunk_lines: list[str], group: str) -> None:
        at = line_cuts(chunk_lines, ITEM_RE)
        if not at:
            emit(group, chunk_lines)
            return
        if at[0] > 0:
            emit(group, chunk_lines[: at[0]])
        for k, start in enumerate(at):
            stop = at[k + 1] if k + 1 < len(at) else len(chunk_lines)
            emit(group, chunk_lines[start:stop])

    if group_no is None:
        by_item(lines, "")
        return out

    group_re = re.compile(rf"^{group_no}-(\d+)\s+(\S.{{0,{GROUP_TITLE_MAX}}})$")
    at = line_cuts(lines, group_re)
    if not at:
        by_item(lines, "")
        return out
    if at[0] > 0:
        by_item(lines[: at[0]], "")
    for k, start in enumerate(at):
        stop = at[k + 1] if k + 1 < len(at) else len(lines)
        by_item(lines[start + 1 : stop], squash(lines[start]))
    return out


def item_chunks(items: list[dict], section: str) -> list[dict]:
    """품목 하나 = 청크 하나. 사용기준이 먼저 오게 둔다 — 질문이 묻는 것이 그것이다."""
    out = []
    for it in items:
        # 이명을 이름 옆에 둔다 — 「비타민C」와 「L-아스코르브산」이 같은 품목이라는 것을
        # 이 줄 말고는 알 데가 없다 (슬라이스 3의 별칭 인덱스가 여기서 나온다)
        head = f"{it['name']} ({it['alias']})" if it["alias"] else it["name"]
        parts = []
        if it["nutrient"]:
            parts.append(f"영양소: {it['nutrient']}")
        if it["purpose"]:
            parts.append(f"주용도: {it['purpose']}")
        parts.append(f"사용기준: {it['use'] or '(따로 정하여진 사용기준 없음)'}")
        if it["define"]:
            parts.append(f"정의: {it['define']}")
        path = f"{section} {it['name']}"
        # 길어서 쪼개지더라도 조각마다 품목명이 앞에 붙는다 — 옆 품목의 사용량과 섞이면 안 된다
        for piece in split_long(squash(" / ".join(parts)), head):
            out.append({"path": path, "text": piece})
    return out


# ────────────────────────────────────────────────────────────── 실행


def source_dates(meta: dict, tree_row: dict) -> dict:
    """공전의 날짜 셋을 갈라 적는다.

    식품안전나라는 **고시일만** 주고 시행일을 주지 않는다. 그리고 지금 서비스되는 라벨
    (예: 식품공전 제2026-55호)은 아직 시행 전일 수 있다 — 실측으로 시행 2026-10-01 이었다.
    본문은 오늘 시행 중인 통합본이므로(미시행 개정은 본문 안에 주석으로 붙어 있다),
    청크의 시행일은 **오늘 시행 중인 판의 시행일**로 적고 라벨은 따로 남긴다.
    """
    ver = resolve_admrul_version(meta)
    picked = ver["picked"]
    if not picked:
        raise RuntimeError(f"오늘 시행 중인 판이 없음: {meta['law_name']}")
    label_no = str(tree_row["ntfcNo"])
    label = f"제{label_no[:4]}-{int(label_no[4:])}호"
    label_date = tree_row["ntfcDe"]
    label_row = next(
        (
            v
            for v in ver["versions"]
            if v["promulgated"] == f"{label_date[:4]}-{label_date[4:6]}-{label_date[6:8]}"
        ),
        None,
    )
    return {
        "effectiveDate": picked["effective"],
        "effectiveNoticeSeq": picked["seq"],
        "promulgated": picked["promulgated"],
        "fsdNoticeNo": label,
        "fsdNoticeDate": f"{label_date[:4]}-{label_date[4:6]}-{label_date[6:8]}",
        "fsdNoticeEffective": (label_row or {}).get("effective"),
    }


def collect_source(meta: dict) -> dict:
    tree = fsd_tree(meta["pubtlgr"])
    known = known_names(tree)
    dates = source_dates(meta, tree[0])
    chunks: list[dict] = []
    pendings: list[dict] = []
    items_report = []
    seq = 0

    for spec in meta["items"]:
        node = resolve_item(tree, spec["path"])
        pdf_path = download(node)
        prose, rows = read_pdf(pdf_path, known)

        prose, prose_pending = split_pending(prose, keep_lines=True)
        for p in prose_pending:
            pendings.append({"section": spec["section"], "where": "줄글", **p})

        parts = outline_chunks(prose, spec["section"], spec.get("group_no"))
        for row in rows:
            for key in ("define", "use", "purpose"):
                row[key], pend = split_pending(row[key])
                for p in pend:
                    pendings.append({"section": spec["section"], "where": row["name"], **p})
        parts += item_chunks(rows, spec["section"])

        for c in parts:
            seq += 1
            chunks.append(
                {
                    "id": f"{meta['code']}-{seq:03d}",
                    "source": meta["short_name"],
                    "sourceKind": meta["kind"],
                    "lawName": meta["law_name"],
                    "path": c["path"],
                    "text": c["text"],
                    "url": item_viewer_url(node),
                    "effectiveDate": dates["effectiveDate"],
                    "inForce": dates["effectiveDate"] <= TODAY,
                }
            )
        items_report.append(
            {
                "section": spec["section"],
                "itemCode": node["itemCode"],
                "chunks": len(parts),
                "products": len(rows),
                "chars": len(prose) + sum(len(r["use"]) for r in rows),
                "pdfBytes": pdf_path.stat().st_size,
                "url": item_viewer_url(node),
            }
        )

    return {"dates": dates, "chunks": chunks, "items": items_report, "pending": pendings}


def write_source(meta: dict, r: dict) -> None:
    (OUT_DIR / f"{meta['code']}.json").write_text(
        json.dumps(
            {
                "code": meta["code"],
                "source": meta["short_name"],
                "sourceKind": meta["kind"],
                "lawName": meta["law_name"],
                "category": meta["category"],
                **r["dates"],
                "collectedAt": TODAY,
                "url": f"{FSD_BASE}/#/ext/Document/{meta['pubtlgr']}",
                "sections": r["items"],
                "pendingAmendments": r["pending"],
                "chunks": r["chunks"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────── 완료 기준 확인
#
# PLAN 슬라이스 2의 완료 기준을 스크립트가 스스로 확인한다. 통과 못 하면 종료 코드 1.

# (a) 이 넷의 정의가 비어 있지 않아야 한다. 가운뎃점 표기가 원문마다 달라(`･` `·`) 느슨히 찾는다
FOOD_TYPES = ["과.?채주스", "과.?채음료", "혼합음료", "액상차"]
# (c) 품목명으로 사용기준이 잡혀야 한다. UC-3 의 비타민C·구연산과 음료 보존료를 본다
ADDITIVES = ["비타민C", "구연산", "안식향산나트륨"]
REQUIRED = ("lawName", "path", "url", "effectiveDate", "text")


def verify(by_code: dict[str, list[dict]]) -> int:
    problems = []
    for chunks in by_code.values():
        for c in chunks:
            missing = [k for k in REQUIRED if not c.get(k)]
            if missing:
                problems.append(f"{c['id']}: 빈 필드 {', '.join(missing)}")
            if c.get("inForce") is not True:
                problems.append(f"{c['id']}: inForce={c.get('inForce')}")

    print("\n완료 기준 확인")
    food = by_code.get("FDC", [])
    for pat in FOOD_TYPES:
        hits = [c for c in food if re.search(pat, c["text"])]
        # 정의문인지 본다 — 유형 이름이 나오고 그 문장이 「…말한다」로 닫히는 자리
        defined = [c for c in hits if re.search(rf"{pat}[^.]{{0,200}}말한다", c["text"])]
        mark = "✓" if defined else "✗"
        print(f"   {mark} (a) {pat}: 청크 {len(hits)}개, 정의문 {len(defined)}개")
        if not defined:
            problems.append(f"(a) 음료류 유형 정의를 못 찾음: {pat}")

    add = by_code.get("FAC", [])
    for name in ADDITIVES:
        hits = [c for c in add if c["path"].endswith(f" {name}") and "사용기준:" in c["text"]]
        mark = "✓" if hits else "✗"
        print(f"   {mark} (c) {name}: 사용기준 섹션 {len(hits)}개")
        if not hits:
            problems.append(f"(c) 첨가물 사용기준 섹션을 못 찾음: {name}")

    if problems:
        print(f"\n✗ 검증 실패 {len(problems)}건")
        for p in problems[:20]:
            print(f"   {p}")
        return 1
    total = sum(len(v) for v in by_code.values())
    print(f"\n✓ 검증 통과 — {total} 청크 모두 필수 필드가 차 있고 inForce=true")
    return 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"collectedAt": TODAY, "sources": []}
    by_code: dict[str, list[dict]] = {}

    for meta in FSD_CODES:
        r = collect_source(meta)
        write_source(meta, r)
        by_code[meta["code"]] = r["chunks"]
        report["sources"].append(
            {
                "code": meta["code"],
                "name": meta["short_name"],
                "lawName": meta["law_name"],
                "category": meta["category"],
                **r["dates"],
                "chunks": len(r["chunks"]),
                "sections": r["items"],
                "pendingAmendments": r["pending"],
            }
        )

    (OUT_DIR / "collection-report-fsd.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n수집일 {TODAY}\n")
    for s in report["sources"]:
        print(f"{s['name']} ({s['lawName']})")
        print(
            f"   본문 시행일 {s['effectiveDate']}"
            f" · 식품안전나라 라벨 {s['fsdNoticeNo']} (고시 {s['fsdNoticeDate']},"
            f" 시행 {s['fsdNoticeEffective'] or '?'})"
        )
        for it in s["sections"]:
            products = f" · 품목 {it['products']}" if it["products"] else ""
            print(
                f"   {it['section']:28s} 청크 {it['chunks']:4d}{products}"
                f"  (PDF {round(it['pdfBytes'] / 1024):,}KB)"
            )
        if s["pendingAmendments"]:
            dates = sorted({p["effectiveDate"] for p in s["pendingAmendments"]})
            cut = sum(1 for p in s["pendingAmendments"] if p["kind"] == "잘라냄")
            print(
                f"   ⚠ 미시행 개정 {len(s['pendingAmendments'])}건 (잘라냄 {cut} ·"
                f" 본문에 겹침 {len(s['pendingAmendments']) - cut}) — 시행일 {', '.join(dates)}"
            )
        print(f"   총 {s['chunks']} 청크")
    print(f"\n→ {OUT_DIR}")
    return verify(by_code)


if __name__ == "__main__":
    raise SystemExit(main())
