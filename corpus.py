"""
수집해 둔 청크를 **하나의 목록으로** 읽어 들이는 로더.

슬라이스 1·2 는 자료를 `data/corpus/<CODE>.json` 과 `tables-<CODE>.json` 두 모양으로 남겼다.
본문 청크는 그대로 쓰면 되지만 별표는 `lines` 배열(ASCII 표 또는 줄글)이라 그대로는
근거가 되지 못한다. 여기서 별표도 같은 스키마의 청크로 만든다 — 도구는 출처가 본문인지
별표인지 신경 쓰지 않고 한 목록만 본다.

다시 수집해도 ID 가 밀리지 않도록 **표 청크 ID 는 본문과 다른 이름공간**을 쓴다:
`LBL-T4-003` = 표시기준 별표 4의 셋째 조각. 별표→T · 별지→F · 별도→D.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"

# 본문 청크 파일. 슬라이스 1(법제처) 4종 + 슬라이스 2(식품안전나라) 2종.
SOURCE_CODES = ["FLA", "FLD", "LBL", "UNF", "FDC", "FAC"]

# 별표 파일이 있는 자료. 법률·시행령은 슬라이스 1이 별표를 받지 않았다 (docs/CATEGORY-MAP.md 참고).
TABLE_CODES = ["LBL", "UNF"]

KIND_TAG = {"별표": "T", "별지": "F", "별도": "D"}

MAX_TABLE_CHUNK = 1100


@dataclass(frozen=True)
class Chunk:
    """근거 한 조각. 본문과 별표가 같은 모양이어야 도구가 한 목록만 보면 된다."""

    id: str
    code: str
    source: str
    source_kind: str
    law_name: str
    path: str
    text: str
    url: str
    effective_date: str
    in_force: bool = True

    @property
    def is_table(self) -> bool:
        return "-T" in self.id or "-F" in self.id or "-D" in self.id

    @property
    def head(self) -> str:
        """본문 첫머리. 청킹 규칙이 조각마다 계층 제목을 앞에 붙여 두었다."""
        return re.sub(r"\s+", " ", self.text[:120])


# ──────────────────────────────────────────────────────────── 별표 파싱
#
# 별표는 두 모양으로 온다.
#   (1) ASCII 상자 표 — `┃칸│칸┃`, 행 사이가 `┠─┼─┨`. 첨가물 표시 별표 4·5·6 이 이 모양이다
#   (2) 줄글 — 별지1(표시사항별 세부표시기준)처럼 계층 표기가 붙은 문단
# 섞여 있는 것(별지1 안의 작은 표)은 줄글로 다룬다. 상자 줄을 그대로 남겨도 읽을 수 있다.

_BOX_EDGE = "┏┣┠┗"
_BOX_ONLY = set("┏┓┗┛┠┨┣┫┯┷┿━─ ")


def _is_box_table(lines: list[str]) -> bool:
    body = [l for l in lines if l.strip()]
    if not body:
        return False
    boxy = sum(1 for l in body if l.lstrip()[:1] in _BOX_EDGE or "┃" in l)
    return boxy / len(body) > 0.7


def _box_rows(lines: list[str]) -> list[list[str]]:
    """구분선(`┠─┼─┨`)을 경계로 **논리 행** 하나씩 돌려준다.

    한 논리 행이 여러 줄에 걸치는 경우가 둘 있고 둘 다 이어 붙이는 것이 맞다.
      - 칸 안에서 줄이 넘친 것 — 「산도조절제,」 / 「영양강화제」
      - 용도 칸이 세로로 병합된 것 — 별표 4의 감미료 묶음. 이름들이 한 행에 모인다
    """
    rows: list[list[str]] = []
    block: list[list[str]] = []

    def flush() -> None:
        if not block:
            return
        width = max(len(r) for r in block)
        merged = []
        for i in range(width):
            parts = [r[i] for r in block if i < len(r) and r[i]]
            merged.append(re.sub(r"\s+,", ",", " ".join(parts)).strip())
        if any(merged):
            rows.append(merged)
        block.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in _BOX_EDGE or set(stripped) <= _BOX_ONLY:
            flush()
            continue
        if "┃" not in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("┃").split("│")]
        block.append(cells)
    flush()
    return rows


def _chunk_box_table(rows: list[list[str]]) -> list[str]:
    """머리 행을 조각마다 붙여 나눈다 — 칸 이름이 없으면 「간략명」인지 「주용도」인지 모른다."""
    if not rows:
        return []
    head = " | ".join(rows[0])
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for row in rows[1:]:
        line = " | ".join(row)
        if buf and size + len(line) > MAX_TABLE_CHUNK:
            out.append("\n".join([head, *buf]))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        out.append("\n".join([head, *buf]))
    return out or [head]


# 줄글 별표의 계층 제목. **짧은 줄만** 제목으로 본다 —
# 「가. 제품명」은 제목이고 「가) 식품의 처리ㆍ제조…」는 본문이다.
_H1 = re.compile(r"^(\d+\.)\s*(.{0,30})$")
_H2 = re.compile(r"^([가-힣]\.)\s*(.{0,30})$")


def _chunk_prose_table(lines: list[str]) -> list[tuple[str, str]]:
    """(위치, 본문) 목록. 위치는 줄글 안에서 읽은 계층 제목이다."""
    out: list[tuple[str, str]] = []
    h1 = h2 = ""
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            out.append((" ".join(x for x in (h1, h2) if x).strip(), "\n".join(buf)))
        buf, size = [], 0

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        m1, m2 = _H1.match(line), _H2.match(line)
        if m1:
            flush()
            h1, h2 = f"{m1.group(1)} {m1.group(2)}".strip(), ""
        elif m2:
            flush()
            h2 = f"{m2.group(1)} {m2.group(2)}".strip()
        elif size + len(line) > MAX_TABLE_CHUNK:
            flush()
        buf.append(line)
        size += len(line) + 1
    flush()
    return out


def _table_chunks(code: str, meta: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for table in meta["tables"]:
        tag = KIND_TAG.get(table["kind"], "X")
        no = str(int(table["no"] or 0))
        label = f"{table['kind']} {no} {table['title']}".strip()
        url = table.get("pdf") or meta["url"]
        lines = table["lines"]

        if _is_box_table(lines):
            pieces = [(label, text) for text in _chunk_box_table(_box_rows(lines))]
        else:
            pieces = [
                (f"{label} {where}".strip(), text)
                for where, text in _chunk_prose_table(lines)
            ]

        for i, (path, text) in enumerate(pieces, start=1):
            chunks.append(
                Chunk(
                    id=f"{code}-{tag}{no}-{i:03d}",
                    code=code,
                    source=meta["lawName"],
                    source_kind="고시 별표",
                    law_name=meta["lawName"],
                    path=path,
                    text=text,
                    url=url,
                    effective_date=meta["effectiveDate"],
                )
            )
    return chunks


# ──────────────────────────────────────────────────────────────── 로더


@dataclass
class Corpus:
    chunks: list[Chunk]
    sources: dict[str, dict]  # code → 자료 메타(시행일·출처 URL 등)

    @functools.cached_property
    def by_id(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}

    def select(self, code: str, path_pattern: str = "") -> list[Chunk]:
        """자료 코드 + 위치 정규식으로 범위를 자른다.

        ID 가 아니라 **위치**로 고르는 이유 — 코퍼스를 다시 수집하면 ID 는 밀리지만
        「제5. 9. 음료류」라는 자리는 그대로다 (PLAN 검증절의 함정 2).

        `path` 뿐 아니라 **본문 첫머리**로도 맞춰 본다. 위치 표기는 단정할 수 없으면
        조상을 접는 규칙이라(`chunking.section_path`), 같은 절의 청크 하나가
        「Ⅲ. 개별표시사항 및 표시기준 1. 식품」까지만 남는 일이 있다. 그 청크의 본문은
        「… 자. 음료류 2) 표시사항 거) 기타표시사항」으로 시작하므로 첫머리를 같이 보면
        절이 통째로 잡힌다 — 실측으로 음료류 「기타표시사항」이 이 경우였다.
        """
        rx = re.compile(path_pattern) if path_pattern else None
        return [
            c
            for c in self.chunks
            if c.code == code
            and (rx is None or rx.search(c.path) or rx.search(c.head))
        ]


@functools.cache
def load() -> Corpus:
    chunks: list[Chunk] = []
    sources: dict[str, dict] = {}

    for code in SOURCE_CODES:
        raw = json.loads((CORPUS_DIR / f"{code}.json").read_text(encoding="utf-8"))
        sources[code] = {k: v for k, v in raw.items() if k != "chunks"}
        for c in raw["chunks"]:
            chunks.append(
                Chunk(
                    id=c["id"],
                    code=code,
                    source=c["source"],
                    source_kind=c["sourceKind"],
                    law_name=c["lawName"],
                    path=c["path"],
                    text=c["text"],
                    url=c["url"],
                    effective_date=c["effectiveDate"],
                    in_force=c.get("inForce", True),
                )
            )

    for code in TABLE_CODES:
        path = CORPUS_DIR / f"tables-{code}.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(_table_chunks(code, meta))
        sources[code]["tables"] = len(meta["tables"])

    return Corpus(chunks=chunks, sources=sources)


if __name__ == "__main__":  # 적재 상태를 눈으로 보는 용도
    import collections

    corpus = load()
    n = collections.Counter(c.code for c in corpus.chunks)
    t = collections.Counter(c.code for c in corpus.chunks if c.is_table)
    print(f"{'code':6}{'본문':>7}{'별표':>7}{'글자':>10}  시행일")
    for code in SOURCE_CODES:
        chars = sum(len(c.text) for c in corpus.chunks if c.code == code)
        print(
            f"{code:6}{n[code] - t[code]:>7}{t[code]:>7}{chars:>10,}  "
            f"{corpus.sources[code]['effectiveDate']}"
        )
    print(f"{'합계':6}{len(corpus.chunks) - sum(t.values()):>7}{sum(t.values()):>7}")
