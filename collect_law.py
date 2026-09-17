"""
법제처 국가법령정보센터 OPEN API → 사실 단위 청크 JSON.

    uv run python collect_law.py

환경변수 `LAW_OC` 에 본인 OC(law.go.kr 회원 아이디 앞부분)를 넣는다. 없으면 'test' 로 돈다.
산출물은 `data/corpus/` 아래 자료별 JSON 과 수집 리포트, 그리고 고시 별표 파일이다.

왜 스크립트인가 — 수집·파싱은 판단이 아니라 결정론적 작업이다. 한 번 받아 커밋해 두면
사이트가 바뀌어도 구현과 채점이 계속 돌아간다 (PLAN D6).

**스크래핑하지 않는다.** 법제처는 JSON API 를 준다 — HTML 을 긁으면 1.3KB 껍데기만 온다.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

from chunking import (
    as_list,
    iso_date,
    marker_cuts,
    section_path,
    split_long,
    squash,
    url_name,
)
from sources import ADMRULS, LAWS, OUT_OF_SCOPE

# 한국어 리포트가 콘솔 인코딩(cp949)에 걸려 깨지지 않게 한다
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

OC = os.environ.get("LAW_OC") or "test"
SERVICE = "https://www.law.go.kr/DRF/lawService.do"
SEARCH = "https://www.law.go.kr/DRF/lawSearch.do"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "corpus"

# 오늘 (KST). 시행 여부 판정 기준.
TODAY = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")

client = httpx.Client(timeout=60.0, follow_redirects=True)


def fetch_json(base: str, params: dict[str, str]) -> dict:
    url = f"{base}?{urlencode({'OC': OC, 'type': 'JSON', **params})}"
    res = client.get(url)
    res.raise_for_status()
    body = res.text
    # 잘못된 target·필수 파라미터 누락은 200 에 로그인 HTML 이나 빈 응답으로 온다
    if not body.lstrip().startswith("{"):
        raise RuntimeError(f"JSON 이 아닌 응답 ({len(body)}B) — {url}")
    return json.loads(body)


def future_targets(s: object) -> list[dict]:
    """`조문시행일자문자열` 을 파싱한다 — "20260212:제47조의3제1항  20260621:제27조의2".

    한 법령 안에서 일부 조문만 나중에 시행되는 경우가 여기 드러난다.
    `조문단위[].조문시행일자` 로는 알 수 없다 — 스냅샷 안에서는 항상 단일값이다.
    """
    out = []
    for m in re.finditer(r"(\d{8})\s*:([^\d]*(?:\d(?!\d{7}:)[^\d]*)*)", str(s or "")):
        date = iso_date(m.group(1))
        if date and date > TODAY:
            out.append({"date": date, "targets": squash(m.group(2))})
    return out


# ────────────────────────────────────────────────────────── 법령 파서


def chunks_from_law(doc: dict, meta: dict) -> dict:
    """조 → 항 → 호 → 목 중첩 구조를 사실 단위로 자른다.

    자르는 단위는 "호"다. 단 호만 떼면 주어가 사라진다 —
      "1. 질병의 예방·치료에 효능이 있는 것으로 인식할 우려가 있는 표시 또는 광고"
    이것만으로는 무엇이 금지되는지 알 수 없다. 그래서 청크 본문에 **항의 앞머리(stem)를
    항상 함께 넣는다.** 단독으로 읽어도 맥락이 서야 한다.
    """
    root = doc["법령"]
    basic = root["기본정보"]
    law_name = squash(basic["법령명_한글"])
    law_eff = iso_date(basic["시행일자"])
    out: list[dict] = []
    seq = 0

    # 인용 URL 은 조 단위까지만 성립한다. 항·호를 붙이면 엉뚱한 곳으로 간다.
    def push(art_label: str, path: str, text: str, effective: str | None) -> None:
        nonlocal seq
        if not text or len(text) < 10:
            return
        for body in split_long(text):
            seq += 1
            out.append(
                {
                    "id": f"{meta['code']}-{seq:03d}",
                    "source": meta["short_name"],
                    "sourceKind": meta["kind"],
                    "lawName": law_name,
                    "path": path,
                    "text": body,
                    "url": f"https://www.law.go.kr/법령/{url_name(law_name)}/{art_label}",
                    "effectiveDate": effective or law_eff,
                }
            )

    for art in as_list(root.get("조문", {}).get("조문단위")):
        # 편·장·절 머리글은 조문이 아니다. 제목이 없고 본문이 "제1장 총칙" 같은 형태다.
        title = squash(art.get("조문제목"))
        if not title:
            continue

        no = str(art.get("조문번호"))
        if meta.get("max_article") is not None and int(no) > meta["max_article"]:
            continue
        branch = art.get("조문가지번호")  # 제8조 vs 제8조의2 를 가르는 유일한 필드
        art_label = f"제{no}조의{branch}" if branch else f"제{no}조"
        art_eff = iso_date(art.get("조문시행일자"))
        paras = as_list(art.get("항"))

        if not paras:
            # 항이 없는 조 — 조문내용 하나가 곧 사실 단위다 (예: 제1조 목적)
            push(art_label, art_label, squash(art.get("조문내용")), art_eff)
            continue

        for para in paras:
            para_label = squash(para.get("항번호"))  # ① ② …
            para_stem = squash(para.get("항내용"))
            items = as_list(para.get("호"))

            if not items:
                push(
                    art_label,
                    f"{art_label}{para_label}",
                    f"{art_label}({title}) {para_stem}",
                    art_eff,
                )
                continue

            for item in items:
                item_no = re.sub(r"\.$", "", squash(item.get("호번호")))
                subs = [squash(m.get("목내용")) for m in as_list(item.get("목"))]
                body = " ".join([squash(item.get("호내용")), *[s for s in subs if s]])
                push(
                    art_label,
                    f"{art_label}{para_label}제{item_no}호",
                    # 항 stem 을 함께 넣는 이유는 위 주석 참고
                    f"{art_label}({title}) {para_stem} {body}",
                    art_eff,
                )

    return {
        "chunks": out,
        "lawName": law_name,
        "effectiveDate": law_eff,
        "url": f"https://www.law.go.kr/법령/{url_name(law_name)}",
        "futureArticles": future_targets(basic.get("조문시행일자문자열")),
        "futureTables": future_targets(basic.get("별표시행일자문자열")),
    }


# ──────────────────────────────────────────────────────── 행정규칙 파서


def resolve_admrul_version(meta: dict) -> dict:
    """고시의 "오늘 적용 판" 을 판 목록에서 고른다.

    시행일 기준 조회 API 가 없어서 직접 골라야 한다 (sources.py 주석 참고).
    """
    versions = []
    for page in range(1, 6):
        doc = fetch_json(
            SEARCH,
            {
                "target": "admrul",
                "nw": "2",
                "search": "1",
                "display": "100",
                "page": str(page),
                "query": meta["search_name"],
            },
        )
        rows = as_list(doc.get("AdmRulSearch", {}).get("admrul"))
        if not rows:
            break
        for r in rows:
            if str(r.get("행정규칙ID")) != meta["rule_id"]:
                continue
            versions.append(
                {
                    "seq": str(r.get("행정규칙일련번호")),
                    "effective": iso_date(r.get("시행일자")),
                    "promulgated": iso_date(r.get("발령일자")),
                    "current": squash(r.get("현행연혁구분") or r.get("현행여부")),
                }
            )
        total = int(doc.get("AdmRulSearch", {}).get("totalCnt") or 0)
        if page * 100 >= total:
            break
    if not versions:
        raise RuntimeError(f"판 목록을 못 찾음: {meta['short_name']} (rule_id={meta['rule_id']})")

    live = [v for v in versions if v["effective"] and v["effective"] <= TODAY]
    live.sort(key=lambda v: (v["promulgated"] or "", v["seq"]))
    picked = live[-1] if live else None
    api_current = next((v for v in versions if v["current"] == "현행"), None)
    return {
        "versions": versions,
        "picked": picked,
        "apiCurrent": api_current,
        # API 가 현행이라 부르는 판이 아직 시행 전이면, 오늘 본문은 그 판이 아니다
        "pendingAmendment": (
            {"seq": api_current["seq"], "effective": api_current["effective"]}
            if api_current and api_current["effective"] and api_current["effective"] > TODAY
            else None
        ),
    }


# 호(`1. `) 마커. 고시 원문은 마커가 앞 단어에 붙어서 온다 — 「…표시ㆍ광고2. 건강기능식품…」.
# m05(JS)는 `\b` 로 경계를 잡았지만 그 규칙은 Python 으로 그대로 옮겨지지 않는다:
# JS `\b` 는 ASCII 기준이라 한글 뒤 숫자가 경계지만, Python `\b` 는 유니코드 기준이라
# 「광고2」 가 한 단어로 붙어 경계가 아니다. 그대로 두면 호 경계를 통째로 놓쳐
# 제2조 전체가 「제2조제1호」 하나로 뭉친다 (실측). 그래서 ASCII 단어 문자만 배제하고,
# 「제8조제1항」 같은 조문 참조와 「[별표 2] Ⅰ. 1. 가.」 같은 인용은 순번 검사로 거른다.
ITEM_MARKER = (re.compile(r"(?<![제항호조0-9A-Za-z_])(\d+)\.\s"), lambda m: int(m.group(1)))

# 조 번호가 없는 고시(「식품등의 표시기준」)는 본문 전체가 **문자열 하나**로 온다 (실측 53,216자).
# 장(Ⅰ·Ⅱ·Ⅲ) 경계를 먼저 끊지 않으면 총칙의 마지막 호 안으로 Ⅱ·Ⅲ 장 전체가 딸려 들어가고,
# 그러면 공통표시기준·개별표시사항 청크의 인용 위치가 전부 「Ⅰ. 총 칙」으로 찍힌다.
#
# 문제는 로마숫자가 본문 안에서 **참조로도** 쓰인다는 것이다 —
# 「…표시방법은 Ⅱ. 공통표시기준에 따른다」가 44곳. 장 제목만 고르는 단서는 두 가지다:
#   (1) 제목 뒤에 바로 첫 항목 `1.` 이 붙는다 (「Ⅲ. 개별표시사항 및 표시기준1. 식품…」)
#   (2) Ⅰ→Ⅱ→Ⅲ 순서로 오른다 (marker_cuts 의 순번 검사)
ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
CHAPTER_MARKER = (
    re.compile(r"([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])\.\s?[^0-9]{0,30}?1\.\s?"),
    lambda m: ROMAN.index(m.group(1)) + 1,
)


def chapter_sections(body: str) -> list[str]:
    """장 경계로 먼저 끊는다. 장이 없으면(제N조로 된 고시) 통째로 하나."""
    at = marker_cuts(body, CHAPTER_MARKER)
    if len(at) < 2:
        return [body]
    starts = ([0] if at[0] > 0 else []) + at
    return [
        body[s: starts[i + 1] if i + 1 < len(starts) else len(body)]
        for i, s in enumerate(starts)
    ]


def chunks_from_admrul(doc: dict, meta: dict) -> dict:
    """고시는 구조가 없다. `조문내용` 이 문자열 배열이고 항·호·목이 한 문자열에 뭉쳐 온다.

    그래서 정규식으로 호(`1. `) 경계를 찾아 자른다. 자를 때도 조의 앞머리를 함께 넣는다.
    목(`가. `)은 호에 붙여 둔다 — 목만 떼면 무엇에 대한 예시인지 사라진다.
    """
    root = doc.get("AdmRulService", doc)
    basic = root.get("행정규칙기본정보", {})
    rule_name = squash(basic.get("행정규칙명"))
    eff = iso_date(basic.get("시행일자"))
    promulgated = iso_date(basic.get("발령일자"))
    url = f"https://www.law.go.kr/행정규칙/{url_name(rule_name)}"
    out: list[dict] = []
    seq = 0

    def push(path: str, text: str) -> None:
        nonlocal seq
        if not text or len(text) < 10:
            return
        for body in split_long(text):
            seq += 1
            out.append(
                {
                    "id": f"{meta['code']}-{seq:03d}",
                    "_rawPath": path,
                    "source": meta["short_name"],
                    "sourceKind": meta["kind"],
                    "lawName": rule_name,
                    "path": path,
                    "text": body,
                    "url": url,
                    "effectiveDate": eff,
                }
            )

    for raw in as_list(root.get("조문내용")):
        for section in chapter_sections(squash(raw)):
            body = squash(section)
            if not body:
                continue

            head = re.match(r"^(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?", body)
            art_label = head.group(1) if head else "본문"

            # "1. …", "2. …" 경계로 자른다. 첫 호 앞까지가 조(또는 장)의 앞머리(stem).
            at = marker_cuts(body, ITEM_MARKER)
            if len(at) < 2:
                push(art_label, body)
                continue

            stem = squash(body[: at[0]])
            for i, start in enumerate(at):
                stop = at[i + 1] if i + 1 < len(at) else len(body)
                part = squash(body[start:stop])
                n = re.match(r"^(\d+)\.", part)
                push(f"{art_label}제{n.group(1)}호" if n else art_label, f"{stem} {part}")

    # 별표 — 실제 표시 방법이 여기 있다. ASCII 표가 섞여 있어 본문 청크와 따로 둔다.
    tables = []
    for t in as_list(root.get("별표", {}).get("별표단위")):
        raw_lines = as_list(t.get("별표내용"))
        flat: list[str] = []
        for line in raw_lines:
            flat.extend(line if isinstance(line, list) else [line])
        lines = [x for x in (squash(v) for v in flat) if x]
        tables.append(
            {
                # 별표·별지·도표가 각각 0001 부터 다시 번호를 매긴다 — 구분을 같이 둬야 가리킬 수 있다
                "kind": squash(t.get("별표구분")),
                "title": squash(t.get("별표제목")),
                "no": squash(t.get("별표번호")),
                "chars": len("\n".join(lines)),
                "pdf": (
                    f"https://www.law.go.kr{t['별표서식PDF파일링크']}"
                    if t.get("별표서식PDF파일링크")
                    else None
                ),
                "lines": lines,
            }
        )

    # 조 번호가 없는 고시는 청크 본문에서 위치를 읽어 path 를 만든다.
    # 못 찾으면 '본문'. `본문제1호` 는 없는 호 번호를 만들어 내는 것이라 오히려 오인시킨다.
    for c in out:
        if str(c.pop("_rawPath", "")).startswith("본문"):
            c["path"] = section_path(c["text"], "본문")

    return {
        "chunks": out,
        "ruleName": rule_name,
        "effectiveDate": eff,
        "promulgated": promulgated,
        "url": url,
        "tables": tables,
    }


# ────────────────────────────────────────────────────────────── 실행


def write_source(meta: dict, r: dict, all_chunks: list[dict]) -> None:
    """청크마다 시행 여부를 계산해 두고 자료별 파일로 쓴다.

    이 필드가 없으면 답변이 시행 예정 기준을 현행으로 말하게 된다 — PRD 4절이
    "현행 시행본을 쓰고 미시행 개정본은 제외한다"고 못박은 자리다.
    """
    for c in r["chunks"]:
        c["inForce"] = (c["effectiveDate"] <= TODAY) if c["effectiveDate"] else None
    kept = [c for c in r["chunks"] if c["inForce"] is not False]
    r["chunks"] = kept
    all_chunks.extend(kept)

    (OUT_DIR / f"{meta['code']}.json").write_text(
        json.dumps(
            {
                "code": meta["code"],
                "source": meta["short_name"],
                "sourceKind": meta["kind"],
                "lawName": r["lawName"],
                "category": meta["category"],
                "effectiveDate": r["effectiveDate"],
                "promulgated": r.get("promulgated"),
                "url": r["url"],
                "collectedAt": TODAY,
                "chunks": kept,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_report(report: dict, all_chunks: list[dict]) -> None:
    def w(s, n):
        return str(s).ljust(n)

    print(f"\n수집일 {TODAY} · OC={report['oc']}\n")
    print(w("code", 6) + w("종류", 8) + w("시행일", 13) + w("시행중", 8) + w("청크", 6) + "별표")
    for s in report["sources"]:
        forced = "?" if s["inForce"] is None else ("예" if s["inForce"] else "아니오 ⚠")
        tables = (
            f"{s['tables']}개 {round((s.get('tableChars') or 0) / 1024)}KB"
            if s.get("tables") is not None
            else "-"
        )
        print(
            w(s["code"], 6)
            + w(s["kind"], 8)
            + w(s["effectiveDate"] or "?", 13)
            + w(forced, 8)
            + w(s["chunks"], 6)
            + tables
        )
    print(f"\n총 {len(all_chunks)} 청크 → {OUT_DIR}")

    pending_rules = [s for s in report["sources"] if s.get("pendingAmendment")]
    if pending_rules:
        print("\n⚠ 시행 예정 개정이 대기 중인 고시 — 본문은 오늘 판을 썼다:")
        for s in pending_rules:
            print(
                f"   {s['name']}: 오늘 판 {s['pickedSeq']} (시행 {s['effectiveDate']})"
                f" / API 현행 {s['apiCurrentSeq']} (시행 {s['pendingAmendment']['effective']})"
            )

    pending_laws = [s for s in report["sources"] if s.get("pendingEffective")]
    if pending_laws:
        print("\n⚠ 시행 예정 개정이 대기 중인 법령:")
        for s in pending_laws:
            print(f"   {s['name']}: 시행 {', '.join(s['pendingEffective'])}")

    partial = [s for s in report["sources"] if s.get("futureArticles") or s.get("futureTables")]
    if partial:
        print("\n⚠ 일부 조문·별표가 미시행인 법령 — 미시행 청크는 제외했다:")
        for s in partial:
            for f in s.get("futureArticles") or []:
                print(f"   {s['name']} 조문 {f['date']}: {f['targets'][:90]}")
            for f in s.get("futureTables") or []:
                print(f"   {s['name']} 별표 {f['date']}: {f['targets'][:90]}")


REQUIRED = ("lawName", "path", "url", "effectiveDate")


def verify(chunks: list[dict]) -> int:
    """완료 기준 (b)·(d) 를 스크립트가 스스로 확인한다. 통과 못 하면 0 이 아닌 코드로 끝난다."""
    problems = []
    for c in chunks:
        missing = [k for k in REQUIRED if not c.get(k)]
        if missing:
            problems.append(f"{c['id']}: 빈 필드 {', '.join(missing)}")
        if c.get("inForce") is not True:
            problems.append(f"{c['id']}: inForce={c.get('inForce')}")
    if problems:
        print(f"\n✗ 검증 실패 {len(problems)}건")
        for p in problems[:20]:
            print(f"   {p}")
        return 1
    print(f"\n✓ 검증 통과 — {len(chunks)} 청크 모두 lawName·path·url·effectiveDate 가 차 있고 inForce=true")
    return 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "collectedAt": TODAY,
        "oc": "test (본인 OC 미설정)" if OC == "test" else "set",
        "sources": [],
    }
    all_chunks: list[dict] = []

    # ── 법령: eflaw&ID = 오늘 시행 중인 본문 (efYd 를 주지 않는다)
    for law in LAWS:
        doc = fetch_json(SERVICE, {"target": "eflaw", "ID": law["law_id"]})
        r = chunks_from_law(doc, law)

        # 시행 예정 개정이 있는지 확인 (코퍼스에는 넣지 않고 알림만)
        pending = None
        try:
            nw2 = fetch_json(SEARCH, {"target": "eflaw", "LID": law["law_id"], "nw": "2"})
            rows = [
                x
                for x in as_list(nw2.get("LawSearch", {}).get("law"))
                if squash(x.get("현행연혁코드")) == "시행예정"
            ]
            pending = [d for d in (iso_date(x.get("시행일자")) for x in rows) if d] or None
        except Exception:
            pass  # 시행예정이 없으면 빈 응답이 온다

        write_source(law, r, all_chunks)
        report["sources"].append(
            {
                "code": law["code"],
                "kind": law["kind"],
                "name": r["lawName"],
                "category": law["category"],
                "effectiveDate": r["effectiveDate"],
                "inForce": (r["effectiveDate"] <= TODAY) if r["effectiveDate"] else None,
                "chunks": len(r["chunks"]),
                "futureArticles": r["futureArticles"],
                "futureTables": r["futureTables"],
                "pendingEffective": pending,
            }
        )

    # ── 고시: 판 목록에서 오늘 적용 판을 직접 고른다
    for rule in ADMRULS:
        v = resolve_admrul_version(rule)
        if not v["picked"]:
            raise RuntimeError(f"오늘 시행 중인 판이 없음: {rule['short_name']}")

        doc = fetch_json(SERVICE, {"target": "admrul", "ID": v["picked"]["seq"]})
        r = chunks_from_admrul(doc, rule)
        r["lawName"] = r["ruleName"]
        write_source(rule, r, all_chunks)

        if r["tables"]:
            (OUT_DIR / f"tables-{rule['code']}.json").write_text(
                json.dumps(
                    {
                        "code": rule["code"],
                        "lawName": r["ruleName"],
                        "effectiveDate": r["effectiveDate"],
                        "promulgated": r["promulgated"],
                        "url": r["url"],
                        "collectedAt": TODAY,
                        "tables": r["tables"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        report["sources"].append(
            {
                "code": rule["code"],
                "kind": rule["kind"],
                "name": r["ruleName"],
                "category": rule["category"],
                "effectiveDate": r["effectiveDate"],
                "promulgated": r["promulgated"],
                "inForce": (r["effectiveDate"] <= TODAY) if r["effectiveDate"] else None,
                "chunks": len(r["chunks"]),
                "tables": len(r["tables"]),
                "tableChars": sum(t["chars"] for t in r["tables"]),
                "pickedSeq": v["picked"]["seq"],
                "versionCount": len(v["versions"]),
                "apiCurrentSeq": (v["apiCurrent"] or {}).get("seq"),
                "pendingAmendment": v["pendingAmendment"],
            }
        )

    report["totalChunks"] = len(all_chunks)
    report["outOfScope"] = OUT_OF_SCOPE
    (OUT_DIR / "collection-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_report(report, all_chunks)
    return verify(all_chunks)


if __name__ == "__main__":
    raise SystemExit(main())
