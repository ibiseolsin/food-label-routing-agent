# 식품 표시·규격 라우팅 에이전트

음료 브랜드를 준비하는 비전문가가 **내 제품의 유형·표시·첨가물·광고 문구를 조문 근거와 함께**
확인하고, 근거가 없으면 지어내지 않고 넘겨받게 하는 고객 응대 에이전트.

아이펠 AI 에이전트 1기 — 에이전트 팀 꾸리기_Agt1 5번 노드 「고객 응대 에이전트 만들기 [프로젝트]」

- **제출 리포트 7항목: [REPORT.md](REPORT.md)**
- 요구사항: [PRD.md](PRD.md) · 작업계획: [PLAN.md](PLAN.md)
- 수집 기록: [docs/COLLECT-NOTES.md](docs/COLLECT-NOTES.md) · 카테고리 매핑표: [docs/CATEGORY-MAP.md](docs/CATEGORY-MAP.md)
- 기준 측정: [docs/BASELINE.md](docs/BASELINE.md) · 개선 기록: [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md)
- 배포 주소: 없음 (PRD 5절 — 공개 배포는 선택). 데모는 로컬에서 `uv run streamlit run app.py`

## 현재 상태

**슬라이스 1~13 전부 완료** — 1단계(1~5) · 2단계 검증·측정·개선(6~10) ·
3단계 데모·리포트·제출 점검(11~13).
법제처 API 수집 · 공전 수집 · 조회 도구 4종 · 평가셋 18문항 ·
`판정 → 근거 조립 → 답변 → 검증 → (넘기기)` 파이프라인이 돌아간다. 넘기기는 두 갈래로
갈린다 — 소관 밖(`out_of_scope`)은 판정 노드가, 소관이지만 근거에 답이 없는 것
(`no_evidence`)은 답변 노드가 정한다. 채점기(S1~S4)와 모범 답안 18문항이 있고 모범 답안
역검증(S5)·역대조 7건을 통과한다.

**측정 상태** (18문항, 기준은 [PRD 4절](PRD.md), 같은 채점기로 잰 값):

| 지표 | 기준 | 기준선 3회 | 최종 5회 |
|---|---|---|---|
| S1 도구 집합 | ≥ 0.75 | 0.78 · 0.78 · 0.78 | 0.78 (5회 모두) |
| S2 답변 적절성 | ≥ 0.75 | 0.67 · 0.61 · 0.67 (평균 0.648) | 0.72 · 0.78 · 0.78 · 0.72 · 0.83 (평균 **0.767**) |
| S3 근거 밖 진술 | 0건 | 0 | 0 (5회 모두) |
| S4 넘기기 | 0건 | 0 | 0 (5회 모두) |

개선 5회를 `answer_prompt` 축에 태워 **한 회차에 한 축만** 바꿨고(`evaluate.py --against` 가
두 축이면 종료 코드 2로 거부한다), 남긴 둘이 S2 를 0.648 → 0.767 로 올렸다.
기준선의 최고(0.67)가 최종의 최저(0.72)보다 낮아 두 분포가 겹치지 않는다.
**5회 중 2회(0.72)는 기준 아래**라 평균·중앙값으로 읽은 값이다 — 회차별 기록·채점기를 고친
이유와 검증은 [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) 에 있다.

데모 화면은 `app.py` 에 있다 — 답변 · 호출한 도구 · **인용한 조문 원문(법제처 링크)** ·
환각 검증 결과가 한 화면에 나오고, 넘긴 답변에는 어느 갈래로 넘겼는지 배지가 붙는다 (PRD S7).
화면 캡처는 [REPORT.md](REPORT.md) 6절과 `docs/img/`.

과제가 요구하는 7항목(주제·카테고리 설계·평가셋·측정 결과·구조도·데모·회고)은
**[REPORT.md](REPORT.md)** 에 있다.

## 실행

```bash
uv run python collect_law.py    # 법제처 OPEN API — 표시·광고 법률·시행령·고시 4종
uv run python collect_code.py   # 식품안전나라 공전 서비스 — 식품공전·식품첨가물공전
uv run python tools.py          # 조회 도구 4종 자체 점검
uv run python tools.py lookup_food_type "유자즙 28%"   # 도구 하나를 직접 호출
uv run python validate_goldenset.py                  # 평가셋 스키마·분포·중복·goldCheck 검증
uv run python verify.py --selftest                   # 환각 검증 규칙 자체 확인 (API 키 불필요)
uv run python verify.py --replay data/runs/slice7.json   # 기록된 답변에 검증 규칙을 다시 건다

uv run python -m agent "유자즙 28% 음료를 유자주스로 팔아도 되나요?"   # 답변 + 호출 도구 + 인용 근거 + 검증
uv run python -m agent --goldenset --out data/runs/slice7.json       # 평가셋 전 문항 관통

uv run python evaluate.py --selftest                  # 채점 규칙 층 자체 확인 (API 키 불필요)
uv run python evaluate.py --reference                 # 모범 답안 채점 = S5 역검증
uv run python evaluate.py --negative                  # 망친 답변이 0점인지 역대조
uv run python evaluate.py --score data/runs/slice7.json   # 기록된 실행을 채점
uv run python evaluate.py --out data/runs/baseline.json   # 파이프라인을 돌려 채점
uv run python evaluate.py --against data/runs/r4-c.json --out data/runs/next.json
                                                      # 직전 회차와 개선 축을 대조하고 (두 축이면 거부) 측정

uv run streamlit run app.py                           # 데모 화면 (http://127.0.0.1:8501)
```

에이전트 실행에는 `OPENAI_API_KEY` 가 필요하다. 모델은 `OPENAI_MODEL` 로 바꾼다 (기본 `gpt-4o-mini`).
채점기 모델은 `EVAL_JUDGE_MODEL` 로 따로 둔다 (기본 `gpt-4o-mini`) — 파이프라인 모델은 개선 축이라
같이 움직이면 회차 비교가 무의미해진다.

수집 스크립트 둘은 받은 원문을 사실 단위 청크로 자른다. 산출물은 `data/corpus/` 에 있고
**커밋되어 있으므로 다시 받지 않아도 된다** — 다시 받으면 스냅샷이 오늘 날짜로 갱신되고
청크 ID 가 밀린다. 두 스크립트 모두 끝에서 완료 기준을 스스로 확인하고, 실패하면 종료 코드 1 이다.

`.env` 는 `.env.example` 을 복사해 만든다. `LAW_OC` 가 없으면 `test` 로 돌아간다(수집은 된다).
공전 수집에는 키가 필요 없다.

## 근거 문서 (스냅샷 2026-09-17)

| 코드 | 문서 | 시행일 | 청크 | 별표 |
|---|---|---|---|---|
| FLA | 식품 등의 표시·광고에 관한 법률 (제1~10조) | 2025-09-19 | 64 | - |
| FLD | 같은 법 시행령 (제1~6조) | 2025-09-19 | 20 | - |
| LBL | 식품등의 표시기준 (고시) | 2025-08-29 | 114 | 14개 87KB |
| UNF | 식품등의 부당한 표시 또는 광고의 내용 기준 (고시) | 2025-12-04 | 17 | 2개 5KB |
| FDC | 식품의 기준 및 규격 — 제5. 9. 음료류 + 제2. 식품일반 공통기준 | 2026-05-19 | 143 | - |
| FAC | 식품첨가물의 기준 및 규격 — 일반사용기준 + 품목별 사용기준 | 2025-11-26 | 883 | - |

합계 1,241 청크. 공전 두 종은 법제처로 본문을 받을 수 없어(본문 329자, 실질 내용이 첨부파일)
식품안전나라 「식품분야 공전 온라인 서비스」의 항목별 PDF 로 받는다. 첨가물은
**품목 하나 = 청크 하나**(716종)라 품목명으로 사용기준이 바로 잡힌다.

공전 본문에는 아직 시행되지 않은 개정이 주석으로 덧붙어 있다. 표 칸에 든 것은 잘라내고,
줄글에 겹친 것은 잘라낼 자리가 없어 기록만 했다 — 어느 청크가 그런지는
[docs/COLLECT-NOTES.md](docs/COLLECT-NOTES.md) 와 `collection-report-fsd.json` 에 있다.

## 카테고리 = 도구 4종

과제 지표가 「호출한 도구 집합이 기대 집합과 정확히 일치하면 1점」이라 카테고리와 도구를
1:1 로 붙였다. 어느 도구가 문서의 어느 부분을 보는지는 [docs/CATEGORY-MAP.md](docs/CATEGORY-MAP.md) 가 원본이다.

| 도구 | 답하는 것 | 근거 |
|---|---|---|
| `lookup_food_type` | 내 제품이 어느 식품유형인가 | 식품공전 음료류 + 식품원료 기준 + 제조·가공기준 |
| `lookup_labeling` | 라벨에 뭐가 반드시 들어가나 | 표시기준 본문 + 별표·별지·별도 |
| `lookup_additive` | 이 첨가물을 얼마까지 쓰고 뭐라 적나 | 첨가물공전 + 표시기준 별표 4~6 |
| `lookup_ad_claims` | 이 문구가 부당한 표시·광고인가 | 표시광고법·시행령 + 부당표시광고 고시 |

발췌는 **주제 색인 + 키워드 점수 + (첨가물) 별칭 색인**으로 한다. 벡터 검색을 쓰지 않는다 —
같은 질의가 늘 같은 근거를 돌려줘야 개선 축을 하나씩 바꾼 효과를 잴 수 있다.
도구 1회 반환은 8,000자 이하다.

## 구조

```
collect_law.py            법제처 API 수집 드라이버
collect_code.py           식품안전나라 공전 수집 드라이버 (PDF → 청크)
chunking.py               조/항/호·계층 표기 파싱 규칙 (두 드라이버가 같이 쓴다)
sources.py                수집 대상 목록과 범위 밖 주제
corpus.py                 청크 로더 — 본문과 별표를 한 목록으로 (별표는 여기서 청크가 된다)
tools.py                  카테고리별 근거 조회 도구 4종 + 자체 점검
validate_goldenset.py     평가셋 검증기 (스키마·분포·중복·goldCheck·금지 표현)
verify.py                 환각 검증 규칙 — 수치·조문·유형명을 근거와 대조 + 자체 확인
app.py                    streamlit 데모 — 답변·도구·인용 원문·검증을 한 화면에 (PRD S7)
evaluate.py               채점기 — S1(집합)·S2(필수 사실 LLM + 금지 표현 규칙)·S3(verify 재사용)·S4(넘기기)
agent.py                  LangGraph 파이프라인 — 판정 → 근거 조립 → 답변 → 검증 (+재생성 1회) → 넘기기 + CLI
data/goldenset.json       채점용 평가셋 18문항 (기대 도구·필수 사실·금지 표현·gold·goldCheck)
data/prompt-examples.json 프롬프트 예시용 4문항 — **채점에 넣지 않는다**
data/reference-answers.json 모범 답안 18문항 — 채점기 역검증(S5)용. gold 청크만 인용한다
data/corpus/<CODE>.json   자료별 청크
data/corpus/tables-*.json 고시 별표 (표시사항별 세부표시기준, 첨가물 표시 별표 등)
data/corpus/collection-report.json      법제처 수집 리포트
data/corpus/collection-report-fsd.json  공전 수집 리포트 (항목별 청크 수·미시행 개정)
data/runs/slice5.json     슬라이스 5 관통 기록 (문항별 도구·근거 ID·답변) — 기준 측정의 출발점
data/runs/slice6.json     슬라이스 6 관통 기록 (+ 위반 목록·재생성 여부·답변 이력)
data/runs/slice7.json     슬라이스 7 관통 기록 (+ 넘기기 갈래와 기대 라벨)
data/runs/slice8-reference.json  모범 답안 채점 기록 (S5) — 문항별 S1~S4 와 왜 0점인지
docs/VERIFY-NOTES.md      검증 규칙이 무엇을 잡고 무엇을 못 잡나 (실측과 한계)
docs/img/demo-*.png       데모 화면 캡처 (REPORT 6절)
REPORT.md                 제출 리포트 7항목 — 주제·카테고리·평가셋·측정·구조도·데모·회고
.cache/fsd/               받은 공전 PDF (gitignore — 4.8MB 짜리가 있다)
```

## 제출물 체크리스트

제출 직전에 **실행으로** 확인한다. 아래는 **2026-09-18 확인** 결과다 (슬라이스 13).

- [x] `collect_law.py` · `collect_code.py` 가 검증 통과로 끝난다 — 종료 코드 0.
      215 청크(FLA 64·FLD 20·LBL 114·UNF 17) · 1,026 청크(FDC 143·FAC 883) 로 **커밋된
      스냅샷과 같은 수**가 나왔다. ⚠ **`data/corpus/` 를 덮지 말고 돌려야 한다** — 아래 주의
- [x] `uv run python tools.py` 가 도구 4종 점검을 통과한다 — 완료 기준 (a)(b)(c)(d) 통과
- [x] `uv run python validate_goldenset.py` 가 평가셋 검증을 통과한다 — 18문항, gold 도달 23/23
- [x] `uv run python -m agent "<질문>"` 이 답변 + 호출 도구 + 인용 근거를 출력한다 —
      「비타민C 사용량 한도」 1건: `lookup_additive` · `[FAC-001]` 원문·링크 · 검증 통과
- [x] `uv run python evaluate.py` 가 S1·S2 수치를 출력하고 기준(≥0.75)을 넘는다 —
      **S1 0.78 · S2 0.78 · S3 0건 · S4 0건** (`data/runs/submit.json`).
      S5 역검증(`--reference`)도 18문항 전부 1점. **단 S2 는 회차마다 흔들려 이 1회로
      기준 충족을 말하지 않는다** — 근거는 5회 평균 0.767 ([docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) 3-0-2)
- [x] `uv run streamlit run app.py` 데모에서 답변·근거·검증 결과가 한 화면에 보인다 —
      127.0.0.1:8701 에 임의 질문(「'무첨가'라고 패키지에 써도 되나요?」) 1건을 직접 입력해
      답변·인용한 조문 원문·호출한 도구·환각 검증 네 자리를 다 확인했다
- [x] `REPORT.md` 7항목 — 7개 절 전부 있고, 이미지 2개·상대 링크 8개가 전부 실재한다.
      Mermaid 구조도는 mermaid@11 로 실제 렌더해 확인했다
- [x] `.env` 가 커밋되지 않았고 저장소에 키 문자열이 없다 — 추적 파일 334개에서
      `.env` 의 실제 값 2개와 `sk-` 모양 문자열을 찾아 **0건**. 추적되는 것은 `.env.example` 뿐

> ⚠ **수집 스크립트를 점검할 때 `data/corpus/` 를 덮지 않는다.** 두 스크립트는 출력 경로
> 옵션이 없어 그냥 돌리면 스냅샷 날짜가 오늘로 갱신되고 청크 ID 가 밀려 평가셋의
> `gold`·`goldCheck` 가 깨진다. 점검은 `OUT_DIR` 만 임시 폴더로 바꿔 돌린다:
>
> ```python
> import collect_law; collect_law.OUT_DIR = <임시 폴더>; collect_law.main()
> ```
