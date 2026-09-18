"""streamlit 데모 (PLAN 슬라이스 11 · PRD S7).

**한 화면에 넷을 같이 둔다** — 답변 · 호출한 도구 · 인용한 조문 원문 · 환각 검증 결과.
넷이 흩어져 있으면 「근거를 보여준다」가 말뿐이 된다. 사용자가 답변 한 줄을 읽고
바로 아래에서 그 근거의 원문과 법제처 링크를 열 수 있어야 한다.

CLI 의 `agent.show()` 와 같은 것을 보여준다. 화면 쪽에만 있는 것은 둘이다 —
**예시 질문 버튼**(처음 열었을 때 무엇을 물어야 할지 모른다)과 **넘기기 배지**
(왜 답을 안 줬는지가 답변만큼 중요하다).

    uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import agent

st.set_page_config(page_title="식품 표시·규격 안내", page_icon="🥤", layout="wide")


@st.cache_resource
def graph():
    """그래프는 한 번만 만든다 — streamlit 은 위젯을 건드릴 때마다 스크립트를 다시 돈다."""
    return agent.build_graph()


EXAMPLES = [
    "유자즙 28% 음료는 식품유형이 뭔가요?",
    "과채음료 라벨에 꼭 들어가야 하는 표시사항이 뭔가요?",
    "비타민C를 음료에 넣으려는데 사용량 한도가 있나요?",
    "'면역력에 좋은 유자차'라고 광고해도 되나요?",
    "유자 발효추출물을 식품원료로 쓸 수 있나요?",
    "HACCP 인증은 어떻게 받나요?",
]

BADGE = {
    "out_of_scope": ("🔁", "소관 밖이라 넘겼습니다 (UC-5)"),
    "no_evidence": ("🔁", "근거에 답이 없어 넘겼습니다 (UC-6)"),
}


def main() -> None:
    st.title("🥤 식품 표시·규격 안내 에이전트")
    st.caption(
        "음료류 식품의 **유형 · 표시 · 첨가물 · 광고 문구**를 조문 근거와 함께 안내합니다. "
        "근거에 없으면 지어내지 않고 넘깁니다."
    )

    with st.sidebar:
        st.subheader("예시 질문")
        st.caption("네 카테고리 + 넘기기 두 갈래")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state["q"] = ex
        st.divider()
        st.caption(
            f"모델 `{agent.MODEL}` · 근거는 스냅샷 시점의 현행 시행본입니다. "
            "개별 제품 판단은 관할 부서에 확인하세요."
        )

    question = st.text_input(
        "질문", value=st.session_state.get("q", ""), placeholder="예) 유자즙 28% 음료는 식품유형이 뭔가요?"
    )
    if not st.button("물어보기", type="primary") and not question:
        return
    if not question.strip():
        st.warning("질문을 입력하세요.")
        return

    with st.spinner("근거를 찾아 답변을 만들고 검증하는 중…"):
        run = agent.ask(question, graph())

    # ── 넘기기 배지 — 답변보다 먼저. 왜 답을 안 줬는지가 답변만큼 중요하다
    if run.escalation:
        icon, label = BADGE[run.escalation]
        st.info(f"{icon} {label}")

    st.subheader("답변")
    st.markdown(run.answer)

    left, right = st.columns([3, 2])

    # ── 인용 원문 — 답변의 [청크ID] 를 실제 조문으로 펼친다
    with left:
        st.subheader("인용한 조문 원문")
        by_id = {ev.chunk.id: ev.chunk for r in run.results for ev in r.evidence}
        if not run.valid_citations:
            st.caption("인용한 조문이 없습니다 (넘긴 답변이거나 근거가 조립되지 않았습니다).")
        for cid in run.valid_citations:
            c = by_id[cid]
            with st.expander(f"[{cid}] {c.law_name} · {c.path}", expanded=True):
                st.caption(f"시행 {c.effective_date}")
                st.text(c.text[:1500])
                st.markdown(f"[법제처 원문 보기]({c.url})")
        if run.invented_citations:
            st.error(f"근거에 없는 ID 를 인용했습니다: {', '.join(run.invented_citations)}")

    with right:
        # ── 호출한 도구 — 어느 카테고리로 판정했고 무엇을 근거로 골랐나
        st.subheader("호출한 도구")
        if not run.tools:
            st.caption("없음 — 소관 밖으로 판정해 조회하지 않았습니다.")
        for r in run.results:
            st.markdown(f"**`{r.tool}`** — {r.label}")
            st.caption(f"{r.scope_desc} · 근거 {len(r.evidence)}청크" + (f" (한도로 {r.dropped}개 제외)" if r.dropped else ""))
        if run.routing_note:
            st.caption(f"판정 근거: {run.routing_note}")

        # ── 환각 검증 — 답변의 수치·조문·유형명이 근거 안에 있는가
        st.subheader("환각 검증")
        if run.attempts == 0:
            st.caption("생략 — 소관 밖이라 답변을 생성하지 않았습니다.")
        elif run.verify_ok:
            st.success(f"통과 — 수치·조문·유형명 {run.verify_checked}건을 근거와 대조했습니다.")
        else:
            st.error(f"위반 {len(run.violations)}건 (재생성 뒤에도 남았습니다)")
            for v in run.violations:
                st.markdown(f"- {v['kind']} 「{v['token']}」 — 근거에 없습니다")
        if run.regenerated:
            with st.expander("재생성 1회 — 첫 답변"):
                st.text(run.answers[0])

        st.caption(f"근거 {len(run.evidence_ids)}청크 · {run.seconds:.1f}초")
        with st.expander("실행 기록"):
            for line in run.trace:
                st.text(line)


main()
