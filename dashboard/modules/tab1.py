import streamlit as st

def _empty_card(title: str, height_class: str):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-body {height_class}"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render(tab_name: str):
    st.subheader(tab_name)
    st.caption("스타일 전용 빈 카드 레이아웃")

    # 상단 3
    c1, c2, c3 = st.columns(3, gap="large")
    with c1: _empty_card("Top - Card 1", "h-top")
    with c2: _empty_card("Top - Card 2", "h-top")
    with c3: _empty_card("Top - Card 3", "h-top")

    st.write("")

    # 중단 2
    m1, m2 = st.columns([3, 2], gap="large")
    with m1: _empty_card("Middle - Card 1", "h-mid")
    with m2: _empty_card("Middle - Card 2", "h-mid")

    st.write("")

    # 하단 1
    _empty_card("Bottom - Large Card", "h-large")
