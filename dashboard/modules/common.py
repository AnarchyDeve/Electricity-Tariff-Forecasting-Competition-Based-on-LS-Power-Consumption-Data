# dashboard/modules/common.py
from contextlib import contextmanager
import streamlit as st

# ===== 색상 토큰 (메인/사이드바 원래 톤으로 복원) =====
COLORS = {
    "bg":   "#1d293d",   # 메인 배경
    "bg2":  "#0f172b",   # 사이드바 배경
    "text": "#e2e8f0",
    "primary": "#615fff",
    "border":  "#314158",
}

def inject_css():
    st.markdown(f"""
    <style>
    :root {{
      --bg: {COLORS['bg']}; --bg2: {COLORS['bg2']}; --text: {COLORS['text']};
      --primary: {COLORS['primary']}; --border: {COLORS['border']};
    }}
    html, body, [data-testid="stAppViewContainer"] {{
      background: var(--bg); color: var(--text);
      font-family: 'Space Grotesk', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;
      font-weight: 300; font-size: 14px;
    }}
    /* 상단 툴바 색 통일 */
    [data-testid="stHeader"], [data-testid="stHeader"] > div {{ background: var(--bg); }}

    /* ===== 사이드바 ===== */
    [data-testid="stSidebar"] {{
      background: var(--bg2);
      border-right: 1px solid var(--border);
    }}
    /* 사이드바의 모든 텍스트를 흰색으로 강제 (대시보드 헤더 포함) */
    [data-testid="stSidebar"] * {{ color: #ffffff !important; }}
    /* 라디오의 기본 원 숨김 + 링크형 */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
      display: none !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
      display:block; padding:8px 10px; border-radius:10px;
      opacity:.70; font-weight:300; cursor:pointer;
      border:1px solid transparent; background: transparent;
      text-decoration:none !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
      background: rgba(97,95,255,.12);
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label.sb-active {{
      opacity:1; font-weight:700;
      background: rgba(97,95,255,.12); border: 1px solid var(--primary);
    }}

    /* ===== 카드 공통 ===== */
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(0,0,0,.06));
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,.20);
      padding: 14px 16px;
      margin-bottom: 18px;
    }}
    .card-title {{
      font-weight: 600; opacity:.95; margin-bottom: 10px;
    }}
    /* 점선 → 실선 */
    .placeholder {{
      border:1px solid rgba(226,232,240,.35);
      border-radius:12px; width:100%;
      background: rgba(0,0,0,.08);
    }}
    </style>
    """, unsafe_allow_html=True)

@contextmanager
def card(title: str):
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown('</div>', unsafe_allow_html=True)

def placeholder(height: int = 160):
    st.markdown(f'<div class="placeholder" style="height:{height}px"></div>', unsafe_allow_html=True)

def section_header(title: str, subtitle: str = "스타일 전용 빈 카드 레이아웃"):
    st.markdown(f"### {title}")
    st.caption(subtitle)
