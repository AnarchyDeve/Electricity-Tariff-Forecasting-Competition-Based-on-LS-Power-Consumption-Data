import streamlit as st
from modules import tab1, tab2, tab3, tab4

st.set_page_config(page_title="Styled Empty Dashboard", page_icon="📊", layout="wide")

# ---- 색상/폰트: .streamlit/config.toml 과 맞춤 ----------------------
TEXT = "#e2e8f0"
PRIMARY = "#615fff"
BG = "#1d293d"
BG2 = "#0f172b"
BORDER = "#314158"
FONT_FAMILY = "Space Grotesk, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial"

# ---- 공통 CSS (전 탭 공용) ------------------------------------------
st.markdown(f"""
<style>
:root {{
  --bg: {BG};
  --bg2: {BG2};
  --text: {TEXT};
  --primary: {PRIMARY};
  --border: {BORDER};
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bg);
  color: var(--text);
  font-family: {FONT_FAMILY};
  font-weight: 300;
  font-size: 14px;
}}
.main .block-container {{ padding-top: 14px; }}

/* 카드 */
.card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow:
    0 12px 28px rgba(0,0,0,.28),
    0 4px 10px rgba(0,0,0,.25);
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.card-title {{ font-weight: 400; font-size: 15px; color: {TEXT}; opacity: .95; margin: 0 0 6px 0; }}
.card-body {{
  width: 100%;
  border: 1px dashed rgba(226,232,240,.18);
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(255,255,255,.02), transparent);
}}

/* 높이 프리셋 */
.h-top   {{ height: 110px; }}
.h-mid   {{ height: 260px; }}
.h-large {{ height: 400px; }}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"]{{ gap: 6px; }}
.stTabs [data-baseweb="tab"]{{
  background: var(--bg2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 10px;
  padding: 8px 12px;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"]{{
  background: linear-gradient(180deg, rgba(97,95,255,.18), rgba(97,95,255,.08));
  border-color: {PRIMARY};
}}
</style>
""", unsafe_allow_html=True)

TAB_TITLES = ["실시간 데이터 확인", "핵심요소 관리", "과거 데이터 분석", "부록"]

tabs = st.tabs(TAB_TITLES)

with tabs[0]:
    tab1.render(TAB_TITLES[0])
with tabs[1]:
    tab2.render(TAB_TITLES[1])
with tabs[2]:
    tab3.render(TAB_TITLES[2])
with tabs[3]:
    tab4.render(TAB_TITLES[3])
