import os, json, traceback
from pathlib import Path
import streamlit as st

# ===== 페이지 기본 =====
st.set_page_config(page_title="성동구 소상공인 비밀상담사 (MVP)", page_icon="💬", layout="wide")
st.title("성동구 소상공인 비밀상담사 (MVP)")
st.caption("Agent-1: 데이터 집계/요약 → Agent-2: 실행카드(JSON) 생성")

# ===== 경로 & 키 =====
DATA_DIR = Path("data")
SHINHAN_DIR = DATA_DIR / "shinhan"
EXTERNAL_DIR = DATA_DIR / "external"

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    st.warning("GEMINI_API_KEY가 설정되지 않았습니다. (앱 설정의 App secrets에 등록하세요)")

# ===== 사이드바: 데이터 상태 =====
st.sidebar.header("데이터 상태")
st.sidebar.write(f"📁 SHINHAN_DIR 존재: {SHINHAN_DIR.exists()}")
st.sidebar.write(f"📁 EXTERNAL_DIR 존재: {EXTERNAL_DIR.exists()}")

# ===== 질문 입력 =====
default_q = "비 오는 날 20대 이하 카페 홍보 전략 제시해줘"
question = st.text_input("질문을 입력하세요", value=default_q)

# ===== 실행 버튼 =====
if st.button("분석 실행", type="primary"):
    # 지연 로딩 임포트 (배포 런타임 문제 회피)
    from bigcon_2agent_mvp_v3 import agent1_pipeline, build_agent2_prompt, call_gemini_agent2

    # Agent-1
    with st.spinner("Agent-1: 데이터 집계/요약 중..."):
        try:
            a1 = agent1_pipeline(question, SHINHAN_DIR, EXTERNAL_DIR)
            st.success("Agent-1 JSON 생성 완료")
            with st.expander("🔎 Agent-1 출력(JSON) 보기", expanded=False):
                st.json(a1)
        except Exception:
            st.error("Agent-1 실행 오류")
            st.code(traceback.format_exc())
            st.stop()

    # Agent-2
    with st.spinner("Agent-2: 카드 생성 중..."):
        try:
            os.environ["GEMINI_API_KEY"] = API_KEY  # 내부 함수가 env 읽도록 주입
            prompt_text = build_agent2_prompt(a1)
            result = call_gemini_agent2(prompt_text)
            st.success("Agent-2 카드 생성 완료")
        except Exception:
            st.error("Agent-2 실행 오류")
            st.code(traceback.format_exc())
            st.stop()

    # 출력
    st.subheader("🧾 추천 액션카드")
    st.json(result)

# 최초 안내
if not st.session_state.get("_intro_shown"):
    st.info("✅ 업로드 성공! 이제 질문 입력 후 [분석 실행]을 눌러 카드 결과를 확인해보세요.")
    st.session_state["_intro_shown"] = True
