import streamlit as st

def render_sidebar():
    st.sidebar.title("📌 네비게이션")
    st.sidebar.markdown("---")
    
    # 메뉴 리스트 정의
    menu_options = [
        "1. 기초 통계량 (Data Summary)", 
        "2. EDA 시각화 (KDE, Heatmap, Trend)", 
        "3. 모델 성능 비교 (RF vs XGB vs CatBoost)", 
        "4. 예측 지도 & XAI (SHAP)"
    ]
    
    # 사용자의 선택값 저장
    page = st.sidebar.radio("이동할 페이지를 선택하세요:", menu_options)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**\n\n4번 탭에서 지도를 클릭하면 XAI 기반 지역별 정책 제언을 볼 수 있습니다.")
    
    return page