import streamlit as st
import sidebar
import main

# 전체 페이지 기본 설정 (가장 먼저 호출되어야 함)
st.set_page_config(
    page_title="지방소멸 위기 예측 및 XAI 대시보드", 
    page_icon="🗺️", 
    layout="wide"
)

def run():
    # 1. 사이드바를 렌더링하고, 사용자가 선택한 페이지 값을 받아옵니다.
    selected_page = sidebar.render_sidebar()
    
    # 2. 선택된 페이지 값에 따라 메인 화면을 렌더링합니다.
    main.render_main(selected_page)

if __name__ == "__main__":
    run()