import streamlit as st

def render_sidebar():
    st.sidebar.title("페이지 선택")
    
    pages = {
        "지도 페이지": "map_page",
        "EDA 페이지": "eda_page"
    }
    
    selected_page_name = st.sidebar.radio("이동할 페이지를 선택하세요:", list(pages.keys()))
    
    return pages[selected_page_name]
