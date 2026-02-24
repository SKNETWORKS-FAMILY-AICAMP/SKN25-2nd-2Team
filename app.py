import streamlit as st
import pandas as pd
from mainpages.eda_page import eda_page

# src 경로를 import 할 수 있도록 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/mainpages')

from src.data_loader import load_and_clean_data
from src.feature_forecaster import forecast_features
from src.model_trainer import train_and_predict_target
from sidebar import render_sidebar
from mainpages.map_page import map_page

class App:
    def __init__(self):
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'final_df' not in st.session_state:
            with st.spinner("데이터 로딩 중... 최초 1회만 실행됩니다."):
                df = load_and_clean_data()
                forecasted_df = forecast_features(df)
                final_df, model, X_train, y_train = train_and_predict_target(forecasted_df)
                st.session_state.final_df = final_df
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.success("데이터 로드 및 모델 학습이 완료되었습니다!")
        else:
            st.info("캐시된 데이터를 사용합니다.")

    def run(self):
        selected_page = render_sidebar()

        if selected_page == "map_page":
            map_page()
        
        # 다른 페이지가 추가될 경우 여기에 elif 문 추가
        
        else:
            st.error("선택된 페이지를 찾을 수 없습니다.")

if __name__ == '__main__':
    app = App()
    app.run()