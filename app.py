import streamlit as st
import pandas as pd
import joblib
from mainpages.eda_page import eda_page

# src 경로를 import 할 수 있도록 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/mainpages')

from src.data_loader import load_and_clean_data
from src.feature_forecaster import forecast_features, get_x_variable_names
from src.model_trainer import train_and_predict_target
from sidebar import render_sidebar
from mainpages.map_page import map_page

class App:
    def __init__(self):
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'final_df' in st.session_state:
            # st.info("캐시된 데이터를 사용합니다.") # 주석 처리하여 메시지 간소화
            return

        model_path = 'analysis/lgbm_model.joblib'
        data_path = 'analysis/final_predictions.csv'
        
        # analysis 폴더가 없으면 생성
        os.makedirs('analysis', exist_ok=True)

        if os.path.exists(model_path) and os.path.exists(data_path):
            with st.spinner("학습된 모델과 데이터를 로딩합니다..."):
                try:
                    st.session_state.model = joblib.load(model_path)
                    final_df = pd.read_csv(data_path)
                    st.session_state.final_df = final_df
                    
                    # SHAP 분석에 필요한 X_train 재구성
                    X_cols = get_x_variable_names()
                    train_data = final_df[final_df['year'] <= 2024].dropna(subset=['개선 인구 소멸 지수'])
                    st.session_state.X_train = train_data[X_cols]
                    
                    st.success("데이터 로딩 완료!")
                except Exception as e:
                    st.warning(f"저장된 데이터 로딩 중 오류 발생: {e}\n\n다시 모델을 학습합니다.")
                    # 로딩 실패 시, 학습 파이프라인 실행
                    self._run_training_pipeline(data_path)
        else:
            self._run_training_pipeline(data_path)

    def _run_training_pipeline(self, data_path):
        with st.spinner("데이터 로딩 및 모델 학습 중... (최초 1회 또는 파일 부재 시 실행)"):
            df = load_and_clean_data()
            if df is None:
                st.error("데이터 파일(최종_전처리완료.csv) 로딩에 실패했습니다.")
                return
            
            forecasted_df = forecast_features(df)
            final_df, model, X_train, y_train = train_and_predict_target(forecasted_df)
            
            # 결과 저장 (다음 실행을 위해)
            final_df.to_csv(data_path, index=False, encoding='utf-8-sig')
            # 모델은 train_and_predict_target 내부에서 이미 저장됨
            
            # 세션 상태 저장
            st.session_state.final_df = final_df
            st.session_state.model = model
            st.session_state.X_train = X_train
            
            st.success("데이터 로드 및 모델 학습이 완료되었습니다!")

    def run(self):
        # 세션 초기화가 실패했으면 여기서 중단
        if 'final_df' not in st.session_state:
            st.error("데이터 초기화에 실패하여 앱을 실행할 수 없습니다.")
            return

        selected_page = render_sidebar()

        if selected_page == "map_page":
            map_page()
        elif selected_page == "eda_page":
            eda_page()
        else:
            st.error("선택된 페이지를 찾을 수 없습니다.")

if __name__ == '__main__':
    app = App()
    app.run()
