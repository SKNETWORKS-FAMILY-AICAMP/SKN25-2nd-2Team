import streamlit as st
import pandas as pd
import lightgbm as lgb
from mainpages.map_page import map_page
from src.feature_forecaster import forecast_features # forecast_features 함수 임포트

# --- 데이터 로드 함수 (임시) ---
@st.cache_data
def load_data():
    # 실제 데이터 로드 로직을 여기에 구현합니다.
    # 예시로 더미 데이터를 생성합니다.
    st.write("데이터 로드 중... (실제 데이터 로드 로직 필요)")
    
    # 최종 전처리 완료 데이터 로드 (예시 경로)
    try:
        df_full = pd.read_csv('data/최종_전처리완료.csv', encoding='cp949', thousands=',')
    except FileNotFoundError:
        st.error("data/최종_전처리완료.csv 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None, None, None

    # '개선 인구 소멸 지수' 컬럼이 없는 경우 임시로 생성
    if '개선 인구 소멸 지수' not in df_full.columns:
        df_full['개선 인구 소멸 지수'] = df_full['인구총조사 인구 (명)'] / 100000 # 임시 값

    # 예측 변수 목록 가져오기
    from src.feature_forecaster import get_x_variable_names
    X_cols = get_x_variable_names()

    # X_train 데이터 준비 (예측에 사용될 피처들)
    # 실제 X_train은 모델 학습 시 사용된 피처들로 구성되어야 합니다.
    # 여기서는 final_df에서 X_cols에 해당하는 컬럼들을 X_train으로 사용합니다.
    X_train_data = df_full[['sigun_nm', 'year'] + X_cols].copy()

    # final_df 생성 (2024년 실측 + 2025~2027년 예측)
    # forecast_features 함수를 사용하여 미래 데이터 예측
    final_df = forecast_features(df_full)
    
    # 모델 로드 (임시)
    # 실제 모델 로드 로직을 여기에 구현합니다.
    st.write("모델 로드 중... (실제 모델 로드 로직 필요)")
    model = lgb.LGBMClassifier() # 더미 모델
    # model = joblib.load('path/to/your/model.pkl') # 실제 모델 로드 예시

    return final_df, model, X_train_data

class App:
    def __init__(self):
        self.pages = {
            "지도 시각화": map_page,
            # 다른 페이지가 있다면 여기에 추가
        }

    def run(self):
        st.sidebar.title("메인 메뉴")
        selection = st.sidebar.radio("페이지 선택", list(self.pages.keys()))

        # 데이터 및 모델 로드 (한 번만 실행)
        if 'final_df' not in st.session_state:
            final_df, model, X_train = load_data()
            if final_df is not None:
                st.session_state.final_df = final_df
                st.session_state.model = model
                st.session_state.X_train = X_train
            else:
                st.error("데이터 로드에 실패했습니다. 애플리케이션을 실행할 수 없습니다.")
                return

        # 선택된 페이지 실행
        page = self.pages[selection]
        page()

if __name__ == "__main__":
    # 이 부분은 main.py에서 호출되므로 직접 실행되지 않습니다.
    # 테스트를 위해 App().run()을 여기에 넣을 수 있지만, 일반적으로 main.py가 시작점입니다.
    pass
