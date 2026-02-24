import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import shap
import matplotlib.pyplot as plt
import platform
import os

# SHAP 시각화 함수 임포트 (경로에 따라 수정 필요)
# Assuming utils is a package or directly accessible
try:
    from src.explainer import show_shap_waterfall_plot
except ImportError:
    st.error("Could not import show_shap_waterfall_plot. Make sure src/explainer.py is accessible.")
    show_shap_waterfall_plot = None

# 한글 폰트 설정 (Streamlit 환경에 맞게 조정)
def set_korean_font():
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin': # Mac
        plt.rcParams['font.family'] = 'AppleGothic'
    else: # Linux (e.g. Streamlit Cloud)
        # 나눔 폰트 설치 및 설정 (Streamlit Cloud 환경을 위한 예시)
        # 이 부분은 Streamlit 앱 배포 시 추가적인 설정이 필요할 수 있습니다.
        # 예를 들어, Dockerfile에 폰트 설치 명령 추가 등
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            st.warning("나눔고딕 폰트를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
            plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

set_korean_font()

# --- 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_geojson(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        return geojson_data
    except FileNotFoundError:
        st.error(f"GeoJSON 파일을 찾을 수 없습니다: {path}")
        return None
    except json.JSONDecodeError:
        st.error(f"GeoJSON 파일 형식이 올바르지 않습니다: {path}")
        return None

# --- 인구 소멸 지수 등급 분류 함수 ---
def classify_extinction_index(score):
    # score가 유효한 숫자가 아닌 경우 기본값 반환
    if pd.isna(score):
        return "알 수 없음", "#808080" # 회색
    
    if score <= 0.5: # 예시 기준, 실제 기준은 데이터 분포에 따라 조정 필요
        return "고위험", "#FF0000"  # 진한 빨강
    elif score <= 1.0:
        return "위험", "#FFA500"    # 주황
    elif score <= 1.5:
        return "주의", "#FFFF00"    # 노랑
    else:
        return "보통", "#008000"    # 초록

def map_page():
    st.title("전국 시군구 주민 이탈 예측 지도")

    # --- 세션 상태 확인 ---
    if 'final_df' not in st.session_state or st.session_state.final_df.empty:
        st.warning("데이터가 로드되지 않았습니다. '데이터 로드' 페이지에서 데이터를 먼저 로드해주세요.")
        return
    if 'model' not in st.session_state:
        st.warning("모델이 학습되지 않았습니다. '모델 학습' 페이지에서 모델을 먼저 학습시켜주세요.")
        return
    if 'X_train' not in st.session_state or st.session_state.X_train.empty:
        st.warning("SHAP 분석을 위한 X_train 데이터가 없습니다. '데이터 로드' 페이지에서 데이터를 먼저 로드해주세요.")
        return

    final_df = st.session_state.final_df
    model = st.session_state.model
    X_train = st.session_state.X_train

    # --- 연도 선택 바 ---
    years = sorted(final_df['year'].unique())
    selected_year = st.sidebar.radio("연도 선택", years, index=years.index(2024) if 2024 in years else 0)

    st.subheader(f"{selected_year}년 전국 시군구 주민 이탈 예측")

    # --- GeoJSON 데이터 로드 ---
    geojson_path = "data/sigungu_excel_matched.geojson"
    geojson_data = load_geojson(geojson_path)

    if geojson_data is None:
        return

    # --- 선택된 연도 데이터 필터링 및 등급 분류 ---
    df_year = final_df[final_df['year'] == selected_year].copy()
    
    if df_year.empty:
        st.warning(f"{selected_year}년에 대한 데이터가 없습니다. 다른 연도를 선택해주세요.")
        return

    # GeoJSON의 sigun_cd와 DataFrame의 sigun_nm을 매핑하기 위한 전처리
    # GeoJSON의 properties에서 'sigun_cd'를 찾고, df_year의 'sigun_nm'과 매핑
    # GeoJSON의 'sigun_cd'는 숫자형, df_year의 'sigun_nm'은 문자열일 수 있으므로 통일 필요
    # 여기서는 GeoJSON의 'sigun_cd'를 기준으로 df_year에 'sigun_cd' 컬럼을 추가하는 방식으로 진행
    
    # GeoJSON의 'sigun_cd'와 'sigun_nm' 매핑 딕셔너리 생성
    geojson_sigungu_map = {}
    for feature in geojson_data['features']:
        props = feature['properties']
        # GeoJSON에 'sigun_nm'이 없으면 'source_SIG_KOR_NM'을 사용
        region_name_in_geojson = props.get('sigun_nm')
        if not region_name_in_geojson: # sigun_nm이 없으면 source_SIG_KOR_NM 시도
            region_name_in_geojson = props.get('source_SIG_KOR_NM')

        if 'sigun_cd' in props and region_name_in_geojson:
            geojson_sigungu_map[str(region_name_in_geojson)] = str(props['sigun_cd']) # region_name을 키로, sigun_cd를 값으로

    # df_year에 'sigun_cd' 컬럼 추가
    df_year['sigun_cd'] = df_year['sigun_nm'].map(geojson_sigungu_map)
    df_year.dropna(subset=['sigun_cd'], inplace=True) # 매핑되지 않은 지역 제거

    # '개선 인구 소멸 지수' 컬럼의 NaN 값 처리 (classify_extinction_index 함수에서 처리)
    if '개선 인구 소멸 지수' not in df_year.columns:
        st.error("데이터에 '개선 인구 소멸 지수' 컬럼이 없습니다.")
        return

    # 인구 소멸 지수 등급 및 색상 적용
    # apply 함수가 Series를 반환하도록 명시적으로 처리
    classified_data = df_year['개선 인구 소멸 지수'].apply(lambda x: classify_extinction_index(x))
    df_year['등급'] = classified_data.apply(lambda x: x[0])
    df_year['색상'] = classified_data.apply(lambda x: x[1])

    # --- Folium 지도 생성 ---
    m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles="cartodbpositron")

    # Choropleth 레이어 추가
    # 기존 Choropleth 레이어와 툴팁 추가 로직을 통합
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        data=df_year,
        columns=['sigun_cd', '개선 인구 소멸 지수'],
        key_on='feature.properties.sigun_cd',
        fill_color='YlOrRd', # 색상 스케일 (예: 노랑-주황-빨강)
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'{selected_year}년 개선 인구 소멸 지수',
        highlight=True,
        name='Choropleth'
    ).add_to(m)

    # Choropleth에 툴팁 추가
    # GeoJSON 데이터의 'source_SIG_KOR_NM'과 df_year의 '개선 인구 소멸 지수', '등급'을 매핑하여 툴팁 생성
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['source_SIG_KOR_NM', '개선 인구 소멸 지수', '등급'],
            aliases=['지역', '개선 인구 소멸 지수', '등급'],
            localize=True,
            labels=True,
            sticky=False,
            style="background-color: #F0EFE9; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )

    # --- Streamlit에 Folium 지도 렌더링 ---
    st.session_state.selected_region = None # 초기화
    map_output = st_folium(m, width=700, height=500, returned_objects=["last_active_drawing", "last_object_clicked"])

    # --- 지역 클릭 인터랙션 처리 ---
    if map_output and map_output["last_object_clicked"]:
        clicked_region_props = map_output["last_object_clicked"]["properties"]
        # GeoJSON의 'source_SIG_KOR_NM'을 사용하여 클릭된 지역명 가져오기
        clicked_sigun_nm = clicked_region_props.get('source_SIG_KOR_NM')
        
        if clicked_sigun_nm:
            st.session_state.selected_region = clicked_sigun_nm
            st.sidebar.write(f"선택된 지역: **{st.session_state.selected_region}**")

            # 선택된 지역의 데이터 가져오기
            region_data_for_shap = df_year[df_year['sigun_nm'] == clicked_sigun_nm]

            if not region_data_for_shap.empty:
                extinction_score = region_data_for_shap['개선 인구 소멸 지수'].iloc[0]
                grade = region_data_for_shap['등급'].iloc[0]
                color = region_data_for_shap['색상'].iloc[0]

                st.subheader(f"'{clicked_sigun_nm}' ({selected_year}년) 상세 분석")
                st.markdown(f"**개선 인구 소멸 지수:** {extinction_score:.2f} (등급: <span style='color:{color};'><b>{grade}</b></span>)", unsafe_allow_html=True)

                # SHAP 분석을 위한 X 변수 추출
                # X_train의 컬럼 순서와 모델 학습 시 사용된 컬럼 순서가 동일하다고 가정
                X_region_year = X_train[(X_train['sigun_nm'] == clicked_sigun_nm) & (X_train['year'] == selected_year)]
                
                if not X_region_year.empty:
                    # 'sigun_nm', 'year' 컬럼 제외하고 모델 입력에 사용될 피처만 추출
                    features_for_model = X_region_year.drop(columns=['sigun_nm', 'year'], errors='ignore')
                    
                    if show_shap_waterfall_plot:
                        st.write("---")
                        st.subheader("SHAP Waterfall Plot")
                        # show_shap_waterfall_plot 함수는 내부적으로 explainer를 생성하고 플롯을 그림
                        # X_target은 단일 샘플 (Series 또는 DataFrame)이어야 함
                        show_shap_waterfall_plot(model, features_for_model.iloc[0]) # 첫 번째 행 (단일 샘플) 전달
                        st.pyplot(plt) # matplotlib 그림을 Streamlit에 표시
                        plt.clf() # 그림 초기화

                        # SHAP 상위 2개 변수 요약
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(features_for_model.iloc[0])
                        
                        # LightGBM의 경우 shap_values가 리스트로 반환될 수 있음 (클래스별)
                        # 이진 분류의 경우 보통 두 번째 요소 (클래스 1에 대한 shap_values)를 사용
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1] # Assuming binary classification, focus on positive class

                        shap_df = pd.DataFrame({
                            'feature': features_for_model.columns,
                            'shap_value': shap_values
                        })
                        shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
                        shap_df = shap_df.sort_values(by='abs_shap_value', ascending=False).head(2)

                        st.write("---")
                        st.subheader("SHAP 분석 요약")
                        if not shap_df.empty:
                            for index, row in shap_df.iterrows():
                                feature_name = row['feature']
                                shap_value = row['shap_value']
                                effect = "증가" if shap_value > 0 else "감소"
                                st.markdown(f"- **{feature_name}**: 이 요인이 주민 이탈 지수 예측에 **{effect}** 영향을 미쳤습니다. (SHAP 값: {shap_value:.2f})")
                        else:
                            st.write("SHAP 상위 변수를 찾을 수 없습니다.")
                    else:
                        st.error("SHAP 시각화 함수를 로드할 수 없어 SHAP 분석을 수행할 수 없습니다.")
                else:
                    st.warning(f"선택된 '{clicked_sigun_nm}' ({selected_year}년)에 대한 SHAP 분석 데이터(X_train)를 찾을 수 없습니다.")
            else:
                st.warning(f"선택된 '{clicked_sigun_nm}' ({selected_year}년)에 대한 데이터가 없습니다.")
            
            # st.session_state.selected_region = None # SHAP 분석 후 선택 초기화 (필요에 따라)

# 페이지 실행
if __name__ == "__main__":
    map_page()
