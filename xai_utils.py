import streamlit as st
import folium
from streamlit_folium import st_folium
import shap
import matplotlib.pyplot as plt

def draw_interactive_map(df_model, model, target_year):
    # 실제 환경에서는 geo_json 파일과 모델 예측 확률을 매핑합니다.
    m = folium.Map(location=[36.5, 127.5], zoom_start=7)
    
    # TODO: folium.Choropleth 추가 로직 작성
    
    # st_folium을 통해 클릭 이벤트 포착
    map_data = st_folium(m, width=700, height=600)
    
    # 클릭한 지역의 이름 반환 (geojson의 속성 구조에 따라 'name' 등 수정)
    if map_data and map_data.get('last_active_drawing'):
        return map_data['last_active_drawing']['properties']['name']
    return None

def show_shap_and_insight(df_model, model, region_name, target_year):
    # 1. 지역 데이터 필터링 (가장 최신 연도 기준)
    region_data = df_model[(df_model['sigun_nm'] == region_name) & (df_model['year'] == 2024)]
    
    if region_data.empty:
        st.warning(f"{region_name}의 데이터를 찾을 수 없습니다.")
        return
        
    X_target = region_data.drop(columns=['year', 'sigun_nm', 'target_T1', 'target_T2', 'target_T3', 'y_T1', 'y_T2', 'y_T3'], errors='ignore')
    
    # 2. 예측 확률 도출
    prob = model.predict_proba(X_target)[0][1] * 100
    
    # 3. SHAP 계산 및 시각화
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_target)
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    
    # 4. 동적 텍스트 생성 (가장 영향력이 큰 상위 2개 변수 추출)
    feature_names = X_target.columns
    shap_vals = shap_values[0].values
    
    # SHAP 값이 양수(위험도 증가)인 상위 변수 찾기
    positive_impacts = sorted(zip(feature_names, shap_vals), key=lambda x: x[1], reverse=True)
    top_reasons = f"'{positive_impacts[0][0]}' 및 '{positive_impacts[1][0]}'"
    
    # 최종 인사이트 출력
    st.markdown(f"""
    ### 💡 [{region_name}] 지역 정책 제언 인사이트
    보시는 것처럼 **{region_name}**은(는) **{prob:.1f}%** 확률로 고위험군 진입이 예상되며,  
    모델은 그 핵심 원인을 위 차트의 붉은색 막대로 표시된 변수(**{top_reasons}** 등)들로 지목했습니다.
    
    따라서 중앙정부는 **{region_name}**에 일회성 지원금이 아닌, 
    해당 핵심 취약 인프라를 확충하는 정책을 우선순위로 투입해야 합니다.
    """)