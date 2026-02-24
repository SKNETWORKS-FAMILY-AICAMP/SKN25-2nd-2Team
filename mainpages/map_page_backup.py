import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import platform
import re


# --- 한글 폰트 설정 ---
def set_korean_font():
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


# --- 구 단위 → 시 단위 변환 함수 ---
def extract_city_name(gu_name):
    first = gu_name.split(',')[0].strip()
    if '세종' in first:
        return '세종특별자치시'
    match = re.match(r'^(.+?시)(?=[가-힣])', first)
    if match:
        return match.group(1)
    match = re.match(r'^(.+?시)\s', first)
    if match:
        return match.group(1)
    match = re.match(r'^(.+?시)$', first)
    if match:
        return match.group(1)
    match = re.match(r'^(.+?군)', first)
    if match:
        return match.group(1)
    return first


# --- 등급 분류 함수 ---
def classify_extinction_index(score):
    if pd.isna(score):
        return "알 수 없음", "#808080"
    if score <= 1.1:
        return "고위험", "#d73027"
    elif score <= 1.5:
        return "위험", "#fc8d59"
    elif score <= 2.0:
        return "주의", "#fee08b"
    else:
        return "보통", "#1a9850"


# --- 메인 페이지 함수 ---
def map_page():
    st.title("전국 시군구 주민 이탈 예측 지도")

    # --- 세션 상태 확인 ---
    if 'final_df' not in st.session_state or st.session_state.final_df.empty:
        st.warning("데이터가 로드되지 않았습니다.")
        return
    if 'model' not in st.session_state:
        st.warning("모델이 학습되지 않았습니다.")
        return
    if 'X_train' not in st.session_state or st.session_state.X_train.empty:
        st.warning("X_train 데이터가 없습니다.")
        return

    final_df = st.session_state.final_df
    model = st.session_state.model
    X_train = st.session_state.X_train

    # selected_region 초기화 (최초 1회만)
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None

    # --- 연도 선택 ---
    years = sorted(final_df['year'].unique())
    selected_year = st.sidebar.radio(
        "연도 선택",
        years,
        index=years.index(2024) if 2024 in years else 0
    )

    st.subheader(f"{selected_year}년 전국 시군구 주민 이탈 예측")

    # --- 선택 연도 데이터 필터링 ---
    df_year = final_df[final_df['year'] == selected_year].copy()

    if df_year.empty:
        st.warning(f"{selected_year}년 데이터가 없습니다.")
        return

    if '개선 인구 소멸 지수' not in df_year.columns:
        st.error("'개선 인구 소멸 지수' 컬럼이 없습니다.")
        return

    # 등급/색상 컬럼 추가
    classified = df_year['개선 인구 소멸 지수'].apply(classify_extinction_index)
    df_year['등급'] = classified.apply(lambda x: x[0])
    df_year['색상'] = classified.apply(lambda x: x[1])

    # 세종시 이름 통일
    df_year['sigun_nm'] = df_year['sigun_nm'].replace('세종시', '세종특별자치시')

    # --- GeoJSON 로드 (매번 새로 로드) ---
    geojson_path = "data/sigungu_excel_matched.geojson"
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON 파일을 찾을 수 없습니다: {geojson_path}")
        return

    # --- GeoJSON에 소멸 지수 데이터 병합 ---
    sigun_dict = (
        df_year.drop_duplicates(subset=['sigun_nm'])
        .set_index('sigun_nm')[['개선 인구 소멸 지수', '등급']]
        .to_dict('index')
    )

    for feature in geojson_data['features']:
        original_name = feature['properties'].get('source_SIG_KOR_NM', '')
        city_name = extract_city_name(original_name)
        feature['properties']['city_nm'] = city_name
        if city_name in sigun_dict:
            feature['properties']['개선 인구 소멸 지수'] = round(
                sigun_dict[city_name]['개선 인구 소멸 지수'], 4
            )
            feature['properties']['등급'] = sigun_dict[city_name]['등급']
        else:
            feature['properties']['개선 인구 소멸 지수'] = '데이터 없음'
            feature['properties']['등급'] = '데이터 없음'

    # --- Folium 지도 생성 ---
    m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles="cartodbpositron")

    # Choropleth - 색상 담당
    folium.Choropleth(
        geo_data=geojson_data,
        data=df_year.drop_duplicates(subset=['sigun_nm']),
        columns=['sigun_nm', '개선 인구 소멸 지수'],
        key_on='feature.properties.city_nm',
        fill_color='RdYlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='개선 인구 소멸 지수',
        nan_fill_color='lightgray'
    ).add_to(m)

    # GeoJson - 클릭 및 툴팁 담당
    folium.GeoJson(
        geojson_data,
        name='clickable',
        style_function=lambda x: {
            'fillOpacity': 0,
            'weight': 0.5,
            'color': 'gray'
        },
        highlight_function=lambda x: {
            'fillOpacity': 0.3,
            'weight': 2,
            'color': 'black',
            'fillColor': 'yellow'
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['city_nm', '개선 인구 소멸 지수', '등급'],
            aliases=['지역', '소멸 지수', '등급'],
            localize=True,
            sticky=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['city_nm'],
            aliases=['지역'],
            localize=True
        )
    ).add_to(m)

    # --- 지도 렌더링 ---
    map_output = st_folium(m, width=1200, height=600, returned_objects=["last_object_clicked_popup"])

    # --- 클릭 이벤트 처리 ---
    popup_data = map_output.get('last_object_clicked_popup')

    if popup_data:
        # 문자열로 오는 경우: "지역단양군" → "단양군" 추출
        if isinstance(popup_data, str):
            region_name = popup_data.replace('지역', '').strip()
            if region_name:
                st.session_state.selected_region = region_name
        # 딕셔너리로 오는 경우
        elif isinstance(popup_data, dict):
            region_name = popup_data.get('city_nm') or popup_data.get('지역')
            if region_name:
                st.session_state.selected_region = region_name

    # --- SHAP 분석 표시 ---
    if st.session_state.selected_region:
        selected_region = st.session_state.selected_region

        region_data = df_year[df_year['sigun_nm'] == selected_region]

        if not region_data.empty:
            extinction_score = region_data['개선 인구 소멸 지수'].iloc[0]
            grade = region_data['등급'].iloc[0]
            color = region_data['색상'].iloc[0]

            st.markdown("---")
            st.subheader(f"📍 {selected_region} ({selected_year}년) 상세 분석")
            st.markdown(
                f"**개선 인구 소멸 지수:** {extinction_score:.4f} &nbsp;&nbsp; "
                f"**등급:** <span style='color:{color}; font-weight:bold;'>{grade}</span>",
                unsafe_allow_html=True
            )

            # SHAP 분석
            X_cols = X_train.columns.tolist()
            target = final_df[
                (final_df['year'] == selected_year) &
                (final_df['sigun_nm'] == selected_region)
            ][X_cols]

            if not target.empty:
                with st.spinner("SHAP 분석 중..."):
                    if "shap_explainer" not in st.session_state:
                        st.session_state.shap_explainer = shap.TreeExplainer(model)
                    explainer = st.session_state.shap_explainer
                    shap_values = explainer(target)
                    vals = shap_values.values[0]
                    indices = np.argsort(np.abs(vals))[-10:]
                    top_vals = vals[indices]
                    top_feats = [X_cols[i] for i in indices]
                    feat_vals = target.values[0]
                    top_feat_labels = [
                        f"{X_cols[i]}\n= {feat_vals[i]:.1f}" for i in indices
                    ]
                    colors = ['#2196F3' if v > 0 else '#F44336' for v in top_vals]

                    fig, ax = plt.subplots(figsize=(10, 7))
                    bars = ax.barh(top_feat_labels, top_vals, color=colors)

                    for bar, val in zip(bars, top_vals):
                        ax.text(
                            val + (0.005 if val > 0 else -0.005),
                            bar.get_y() + bar.get_height() / 2,
                            f'{val:+.3f}',
                            va='center',
                            ha='left' if val > 0 else 'right',
                            fontsize=10,
                            fontweight='bold',
                            color='#2196F3' if val > 0 else '#F44336'
                        )

                    ax.axvline(x=0, color='black', linewidth=0.8)
                    legend_handles = [
                        mpatches.Patch(facecolor='#2196F3', label='소멸 위험 감소 (긍정)'),
                        mpatches.Patch(facecolor='#F44336', label='소멸 위험 증가 (부정)')
                    ]
                    ax.legend(handles=legend_handles, loc='lower right', fontsize=10)
                    ax.set_title(
                        f'{selected_year}년 {selected_region} SHAP 분석',
                        fontsize=14, fontweight='bold', pad=15
                    )
                    ax.set_xlabel('SHAP 값', fontsize=11)
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning(f"'{selected_region}' ({selected_year}년) SHAP 분석 데이터를 찾을 수 없습니다.")
        else:
            st.warning(f"'{selected_region}' ({selected_year}년) 데이터가 없습니다.")


if __name__ == "__main__":
    map_page()