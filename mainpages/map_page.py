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


# ──────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────

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


def get_grade_emoji(grade):
    return {"고위험": "🔴", "위험": "🟠", "주의": "🟡", "보통": "🟢"}.get(grade, "⚪")


# ──────────────────────────────────────────
# SHAP 차트
# ──────────────────────────────────────────

def render_shap_chart(model, final_df, selected_year, selected_region, X_train):
    X_cols = X_train.columns.tolist()
    target = final_df[
        (final_df['year'] == selected_year) &
        (final_df['sigun_nm'] == selected_region)
    ][X_cols]

    if target.empty:
        st.warning("SHAP 분석 데이터를 찾을 수 없습니다.")
        return

    with st.spinner("SHAP 분석 중..."):
        if 'shap_explainer' not in st.session_state:
            st.session_state.shap_explainer = shap.TreeExplainer(model)
        explainer = st.session_state.shap_explainer
        shap_values = explainer(target)

    vals = shap_values.values[0]
    feat_vals = target.values[0]
    indices = np.argsort(np.abs(vals))[-10:]
    top_vals = vals[indices]
    top_feat_labels = [f"{X_cols[i]}  (= {feat_vals[i]:.1f})" for i in indices]
    colors = ['#2196F3' if v > 0 else '#F44336' for v in top_vals]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    bars = ax.barh(top_feat_labels, top_vals, color=colors,
                   height=0.6, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, top_vals):
        offset = 0.008 if val > 0 else -0.008
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}',
            va='center',
            ha='left' if val > 0 else 'right',
            fontsize=10,
            fontweight='bold',
            color='#2196F3' if val > 0 else '#F44336'
        )

    ax.axvline(x=0, color='#333333', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)

    legend_handles = [
        mpatches.Patch(facecolor='#2196F3', label='소멸 위험 감소 (긍정적 영향)'),
        mpatches.Patch(facecolor='#F44336', label='소멸 위험 증가 (부정적 영향)')
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='#cccccc')
    ax.set_title(
        f'{selected_year}년 {selected_region} — SHAP 변수 영향도 분석',
        fontsize=13, fontweight='bold', pad=15, color='#1a1a2e'
    )
    ax.set_xlabel('SHAP 값  (← 위험 증가  |  위험 감소 →)', fontsize=10, color='#555555')
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────
# 메인 페이지
# ──────────────────────────────────────────

def map_page():

    st.markdown("""
    <style>
        .main-title {
            font-size: 2rem; font-weight: 800;
            color: #1a1a2e; margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1rem; color: #666; margin-bottom: 1.5rem;
        }
        .metric-card {
            background: white; border-radius: 12px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        }
        .metric-label { font-size: 0.85rem; color: #888; margin-bottom: 0.3rem; }
        .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
        .legend-row {
            display: flex; gap: 1rem; align-items: center;
            flex-wrap: wrap; margin-bottom: 0.8rem;
        }
        .legend-item {
            display: flex; align-items: center;
            gap: 0.3rem; font-size: 0.85rem; color: #444;
        }
        .legend-dot {
            width: 12px; height: 12px;
            border-radius: 50%; display: inline-block;
        }
        .section-header {
            font-size: 1.2rem; font-weight: 700; color: #1a1a2e;
            padding: 0.5rem 0; border-bottom: 2px solid #e0e0e0;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── 세션 확인 ──
    if 'final_df' not in st.session_state or st.session_state.final_df.empty:
        st.warning("⚠️ 데이터가 로드되지 않았습니다.")
        return
    if 'model' not in st.session_state:
        st.warning("⚠️ 모델이 학습되지 않았습니다.")
        return
    if 'X_train' not in st.session_state or st.session_state.X_train.empty:
        st.warning("⚠️ X_train 데이터가 없습니다.")
        return

    final_df = st.session_state.final_df
    model = st.session_state.model
    X_train = st.session_state.X_train

    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None

    # ── 타이틀 ──
    st.markdown('<div class="main-title">🗺️ 전국 주민 이탈 예측 지도</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">지역을 클릭하거나 검색하면 SHAP 분석 결과를 확인할 수 있습니다.</div>', unsafe_allow_html=True)

    # ── 상단 컨트롤: 연도 슬라이더 + 지역 검색 ──
    col_year, col_search = st.columns([2, 1])

    with col_year:
        years = sorted([int(y) for y in final_df['year'].unique()])
        selected_year = st.select_slider(
            "📅 연도 선택",
            options=years,
            value=2024
        )

    with col_search:
        all_regions = sorted(final_df['sigun_nm'].unique().tolist())
        search_options = ["선택 안 함"] + all_regions
        
        try:
            current_index = search_options.index(st.session_state.selected_region)
        except ValueError:
            current_index = 0

        search_region = st.selectbox(
            "🔍 지역 검색",
            options=search_options,
            index=current_index,
            key='region_search_box'
        )
        
        if search_region != "선택 안 함":
            st.session_state.selected_region = search_region
        else:
            st.session_state.selected_region = None

    # ── 등급 범례 ──
    st.markdown("""
    <div class="legend-row">
        <span style="font-size:0.85rem; color:#555; font-weight:600;">등급 기준 :</span>
        <span class="legend-item"><span class="legend-dot" style="background:#d73027;"></span> 고위험 (1.1 이하)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#fc8d59;"></span> 위험 (1.1 ~ 1.5)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#fee08b;"></span> 주의 (1.5 ~ 2.0)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#1a9850;"></span> 보통 (2.0 초과)</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 데이터 필터링 ──
    df_year = final_df[final_df['year'] == selected_year].copy()
    if df_year.empty:
        st.warning(f"{selected_year}년 데이터가 없습니다.")
        return

    classified = df_year['개선 인구 소멸 지수'].apply(classify_extinction_index)
    df_year['등급'] = classified.apply(lambda x: x[0])
    df_year['색상'] = classified.apply(lambda x: x[1])
    df_year['sigun_nm'] = df_year['sigun_nm'].replace('세종시', '세종특별자치시')

    # ── GeoJSON 로드 및 병합 ──
    try:
        with open("data/sigungu_excel_matched.geojson", 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error("GeoJSON 파일을 찾을 수 없습니다.")
        return

    region_coords = {}
    for feature in geojson_data['features']:
        city_name = extract_city_name(feature['properties'].get('source_SIG_KOR_NM', ''))
        try:
            coords = feature['geometry']['coordinates'][0][0]
            if feature['geometry']['type'] == 'MultiPolygon':
                 coords = feature['geometry']['coordinates'][0][0][0]
            region_coords[city_name] = [coords[1], coords[0]]
        except (IndexError, TypeError):
            continue

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
                sigun_dict[city_name]['개선 인구 소멸 지수'], 4)
            feature['properties']['등급'] = sigun_dict[city_name]['등급']
        else:
            feature['properties']['개선 인구 소멸 지수'] = '데이터 없음'
            feature['properties']['등급'] = '데이터 없음'

    # ── 지도 생성 ──
    map_center = [36.5, 127.5]
    map_zoom = 7
    selected_region_name = st.session_state.get('selected_region')

    if selected_region_name and selected_region_name in region_coords:
        map_center = region_coords[selected_region_name]
        map_zoom = 9

    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=geojson_data,
        data=df_year.drop_duplicates(subset=['sigun_nm']),
        columns=['sigun_nm', '개선 인구 소멸 지수'],
        key_on='feature.properties.city_nm',
        fill_color='RdYlGn',
        fill_opacity=0.75,
        line_opacity=0.2,
        legend_name='개선 인구 소멸 지수',
        nan_fill_color='lightgray'
    ).add_to(m)

    folium.GeoJson(
        geojson_data,
        name='clickable',
        style_function=lambda x: {'fillOpacity': 0, 'weight': 0.5, 'color': '#999'},
        highlight_function=lambda x: {
            'fillOpacity': 0.35, 'weight': 2.5,
            'color': '#1a1a2e', 'fillColor': '#ffeb3b'
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['city_nm', '개선 인구 소멸 지수', '등급'],
            aliases=['📍 지역', '📊 소멸 지수', '🏷️ 등급'],
            localize=True,
            sticky=True,
            style="font-family: sans-serif; font-size: 13px;"
        ),
        popup=folium.GeoJsonPopup(
            fields=['city_nm'],
            aliases=['지역'],
            localize=True
        )
    ).add_to(m)

    # ── 지도 렌더링 ──
    map_output = st_folium(m, width="100%", height=580,
                           returned_objects=["last_object_clicked_popup"])

    # ── 클릭 이벤트 처리 ──
    popup_data = map_output.get('last_object_clicked_popup')
    if popup_data:
        region_name_from_map = None
        if isinstance(popup_data, str):
            region_name_from_map = popup_data.replace('지역', '').strip()
        elif isinstance(popup_data, dict):
            region_name_from_map = popup_data.get('city_nm') or popup_data.get('지역')
        
        if region_name_from_map and st.session_state.selected_region != region_name_from_map:
            st.session_state.selected_region = region_name_from_map

    # ── SHAP 분석 패널 ──
    if st.session_state.selected_region:
        selected_region = st.session_state.selected_region
        region_data = df_year[df_year["sigun_nm"] == selected_region]

        st.markdown("--- ")
        st.markdown(
            f'<div class="section-header">📍 {selected_region} 상세 분석 ({selected_year}년)</div>',
            unsafe_allow_html=True
        )

        status_placeholder = st.empty()
        status_placeholder.info("⏳ SHAP 분석 중입니다. 잠시만 기다려 주세요...")

        if not region_data.empty:
            extinction_score = region_data['개선 인구 소멸 지수'].iloc[0]
            grade = region_data['등급'].iloc[0]
            color = region_data['색상'].iloc[0]
            emoji = get_grade_emoji(grade)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">개선 인구 소멸 지수</div>
                    <div class="metric-value">{extinction_score:.4f}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">소멸 위험 등급</div>
                    <div class="metric-value">
                        <span style="color:{color};">{emoji} {grade}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with c3:
                rank_df = df_year.sort_values('개선 인구 소멸 지수')
                region_list = rank_df['sigun_nm'].tolist()
                rank = region_list.index(selected_region) + 1 \
                    if selected_region in region_list else '-'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">전국 위험 순위</div>
                    <div class="metric-value">{rank}위 / {len(rank_df)}개</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="section-header">🔬 SHAP 변수 영향도 분석</div>',
                        unsafe_allow_html=True)
            st.caption("각 변수가 소멸 지수 예측에 얼마나 영향을 미쳤는지 나타냅니다. 파란색은 위험 감소, 빨간색은 위험 증가 방향입니다.")
            render_shap_chart(model, final_df, selected_year, selected_region, X_train)
            status_placeholder.empty()

        else:
            st.warning(f"'{selected_region}' 데이터를 찾을 수 없습니다.")


if __name__ == "__main__":
    map_page()
