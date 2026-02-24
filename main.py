import streamlit as st
from data_handler import load_and_preprocess_data
import plot_utils
import model_trainer
import xai_utils

# 데이터를 한 번만 로드하도록 캐싱 (속도 최적화)
@st.cache_data
def get_data():
    # 파일 경로는 실제 환경에 맞게 조정해 주세요.
    return load_and_preprocess_data('data/최종_전처리완료.csv')

def render_main(page):
    # 1. 데이터 불러오기
    try:
        df_full, df_model = get_data()
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 2. 페이지 라우팅 로직
    if page == "1. 기초 통계량 (Data Summary)":
        st.title("📊 기초 통계량 조회")
        st.markdown("특정 변수를 선택하여 5년간의 요약 통계량을 확인합니다.")
        
        selected_cols = st.multiselect(
            "확인할 변수를 선택하세요:", 
            df_full.columns.tolist(), 
            default=[df_full.columns[5], df_full.columns[11]]
        )
        if selected_cols:
            st.dataframe(df_full[selected_cols].describe(), use_container_width=True)

    elif page == "2. EDA 시각화 (KDE, Heatmap, Trend)":
        st.title("📈 탐색적 데이터 분석 (EDA)")
        
        st.subheader("2-1. Target(개선인구소멸지수) KDE Histogram")
        plot_utils.plot_target_kde(df_full)
        
        st.markdown("---")
        st.subheader("2-2. 동적 상관관계 Heatmap")
        feature_cols = st.multiselect(
            "상관관계를 볼 변수를 선택하세요:", 
            df_full.columns.tolist()[12:81], 
            default=df_full.columns.tolist()[12:17]
        )
        if feature_cols:
            plot_utils.plot_dynamic_heatmap(df_full, feature_cols)
            
        st.markdown("---")
        st.subheader("2-3. 동적 시계열 트렌드 (Area Chart)")
        trend_cols = st.multiselect(
            "시계열 트렌드를 볼 변수를 선택하세요:", 
            df_full.columns.tolist()[12:81], 
            default=df_full.columns.tolist()[12:16]
        )
        if trend_cols:
            plot_utils.plot_time_series_trend(df_full, trend_cols)

    elif page == "3. 모델 성능 비교 (RF vs XGB vs CatBoost)":
        st.title("🤖 다중 스텝 예측 모델 성능 비교")
        st.write("2020~2024년 롤링 윈도우 교차 검증을 통해 학습된 각 모델의 성능(T+1 예측) 비교입니다.")
        
        with st.spinner("모델 학습 및 교차 검증 평가 중..."):
            metrics_df, best_model = model_trainer.train_and_evaluate(df_model)
            st.table(metrics_df)
            
            # 1등 모델 기반 자동 스토리텔링 (이전 단계에서 작성한 인사이트)
            st.success(model_trainer.get_storytelling(metrics_df.iloc[0]['Model']))

    elif page == "4. 예측 지도 & XAI (SHAP)":
        st.title("🗺️ 연도별 예측 지도 및 XAI 인사이트")
        
        with st.spinner("모델을 불러오는 중..."):
            _, best_model = model_trainer.train_and_evaluate(df_model)
        
        # 상단 슬라이더
        target_year = st.slider("예측 대상 연도 (Target Year)", 2025, 2027, 2025)
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📍 {target_year}년 지방소멸 고위험 예측 지도")
            # 지도 표출 후 클릭된 지역 이름 반환
            clicked_region = xai_utils.draw_interactive_map(df_model, best_model, target_year)
        
        with col2:
            st.subheader("🔍 SHAP 기반 핵심 요인 분석")
            if clicked_region:
                # 클릭된 지역명에 맞춰 SHAP 그래프 및 인사이트 텍스트 표출
                xai_utils.show_shap_and_insight(df_model, best_model, clicked_region, target_year)
            else:
                st.info("👈 왼쪽 지도에서 분석하고 싶은 지역(시군구)을 클릭해주세요.")