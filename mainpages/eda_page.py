import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import platform
import warnings

warnings.filterwarnings('ignore')


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


# ──────────────────────────────────────────
# EDA 페이지
# ──────────────────────────────────────────

def eda_page():

    st.markdown("""
    <style>
        .main-title {
            font-size: 2rem; font-weight: 800;
            color: #1a1a2e; margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1rem; color: #666; margin-bottom: 1.5rem;
        }
        .section-header {
            font-size: 1.2rem; font-weight: 700; color: #1a1a2e;
            padding: 0.5rem 0; border-bottom: 2px solid #e0e0e0;
            margin-bottom: 1rem; margin-top: 1.5rem;
        }
        .insight-box {
            background: #f0f4ff;
            border-left: 4px solid #2196F3;
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1.2rem;
            margin: 0.8rem 0;
            font-size: 0.92rem;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">📊 탐색적 데이터 분석 (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">데이터의 분포, 상관관계, 변수 중요도를 탐색합니다.</div>', unsafe_allow_html=True)

    # ── 세션 확인 ──
    if 'final_df' not in st.session_state or st.session_state.final_df.empty:
        st.warning("데이터가 로드되지 않았습니다.")
        return

    final_df = st.session_state.final_df
    df = final_df[final_df['year'] <= 2024].copy()

    year_col = 'year'
    sigungu_col = 'sigun_nm'
    target_col = '개선 인구 소멸 지수'

    if target_col not in df.columns:
        st.error(f"'{target_col}' 컬럼이 없습니다.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [year_col, target_col]]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 연도별 분포",
        "🔥 상관관계 히트맵",
        "📈 시계열 트렌드",
        "🌲 RF 변수 중요도"
    ])

    # ── TAB 1 - 연도별 분포 ──
    with tab1:
        st.markdown('<div class="section-header">📦 연도별 개선 인구 소멸 지수 분포</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">💡 각 연도별로 소멸 지수 분포를 확인합니다. 박스의 위치와 이상치를 통해 위험 지역 변화 추이를 파악할 수 있습니다.</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        sns.boxplot(data=df, x=year_col, y=target_col, palette='husl', ax=ax)
        ax.set_title('연도별 개선 인구 소멸 지수 분포', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('연도', fontsize=12)
        ax.set_ylabel('개선 인구 소멸 지수', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('<div class="section-header">📋 연도별 기초 통계량</div>', unsafe_allow_html=True)
        stats_df = df.groupby(year_col)[target_col].describe().round(4)
        st.dataframe(stats_df, use_container_width=True)

    # ── TAB 2 - 상관관계 히트맵 ──
    with tab2:
        st.markdown('<div class="section-header">🔥 Target 및 주요 Feature 상관관계 히트맵</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">💡 소멸 지수와 상관관계가 높은 상위 변수를 히트맵으로 시각화합니다. 붉은색=양의 상관관계, 파란색=음의 상관관계입니다.</div>', unsafe_allow_html=True)

        top_n = st.slider("상위 변수 개수 선택", min_value=5, max_value=20, value=16, step=1)

        corr_matrix = df[feature_cols + [target_col]].corr(method='pearson', min_periods=1)
        top_corr_features = (
            corr_matrix[target_col].abs()
            .sort_values(ascending=False)
            .index[1:top_n + 1]
        )
        selected_cols = top_corr_features.insert(0, target_col)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#f8f9fa')
        sns.heatmap(
            df[selected_cols].corr(),
            annot=True, fmt=".2f", cmap='coolwarm',
            vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 9}
        )
        ax.set_title(f'상관관계 히트맵 (Top {top_n})', fontsize=15, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('<div class="section-header">📋 소멸 지수와의 상관계수 순위</div>', unsafe_allow_html=True)
        corr_with_target = (
            corr_matrix[target_col].drop(target_col)
            .sort_values(ascending=False)
            .reset_index()
        )
        corr_with_target.columns = ['변수명', '상관계수']
        corr_with_target['상관계수'] = corr_with_target['상관계수'].round(4)
        st.dataframe(corr_with_target, use_container_width=True, height=300)

    # ── TAB 3 - 시계열 트렌드 ──
    with tab3:
        st.markdown('<div class="section-header">📈 주요 변수 연도별 트렌드</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">💡 상관관계 상위 변수들의 연도별 평균값 변화를 Area Chart로 확인합니다.</div>', unsafe_allow_html=True)

        corr_matrix2 = df[feature_cols + [target_col]].corr(method='pearson', min_periods=1)
        top_features_ts = (
            corr_matrix2[target_col].abs()
            .sort_values(ascending=False)
            .index[1:17].tolist()
        )

        selected_features = st.multiselect(
            "시계열 트렌드를 볼 변수 선택 (최대 12개)",
            options=top_features_ts,
            default=top_features_ts[:8],
            max_selections=12
        )

        if not selected_features:
            st.info("변수를 선택해주세요.")
        else:
            years_list = sorted(df[year_col].unique())
            df_grouped = df.groupby(year_col)[selected_features].mean().reset_index()
            cols_per_row = 3
            rows = (len(selected_features) + cols_per_row - 1) // cols_per_row

            fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, 4 * rows))
            fig.patch.set_facecolor('#f8f9fa')

            if rows == 1 and len(selected_features) == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            colors = plt.cm.Set2(np.linspace(0, 1, len(selected_features)))

            for idx, (feat, color) in enumerate(zip(selected_features, colors)):
                ax = axes[idx]
                ax.set_facecolor('#f8f9fa')
                y_vals = df_grouped[feat]
                ax.plot(df_grouped[year_col], y_vals, linewidth=2.5, color=color, alpha=0.9)
                y_min, y_max = y_vals.min(), y_vals.max()
                y_margin = (y_max - y_min) * 0.2 if y_max != y_min else abs(y_min) * 0.2 + 0.1
                lower = y_min - y_margin
                ax.fill_between(df_grouped[year_col], lower, y_vals, color=color, alpha=0.25)
                ax.set_ylim(lower, y_max + y_margin)
                ax.set_title(feat, fontsize=11, fontweight='bold')
                ax.set_xticks(years_list)
                ax.set_xlabel('연도', fontsize=9)
                ax.set_ylabel('평균값', fontsize=9)
                ax.tick_params(labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.2, linestyle='--')

            for idx in range(len(selected_features), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle('주요 변수 연도별 평균 트렌드', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── TAB 4 - RF 변수 중요도 ──
    with tab4:
        st.markdown('<div class="section-header">🌲 RandomForest 변수 중요도 분석</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">💡 연도별 RandomForest로 변수 중요도 Top N을 추출하고, 모든 연도에 공통된 핵심 변수를 선정합니다. 빨간색 막대가 최종 선정된 공통 변수입니다.</div>', unsafe_allow_html=True)

        top_n_rf = st.slider("연도별 Top N 변수 선택", min_value=10, max_value=60, value=40, step=5)

        if st.button("🌲 RF 변수 중요도 분석 시작", type="primary"):
            from sklearn.ensemble import RandomForestClassifier

            threshold = 1.04
            years_rf = sorted(df[year_col].unique())
            top_features_per_year = {}
            importance_per_year = {}

            progress = st.progress(0)
            status = st.empty()

            for i, year in enumerate(years_rf):
                status.info(f"⏳ {year}년 RF 학습 중... ({i+1}/{len(years_rf)})")

                df_yr = df[df[year_col] == year].copy()
                y_yr = (df_yr[target_col] < threshold).astype(int)

                X_yr = df_yr[feature_cols].copy()
                X_yr = X_yr.apply(pd.to_numeric, errors='coerce')
                X_yr = X_yr.dropna(axis=1, how='all')
                X_yr = X_yr.fillna(X_yr.mean())

                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_yr, y_yr)

                importances = pd.Series(rf.feature_importances_, index=X_yr.columns)
                top_n_feats = importances.sort_values(ascending=False).head(top_n_rf)
                top_features_per_year[year] = set(top_n_feats.index.tolist())
                importance_per_year[year] = importances.sort_values(ascending=False).head(20)

                progress.progress((i + 1) / len(years_rf))

            status.success("✅ 분석 완료!")

            common_features = set.intersection(*top_features_per_year.values())

            st.markdown(f'<div class="insight-box">✅ 모든 연도 Top {top_n_rf} 내 <b>공통 핵심 변수 {len(common_features)}개</b> 선정 완료</div>',
                        unsafe_allow_html=True)

            if common_features:
                st.write("**공통 핵심 변수 목록:**")
                st.write(sorted(list(common_features)))

            # 연도별 중요도 차트
            st.markdown('<div class="section-header">연도별 Top 20 변수 중요도</div>', unsafe_allow_html=True)
            selected_year_rf = st.selectbox("연도 선택", options=years_rf, index=len(years_rf) - 1)

            imp_data = importance_per_year[selected_year_rf]
            colors_rf = ['#d73027' if feat in common_features else '#2196F3' for feat in imp_data.index]

            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            ax.barh(imp_data.index[::-1], imp_data.values[::-1],
                    color=colors_rf[::-1], height=0.6, edgecolor='white', linewidth=0.5)
            ax.set_title(f'{selected_year_rf}년 RF 변수 중요도 Top 20',
                         fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Feature Importance', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(axis='x', alpha=0.2, linestyle='--')
            legend_handles = [
                mpatches.Patch(facecolor='#d73027', label='공통 핵심 변수 (교집합)'),
                mpatches.Patch(facecolor='#2196F3', label='해당 연도 선택 변수')
            ]
            ax.legend(handles=legend_handles, loc='lower right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("위 버튼을 클릭하면 RF 분석을 시작합니다. (수 분 소요될 수 있습니다)")


if __name__ == "__main__":
    eda_page()