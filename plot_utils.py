import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import platform
import os
import pandas as pd
import numpy as np

# --- 1. 한글 폰트 및 기본 설정 ---
# OS별 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else:
    # Linux/Docker 환경 등에서 폰트가 없을 경우를 대비
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)

# 이미지 저장 폴더 생성
os.makedirs('plots', exist_ok=True)

# --- 2. 시각화 함수 정의 ---
def plot_target_boxplot(df):
    """Target(개선인구소멸지수)의 연도별 Boxplot 출력 및 저장"""
    # 기존 코드의 컬럼 인덱스를 그대로 활용
    target_col = df.columns[11]  
    year_col = df.columns[5]     
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # KDE 대신 Boxplot 적용, ax 객체에 바로 그리기
    sns.boxplot(data=df, x=year_col, y=target_col, palette='husl', ax=ax)
    
    # ax 객체를 사용하여 Title, Label, Y-limit 설정
    ax.set_title(f'연도별 [{target_col}] 분포', fontsize=16)
    ax.set_xlabel('연도', fontsize=12)
    ax.set_ylabel('개선 인구 소멸 지수', fontsize=12)
    ax.set_ylim(bottom=0)
    
    # PPT용 저장 및 Streamlit 출력
    plt.savefig('plots/1_target_boxplot_distribution.png', dpi=300, bbox_inches='tight')
    st.pyplot(fig)
    plt.close(fig)

def plot_dynamic_heatmap(df, feature_cols):
    """선택된 Feature들 간의 상관관계 히트맵 출력 및 저장"""
    target_col = df.columns[11]
    cols_to_plot = [target_col] + feature_cols
    
    # 숫자형 데이터만 선택하여 상관관계 계산
    corr_data = df[cols_to_plot].select_dtypes(include=[np.number]).corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Target 및 주요 Feature 간의 상관관계 히트맵', fontsize=16, pad=15)
    
    # PPT용 저장 및 Streamlit 출력
    plt.savefig('plots/2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    st.pyplot(fig)
    plt.close(fig)

def plot_time_series_trend(df, feature_cols):
    """패싯(Faceting)과 영역 차트(Area Chart)를 활용한 시계열 트렌드 분석"""
    year_col_name = df.columns[5]
    df_ts = df.copy()

    # 연도별 평균값 계산 및 melt
    df_grouped_mean = df_ts.groupby(year_col_name)[feature_cols].mean().reset_index()
    df_plot_melted = df_grouped_mean.melt(id_vars=year_col_name, var_name='Feature', value_name='Average Value')

    # --- 💡 커스텀 매핑 함수: 변수별 min/max에 맞춰 y축 최적화 ---
    def custom_area_plot(x, y, color, **kwargs):
        ax = plt.gca()
        
        # 선 그리기
        ax.plot(x, y, linewidth=2.5, color=color, alpha=0.9)
        
        # 데이터의 min, max 계산 후 위아래로 20% 여백 추가
        y_min, y_max = y.min(), y.max()
        y_margin = (y_max - y_min) * 0.2
        if y_margin == 0:  # 데이터 변화가 없을 경우 방어코드
            y_margin = y_min * 0.2 if y_min != 0 else 0.1
            
        lower_bound = y_min - y_margin
        upper_bound = y_max + y_margin
        
        # y축 범위를 타이트하게 재설정
        ax.set_ylim(lower_bound, upper_bound)
        
        # 영역 칠하기 (바닥을 0이 아닌 lower_bound로 설정)
        ax.fill_between(x, lower_bound, y, color=color, alpha=0.3)

    # FacetGrid 생성 (sharey=False로 독립적 y축 허용)
    g = sns.FacetGrid(df_plot_melted, col='Feature', hue='Feature', 
                      col_wrap=4, height=4, aspect=1.2, palette='Set2', sharey=False)

    # 커스텀 함수 매핑
    g.map(custom_area_plot, year_col_name, 'Average Value')

    # x축 연도 설정
    g.set(xticks=df_ts[year_col_name].unique())

    # 레이아웃 및 폰트 크기 설정
    g.set_titles(col_template="{col_name}", size=15, fontweight='bold')
    g.set_axis_labels('연도', '평균값', fontsize=14)

    for ax in g.axes.flatten():
        ax.tick_params(labelsize=12)

    plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3)
    g.fig.suptitle('주요 Feature의 연도별 트렌드 (변수별 스케일 최적화)', fontsize=22, fontweight='bold')

    # PPT용 저장 및 Streamlit 출력
    plt.savefig('plots/3_timeseries_trend_area_chart.png', dpi=300, bbox_inches='tight')
    st.pyplot(g.fig)
    plt.close(g.fig)