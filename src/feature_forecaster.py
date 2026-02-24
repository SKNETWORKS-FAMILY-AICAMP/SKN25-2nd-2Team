
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from tqdm import tqdm

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from tqdm import tqdm

def get_x_variable_names():
    """
    EDA 및 VIF 분석을 통해 최종 선정된 23개의 X 변수 이름을 반환합니다.
    """
    return [
        '고용률', '노인여가복지시설수', '초등학교 학생수 (명)', '가구수 (가구)',
        '도소매업 사업체수(등록기반) (개)', '도소매업 종사자수(등록기반) (명)',
        '서비스업 사업체수(등록기반) (개)', '평균연령 (세)', '유치원 교원수 (명)',
        '혼인건수 (건)', '학급당 학생수 (명)', '출생아수 (명)', '신혼부부수 (쌍)',
        '제조업 사업체수(등록기반) (개)', '인구총조사 인구 (명)', '주민등록인구 (명)',
        '하수도보급률 (%)', '노인인구 1000명 당 여가복지시설',
        '운수업 종사자수(등록기반) (명)', '초등학교 교원수 (명)',
        '조혼인율 (천명당)', '이혼건수 (건)', '유치원 원아수 (명)'
    ]

def forecast_features(df):
    """
    각 지역별, 변수별로 ARIMA 모델을 사용하여 미래의 X 변수 값을 예측합니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임

    Returns:
        pd.DataFrame: 미래 X 변수 예측치가 포함된 데이터프레임
    """
    warnings.filterwarnings('ignore')

    X_cols = get_x_variable_names()

    future_years = [2025, 2026, 2027]
    forecast_steps = len(future_years)
    regions = df['sigun_nm'].unique()
    all_forecasts = []

    print(f'Starting feature forecasting for {len(regions)} regions...')
    # tqdm을 사용하여 진행 상황 표시
    for region in tqdm(regions, desc="Forecasting Regions"):
        region_df = df[df['sigun_nm'] == region]
        
        for col in X_cols:
            time_series_data = region_df[['year', col]].set_index('year')[col].sort_index()
            
            try:
                # 데이터가 상수 값인지 확인 (분산이 0인지)
                if time_series_data.var() == 0:
                    # 상수 값인 경우, 마지막 값으로 예측
                    forecast = pd.Series([time_series_data.iloc[-1]] * forecast_steps, index=pd.to_datetime([str(y) for y in future_years]).to_period('A'))
                else:
                    # ARIMA 모델 학습 및 예측
                    model = ARIMA(time_series_data, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=forecast_steps)
            except Exception:
                # 어떤 에러든 발생 시, 마지막 값으로 예측 (Fallback)
                last_value = time_series_data.iloc[-1]
                forecast = pd.Series([last_value] * forecast_steps, index=pd.to_datetime([str(y) for y in future_years]).to_period('A'))

            for i, year in enumerate(future_years):
                forecast_data = {
                    'sigun_nm': region,
                    'year': year,
                    'variable': col,
                    'forecasted_value': forecast.iloc[i]
                }
                all_forecasts.append(forecast_data)

    print('Feature forecasting complete.')

    forecast_df = pd.DataFrame(all_forecasts)
    forecast_pivot = forecast_df.pivot_table(index=['sigun_nm', 'year'], columns='variable', values='forecasted_value').reset_index()

    # 원본 데이터프레임에서 예측에 사용되지 않은 컬럼들 선택
    original_cols_to_keep = [col for col in df.columns if col not in X_cols and col != '개선 인구 소멸 지수']
    
    new_rows = []
    for region in regions:
        # 각 지역의 가장 최신 데이터(2024년)를 기반으로 미래 정보 생성
        base_info_row = df[(df['sigun_nm'] == region) & (df['year'] == 2024)]
        if base_info_row.empty:
             base_info_row = df[df['sigun_nm'] == region].iloc[-1:] # 2024년 데이터가 없는 경우 최신 데이터 사용
        
        for year in future_years:
            new_row = base_info_row[original_cols_to_keep].iloc[0].to_dict()
            new_row['year'] = year
            new_rows.append(new_row)

    future_base_df = pd.DataFrame(new_rows)
    future_df_final = pd.merge(future_base_df, forecast_pivot, on=['sigun_nm', 'year'])
    
    combined_df = pd.concat([df, future_df_final], ignore_index=True)
    
    # '개선 인구 소멸 지수' 컬럼이 없다면 생성하고 NaN으로 채움
    if '개선 인구 소멸 지수' not in combined_df.columns:
        combined_df['개선 인구 소멸 지수'] = pd.NA

    return combined_df

if __name__ == '__main__':
    from data_loader import load_and_clean_data
    
    initial_df = load_and_clean_data()
    if initial_df is not None:
        forecasted_df = forecast_features(initial_df)
        print("\n--- Forecasted DataFrame Head (2025 and after) ---")
        print(forecasted_df[forecasted_df['year'] >= 2025].head())
        
        # Save to a file for inspection
        output_path = 'analysis/X_variables_forecasted.csv'
        forecasted_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nForecasted data saved to {output_path}")
