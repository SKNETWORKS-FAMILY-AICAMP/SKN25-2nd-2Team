import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
import joblib
import os

def train_and_predict_target(df):
    """
    LightGBM 모델을 학습하고, 성능을 검증한 뒤, 미래의 목표 변수(y)를 예측합니다.

    Args:
        df (pd.DataFrame): X 변수 예측이 완료된 전체 데이터프레임

    Returns:
        tuple: (최종 예측이 포함된 데이터프레임, 학습된 최종 모델 객체)
    """
    warnings.filterwarnings('ignore')

    y_col = '개선 인구 소멸 지수'
    X_cols = [
        '고용률', '노인여가복지시설수', '초등학교 학생수 (명)', '가구수 (가구)', 
        '도소매업 사업체수(등록기반) (개)', '도소매업 종사자수(등록기반) (명)', 
        '서비스업 사업체수(등록기반) (개)', '평균연령 (세)', '유치원 교원수 (명)', 
        '혼인건수 (건)', '학급당 학생수 (명)', '출생아수 (명)', '신혼부부수 (쌍)', 
        '제조업 사업체수(등록기반) (개)', '인구총조사 인구 (명)', '주민등록인구 (명)', 
        '하수도보급률 (%)', '노인인구 1000명 당 여가복지시설', 
        '운수업 종사자수(등록기반) (명)', '초등학교 교원수 (명)', 
        '조혼인율 (천명당)', '이혼건수 (건)', '유치원 원아수 (명)'
    ]

    # 학습/검증/미래 데이터 분리
    train_df = df[df['year'] <= 2023].dropna(subset=[y_col])
    val_df = df[df['year'] == 2024].dropna(subset=[y_col])
    future_df = df[df['year'] > 2024]

    X_train = train_df[X_cols]
    y_train = train_df[y_col]
    X_val = val_df[X_cols]
    y_val = val_df[y_col]
    X_future = future_df[X_cols]

    # --- 모델 학습 및 성능 검증 ---
    print('--- Model Performance Evaluation on 2024 Data ---')
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    y_pred_val = lgbm.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}')

    # --- 전체 데이터로 재학습 및 미래 예측 ---
    print('\n--- Predicting target variable for 2025-2027 ---')
    full_train_df = df[df['year'] <= 2024].dropna(subset=[y_col])
    X_full_train = full_train_df[X_cols]
    y_full_train = full_train_df[y_col]

    lgbm_final = lgb.LGBMRegressor(random_state=42)
    lgbm_final.fit(X_full_train, y_full_train)
    y_pred_future = lgbm_final.predict(X_future)

    # 예측 결과를 데이터프레임에 채우기
    df.loc[df['year'] > 2024, y_col] = y_pred_future
    
    # 모델 저장
    model_path = 'analysis/lgbm_model.joblib'
    os.makedirs('analysis', exist_ok=True)
    joblib.dump(lgbm_final, model_path)
    print(f'Model saved to {model_path}')
    
    print('Target variable prediction complete.')
    return df, lgbm_final, X_full_train, y_full_train

if __name__ == '__main__':
    from data_loader import load_and_clean_data
    from feature_forecaster import forecast_features

    initial_df = load_and_clean_data()
    if initial_df is not None:
        forecasted_df = forecast_features(initial_df)
        final_df, _, _, _ = train_and_predict_target(forecasted_df)
        
        output_path = 'analysis/final_predictions.csv'
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nFinal predictions saved to {output_path}")
        print(final_df[final_df['year'] >= 2024].head())