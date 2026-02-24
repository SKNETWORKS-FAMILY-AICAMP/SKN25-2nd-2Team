import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    데이터를 로드하고 분석 및 모델링에 적합하게 전처리하는 함수
    """
    # 1. 데이터 로드 (천 단위 쉼표 제거)
    df_full = pd.read_csv(file_path, encoding='cp949', thousands=',')
    
    # 2. 분석에 필요한 주요 컬럼 인덱스 정의
    year_col = df_full.columns[5]       # F열: 연도
    sigungu_col = df_full.columns[4]    # E열: 시군구
    target_col = df_full.columns[11]    # L열: 개선 소멸위험지수
    
    # 3. 분석용 데이터프레임 생성 (M열~CC열 Feature 포함)
    feature_cols = df_full.columns[12:81].tolist()
    df = df_full[[year_col, sigungu_col, target_col] + feature_cols].copy()
    
    # 4. 데이터 타입 변환 및 정렬
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[year_col] = df[year_col].astype(int)
    df = df.sort_values(by=[sigungu_col, year_col]).reset_index(drop=True)
    
    # 5. 다중 스텝 직접 예측을 위한 미래 타겟 생성 (T+1, T+2, T+3)
    # 1.04 미만인 경우를 소멸 위기(1)로 정의
    threshold = 1.04
    df['y_T1'] = (df.groupby(sigungu_col)[target_col].shift(-1) < threshold).astype(float)
    df['y_T2'] = (df.groupby(sigungu_col)[target_col].shift(-2) < threshold).astype(float)
    df['y_T3'] = (df.groupby(sigungu_col)[target_col].shift(-3) < threshold).astype(float)
    
    return df_full, df