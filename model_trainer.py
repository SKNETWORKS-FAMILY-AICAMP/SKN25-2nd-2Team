import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score

def train_and_evaluate(df_model):
    """교차 검증을 통해 모델별 성능을 평가하고 결과를 반환합니다."""
    # Feature 정의 (타겟 및 식별 컬럼 제외)
    drop_cols = ['year', 'sigun_nm', '개선 인구 소멸 지수 ', 'target_T1', 'target_T2', 'target_T3', 'y_T1', 'y_T2', 'y_T3']
    X_features = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors='ignore')
    
    # 숫자형 변환 및 결측치 처리
    X_features = X_features.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    X_features = X_features.fillna(X_features.mean())

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, random_state=42, verbose=0)
    }

    # 롤링 윈도우 설정 (Train: ~2022 -> Test: 2023 / Train: ~2023 -> Test: 2024)
    windows = [([2020, 2021, 2022], 2023), ([2020, 2021, 2022, 2023], 2024)]
    
    final_results = []
    trained_models = {}

    for name, model in models.items():
        f1_list = []
        for train_years, test_year in windows:
            train_idx = df_model[(df_model['year'].isin(train_years)) & (df_model['y_T1'].notnull())].index
            test_idx = df_model[(df_model['year'] == test_year) & (df_model['y_T1'].notnull())].index
            
            X_train, y_train = X_features.loc[train_idx], df_model.loc[train_idx, 'y_T1']
            X_test, y_test = X_features.loc[test_idx], df_model.loc[test_idx, 'y_T1']
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1_list.append(f1_score(y_test, preds, average='macro'))
        
        avg_f1 = np.mean(f1_list)
        final_results.append({'Model': name, 'Average F1-Score': round(avg_f1, 4)})
        
        # 전체 데이터로 최종 학습 (예측용)
        model.fit(X_features, df_model['y_T1'].fillna(0)) 
        trained_models[name] = model

    results_df = pd.DataFrame(final_results).sort_values(by='Average F1-Score', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    
    return results_df, trained_models[best_model_name]

def get_storytelling(model_name):
    """선정된 모델에 따른 분석 인사이트 텍스트를 반환합니다."""
    stories = {
        "Random Forest": "데이터의 샘플 수가 적고 이상치가 많은 특성상, 여러 나무의 결과를 평균 내어 과적합을 방어하는 Random Forest가 가장 안정적인 성능을 보였습니다.",
        "XGBoost": "정형 데이터의 미세한 잔차를 정교하게 학습하는 XGBoost가 지자체별 통계 수치의 트렌드를 가장 잘 포착하였습니다.",
        "CatBoost": "지자체별 특화 데이터에 존재하는 복잡한 속성들을 내부적으로 최적화하여 처리하는 CatBoost의 성능이 가장 우수했습니다."
    }
    return stories.get(model_name, "")