import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd # pandas 임포트 추가

def show_shap_waterfall_plot(model, X_target):
    """
    주어진 데이터 포인트(X_target)에 대한 SHAP Waterfall Plot을 생성합니다.
    X_target은 단일 샘플의 Series 또는 DataFrame 형태일 수 있습니다.
    """
    if X_target.empty:
        st.warning("SHAP 분석을 위한 데이터가 없습니다.")
        return

    # X_target이 Series인 경우 DataFrame으로 변환
    if isinstance(X_target, pd.Series):
        X_target_df = X_target.to_frame().T
    else:
        X_target_df = X_target

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_target_df) # DataFrame 형태로 전달

    fig, ax = plt.subplots()
    # shap_values는 Explanation 객체이므로, 단일 인스턴스에 대한 waterfall plot은 shap_values[0]으로 접근
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close()

    # SHAP 값이 가장 큰 변수들을 이용한 텍스트 설명
    feature_names = X_target_df.columns
    # shap_values.values는 Explanation 객체 내의 SHAP 값 배열
    # 단일 인스턴스이므로 첫 번째 행의 값을 사용
    shap_vals = shap_values.values[0]
    
    # SHAP 값이 양수(위험도 증가)인 상위 변수 찾기
    # zip을 사용하여 feature_names와 shap_vals를 묶고, shap_vals 기준으로 정렬
    feature_shap_pairs = sorted(zip(feature_names, shap_vals), key=lambda x: x[1], reverse=True)
    
    if len(feature_shap_pairs) >= 2:
        # 상위 2개 변수 이름 추출
        top_feature1 = feature_shap_pairs[0][0]
        top_feature2 = feature_shap_pairs[1][0]
        
        st.markdown(f"""
        #### 💡 분석 요약
        모델은 **'{top_feature1}'** 및 **'{top_feature2}'** 등의 변수가 소멸 지수 예측에 가장 큰 영향을 미친다고 분석했습니다.
        """ ) 
    else:
        st.markdown("#### 💡 분석 요약\nSHAP 분석을 위한 충분한 변수가 없습니다.")
