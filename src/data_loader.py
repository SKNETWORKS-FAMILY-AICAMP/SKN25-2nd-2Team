
import pandas as pd
import warnings

def load_and_clean_data():
    """
    데이터를 로드하고 숫자형으로 변환해야 할 컬럼들을 정제합니다.
    
    Returns:
        pd.DataFrame: 정제된 데이터프레임
    """
    warnings.filterwarnings('ignore')

    # 상대 경로 사용
    file_path = 'data/최종_전처리완료.csv'

    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # 모든 컬럼명의 앞뒤 공백 제거
    df.columns = df.columns.str.strip()

    # 숫자 변환이 필요한 object 타입 컬럼 리스트
    cols_to_convert = [
        '가구수 (가구)', '도소매업 종사자수(등록기반) (명)', '도시지역면적 (㎡)',
        '등록장애인수 (명)', '사망자수 (명)', '토지거래면적 (천㎡)'
    ]

    # 컬럼 타입 변환
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # 결측치 발생 시 0으로 채움 (단순한 예시, 실제로는 더 정교한 처리 필요 가능)
    df.fillna(0, inplace=True)

    print("Data loading and cleaning complete.")
    return df

if __name__ == '__main__':
    cleaned_df = load_and_clean_data()
    if cleaned_df is not None:
        print("\n--- Data Info After Cleaning ---")
        cleaned_df.info()
        print("\n--- First 5 Rows ---")
        print(cleaned_df.head())
