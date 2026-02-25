# SKN25-2nd-2Team

# 1. 팀 소개
<img width="350" height="350" alt="ChatGPT Image 2026년 2월 24일 오전 10_26_26" src="https://github.com/user-attachments/assets/7f20875c-9ef7-4114-8030-92fe68a25be3" />


| 이름 | GitHub | 역할 |
| :--- | :---: | :--- |
| 권가영 | [@Gayoung03](https://github.com/Gayoung03) | 데이터 수집 및 전처리, 모델링, streamlit 구현 |
| 여해준 | [@inoocap-ux](https://github.com/inoocap-ux) | 데이터 수집 및 전처리 |
| 유행운 | [@happyhippo-cmd](https://github.com/아이디) | 데이터 수집 및 전처리 |
| 전운열 | [@cudaboy](https://github.com/cudaboy) | 데이터 전처리, 모델 구현, 시각화 생성 |
| 조은석 | [@silverstone-1004](https://github.com/silverstone-1004) | 데이터 수집, ppt 작성 |

# 2. 프로젝트 기간
2026.02.23. - 2026.02.24.

# 3. 프로젝트 개요

📕 프로젝트명

기초자치단체별 인구소멸위험지수 예측 및 소멸 요인 분석 프로젝트: **지방소멸 레이더 (지켜줘, 로컬)**

<br>

✅ 프로젝트 배경 및 목적

현재 대한민국은 급격한 고령화와 청년층의 수도권 집중 현상으로 인해 수많은 지역이 인구 소멸(인구 이탈) 위기에 직면해 있습니다. 단순히 소멸 위험을 인지하는 것을 넘어, '언제, 어느 지역이 가장 위험해지는가' 그리고 '**어떤 인프라(문화, 복지 등)의 부재가 소멸을 가속하는가**'를 데이터 기반으로 파악할 필요가 있습니다.

<br>

**주제 선정 배경**: 기업이 '**가입 고객 이탈 예측**'을 통해 이탈 가능성을 미리 방어하듯, 지역 주민을 하나의 고객으로 보고 이탈을 예측하여 선제적으로 대응하는 모형을 구축하고자 했습니다. 인구 소멸은 지자체별 데이터 기반의 선제적 대응이 필요하다고 판단하여 지역 별 인구 이탈에 관계성이 높은 인자를 파악하고자 하였습니다.

<br>

**프로젝트 3대 목적**:
1. 주민 이탈에 영향을 주는 핵심 변수 도출
2. 실질적인 전출입 인구 요인을 반영하기 위해, 전입/전출 인구 비율을 로그 값으로 반영한 '**개선 인구소멸지수**'를 새롭게 정의하고 이를 **Target 값**으로 활용하여 n년 후의 위험도 예측
3. SHAP 알고리즘을 활용한 지역별 소멸 위험 영향 요인 직관적 해석

<br>

🖐️ 프로젝트 소개

2020년부터 2024년까지 5개년의 229개 전국 기초자치단체(시/군/구) 데이터를 결합하여 미래(2025~2027년)의 지역별 소멸위험지수를 예측하고, 인구 이탈에 영향을 미치는 주요한 원인을 파악하는 머신러닝 프로젝트입니다.

<br>

❤️ 기대효과

• 2026, 2027년 소멸위험지수가 높은 지역 예측
• 소멸위험지수에 높은 영향을 미치는 요인 도출
• 시각화를 통한 분석 결과 공유

<br>

👤 대상 사용자
• 중앙부처의 인구 대응 TF
• 각 지자체의 정책 입안자 및 예산 담당자 등

<br>

# 4. 기술 스택
- **Language**
  - ![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)

- **Data Preprocessing**
  - ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  - ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

- **Machine Learning**
  - ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

- **Visualization**
  - ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
  - ![Folium](https://img.shields.io/badge/Folium-77B829?style=for-the-badge&logo=Leaflet&logoColor=white)
  - ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
  - ![Seaborn](https://img.shields.io/badge/Seaborn-%234479A1.svg?style=for-the-badge&logo=Seaborn&logoColor=white)

# 5. 수행결과

## 1️⃣ ERD
<img width="624" height="825" alt="ERD_" src="https://github.com/user-attachments/assets/5f616ace-37fd-43d4-948d-e712fae8fc60" />


## 2️⃣ 분석 결과
<img width="960" height="540" alt="16" src="https://github.com/user-attachments/assets/4de91cec-d731-4120-b668-fea7dd1ffa3f" />
<img width="960" height="540" alt="17" src="https://github.com/user-attachments/assets/a4e68255-581f-42a1-a240-30b3b36e7db6" />
<img width="960" height="540" alt="18" src="https://github.com/user-attachments/assets/4c64caac-6e44-4897-a908-1326367d976c" />


## 3️⃣ 대시보드

## 4️⃣ 아쉬운 점 및 향후 개선 방향
1. 데이터 범위: 5개년 연단위 데이터의 한계를 극복하기 위해 분기/월 단위 세밀한 데이터 및 외부 지표 추가 필요.
2. 모델링 고도화: 예측된 Feature 값으로 미래 Target을 예상하는 현재의 시계열 검증 방식을 고도화하고, 데이터 불균형 대응 및 다른 머신러닝 최적화 기법과의 추가 비교 필요.
3. 정책 연계 및 시각화: 데이터 전처리 소요 시간으로 인해 미완성된 Streamlit 웹 대시보드를 고도화하여, 실제 지자체의 맞춤형 대응 시나리오에 활용할 수 있도록 발전시킬 계획

# 6. 한 줄 회고
| 이름 | 회고 내용 |
|--------|-------|
| 권가영 | 데이터셋을 X,y 전부를 처음부터 구성하는 경험은 처음이라 선행 연구 분석을 많이 했습니다. 선행 연구로부터 얻은 게 많았고 분류, 회귀 모델을 같이 써본 연구라 뿌듯합니다. 함께 고생해주신 팀원분들께 감사합니다.|
| **여해준&nbsp;** | 코시스에서 데이터들을 수집할때 결측치가 많아서 왜 그런지 했는데 코시스 내부 데이터 구조에서 이상하게 띄어진 구조들이 많았습니다.이 경험에서 한번에 데이터를 깔끔하게 수집하는 것은 불가능하고 여러번의 전처리를 통해 보석같이 빛나는 데이터를 손에 넣을 수 있다는 사실을 깨달았습니다. |
| **유행운&nbsp;** | 데이터를 직접 수집하고 전처리하는 과정이 특히 인상 깊었습니다. 수업 시간에 보던 코드를 넘어 실제로 학습까지 진행해보니 공부를 더 열심히 해야겠다는 동기가 생겼습니다. 또한 현실적으로 중요한 주제를 다룬 프로젝트라 더욱 흥미로웠고, 팀원분들께서 적극적으로 참여해 주셔서 즐겁게 진행할 수 있었습니다. |
| **전운열&nbsp;** | 지방 소멸을 고객 이탈과 연결 지어서 주제를 선정한 점은 신선한 아이디어라고 생각합니다. 데이터 전처리 과정에서 KOSIS의 e-지방지표 통계자료를 수집할 때, 자치단체의 행정구역 변경으로 인해 지역명을 mapping하는 시간이 생각보다 길게 걸렸던 것 같습니다. 따라서 시계열 자료를 모델링하고 비교하거나, 가중치 최적화하는 과정에 많은 시간을 집중하지 못한 점은 추후 보완할 점이라고 생각합니다. 팀원 분들 모두 짧은 일정 중에 따라오느라 고생 많으셨고, 바쁘게 작업하시느라 모두 수고 많으셨습니다! |
| **조은석&nbsp;** | 열정적인 팀원들이 멋졌고, 공부해야 할 것이 많다는 것을 많이 느끼게 된 프로젝트였습니다. |
