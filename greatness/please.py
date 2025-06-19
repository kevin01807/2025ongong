import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import deque
import numpy as np
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="ICT 역량 분류 및 격차 분석", layout="wide")

# ----------------------
# 1. 데이터 불러오기 및 전처리
# ----------------------

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "data", "4-4-1.csv")
    st.write("📂 데이터 경로 확인:", os.path.abspath(file_path))
    df = pd.read_csv(file_path, encoding="utf-8")
    df.rename(columns={'기술유형': 'Skill_Type', '성별': 'Gender'}, inplace=True)

    skill_map = {
        'ARSP': '문서 편집',
        'EMAIL': '이메일 사용',
        'COPY': '파일 복사',
        'SEND': '파일 전송',
        'INST': 'SW 설치',
        'COMM': '온라인 커뮤니케이션',
        'BUY': '온라인 구매',
        'BANK': '온라인 뱅킹',
        'USEC': '보안 설정'
    }
    df['Skill_KR'] = df['Skill_Type'].map(skill_map)
    df['Gender'] = df['Gender'].fillna('전체')
    return df

# 이하 동일...


df = load_data()

# ----------------------
# 2. 데이터 시각화
# ----------------------

st.title("\ud83d\udcca ICT 역량 데이터 시각화 및 분류 시스템")
st.markdown("SDGs 4.4.1 - ICT 기술 역량 보유자 비율 시계열 및 분류 예측")

selected_skill = st.selectbox("기술 유형 선택", sorted(df['Skill_KR'].unique()))
filtered_df = df[df['Skill_KR'] == selected_skill]

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=filtered_df, x='Year', y='Value', hue='Gender', marker='o', ax=ax)
plt.title(f"{selected_skill} 비율 변화 (연도별, 성별)")
plt.ylabel("비율 (%)")
st.pyplot(fig)

# ----------------------
# 3. 분류용 샘플 데이터 생성 및 나이브 베이즈 분류기 학습
# ----------------------

np.random.seed(42)
sample_data = pd.DataFrame({
    'Age': np.random.randint(10, 70, 200),
    'Gender': np.random.choice(['남성', '여성'], 200),
    'Edu_Level': np.random.choice(['중졸 이하', '고졸', '대졸 이상'], 200),
    'Internet_Hour': np.random.randint(0, 8, 200),
    'ICT_Skilled': np.random.choice([0, 1], 200, p=[0.4, 0.6])
})

gender_map = {'남성': 0, '여성': 1}
edu_map = {'중졸 이하': 0, '고졸': 1, '대졸 이상': 2}

sample_data['Gender_num'] = sample_data['Gender'].map(gender_map)
sample_data['Edu_num'] = sample_data['Edu_Level'].map(edu_map)

X = sample_data[['Age', 'Gender_num', 'Edu_num', 'Internet_Hour']]
y = sample_data['ICT_Skilled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GaussianNB()
model.fit(X_train, y_train)

# ----------------------
# 4. 사용자 입력 - 분류 예측
# ----------------------

st.header("\ud83e\uddd0 ICT 역량 보유 예측 (나이브 베이즈)")
age = st.slider("나이", 10, 70, 25)
gender = st.radio("성별", ['남성', '여성'])
edu = st.selectbox("교육 수준", ['중졸 이하', '고졸', '대졸 이상'])
hour = st.slider("하루 인터넷 이용 시간", 0, 10, 2)

input_df = pd.DataFrame([[
    age, gender_map[gender], edu_map[edu], hour
]], columns=X.columns)

prediction = model.predict(input_df)[0]
result_text = "\u2705 ICT 역량 **보유**로 예측됩니다." if prediction else "\u274c ICT 역량 **미보유**로 예측됩니다."
st.subheader(result_text)

# ----------------------
# 5. 자료구조 시뮬레이션 (큐 / 스택)
# ----------------------

st.header("\ud83e\udde9 자료구조 시뮬레이션")

st.subheader("큐 구조 (대기열 시뮬레이션)")
queue = deque()
queue.append("사용자1")
queue.append("사용자2")
queue.append("사용자3")

if st.button("대기열에서 다음 사용자 꺼내기 (popleft)"):
    if queue:
        st.success(f"처리 중: {queue.popleft()}")
    else:
        st.warning("큐가 비어 있습니다.")

st.subheader("스택 구조 (후입선출 시뮬레이션)")
stack = []
if st.button("결과 스택에 저장"):
    stack.append(result_text)
    st.info(f"스택 크기: {len(stack)}")

if st.button("최근 결과 보기 (pop)"):
    if stack:
        st.success(stack.pop())
    else:
        st.warning("스택이 비어 있습니다.")

# ----------------------
# 6. 정렬 알고리즘 시각화
# ----------------------

st.header("\ud83d\udcc8 지역별 ICT 기술 보유율 정렬 예시")

region_data = pd.DataFrame({
    'Region': [f"지역{i}" for i in range(1, 11)],
    'ICT_Rate': np.random.uniform(40, 90, 10).round(2)
})

sort_algo = st.selectbox("정렬 알고리즘 선택", ['선택 정렬', '버블 정렬'])

if sort_algo == '선택 정렬':
    sorted_data = region_data.copy()
    for i in range(len(sorted_data)):
        min_idx = i
        for j in range(i+1, len(sorted_data)):
            if sorted_data.loc[j, 'ICT_Rate'] < sorted_data.loc[min_idx, 'ICT_Rate']:
                min_idx = j
        sorted_data.iloc[i], sorted_data.iloc[min_idx] = sorted_data.iloc[min_idx], sorted_data.iloc[i]
elif sort_algo == '버블 정렬':
    sorted_data = region_data.copy()
    for i in range(len(sorted_data)):
        for j in range(0, len(sorted_data) - i - 1):
            if sorted_data.loc[j, 'ICT_Rate'] > sorted_data.loc[j+1, 'ICT_Rate']:
                sorted_data.iloc[j], sorted_data.iloc[j+1] = sorted_data.iloc[j+1], sorted_data.iloc[j]

st.dataframe(sorted_data)
st.bar_chart(sorted_data.set_index('Region'))
