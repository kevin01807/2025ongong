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
import re

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 유니코드 오류 방지용 텍스트 정리 함수
def clean_unicode(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = ''.join(c for c in text if c.isprintable())
    return text

st.set_page_config(page_title=clean_unicode("ICT 역량 분류 및 격차 분석"), layout="wide")

# ----------------------
# 1. 데이터 불러오기 및 전처리
# ----------------------

@st.cache_data
def load_data():
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "4-4-1.csv")
    st.write(clean_unicode("데이터 경로 확인:"), file_path)
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

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(clean_unicode)

    return df

# ----------------------
# 2. 제목 및 데이터 불러오기
# ----------------------
st.title(clean_unicode("ICT 역량 분류 및 격차 분석"))
df = load_data()
st.dataframe(df.head())

# ----------------------
# 3. 기술 유형별 시각화
# ----------------------
st.header(clean_unicode("기술 유형별 ICT 활용 격차"))
selected_skill = st.selectbox("기술을 선택하세요", df['Skill_KR'].unique())
filtered = df[df['Skill_KR'] == selected_skill]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=filtered, x='연도', y='값', hue='Gender', ax=ax)
ax.set_title(clean_unicode(f"{selected_skill} 기술 활용도 (성별 비교)"))
st.pyplot(fig)

# ----------------------
# 4. 나이브 베이즈 분류기 적용
# ----------------------
st.subheader(clean_unicode("나이브 베이즈 분류기를 활용한 예측"))

numeric_df = df[['연도', '값']].copy()
numeric_df['성별'] = df['Gender']
numeric_df['기술'] = df['Skill_KR']

numeric_df['성별코드'] = numeric_df['성별'].map({'남자': 0, '여자': 1, '전체': 2})
numeric_df['기술코드'] = numeric_df['기술'].astype('category').cat.codes

X = numeric_df[['연도', '성별코드', '기술코드']]
y = numeric_df['값'] > numeric_df['값'].mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text(clean_unicode("분류 보고서"))
st.text(clean_unicode(classification_report(y_test, y_pred)))

# ----------------------
# 5. 큐 & 스택 시뮬레이션
# ----------------------
st.subheader(clean_unicode("자료구조 시뮬레이션: 큐와 스택"))

tab1, tab2 = st.tabs(["큐 (Queue)", "스택 (Stack)"])

with tab1:
    queue = deque()
    q_input = st.text_input("큐에 추가할 항목 입력")
    if st.button("큐에 추가"):
        queue.append(q_input)
    if st.button("큐에서 제거"):
        if queue:
            queue.popleft()
    st.write("현재 큐 상태:", list(queue))

with tab2:
    stack = []
    s_input = st.text_input("스택에 추가할 항목 입력")
    if st.button("스택에 추가"):
        stack.append(s_input)
    if st.button("스택에서 제거"):
        if stack:
            stack.pop()
    st.write("현재 스택 상태:", stack)

# ----------------------
# 6. 정렬 알고리즘 시각화
# ----------------------
st.subheader(clean_unicode("정렬 알고리즘 시각화"))

sort_data = st.text_input("정렬할 숫자 입력 (쉼표로 구분)", value="5,2,9,1,7")

if st.button("정렬 시작"):
    try:
        nums = [int(x) for x in sort_data.split(',')]
        st.write("원본 배열:", nums)

        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        st.write("정렬된 배열:", nums)

        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(nums)), nums)
        ax2.set_title(clean_unicode("정렬 결과 시각화"))
        st.pyplot(fig2)
    except:
        st.warning("숫자를 올바르게 입력해주세요!")
