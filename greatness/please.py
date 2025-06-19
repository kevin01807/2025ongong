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

# 유니코드 정리 함수
def clean_unicode(text):
    return ''.join(c for c in text if c.isprintable() and ord(c) < 55296 or ord(c) > 57343)

# ----------------------
# 1. 데이터 불러오기 및 전처리
# ----------------------
@st.cache_data
def load_data():
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "4-4-1.csv")
    st.write("데이터 경로 확인:", file_path)
    df = pd.read_csv(file_path, encoding="utf-8")

    # 컬럼명 정리
    df.rename(columns={
        '기술유형': 'Skill_Type',
        '성별': 'Gender',
        'Year': '연도',
        'Value': '값'
    }, inplace=True)

    # 기술코드 매핑
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

# ----------------------
# 2. Streamlit 인터페이스
# ----------------------
df = load_data()

st.title(clean_unicode("ICT 역량 분류 및 격차 분석"))
st.header(clean_unicode("기술 유형별 ICT 활용 격차"))

# 기술 유형 선택
selected_skill = st.selectbox("기술을 선택하세요", df['Skill_KR'].dropna().unique())

# 선택한 기술 필터링
filtered = df[df['Skill_KR'] == selected_skill]
filtered = filtered.dropna(subset=['연도', '값', 'Gender'])

# 시각화
if filtered.empty:
    st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=filtered, x='연도', y='값', hue='Gender', ax=ax)
    ax.set_title(clean_unicode(f"{selected_skill} 기술 활용도 (성별 비교)"))
    st.pyplot(fig)

# ----------------------
# 3. 나이브 베이즈 분류기 적용
# ----------------------
st.subheader(clean_unicode("나이브 베이즈 분류기를 활용한 예측"))

numeric_df = df[['연도', '값']].copy()
numeric_df['성별'] = df['Gender']
numeric_df['기술'] = df['Skill_KR']

# Label Encoding
numeric_df['성별코드'] = numeric_df['성별'].map({'남자': 0, '여자': 1, '전체': 2})
numeric_df['기술코드'] = numeric_df['기술'].astype('category').cat.codes

X = numeric_df[['연도', '성별코드', '기술코드']]
y = numeric_df['값'] > numeric_df['값'].mean()  # 평균 초과 여부

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text(clean_unicode("📌 분류 보고서"))
st.text(clean_unicode(classification_report(y_test, y_pred)))

# ----------------------
# 4. 큐 & 스택 시뮬레이션
# ----------------------
st.subheader(clean_unicode("자료구조 시뮬레이션: 큐와 스택"))

tab1, tab2 = st.tabs(["📥 큐 (Queue)", "📦 스택 (Stack)"])

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
# 5. 간단한 정렬 알고리즘 시각화
# ----------------------
st.subheader(clean_unicode("정렬 알고리즘 시각화"))

sort_data = st.text_input("정렬할 숫자 입력 (쉼표로 구분)", value="5,2,9,1,7")

if st.button("정렬 시작"):
    try:
        nums = [int(x) for x in sort_data.split(',')]
        st.write("원본 배열:", nums)

        # 버블 정렬 구현
        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        st.write("정렬된 배열:", nums)

        # 시각화
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(nums)), nums)
        ax2.set_title("정렬 결과 시각화")
        st.pyplot(fig2)
    except:
        st.warning("숫자를 올바르게 입력해주세요!")
