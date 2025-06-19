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

# 유니코드 오류 방지용 텍스트 정리 함수
def clean_unicode(text):
    return ''.join(c for c in str(text) if c.isprintable())

st.set_page_config(page_title=clean_unicode("ICT 역량 분류 및 격차 분석"), layout="wide")

# ----------------------
# 1. 데이터 불러오기 및 전처리
# ----------------------
@st.cache_data
def load_data():
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "4-4-1.csv")
    st.write("📂 데이터 경로 확인:", file_path)
    df = pd.read_csv(file_path, encoding="utf-8")
    return df

df = load_data()

# ----------------------
# 2. 시각화
# ----------------------
st.title(clean_unicode("ICT 역량 분류 및 격차 분석"))
st.header(clean_unicode("기술 유형별 ICT 활용 격차"))

if '기술유형' in df.columns and '성별' in df.columns and 'Year' in df.columns and 'Value' in df.columns:
    df['기술유형'] = df['기술유형'].astype(str)
    selected_skill = st.selectbox("기술을 선택하세요", df['기술유형'].unique())

    filtered = df[df['기술유형'] == selected_skill]

    if filtered.empty:
        st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
    else:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=filtered, x='Year', y='Value', hue='성별', ax=ax)
            ax.set_title(clean_unicode(f"{selected_skill} 기술 활용도 (성별 비교)"))
            st.pyplot(fig)
        except Exception as e:
            st.error(f"시각화 중 오류 발생: {e}")
else:
    st.error("데이터셋에 필요한 컬럼이 없습니다. 컬럼명을 확인하세요.")

# ----------------------
# 3. 나이브 베이즈 분류기 적용
# ----------------------
st.subheader("나이브 베이즈 분류기를 활용한 예측")

numeric_df = df[['Year', 'Value']].copy()
numeric_df['Gender'] = df['성별']
numeric_df['Skill'] = df['기술유형']

# Label Encoding
numeric_df['Gender_Code'] = numeric_df['Gender'].map({'남자': 0, '여자': 1, '전체': 2})
numeric_df['Skill_Code'] = numeric_df['Skill'].astype('category').cat.codes

# NaN 제거
numeric_df = numeric_df.dropna(subset=['Year', 'Value', 'Gender_Code', 'Skill_Code'])

# 특성과 라벨 설정
X = numeric_df[['Year', 'Gender_Code', 'Skill_Code']]
y = numeric_df['Value'] > numeric_df['Value'].mean()

# 데이터가 충분한지 확인
if len(X) < 2:
    st.warning("📉 학습에 사용할 데이터가 부족합니다. 필터 조건을 변경하거나 데이터를 확인해주세요.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("📌 분류 보고서")
    st.text(classification_report(y_test, y_pred))


# ----------------------
# 4. 큐 & 스택 시뮬레이션
# ----------------------
st.subheader("자료구조 시뮬레이션: 큐와 스택")

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
# 5. 정렬 알고리즘 시각화
# ----------------------
st.subheader("정렬 알고리즘 시각화")

sort_data = st.text_input("정렬할 숫자 입력 (쉼표로 구분)", value="5,2,9,1,7")

if st.button("정렬 시작"):
    try:
        nums = [int(x) for x in sort_data.split(',') if x.strip().isdigit()]
        st.write("원본 배열:", nums)

        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]

        st.write("정렬된 배열:", nums)

        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(nums)), nums)
        ax2.set_title("정렬 결과 시각화")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"입력 오류: {e}")
