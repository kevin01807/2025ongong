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

def clean_unicode(text):
    return ''.join(c for c in str(text) if c.isprintable())

st.set_page_config(page_title=clean_unicode("ICT 역량 분류 및 격차 분석"), layout="wide")

# 1. 데이터 로드
@st.cache_data
def load_data():
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "4-4-1.csv")
    st.write("📂 데이터 경로 확인:", file_path)
    df = pd.read_csv(file_path, encoding="utf-8")
    return df

df = load_data()

# 2. 시각화
st.header("기술 유형별 ICT 활용 격차")

if {"기술유형", "성별", "Year", "Value"}.issubset(df.columns):
    skill_list = df["기술유형"].dropna().unique()
    selected_skill = st.selectbox("기술을 선택하세요", skill_list)

    filtered = df[df["기술유형"] == selected_skill]
    filtered = filtered.dropna(subset=["Year", "Value", "성별"])

    if filtered.empty:
        st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            sns.barplot(data=filtered, x="Year", y="Value", hue="성별", ax=ax)
            ax.set_title(f"{selected_skill} 기술 활용도 (성별 비교)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"시각화 중 오류 발생: {e}")
else:
    st.error("필수 컬럼 누락 ('기술유형', '성별', 'Year', 'Value')")

# 3. 나이브 베이즈 분류기
st.subheader("나이브 베이즈 분류기를 활용한 예측")

df_model = df.dropna(subset=["Year", "Value", "성별", "기술유형"])
df_model['성별코드'] = df_model['성별'].astype('category').cat.codes
df_model['기술코드'] = df_model['기술유형'].astype('category').cat.codes

X = df_model[["Year", "성별코드", "기술코드"]]
y = df_model["Value"] > df_model["Value"].mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("📌 분류 결과")
st.text(classification_report(y_test, y_pred))

# 4. 큐 & 스택
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
    s_input = st.text_input("스택에 추가할 항목 입력", key="stack_input")
    if st.button("스택에 추가"):
        stack.append(s_input)
    if st.button("스택에서 제거"):
        if stack:
            stack.pop()
    st.write("현재 스택 상태:", stack)

# 5. 정렬 알고리즘 시각화
st.subheader("정렬 알고리즘 시각화")

sort_input = st.text_input("정렬할 숫자 입력 (쉼표로 구분)", value="5,2,9,1,7")

if st.button("정렬 시작"):
    try:
        nums = [int(x) for x in sort_input.split(",")]
        st.write("원본 배열:", nums)

        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]

        st.write("정렬된 배열:", nums)
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(nums)), nums)
        ax2.set_title("정렬 결과 시각화")
        st.pyplot(fig2)
    except:
        st.warning("숫자를 쉼표로 구분하여 정확히 입력해주세요.")
