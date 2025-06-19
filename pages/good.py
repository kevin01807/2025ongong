import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import deque
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="ICT 역량 분류 및 격차 분석", layout="wide")

# 데이터 불러오기
@st.cache_data
def load_data():
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "4-4-1.csv")
    st.write("📂 데이터 경로 확인:", file_path)
    df = pd.read_csv(file_path, encoding="utf-8")
    return df

df = load_data()

# -------------------------------
# 1. 시각화
# -------------------------------
st.header("기술 유형별 ICT 활용 격차")
selected_skill = st.selectbox("기술을 선택하세요", df['기술유형'].unique())
filtered = df[df['기술유형'] == selected_skill]

if filtered.empty:
    st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered, x='Year', y='Value', hue='성별', ax=ax)
        ax.set_title(f"{selected_skill} 기술 활용도 (성별 비교)")
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"시각화 중 오류 발생: {e}")

# -------------------------------
# 2. 나이브 베이즈 분류기
# -------------------------------
st.subheader("나이브 베이즈 분류기를 활용한 예측")

try:
    df_nb = df[['Year', 'Value', '성별', '기술유형']].dropna()
    df_nb['Gender_Code'] = df_nb['성별'].map({'남자': 0, '여자': 1, '전체': 2})
    df_nb['Skill_Code'] = df_nb['기술유형'].astype('category').cat.codes

    X = df_nb[['Year', 'Gender_Code', 'Skill_Code']]
    y = df_nb['Value'] > df_nb['Value'].mean()

    if len(X) < 2:
        st.warning("📉 학습에 사용할 데이터가 부족합니다. 필터 조건을 변경하거나 데이터를 확인해주세요.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.text("📌 분류 보고서")
        st.text(classification_report(y_test, y_pred))
except Exception as e:
    st.error(f"나이브 베이즈 실행 중 오류 발생: {e}")

# -------------------------------
# 3. 큐 & 스택 시뮬레이션
# -------------------------------
st.subheader("자료구조 시뮬레이션: 큐와 스택")

tab1, tab2 = st.tabs(["📥 큐 (Queue) - ICT 교육 대기열", "📦 스택 (Stack) - 기술 지원 우선순위"])

with tab1:
    st.write("ICT 교육 프로그램 참여자 대기열을 시뮬레이션한 큐 구조입니다.")
    queue = deque()
    q_input = st.text_input("대기열에 추가할 이름")
    if st.button("큐에 추가"):
        queue.append(q_input)
    if st.button("큐에서 제거"):
        if queue:
            queue.popleft()
    st.write("현재 대기열 상태:", list(queue))

with tab2:
    st.write("긴급 ICT 기술 지원 요청을 스택으로 관리합니다.")
    stack = []
    s_input = st.text_input("긴급 요청 입력")
    if st.button("스택에 추가"):
        stack.append(s_input)
    if st.button("스택에서 제거"):
        if stack:
            stack.pop()
    st.write("현재 요청 스택 상태:", stack)

# -------------------------------
# 4. 정렬 알고리즘 시각화
# -------------------------------
st.subheader("정렬 알고리즘 시각화: ICT 역량 점수 정렬")

sort_data = st.text_input("ICT 역량 점수 입력 (예: 82,95,70)", value="82,95,70")
if st.button("정렬 시작"):
    try:
        nums = [int(x) for x in sort_data.split(',')]
        st.write("원본 점수:", nums)
        # 버블 정렬
        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        st.write("정렬된 점수:", nums)
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(nums)), nums)
        ax2.set_title("정렬된 ICT 점수 시각화")
        st.pyplot(fig2)
    except:
        st.warning("숫자를 올바르게 입력하세요. (쉼표로 구분)")
