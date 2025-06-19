import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import deque

plt.rcParams['font.family'] = 'Malgun Gothic'

def clean_unicode(text):
    return ''.join(c for c in str(text) if c.isprintable())

st.set_page_config(page_title="ICT 역량 분류 및 격차 분석", layout="wide")

@st.cache_data
def load_main_data():
    df = pd.read_csv("data/4-4-1.csv")
    df.rename(columns={'기술유형': 'Skill_Type', '성별': 'Gender'}, inplace=True)
    skill_map = {
        'ARSP': '문서 편집', 'EMAIL': '이메일 사용', 'COPY': '파일 복사',
        'SEND': '파일 전송', 'INST': 'SW 설치', 'COMM': '온라인 커뮤니케이션',
        'BUY': '온라인 구매', 'BANK': '온라인 뱅킹', 'USEC': '보안 설정'
    }
    df['Skill_KR'] = df['Skill_Type'].map(skill_map)
    df['Gender'] = df['Gender'].fillna('전체')
    return df.dropna(subset=['Year', 'Value'])

df = load_main_data()

# 1. 기술 활용 시각화
st.header("기술 유형별 ICT 활용 격차")
selected_skill = st.selectbox("기술을 선택하세요", df['Skill_KR'].dropna().unique())
filtered = df[df['Skill_KR'] == selected_skill]

if filtered.empty:
    st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=filtered, x='Year', y='Value', hue='Gender', ax=ax)
    ax.set_title(f"{selected_skill} 기술 활용도 (성별 비교)")
    st.pyplot(fig)

# 2. 나이브 베이즈 분류기
st.subheader("나이브 베이즈 분류기를 활용한 예측")
try:
    model_df = df[['Year', 'Value', 'Gender', 'Skill_KR']].copy()
    model_df['성별코드'] = model_df['Gender'].map({'남자': 0, '여자': 1, '전체': 2})
    model_df['기술코드'] = model_df['Skill_KR'].astype('category').cat.codes
    model_df.dropna(inplace=True)

    X = model_df[['Year', '성별코드', '기술코드']]
    y = model_df['Value'] > model_df['Value'].mean()

    if len(X) == 0:
        st.error("📉 데이터가 부족합니다. 다른 조건을 선택하거나 전체 데이터를 확인하세요.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.text("📌 분류 보고서")
        st.text(clean_unicode(classification_report(y_test, y_pred)))
except Exception as e:
    st.error(f"나이브 베이즈 실행 중 오류 발생: {e}")

# 3. 큐 & 스택 시뮬레이션
st.subheader("자료구조 시뮬레이션: 큐와 스택")

queue_tab, stack_tab = st.tabs(["📥 큐 (Queue)", "📦 스택 (Stack)"])

with queue_tab:
    try:
        queue_df = pd.read_csv("data/queue_data.csv")
        queue_items = queue_df['Item'].tolist()
        queue = deque(queue_items)
        st.write("초기 큐 상태:", list(queue))
        if st.button("큐에서 제거 (Dequeue)"):
            if queue:
                queue.popleft()
        st.write("현재 큐 상태:", list(queue))
    except Exception as e:
        st.error(f"큐 처리 중 오류 발생: {e}")

with stack_tab:
    try:
        stack_df = pd.read_csv("data/stack_data.csv")
        stack = stack_df['Item'].tolist()
        st.write("초기 스택 상태:", stack)
        if st.button("스택에서 제거 (Pop)"):
            if stack:
                stack.pop()
        st.write("현재 스택 상태:", stack)
    except Exception as e:
        st.error(f"스택 처리 중 오류 발생: {e}")

# 4. 정렬 알고리즘 시각화
st.subheader("정렬 알고리즘 시각화")

sort_input = st.text_input("정렬할 숫자 입력 (쉼표로 구분)", value="5,2,9,1,7")
if st.button("정렬 시작"):
    try:
        nums = [int(x) for x in sort_input.split(',')]
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
        st.warning("숫자를 올바르게 입력해주세요!")
