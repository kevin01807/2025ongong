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

# -------------------
# 데이터 불러오기 함수
# -------------------
@st.cache_data
def load_data():
    base_dir = os.getcwd()
    main_file = os.path.join(base_dir, "data", "4-4-1.csv")
    queue_file = os.path.join(base_dir, "data", "queue_data.csv")
    stack_file = os.path.join(base_dir, "data", "stack_data.csv")

    df = pd.read_csv(main_file, encoding="utf-8")
    queue_df = pd.read_csv(queue_file, encoding="utf-8")
    stack_df = pd.read_csv(stack_file, encoding="utf-8")

    return df, queue_df, stack_df

df, queue_df, stack_df = load_data()

# -------------------
# 1. ICT 기술 시각화
# -------------------
st.title(clean_unicode("ICT 역량 분류 및 격차 분석"))
st.header(clean_unicode("기술 유형별 ICT 활용 격차"))

if '기술유형' in df.columns and '성별' in df.columns and 'Year' in df.columns and 'Value' in df.columns:
    df['기술유형'] = df['기술유형'].astype(str)
    selected_skill = st.selectbox("기술을 선택하세요", df['기술유형'].unique())
    filtered = df[df['기술유형'] == selected_skill]

    if not filtered.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered, x='Year', y='Value', hue='성별', ax=ax)
        ax.set_title(clean_unicode(f"{selected_skill} 기술 활용도 (성별 비교)"))
        st.pyplot(fig)
    else:
        st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    st.error("데이터셋에 필요한 컬럼이 없습니다.")

# 3. 나이브 베이즈 분류기
# ----------------------
st.subheader("나이브 베이즈 분류기를 활용한 예측")
try:
    numeric_df = df[['Year', 'Value']].copy()
    numeric_df['Gender'] = df['Gender']
    numeric_df['Skill'] = df['Skill_KR']

    numeric_df['Gender_Code'] = numeric_df['Gender'].map({'남자': 0, '여자': 1, '전체': 2})
    numeric_df['Skill_Code'] = numeric_df['Skill'].astype('category').cat.codes
    numeric_df.dropna(inplace=True)

    X = numeric_df[['Year', 'Gender_Code', 'Skill_Code']]
    y = numeric_df['Value'] > numeric_df['Value'].mean()

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("데이터가 부족합니다. 다른 기술을 선택해보세요.")
except Exception as e:
    st.error(f"나이브 베이즈 실행 중 오류 발생: {e}")
# -------------------
# 3. 큐/스택 시뮬레이션
# -------------------
st.subheader("자료구조 시뮬레이션: 큐와 스택")
st.markdown("#### 📥 ICT 요청 처리 구조: Queue(선착순) vs Stack(긴급처리)")

tab1, tab2 = st.tabs(["📥 큐 (Queue)", "📦 스택 (Stack)"])

with tab1:
    st.dataframe(queue_df)
    st.write("큐 시뮬레이션: 기술 요청이 먼저 도착한 순서대로 처리됩니다.")

with tab2:
    st.dataframe(stack_df)
    st.write("스택 시뮬레이션: 가장 최근 요청이 우선 처리됩니다.")

# -------------------
# 4. 정렬 알고리즘 시각화
# -------------------
st.subheader("정렬 알고리즘 시각화")
st.markdown("#### 🔢 ICT 기술 우선순위 정렬")

sort_data = st.text_input("정렬할 숫자 입력 (쉼표로 구분)", value="5,2,9,1,7")
if st.button("정렬 시작"):
    try:
        nums = [int(x.strip()) for x in sort_data.split(',') if x.strip().isdigit()]
        st.write("원본 배열:", nums)

        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]

        st.write("정렬된 배열:", nums)
        fig, ax = plt.subplots()
        ax.bar(range(len(nums)), nums)
        ax.set_title("정렬 결과 시각화")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"입력 오류: {e}")
