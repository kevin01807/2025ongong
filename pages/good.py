import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Set base directory for data files
base_dir = "/mnt/data"

# ---------- 1. ICT 활용 격차 시각화 ----------
st.title("ICT 역량 분류 및 격차 분석")

ict_file = os.path.join(base_dir, "4-4-1.csv")
st.text(f"데이터 경로 확인: {ict_file}")

df = pd.read_csv(ict_file)

st.header("기술 유형별 ICT 활용 격차")
skills = df["기술유형"].unique()
selected_skill = st.selectbox("기술을 선택하세요", skills)

filtered = df[df["기술유형"] == selected_skill]

if filtered.empty:
    st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    fig, ax = plt.subplots()
    try:
        sns.barplot(data=filtered, x="Year", y="Value", hue="성별", ax=ax)
        ax.set_title(f"{selected_skill} 기술 활용도 (성별 비교)")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"시각화 중 오류 발생: {e}")

# ---------- 2. 나이브 베이즈 분류 ----------
st.header("ICT 기술 활용도 예측 (나이브 베이즈)")

try:
    features = df[["Year", "기술유형", "성별"]]
    target = df["Value"]

    # 인코딩
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    features["기술유형"] = le1.fit_transform(features["기술유형"])
    features["성별"] = le2.fit_transform(features["성별"])

    # 결측치 처리
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(features)
    y = target

    # 분류기 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    if X_train.shape[0] == 0:
        st.warning("학습에 사용할 데이터가 부족합니다. 데이터를 확인하세요.")
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.success(f"모델 정확도: {score:.2f}")
except Exception as e:
    st.error(f"나이브 베이즈 실행 중 오류 발생: {e}")

# ---------- 3. 큐 처리 ----------
st.header("ICT 업무 우선순위 큐 정렬")

try:
    queue_file = os.path.join(base_dir, "queue_data.csv")
    queue_df = pd.read_csv(queue_file)
    queue_sorted = queue_df.sort_values(by="Priority")
    st.subheader("우선순위 큐 (낮은 숫자일수록 우선순위 높음)")
    st.dataframe(queue_sorted)
except Exception as e:
    st.error(f"큐 처리 중 오류 발생: {e}")

# ---------- 4. 스택 처리 ----------
st.header("ICT 작업 중요도 스택 정렬")

try:
    stack_file = os.path.join(base_dir, "stack_data.csv")
    stack_df = pd.read_csv(stack_file)
    stack_sorted = stack_df.sort_values(by="Importance", ascending=False)
    st.subheader("중요도 스택 (높은 숫자일수록 더 먼저)")
    st.dataframe(stack_sorted)
except Exception as e:
    st.error(f"스택 처리 중 오류 발생: {e}")
