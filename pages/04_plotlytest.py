import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="당뇨병 예측 시스템", layout="wide")
st.title("🩺 당뇨병 예측 Streamlit 프로그램")

# CSV 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_data_upload.csv")
    df["class"] = df["class"].map({"Positive": 1, "Negative": 0})
    binary_cols = df.columns.drop(["Age", "Gender", "class"])
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    return df

df = load_data()

# 📊 데이터 탐색 시각화
st.subheader("📈 나이대별 당뇨병 분포")
fig = px.histogram(df, x="Age", color="class", barmode="group",
                   color_discrete_map={1: "red", 0: "blue"},
                   labels={"class": "당뇨병 여부"})
st.plotly_chart(fig, use_container_width=True)

# 🔍 학습용 분류 모델 준비
X = df.drop(columns=["class"])
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.success(f"🎯 분류 모델 정확도: {acc * 100:.2f}%")

# 🧪 사용자 입력 기반 예측
st.subheader("🧪 내 증상으로 당뇨병 예측해보기")

with st.form("predict_form"):
    age = st.slider("나이", 10, 100, 45)
    gender = st.radio("성별", ["남성", "여성"])
    input_data = {
        "Age": age,
        "Gender": 1 if gender == "남성" else 0
    }
    for col in X.columns:
        if col not in ["Age", "Gender"]:
            input_data[col] = st.radio(f"{col}", ["아님", "있음"]) == "있음"
    submitted = st.form_submit_button("예측하기")

if submitted:
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]
    if prediction == 1:
        st.error(f"⚠️ 당뇨병 위험 있음 (예측 확률 {prob*100:.2f}%)")
    else:
        st.success(f"✅ 당뇨병 위험 낮음 (예측 확률 {prob*100:.2f}%)")
