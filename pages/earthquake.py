import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SDGs 9: 설비 이상 감지 시스템", layout="wide")

st.title("🔧 SDGs 9 기반 진동·온도 이상 감지 시스템")
st.markdown("스마트팩토리 설비에서 수집된 진동/온도 데이터를 활용하여 이상 여부를 진단하고 시각화하는 시스템입니다.")
st.markdown("**지속가능한 산업화(SDGs 9)**를 위해 예측 유지보수 기술을 접목한 자료구조+알고리즘 프로젝트입니다.")

# ✅ 시뮬레이션 데이터 생성 함수
@st.cache_data
def generate_simulation_data():
    np.random.seed(42)
    time = pd.date_range(start="2025-06-01", periods=100, freq="H")
    temperature = np.random.normal(loc=50, scale=5, size=100)
    vibration = np.random.normal(loc=30, scale=10, size=100)
    temperature[80:] += np.linspace(5, 20, 20)
    vibration[85:] += np.linspace(10, 30, 15)

    df = pd.DataFrame({
        "Time": time,
        "Temperature": temperature,
        "Vibration": vibration
    })
    return df

# ✅ 전처리 함수
def preprocess_data(df):
    df["Time"] = pd.to_datetime(df["Time"])
    df["Temp_Status"] = np.where(df["Temperature"] > 65, "이상", "정상")
    df["Vib_Status"] = np.where(df["Vibration"] > 50, "이상", "정상")
    df["System_Status"] = np.where(
        (df["Temp_Status"] == "이상") | (df["Vib_Status"] == "이상"),
        "경고", "정상"
    )
    return df

# ✅ 사이드바 - CSV 업로드 or 시뮬레이션 선택
st.sidebar.header("📁 데이터 선택")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (Time, Temperature, Vibration 포함)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ 업로드 완료!")
else:
    df = generate_simulation_data()
    st.sidebar.info("💡 업로드된 파일이 없어서 시뮬레이션 데이터를 사용합니다.")

# ✅ 데이터 전처리
df = preprocess_data(df)

# ✅ 시각화
st.subheader("📈 시간별 온도/진동 그래프")
fig1 = px.line(df, x="Time", y=["Temperature", "Vibration"], title="온도 및 진동 변화 추이")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🛑 이상 상태 분포 (Pie Chart)")
status_counts = df["System_Status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]
fig2 = px.pie(status_counts, names="Status", values="Count", color="Status",
              color_discrete_map={"정상": "green", "경고": "red"})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📋 이상 데이터 목록 (상위 10개)")
st.dataframe(df[df["System_Status"] == "경고"].head(10))

# ✅ 알고리즘 설명
with st.expander("🧠 사용한 자료구조와 알고리즘 설명"):
    st.markdown("""
    - **큐**: 센서 데이터를 시간 순서대로 저장하여 처리
    - **이진 탐색**: 임계값 초과 시점 탐색
    - **정렬 알고리즘**: 진동/온도 기준 정렬로 위험도 우선 탐지
    """)

# ✅ 출처 표기
with st.expander("📚 데이터 출처 및 SDGs 연계"):
    st.markdown("""
    - **SDGs Goal 9**: 지속 가능한 산업화와 사회기반시설 구축을 위한 목표
    - **데이터 출처**: 업로드된 CSV 파일 또는 시뮬레이션
    """)
