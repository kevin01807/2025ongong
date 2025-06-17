import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import entropy
from scipy.integrate import solve_bvp

# 1. 데이터 업로드 (실제 파일 로딩은 생략, 예시로 설명)
@st.cache_data
def load_data():
    df_power = pd.read_csv("지역별_전력사용량_계약종별_정리본.csv")
    df_temp = pd.read_csv("통계청_SGIS_통계주제도_기상데이터_20240710.csv")
    df_hourly = pd.read_csv("한국전력거래소_시간별 전국 전력수요량_20241231.csv")
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

st.title("지역 간 전력 소비 최적화 분석 시스템")
st.markdown("### SDGs 7.1 & 9.4 기반: 전력 사용 예측, 엔트로피 분석, 경로 최적화")

# 2. 샤논 엔트로피 계산 함수
st.header("1. 샤논 엔트로피 기반 전력 소비 불확실성")
selected_region = st.selectbox("지역 선택", df_power["지역"].unique())
data = df_power[df_power["지역"] == selected_region].iloc[:, 2:].values.flatten()
normalized_data = data / np.sum(data)
region_entropy = entropy(normalized_data)
st.metric("샤논 엔트로피 (불확실성)", f"{region_entropy:.4f}")

# 3. 기온 기반 전력 예측 (단순 선형 회귀)
st.header("2. 기온 기반 전력 소비 예측 모델")
df_temp_power = pd.merge(df_temp, df_power, on="지역")
X = df_temp_power["평균기온(℃)"]
y = df_temp_power["전력사용량"]
a, b = np.polyfit(X, y, deg=1)
fig1 = px.scatter(df_temp_power, x=X, y=y, trendline="ols", labels={"x":"기온(℃)", "y":"전력사용량"})
st.plotly_chart(fig1)
st.latex(f"전력 사용량 예측: y = {a:.2f}x + {b:.2f}")

# 4. 지도 시각화 (불균형 점수)
st.header("3. 지역 간 전력 소비 불균형 지도")
df_power_sum = df_power.groupby("지역")["전력사용량"].sum().reset_index()
score = (df_power_sum["전력사용량"] - df_power_sum["전력사용량"].mean()) / df_power_sum["전력사용량"].std()
df_power_sum["불균형 점수"] = score
fig2 = px.choropleth(df_power_sum, geojson="https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/municipalities.json", 
                     locations="지역", color="불균형 점수", featureidkey="properties.name",
                     color_continuous_scale="Viridis", title="전력 불균형 점수 지도")
st.plotly_chart(fig2)

# 5. 변분법 시뮬레이션
st.header("4. 변분법 기반 최단 경로 시뮬레이션")
def ode_system(x, y):
    return np.vstack((y[1], np.zeros_like(x)))

def boundary_conditions(ya, yb):
    return np.array([ya[0], yb[0] - 1])

x_vals = np.linspace(0, 1, 10)
y_init = np.zeros((2, x_vals.size))
sol = solve_bvp(ode_system, boundary_conditions, x_vals, y_init)
fig3, ax3 = plt.subplots()
ax3.plot(sol.x, sol.y[0], label='최적 경로')
ax3.set_title("변분법 기반 에너지 최소 경로")
ax3.set_xlabel("거리")
ax3.set_ylabel("고도")
ax3.legend()
st.pyplot(fig3)

st.success("✔️ 모든 분석이 완료되었습니다. SDGs 기반 전력 정책 수립에 활용 가능합니다.")
