# 지역 간 전력 소비 분석 및 최적 경로 Streamlit 코드
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2

st.set_page_config(layout="wide")

# --------------------
# 1. 데이터 불러오기
# --------------------
@st.cache_data

def load_data():
    df_power = pd.read_csv("지역별_전력사용량_계약종별_정리본.csv")
    df_temp = pd.read_csv("통계청_SGIS_통계주제도_기상데이터_20240710.csv", encoding='cp949')
    df_hourly = pd.read_csv("한국전력거래소_시간별 전국 전력수요량_20241231.csv", encoding='cp949')
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

# --------------------
# 2. 샤논 엔트로피 분석
# --------------------
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

st.header("🔋 지역 간 전력 소비 샤논 엔트로피 분석")
region_entropy = df_power.groupby('시군구')['사용량'].apply(compute_entropy).reset_index()
region_entropy.columns = ['시군구', '샤논엔트로피']
st.dataframe(region_entropy)
fig = px.bar(region_entropy, x='시군구', y='샤논엔트로피', title="지역별 전력 소비 엔트로피")
st.plotly_chart(fig)

# --------------------
# 3. 기온 기반 전력 예측
# --------------------
st.header("🌡️ 기온 기반 전력 예측 모델")
merged = pd.merge(df_power, df_temp, left_on='시군구', right_on='시도명')
X = merged[['평균기온값']]
y = merged['사용량']
model = LinearRegression().fit(X, y)
pred = model.predict(X)
plt.figure(figsize=(6,4))
plt.scatter(X, y, label='실제값')
plt.plot(X, pred, color='red', label='예측')
plt.xlabel('평균기온')
plt.ylabel('전력 사용량')
plt.legend()
st.pyplot(plt)

# --------------------
# 4. 전력 불균형 점수 지도
# --------------------
st.header("🗺️ 전력 불균형 지도 시각화")
mean_usage = df_power.groupby('시군구')['사용량'].mean()
z_scores = (mean_usage - mean_usage.mean()) / mean_usage.std()
z_df = pd.DataFrame({'시군구': z_scores.index, '불균형점수': z_scores.values})
fig_map = px.bar(z_df, x='시군구', y='불균형점수', color='불균형점수', title='전력 불균형 점수')
st.plotly_chart(fig_map)

# --------------------
# 5. 시간대별 수요 분석
# --------------------
st.header("⏱️ 시간대별 전력 수요 패턴")
df_hourly_melted = df_hourly.melt(id_vars=['날짜'], var_name='시간', value_name='전력수요')
df_hourly_avg = df_hourly_melted.groupby('시간')['전력수요'].mean().reset_index()
fig_line = px.line(df_hourly_avg, x='시간', y='전력수요', title='시간대별 평균 전력 수요')
st.plotly_chart(fig_line)

# --------------------
# 6. 변분법 기반 최적 경로 예시
# --------------------
st.header("📐 변분법 기반 최적 경로 예시")
def ode_system(x, y):
    return np.vstack((y[1], -0.5 * y[0]))
def bc(ya, yb):
    return np.array([ya[0], yb[0] - 1])
x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
y[0] = x
sol = solve_bvp(ode_system, bc, x, y)
x_plot = np.linspace(0, 1, 100)
y_plot = sol.sol(x_plot)[0]
plt.figure(figsize=(6,4))
plt.plot(x_plot, y_plot, label='최적 경로')
plt.xlabel('거리')
plt.ylabel('전력 손실 최소화 경로')
plt.title("변분법 기반 배전 경로 최적화")
plt.legend()
st.pyplot(plt)

st.success("✅ 모든 요소 포함 완료!")
