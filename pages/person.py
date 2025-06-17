
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
    df_power = pd.read_csv("power_by_region.csv")
    df_temp = pd.read_csv("통계청_SGIS_통계주제도_기상데이터_20240710.csv")
    df_hourly = pd.read_csv("hourly_power.csv")
    df_sdg711 = pd.read_csv("7-1-1.csv")
    return df_power, df_temp, df_hourly, df_sdg711

df_power, df_temp, df_hourly, df_sdg711 = load_data()

# --------------------
# 2. 샤논 엔트로피 계산
# --------------------
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

st.title("지역 간 전력 소비 분석 및 배전 경로 최적화")
st.header("🔋 전력 사용량 기반 샤논 엔트로피 분석")

region_entropy = df_power.groupby('시군구')['사용량'].apply(compute_entropy).reset_index()
region_entropy.columns = ['시군구', '샤논엔트로피']
st.dataframe(region_entropy)

fig = px.bar(region_entropy, x='시군구', y='샤논엔트로피', title="지역별 샤논 엔트로피")
st.plotly_chart(fig)

# --------------------
# 3. 온도 기반 전력 예측 회귀
# --------------------
st.header("🌡️ 온도 기반 전력 예측 회귀모델")

merged = pd.merge(df_power, df_temp, on='시군구')
X = merged[['평균기온']]
y = merged['사용량']

model = LinearRegression().fit(X, y)
pred = model.predict(X)

fig2, ax = plt.subplots()
ax.scatter(X, y, label='실제값')
ax.plot(X, pred, color='red', label='예측값')
ax.set_xlabel('평균기온')
ax.set_ylabel('전력 사용량')
ax.legend()
st.pyplot(fig2)

# --------------------
# 4. 전력 불균형 점수 시각화 (지도)
# --------------------
st.header("🗺️ 지역별 전력 불균형 점수 지도 시각화")

mean_usage = df_power.groupby('시군구')['사용량'].mean()
z_scores = (mean_usage - mean_usage.mean()) / mean_usage.std()

z_df = pd.DataFrame({'시군구': z_scores.index, '불균형점수': z_scores.values})
fig_map = px.bar(z_df, x='시군구', y='불균형점수', title="지역 간 전력 불균형 점수 (z-score)")
st.plotly_chart(fig_map)

# --------------------
# 5. 변분법 기반 경로 최적화 예제
# --------------------
st.header("📈 변분법 기반 배전 경로 최적화 (예시)")

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

fig3, ax3 = plt.subplots()
ax3.plot(x_plot, y_plot, label='최적 경로(변분법)')
ax3.set_title("변분법 기반 최적 경로 예시")
ax3.set_xlabel("거리")
ax3.set_ylabel("전압/에너지/손실 등")
ax3.legend()
st.pyplot(fig3)

# --------------------
# 6. SDGs 7.1.1 지표 비교 시각화
# --------------------
st.header("🌍 SDGs 7.1.1: 전력 접근성 국가 비교")

st.dataframe(df_sdg711.head())

fig4 = px.bar(df_sdg711.sort_values('Value', ascending=False),
              x='Country', y='Value', title='SDGs 7.1.1 국가별 전력 접근 비율 (%)')
st.plotly_chart(fig4)
