import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2

# --------------------
# 1. 데이터 불러오기
# --------------------
@st.cache_data
def load_data():
    df_power = pd.read_csv("지역별_전력사용량_계약종별_정리본.csv")
    df_temp = pd.read_csv("통계청_SGIS_통계주제도_기상데이터_20240710.csv")
    df_hourly = pd.read_csv("한국전력거래소_시간별 전국 전력수요량_20241231.csv")
    df_access = pd.read_csv("7-1-1.csv")
    return df_power, df_temp, df_hourly, df_access

df_power, df_temp, df_hourly, df_access = load_data()

# --------------------
# 2. 샤논 엔트로피 계산
# --------------------
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

st.title("📊 지역 간 전력 소비 분석 및 배전 경로 최적화")

st.header("🔹 샤논 엔트로피 기반 지역별 전력 소비 다양도 분석")
region_entropy = df_power.groupby('시군구')['사용량'].apply(compute_entropy).reset_index()
region_entropy.columns = ['시군구', '샤논엔트로피']
st.dataframe(region_entropy)
fig = px.bar(region_entropy, x='시군구', y='샤논엔트로피', title="지역별 샤논 엔트로피")
st.plotly_chart(fig)

# --------------------
# 3. 온도 기반 회귀 예측
# --------------------
st.header("🌡️ 평균기온 기반 전력 사용량 예측")
merged = pd.merge(df_power, df_temp, on='시군구')
X = merged[['평균기온']]
y = merged['사용량']
model = LinearRegression().fit(X, y)
pred = model.predict(X)
plt.figure(figsize=(6,4))
plt.scatter(X, y, label='실제값')
plt.plot(X, pred, color='red', label='회귀 예측')
plt.xlabel('평균기온')
plt.ylabel('전력 사용량')
plt.legend()
st.pyplot(plt)

# --------------------
# 4. 전력 불균형 지도
# --------------------
st.header("🗺️ 지역 전력 불균형 지도 시각화")
mean_usage = df_power.groupby('시군구')['사용량'].mean()
std_usage = df_power.groupby('시군구')['사용량'].std()
z_scores = (mean_usage - mean_usage.mean()) / mean_usage.std()
z_df = pd.DataFrame({'시군구': z_scores.index, '불균형점수': z_scores.values})
fig_map = px.bar(z_df, x='시군구', y='불균형점수', color='불균형점수', title="지역 간 전력 불균형 (z-score)")
st.plotly_chart(fig_map)

# --------------------
# 5. 변분법 최적 경로 예시
# --------------------
st.header("📈 변분법 기반 배전 경로 최적화 예시")
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
plt.ylabel('전압 또는 손실')
plt.legend()
st.pyplot(plt)

# --------------------
# 6. 전력 접근성 비교 (SDGs 7.1.1)
# --------------------
st.header("🔌 SDG 7.1.1: 도시 vs 농촌 전력import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2

# --------------------
# 1. 데이터 불러오기
# --------------------
@st.cache_data
def load_data():
    df_power = pd.read_csv("지역별_전력사용량_계약종별_정리본.csv")
    df_temp = pd.read_csv("통계청_SGIS_통계주제도_기상데이터_20240710.csv")
    df_hourly = pd.read_csv("한국전력거래소_시간별 전국 전력수요량_20241231.csv")
    df_usage = pd.read_csv("전력+사용량(용도별)_20250618005305.csv")
    df_sdg = pd.read_csv("7-1-1.csv")
    return df_power, df_temp, df_hourly, df_usage, df_sdg

df_power, df_temp, df_hourly, df_usage, df_sdg = load_data()

st.title("지역 간 전력 소비 분석 및 배전 경로 최적화")

# --------------------
# 2. 샤논 엔트로피 계산
# --------------------
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

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

plt.figure(figsize=(6,4))
plt.scatter(X, y, label='실제값')
plt.plot(X, pred, color='red', label='예측값')
plt.xlabel('평균기온')
plt.ylabel('전력 사용량')
plt.legend()
st.pyplot(plt)

# --------------------
# 4. 전력 불균형 점수 시각화 (지도)
# --------------------
st.header("🗺️ 지역별 전력 불균형 점수 지도 시각화")
mean_usage = df_power.groupby('시군구')['사용량'].mean()
std_usage = df_power.groupby('시군구')['사용량'].std()
z_scores = (mean_usage - mean_usage.mean()) / mean_usage.std()
z_df = pd.DataFrame({'시군구': z_scores.index, '불균형점수': z_scores.values})
fig_map = px.choropleth(z_df, locations='시군구', locationmode='geojson-id',
                        color='불균형점수',
                        color_continuous_scale='YlOrRd',
                        title="지역 간 전력 불균형 지도 (z-score)")
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

plt.figure(figsize=(6,4))
plt.plot(x_plot, y_plot, label='최적 경로(변분법)')
plt.title("변분법 기반 최적 경로 예시")
plt.xlabel("거리")
plt.ylabel("전압/에너지/손실 등")
plt.legend()
st.pyplot(plt)

# --------------------
# 6. SDG 7.1.1 지표 비교
# --------------------
st.header("📊 SDG 7.1.1 지표: 전력 접근성 비교 분석")
st.dataframe(df_sdg)
fig_sdg = px.bar(df_sdg, x='지역', y='전력접근성지표', color='도시/농촌',
                title="도시와 농촌 간 전력 접근성 비교")
st.plotly_chart(fig_sdg)
