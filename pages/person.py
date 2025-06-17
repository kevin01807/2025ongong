import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2
from collections import deque

# --------------------
# 1. 데이터 불러오기
# --------------------
@st.cache_data

def load_data():
    df_power = pd.read_csv("power_by_region.csv")
    df_temp = pd.read_csv("temperature_by_region.csv")
    df_hourly = pd.read_csv("hourly_power.csv")
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

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

plt.figure(figsize=(6,4))
plt.plot(x_plot, y_plot, label='최적 경로(변분법)')
plt.title("변분법 기반 최적 경로 예시")
plt.xlabel("거리")
plt.ylabel("전압/에너지/손실 등")
plt.legend()
st.pyplot(plt)

# --------------------
# 6. 자료구조 (큐, 스택) 및 탐색, 정렬 예시
# --------------------
st.header("🧠 자료구조 · 알고리즘 적용 예시")

# 큐 (Queue): 전력 수요 대기열
power_queue = deque(df_hourly.iloc[0, 1:11])  # 1시~10시 수요량 큐
st.write("**전력 수요 큐 (1~10시):**", list(power_queue))

# 스택 (Stack): 고부하 시간 히스토리
high_load_stack = []
threshold = df_hourly.iloc[0, 1:25].mean()
for hour, value in enumerate(df_hourly.iloc[0, 1:25], start=1):
    if value > threshold:
        high_load_stack.append((hour, value))
st.write("**고부하 시간 스택:**", high_load_stack[::-1])  # 후입선출 출력

# 탐색: 특정 시간대 전력 수요 이진 탐색
from bisect import bisect_left
sorted_usage = sorted(df_hourly.iloc[0, 1:25])
target = 60000
pos = bisect_left(sorted_usage, target)
st.write(f"**60000 이상 수요 첫 위치 (정렬 후):** {pos}, 값: {sorted_usage[pos] if pos < len(sorted_usage) else '없음'}")

# 정렬: 전력 수요 정렬 결과
sorted_df = df_hourly.iloc[0, 1:25].sort_values(ascending=False)
st.write("**정렬된 시간대별 전력 수요:**")
st.dataframe(sorted_df)

# 트리 구조 예시: 지역별 → 월별 사용량 계층 출력
st.write("**지역별 트리 구조 예시**")
tree_data = df_power.groupby(['시군구', '월'])['사용량'].sum().unstack().fillna(0)
st.dataframe(tree_data)
