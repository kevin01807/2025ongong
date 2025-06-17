import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2
from collections import deque
import os

# --------------------
# 데이터 불러오기
# --------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

# --------------------
# 샤논 엔트로피 계산
# --------------------
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

st.title("📊 지역 간 전력 소비 분석 및 경로 최적화")
st.subheader("1️⃣ 샤논 엔트로피 기반 지역 전력 다양성 분석")

entropy_df = df_power.groupby("시군구")["사용량"].apply(compute_entropy).reset_index()
entropy_df.columns = ["시군구", "샤논엔트로피"]
st.dataframe(entropy_df)

fig_entropy = px.bar(entropy_df, x="시군구", y="샤논엔트로피", title="지역별 전력 사용 샤논 엔트로피")
st.plotly_chart(fig_entropy)

# --------------------
# 온도 기반 전력 예측 회귀
# --------------------
st.subheader("2️⃣ 온도 기반 전력 예측 회귀")

merged = pd.merge(df_power, df_temp, on="시군구")
X = merged[["평균기온"]]
y = merged["사용량"]

model = LinearRegression().fit(X, y)
pred = model.predict(X)

plt.figure()
plt.scatter(X, y, label="실제값")
plt.plot(X, pred, color="red", label="예측값")
plt.xlabel("평균기온")
plt.ylabel("전력 사용량")
plt.legend()
st.pyplot(plt)

# --------------------
# z-score 지도 시각화
# --------------------
st.subheader("3️⃣ 전력 사용량 불균형 지도")

mean_usage = df_power.groupby("시군구")["사용량"].mean()
z_scores = (mean_usage - mean_usage.mean()) / mean_usage.std()
z_df = pd.DataFrame({"시군구": z_scores.index, "불균형점수": z_scores.values})

fig_map = px.bar(z_df, x="시군구", y="불균형점수", title="지역별 전력 사용 불균형 (z-score)")
st.plotly_chart(fig_map)

# --------------------
# 변분법 기반 경로 최적화
# --------------------
st.subheader("4️⃣ 변분법 기반 최적 경로 계산 예시")

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

plt.figure()
plt.plot(x_plot, y_plot, label="최적 경로")
plt.xlabel("거리")
plt.ylabel("전압/손실량")
plt.title("변분법 최적 경로 예시")
plt.legend()
st.pyplot(plt)

# --------------------
# 스택, 큐, 정렬, 탐색 시뮬레이션
# --------------------
st.subheader("5️⃣ 자료구조 기반 분석 시뮬레이션")

# 큐: 시간 순 대기열
power_queue = deque(df_hourly["계통한계예비력(MW)"][:10])
st.write("📦 전력 수요 예비력 대기열 (큐)")
st.write(list(power_queue))

# 스택: 마지막 5개 시간대 위험지역 기록
top_regions = df_power.sort_values("사용량", ascending=False)["시군구"].unique()[:5]
region_stack = list(top_regions)
st.write("🗂️ 최근 고위험 지역 스택")
st.write(region_stack)

# 정렬: 평균기온 기준 정렬
sorted_temp = df_temp.sort_values("평균기온", ascending=False)
st.write("🌡️ 평균기온 내림차순 정렬")
st.dataframe(sorted_temp[["시군구", "평균기온"]])

# 이진 탐색: 특정 온도 이상 지역 찾기
def binary_search_region(df, temp_threshold):
    df_sorted = df.sort_values("평균기온").reset_index()
    left, right = 0, len(df_sorted) - 1
    result = []
    while left <= right:
        mid = (left + right) // 2
        if df_sorted.loc[mid, "평균기온"] >= temp_threshold:
            result.append(df_sorted.loc[mid, "시군구"])
            right = mid - 1
        else:
            left = mid + 1
    return result

search_temp = st.slider("🔍 온도 이상 지역 찾기 (이진 탐색)", min_value=-5, max_value=35, value=25)
found_regions = binary_search_region(df_temp, search_temp)
st.write(f"🌍 {search_temp}℃ 이상 지역:", found_regions)
