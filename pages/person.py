import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from collections import deque
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2

# -----------------------------
# 1. 데이터 로드 (정확한 경로)
# -----------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

st.title("전력 소비 분석 및 자료구조 알고리즘 시각화")
st.header("🔋 1. 샤논 엔트로피 기반 지역 전력소비 다양성 분석")

def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

entropy_df = df_power.groupby("시군구")["사용량"].apply(compute_entropy).reset_index()
entropy_df.columns = ["시군구", "샤논엔트로피"]
st.dataframe(entropy_df)

fig1 = px.bar(entropy_df, x="시군구", y="샤논엔트로피", title="지역별 샤논 엔트로피")
st.plotly_chart(fig1)

# ------------------------
# 2. 회귀분석 (탐색 적용)
# ------------------------
st.header("🌡️ 2. 온도 기반 전력 예측 회귀모델")

merged = pd.merge(df_power, df_temp, on="시군구")
X = merged[["평균기온"]]
y = merged["사용량"]

model = LinearRegression().fit(X, y)
pred = model.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(X, y, label="실제값")
plt.plot(X, pred, color="red", label="예측값")
plt.xlabel("평균기온")
plt.ylabel("전력사용량")
plt.legend()
st.pyplot(plt)

# ------------------------------
# 3. 큐(Stack)/스택(Queue) 적용
# ------------------------------
st.header("📦 3. 큐/스택 기반 전력 소비 분석")

st.subheader("Queue (선입선출): 최근 10시간 전력 소비")
power_queue = deque(df_hourly["소비량"].values[:10])
st.write(list(power_queue))

st.subheader("Stack (후입선출): 마지막 5시간 전력 소비")
power_stack = list(df_hourly["소비량"].values[-5:])
st.write(power_stack[::-1])

# -------------------------------
# 4. 전력 불균형 z-score 시각화
# -------------------------------
st.header("🗺️ 4. 지역 간 전력 불균형 시각화")

mean_usage = df_power.groupby('시군구')['사용량'].mean()
z_scores = (mean_usage - mean_usage.mean()) / mean_usage.std()
z_df = pd.DataFrame({'시군구': z_scores.index, '불균형점수': z_scores.values})
fig2 = px.bar(z_df, x='시군구', y='불균형점수', title="전력 불균형 Z-Score")
st.plotly_chart(fig2)

# -------------------------
# 5. 정렬 알고리즘 적용
# -------------------------
st.header("📊 5. 버블 정렬로 소비량 정렬 (예시)")

def bubble_sort(arr):
    n = len(arr)
    arr = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j][1] > arr[j + 1][1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

top_usage = df_power.groupby("시군구")["사용량"].mean().reset_index()
sorted_data = bubble_sort(list(top_usage.values))
sorted_df = pd.DataFrame(sorted_data, columns=["시군구", "사용량"])
st.dataframe(sorted_df)

# ------------------------
# 6. 이진 탐색 알고리즘
# ------------------------
st.header("🔍 6. 이진 탐색 (특정 사용량 찾기)")

sorted_vals = sorted_df["사용량"].values
target = st.number_input("🔢 탐색할 사용량 입력", min_value=0.0)

def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if abs(arr[mid] - x) < 1e-3:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

if st.button("탐색 실행"):
    result = binary_search(sorted_vals, target)
    if result != -1:
        st.success(f"탐색 결과: 위치 {result}, 사용량 = {sorted_vals[result]}")
    else:
        st.warning("해당 사용량은 존재하지 않음")

# ------------------------
# 7. 트리 기반 전력 분류 예시
# ------------------------
st.header("🌲 7. 간단한 트리 구조 기반 분류 (임계점 기준)")

threshold = st.slider("임계 사용량 설정", min_value=0, max_value=200000, value=80000)
df_power["분류"] = df_power["사용량"].apply(lambda x: "과소비" if x > threshold else "정상")
st.write(df_power[["시군구", "사용량", "분류"]].head())

# ------------------------
# 8. 변분법 기반 경로 최적화
# ------------------------
st.header("🧮 8. 변분법 기반 최적 배전 경로 (예시)")

from scipy.integrate import solve_bvp

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
plt.title("변분법 기반 최적 경로")
plt.xlabel("거리")
plt.ylabel("전력/손실")
plt.legend()
st.pyplot(plt)
