# 📦 라이브러리 불러오기
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2
from collections import deque
import heapq
import os

# 📂 데이터 불러오기
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

st.title("⚡ 전력 사용 분석 + 자료구조 알고리즘 융합 프로젝트")

# 📊 선형 자료구조 - 큐: 전력 수요 대기열 시뮬레이션
st.header("📌 큐 자료구조: 시간 순 전력 수요 대기열 시뮬레이션")
power_queue = deque(df_hourly['수요량'][:10])
st.write("대기열 상태:", list(power_queue))
power_queue.append(50000)
power_queue.popleft()
st.write("변경 후 대기열:", list(power_queue))

# 📦 스택 구조 - 역추적 기반 전력 소비 이력 저장
st.header("📌 스택 자료구조: 지역별 소비 이력 역추적")
selected_city = st.selectbox("지역 선택 (시군구)", df_power['시군구'].unique())
df_selected = df_power[df_power['시군구'] == selected_city]
stack = []
for m in ['1월', '2월', '3월', '4월', '5월']:
    stack.append(df_selected[m].sum())
st.write("소비 이력 (최근→과거):", stack[::-1])

# 📊 정렬 알고리즘: 지역별 평균 사용량 정렬
st.header("📌 정렬 알고리즘: 지역 평균 사용량 내림차순 정렬")
df_power['사용량합'] = df_power[['1월', '2월', '3월', '4월', '5월']].sum(axis=1)
sorted_df = df_power.groupby('시군구')['사용량합'].mean().sort_values(ascending=False).reset_index()
st.dataframe(sorted_df)

# 🔍 이진 탐색: 특정 사용량을 가진 지역 탐색
st.header("📌 이진 탐색: 사용량 기반 지역 탐색")
target = st.number_input("찾고 싶은 사용량", min_value=0)
sorted_list = sorted_df['사용량합'].tolist()

def binary_search(data, target):
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            high = mid - 1
        else:
            low = mid + 1
    return -1

index = binary_search(sorted_list, target)
if index != -1:
    st.success(f"{target} 사용량을 가진 지역: {sorted_df.iloc[index]['시군구']}")
else:
    st.warning("해당 사용량을 가진 지역이 없습니다.")

# 🌲 트리 구조: 지역 간 위계 구조 시각화 (간단 계층 표현)
st.header("📌 트리 구조: 시도 → 시군구 계층 구조")
tree = {}
for _, row in df_power.iterrows():
    sido = row['시도']
    sigungu = row['시군구']
    if sido not in tree:
        tree[sido] = set()
    tree[sido].add(sigungu)
for sido in sorted(tree.keys()):
    st.markdown(f"**{sido}**")
    st.markdown(", ".join(tree[sido]))

# 📈 회귀 분석: 온도 기반 예측
st.header("🌡️ 온도 기반 회귀 분석")
merged = pd.merge(df_power, df_temp, on='시군구')
X = merged[['평균기온']]
y = merged['사용량합']
model = LinearRegression().fit(X, y)
pred = model.predict(X)
plt.figure()
plt.scatter(X, y, label='실제')
plt.plot(X, pred, color='red', label='예측')
plt.legend()
st.pyplot(plt)

# 🔢 샤논 엔트로피
st.header("🧮 샤논 엔트로피 기반 지역 불균형 측정")
def entropy(s):
    p = s.value_counts(normalize=True)
    return -sum(p_i * log2(p_i) for p_i in p if p_i > 0)

df_entropy = df_power.groupby('시군구')['사용량합'].apply(entropy).reset_index()
df_entropy.columns = ['시군구', '샤논엔트로피']
fig = px.bar(df_entropy, x='시군구', y='샤논엔트로피')
st.plotly_chart(fig)

# 🔧 변분법 경로 최적화 예시
st.header("🛠️ 변분법 기반 경로 최적화 (예시)")
from scipy.integrate import solve_bvp

def ode_sys(x, y): return np.vstack((y[1], -0.5 * y[0]))
def bc(ya, yb): return np.array([ya[0], yb[0]-1])
x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
sol = solve_bvp(ode_sys, bc, x, y)
x_plot = np.linspace(0, 1, 100)
y_plot = sol.sol(x_plot)[0]
plt.figure()
plt.plot(x_plot, y_plot)
plt.title("변분법 기반 최적 경로")
st.pyplot(plt)
