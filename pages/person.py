import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_bvp
from math import log2
from collections import deque

# -----------------------------
# 1. 데이터 불러오기 (경로 설정 포함)
# -----------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

# -----------------------------
# 2. 샤논 엔트로피 계산
# -----------------------------
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

st.title("🔌 지역 간 전력 소비 분석 및 알고리즘 적용 최적화 시스템")

st.header("① 샤논 엔트로피 분석")
if "시군구" in df_power.columns and "사용량" in df_power.columns:
    entropy_df = df_power.groupby("시군구")["사용량"].apply(compute_entropy).reset_index()
    entropy_df.columns = ["시군구", "샤논엔트로피"]
    st.dataframe(entropy_df)
    st.plotly_chart(px.bar(entropy_df, x="시군구", y="샤논엔트로피", title="지역별 전력 소비의 정보량(샤논 엔트로피)"))
else:
    st.error("❌ '시군구' 또는 '사용량' 컬럼이 존재하지 않습니다. 파일을 확인하세요.")

# -----------------------------
# 3. 온도 기반 선형 회귀 예측
# -----------------------------
st.header("② 온도에 따른 전력 사용량 선형 회귀")

if "시군구" in df_power.columns and "시군구" in df_temp.columns:
    merged = pd.merge(df_power, df_temp, on="시군구")
    if "평균기온" in merged.columns and "사용량" in merged.columns:
        X = merged[["평균기온"]]
        y = merged["사용량"]
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="실제값")
        ax.plot(X, pred, color="red", label="예측값")
        ax.set_xlabel("평균기온")
        ax.set_ylabel("전력 사용량")
        ax.set_title("기온에 따른 전력 사용량 예측")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ 병합된 데이터에서 '평균기온' 또는 '사용량'이 누락되었습니다.")
else:
    st.error("❌ power 또는 temp 데이터에 '시군구' 컬럼이 없습니다.")

# -----------------------------
# 4. 큐와 스택 시뮬레이션
# -----------------------------
st.header("③ 큐/스택 기반 수요 처리 시뮬레이션")

if "수요량(MWh)" in df_hourly.columns:
    from queue import LifoQueue
    from collections import deque

    power_queue = deque(df_hourly["수요량(MWh)"][:10])
    stack = LifoQueue()
    for v in df_hourly["수요량(MWh)"][:10]:
        stack.put(v)

    st.subheader("Queue 구조: FIFO")
    st.write(list(power_queue))

    st.subheader("Stack 구조: LIFO")
    st.write([stack.get() for _ in range(stack.qsize())])
else:
    st.error("❌ '수요량(MWh)' 컬럼이 존재하지 않습니다. 파일을 확인하세요.")

# -----------------------------
# 5. 정렬 알고리즘 시각화
# -----------------------------
st.header("④ 전력 수요 정렬 알고리즘 (선택 정렬)")

def selection_sort(arr):
    arr = arr.copy()
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

if "수요량(MWh)" in df_hourly.columns:
    sample = df_hourly["수요량(MWh)"].head(20).tolist()
    sorted_sample = selection_sort(sample)
    st.write("정렬 전:", sample)
    st.write("정렬 후:", sorted_sample)
else:
    st.error("❌ '수요량(MWh)' 컬럼이 존재하지 않습니다.")

# -----------------------------
# 6. 트리 기반 전력 탐색 알고리즘 예제
# -----------------------------
st.header("⑤ 이진 탐색 트리 기반 전력 탐색 예제")

class Node:
    def __init__(self, key):
        self.left = self.right = None
        self.val = key

def insert(root, key):
    if root is None:
        return Node(key)
    if key < root.val:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root

def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

if "수요량(MWh)" in df_hourly.columns:
    root = None
    for value in df_hourly["수요량(MWh)"][:15]:
        root = insert(root, value)
    st.write("중위 순회 결과 (오름차순):", inorder(root))
else:
    st.error("❌ '수요량(MWh)' 컬럼이 존재하지 않습니다.")

# -----------------------------
# 7. 변분법 기반 최적 경로
# -----------------------------
st.header("⑥ 변분법 기반 배전 경로 최적화 예시")

def ode(x, y):
    return np.vstack((y[1], -0.5 * y[0]))

def bc(ya, yb):
    return np.array([ya[0], yb[0] - 1])

x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
y[0] = x

sol = solve_bvp(ode, bc, x, y)
x_plot = np.linspace(0, 1, 100)
y_plot = sol.sol(x_plot)[0]

fig, ax = plt.subplots()
ax.plot(x_plot, y_plot, label="최적 경로")
ax.set_title("변분법 기반 배전 최적 경로")
ax.set_xlabel("거리")
ax.set_ylabel("전력/전압")
ax.legend()
st.pyplot(fig)

