import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from math import log2
from collections import deque

# ===== 데이터 불러오기 =====
def load_data():
    base_path = os.path.dirname(__file__)
    power = pd.read_csv(os.path.join(base_path, "power_by_region.csv"))
    temp = pd.read_csv(os.path.join(base_path, "temperature_by_region.csv"))
    hourly = pd.read_csv(os.path.join(base_path, "hourly_power.csv"))
    return power, temp, hourly

df_power, df_temp, df_hourly = load_data()

# ===== 전처리 및 컬럼 생성 =====
df_power['AvgUsage'] = df_power[[f'{i}월' for i in range(1, 13)]].mean(axis=1)

# ===== 샤논 엔트로피 =====
def compute_entropy(group):
    counts = group['AvgUsage'].value_counts()
    prob = counts / counts.sum()
    return -np.sum([p * log2(p) for p in prob if p > 0])

entropy_df = df_power.groupby(['시도', '시군구']).apply(compute_entropy).reset_index(name='Entropy')

# ===== 큐 / 스택 예시 =====
power_queue = deque(df_hourly['Power(MWh)'].head(10))
power_stack = list(df_hourly['Power(MWh)'].tail(10))

# ===== 정렬 알고리즘 예시 (버블 정렬) =====
def bubble_sort(arr):
    arr = arr.copy()
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j][1] < arr[j+1][1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

region_usage = df_power.groupby('시군구')['AvgUsage'].mean().reset_index()
region_sorted = bubble_sort(region_usage.values.tolist())

# ===== 회귀분석 =====
df_merged = pd.merge(df_power, df_temp, how='inner', on=['시도', '시군구'])
fig1 = px.scatter(df_merged, x='평균기온값', y='AvgUsage', trendline='ols', title='Regression: Temperature vs Power')

# ===== 트리 시각화 (간단한 구조) =====
tree_edges = df_power[['시도', '시군구']].drop_duplicates()
tree_edges['root'] = 'Total'
fig2 = px.sunburst(tree_edges, path=['root', '시도', '시군구'], title='Power Distribution Tree')

# ===== Streamlit UI =====
st.title("Energy Data Explorer with Algorithms")

st.header("1. Shannon Entropy by Region")
st.dataframe(entropy_df)

st.header("2. Queue & Stack Example")
st.write(f"Power Queue (First 10 hours): {list(power_queue)}")
st.write(f"Power Stack (Last 10 hours): {power_stack[::-1]}")

st.header("3. Sorted Regions by AvgUsage (Bubble Sort)")
st.dataframe(pd.DataFrame(region_sorted, columns=['Region', 'AvgUsage']))

st.header("4. Regression Analysis")
st.plotly_chart(fig1)

st.header("5. Tree Visualization")
st.plotly_chart(fig2)

