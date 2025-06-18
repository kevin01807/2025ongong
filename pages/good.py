import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from collections import deque
from math import log2
from queue import LifoQueue
import networkx as nx
import statsmodels.api as sm

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
path_power = os.path.join(BASE_DIR, "power_by_region.csv")
path_temp = os.path.join(BASE_DIR, "temperature_by_region.csv")
path_hourly = os.path.join(BASE_DIR, "hourly_power.csv")
path_sdg = os.path.join(BASE_DIR, "7-1-1.csv")

# 데이터 불러오기
df_power = pd.read_csv(path_power)
df_temp = pd.read_csv(path_temp)
df_hourly = pd.read_csv(path_hourly)
df_sdg = pd.read_csv(path_sdg)

# 사용량 컬럼 생성 (월별 합산)
df_power['사용량(kWh)'] = df_power[[f"{i}월" for i in range(1,13)]].sum(axis=1)

# ✅ 샤논 엔트로피 계산 함수
def compute_entropy(group):
    counts = group['사용량(kWh)'].value_counts()
    prob = counts / counts.sum()
    return -np.sum([p * log2(p) for p in prob if p > 0])

entropy_df = df_power.groupby(['시도', '시군구']).apply(compute_entropy).reset_index(name='Entropy')

# ✅ 회귀 시각화
avg_power = df_power.groupby(['시도', '시군구'])['사용량(kWh)'].mean().reset_index(name='AvgUsage')
avg_temp = df_temp.groupby(['시도', '시군구'])['평균기온값'].mean().reset_index()
df_merged = pd.merge(avg_power, avg_temp, on=['시도', '시군구'])
fig1 = px.scatter(df_merged, x='평균기온값', y='AvgUsage', trendline='ols', title='Regression: Temperature vs Power')

# ✅ 큐 구조
power_queue = deque(df_hourly['TotalLoad'][:10])  # 앞 10개 값 대기열 처리

# ✅ 스택 구조
recent_stack = LifoQueue()
for val in df_hourly['TotalLoad'][-10:]:
    recent_stack.put(val)

# ✅ 트리 구조: 시도-시군구-계약종별
tree = nx.DiGraph()
for _, row in df_power.iterrows():
    tree.add_edge("South Korea", row['시도'])
    tree.add_edge(row['시도'], row['시군구'])
    tree.add_edge(row['시군구'], row['계약종별'])

# ✅ 버블 정렬: 평균 사용량 순위
def bubble_sort(data):
    arr = data.copy()
    for i in range(len(arr)):
        for j in range(0, len(arr) - i - 1):
            if arr[j][1] < arr[j + 1][1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

usage_by_region = df_power.groupby('시군구')['사용량(kWh)'].sum().reset_index()
sorted_regions = bubble_sort(list(zip(usage_by_region['시군구'], usage_by_region['사용량(kWh)'])))

# ✅ SDG 7-1-1 데이터 시각화
fig2 = px.bar(df_sdg, x='지역', y='7-1-1 실적', title='SDG 7-1-1 Achievement by Region')

# ✅ Streamlit UI 구성
st.title("Energy Distribution Analysis with Algorithms")
st.plotly_chart(fig1)
st.subheader("Shannon Entropy by Region")
st.dataframe(entropy_df)
st.subheader("Power Queue (First 10 loads)")
st.write(list(power_queue))
st.subheader("Stack (Recent 10 loads)")
st.write(list(recent_stack.queue))
st.subheader("Tree Graph (Distribution Path)")
fig_tree, ax = plt.subplots(figsize=(10, 6))
pos = nx.spring_layout(tree, k=0.5)
nx.draw(tree, pos, with_labels=True, font_size=6, node_size=50, ax=ax)
st.pyplot(fig_tree)
st.subheader("Top Regions by Usage (Bubble Sort)")
st.write(sorted_regions)
st.plotly_chart(fig2)
