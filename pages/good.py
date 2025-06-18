import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from math import log2

# 파일 경로 함수
BASE_DIR = os.path.dirname(__file__)
def get_path(filename): return os.path.join(BASE_DIR, filename)

# 데이터 불러오기
@st.cache_data
def load_data():
    df_power = pd.read_csv(get_path("power_by_region.csv"))  # 시도, 시군구, 사용량(kWh)
    df_temp = pd.read_csv(get_path("temperature_by_region.csv"))  # 시도, 평균기온값
    df_hourly = pd.read_csv(get_path("hourly_power.csv"))  # 시간, 수요량(MWh)
    df_sdg = pd.read_csv(get_path("7-1-1.csv"))  # Year, 거주지역별, Value
    return df_power, df_temp, df_hourly, df_sdg

df_power, df_temp, df_hourly, df_sdg = load_data()

# 1. 샤논 엔트로피 계산
def compute_entropy(group):
    counts = group['사용량(kWh)'].value_counts()
    total = counts.sum()
    prob = counts / total
    return -np.sum([p * log2(p) for p in prob if p > 0])

entropy_df = df_power.groupby(['시도', '시군구']).apply(compute_entropy).reset_index(name='Entropy')
st.subheader("1. Shannon Entropy by Region")
st.dataframe(entropy_df)

# 2. 회귀 분석 (온도 vs 사용량)
df_temp_power = df_power.groupby('시도')['사용량(kWh)'].mean().reset_index(name='AvgUsage')
df_merged = pd.merge(df_temp_power, df_temp, on='시도')
fig1 = px.scatter(df_merged, x='평균기온값', y='AvgUsage', trendline='ols',
                  title='Regression: Temperature vs Power')
st.plotly_chart(fig1)

# 3. 자료구조: 큐/스택
power_queue = deque(df_hourly['수요량(MWh)'][:10])
power_stack = list(df_hourly['수요량(MWh)'][:10])
st.subheader("2. Queue / Stack Structure (First 10 hours)")
st.write("Queue:", list(power_queue))
st.write("Stack:", power_stack)

# 4. 탐색/정렬: 버블 정렬
def bubble_sort(df, column):
    data = df.copy()
    arr = data.to_dict("records")
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j][column] < arr[j+1][column]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return pd.DataFrame(arr)

sorted_df = bubble_sort(df_power, "사용량(kWh)")
st.subheader("3. Bubble Sort - Top Power Usage")
st.dataframe(sorted_df.head(10))

# 5. 트리 구조 시각화
G = nx.DiGraph()
for _, row in df_power.iterrows():
    G.add_edge("Korea", row['시도'])
    G.add_edge(row['시도'], row['시군구'])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=500, font_size=7, arrows=True)
st.subheader("4. Power Distribution Tree (Region)")
st.pyplot(plt)

# 6. SDG 7-1-1 시각화
fig2 = px.line(df_sdg.dropna(), x='Year', y='Value', color='거주지역별',
               title="SDG 7.1.1: Access to Electricity by Region")
st.subheader("5. SDG 7-1-1 Analysis")
st.plotly_chart(fig2)

# 7. 변분법 기반 최적화
def variational_energy(y):
    dy = np.gradient(y)
    return np.sum(dy**2)

energy_curve = df_hourly['수요량(MWh)'][:50].values
energy_loss = variational_energy(energy_curve)
st.subheader("6. Variational Principle - Energy Flow Cost")
st.write(f"∫(dy/dt)^2 = {energy_loss:.2f}")
