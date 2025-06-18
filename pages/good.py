import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from collections import deque
from math import log2
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm

# ✅ 파일 경로 설정
base_dir = os.path.dirname(__file__)
power_path = os.path.join(base_dir, 'power_by_region.csv')
temp_path = os.path.join(base_dir, 'temperature_by_region.csv')
hourly_path = os.path.join(base_dir, 'hourly_power.csv')
sdg_path = os.path.join(base_dir, '7-1-1.csv')

# ✅ 데이터 불러오기
df_power = pd.read_csv(power_path)
df_temp = pd.read_csv(temp_path)
df_hourly = pd.read_csv(hourly_path)
df_sdg = pd.read_csv(sdg_path)

# ✅ 컬럼 정리 및 매핑
df_temp.rename(columns={'시도명': '시도', '관측소명': '시군구'}, inplace=True)
df_power['총사용량'] = df_power[[str(m)+'월' for m in range(1, 13)]].sum(axis=1)

# ✅ 1. 샤논 엔트로피 계산
def compute_entropy(group):
    counts = group['총사용량'].value_counts(normalize=True)
    return -np.sum([p * log2(p) for p in counts if p > 0])

entropy_df = df_power.groupby(['시도', '시군구']).apply(compute_entropy).reset_index(name='Entropy')
st.subheader("1. Shannon Entropy of Power Usage")
st.dataframe(entropy_df)

# ✅ 2. 기온 기반 회귀 분석
avg_power = df_power.groupby(['시도', '시군구'])['총사용량'].mean().reset_index(name='AvgUsage')
avg_temp = df_temp.groupby(['시도', '시군구'])['평균기온값'].mean().reset_index()
df_merged = pd.merge(avg_power, avg_temp, on=['시도', '시군구'], how='inner')

fig1 = px.scatter(df_merged, x='평균기온값', y='AvgUsage', trendline='ols',
                  title='Regression: Temperature vs Power Usage')
st.plotly_chart(fig1)

# ✅ 3. 큐/스택 사용 예시
st.subheader("2. Queue/Stack Simulation")
power_stack = list(df_power['총사용량'].head(5))
power_queue = deque(df_power['총사용량'].head(5))
st.write("Stack (Top 5):", power_stack[::-1])
st.write("Queue (Top 5):", list(power_queue))

# ✅ 4. 버블 정렬 적용
def bubble_sort(df):
    data = df[['시도', '시군구', '총사용량']].copy().reset_index(drop=True)
    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            if data.loc[j, '총사용량'] < data.loc[j+1, '총사용량']:
                data.loc[j], data.loc[j+1] = data.loc[j+1].copy(), data.loc[j].copy()
    return data.head(10)

sorted_df = bubble_sort(df_power)
st.subheader("3. Top 10 Regions by Power Usage (Bubble Sort)")
st.dataframe(sorted_df)

# ✅ 5. 트리 구조 시각화
st.subheader("4. Tree Structure of Power Distribution")

tree_data = df_power[['시도', '시군구']].drop_duplicates()
G = nx.DiGraph()
for _, row in tree_data.iterrows():
    G.add_edge(row['시도'], row['시군구'])

fig2, ax = plt.subplots(figsize=(10, 8))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1200, font_size=8)
st.pyplot(fig2)

# ✅ 6. SDG 7-1-1 시각화
st.subheader("5. SDG 7.1.1 Performance by Region")

try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    pass  # 일부 환경에서는 폰트 설치 안되어 있을 수 있음

fig3 = px.bar(df_sdg, x='지역명', y='보급률', color='보급률',
              title='SDG 7.1.1: 보급률 by Region')
st.plotly_chart(fig3)

# ✅ 7. 변분법 기반 최적 경로(예시: 최소 거리 기반 연결)
st.subheader("6. Variational Optimization (Distance Simulation)")

# 예시 좌표 데이터 추가
sample_coords = df_power[['시도', '시군구']].drop_duplicates().reset_index(drop=True)
sample_coords['x'] = np.random.rand(len(sample_coords)) * 100
sample_coords['y'] = np.random.rand(len(sample_coords)) * 100

fig4, ax4 = plt.subplots(figsize=(8, 6))
for i in range(len(sample_coords)-1):
    x1, y1 = sample_coords.loc[i, ['x', 'y']]
    x2, y2 = sample_coords.loc[i+1, ['x', 'y']]
    ax4.plot([x1, x2], [y1, y2], 'k--')
ax4.scatter(sample_coords['x'], sample_coords['y'], c='red')
for i, row in sample_coords.iterrows():
    ax4.text(row['x']+1, row['y'], f"{row['시도']}-{row['시군구']}", fontsize=8)
ax4.set_title("Optimized Power Path Simulation (Variational Principle)")
st.pyplot(fig4)
