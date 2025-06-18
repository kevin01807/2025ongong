import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import os
from io import BytesIO
from collections import deque
from math import log2
from statsmodels.api import OLS, add_constant

# 경로 설정
def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# 데이터 불러오기 함수
def load_data():
    df_power = pd.read_csv(get_path("power_by_region.csv"))
    df_temp = pd.read_csv(get_path("temperature_by_region.csv"))
    df_hourly = pd.read_csv(get_path("hourly_power.csv"))
    df_sdg = pd.read_csv(get_path("7-1-1.csv"))
    return df_power, df_temp, df_hourly, df_sdg

df_power, df_temp, df_hourly, df_sdg = load_data()

# 샤논 엔트로피 계산
def compute_entropy(group):
    counts = group['Usage'].value_counts()
    prob = counts / counts.sum()
    return -np.sum([p * log2(p) for p in prob if p > 0])

df_power.rename(columns={
    '시도': 'Province',
    '시군구': 'City',
    '사용량': 'Usage'
}, inplace=True)

entropy_df = df_power.groupby(['Province', 'City']).apply(compute_entropy).reset_index(name='Entropy')

# 변분법 기반 분석 (가중합 최소화)
def variational_minimize(data):
    return data['Usage'].mean() + data['Usage'].std() * 0.5

optimized_score = variational_minimize(df_power)

# 탐색 및 정렬 알고리즘 - 버블 정렬 예시
def bubble_sort_regions(df):
    items = df.groupby("City")["Usage"].sum().reset_index().values.tolist()
    n = len(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if items[j][1] < items[j + 1][1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return pd.DataFrame(items, columns=["City", "TotalUsage"])

sorted_df = bubble_sort_regions(df_power)

# 큐 / 스택 예시
power_queue = deque(df_hourly['Usage'].head(10))
power_stack = list(df_hourly['Usage'].tail(10))

# 회귀 분석: 온도 vs 전력 사용량
avg_temp = df_temp.groupby("Region")["평균기온값"].mean().reset_index()
avg_usage = df_power.groupby("City")["Usage"].mean().reset_index()
df_merged = pd.merge(avg_temp, avg_usage, left_on="Region", right_on="City")

fig1 = px.scatter(df_merged, x='평균기온값', y='Usage', trendline='ols', title='Regression: Temperature vs Power')

# 지도 기반 전력 배전 트리 시각화
m = folium.Map(location=[36.5, 127.5], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)
for _, row in df_power.iterrows():
    try:
        if not (np.isnan(row['lat']) or np.isnan(row['lon'])):
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"{row['Province']} > {row['City']}",
                icon=folium.Icon(color='blue', icon='bolt', prefix='fa')
            ).add_to(marker_cluster)
    except:
        continue
m.save("/mnt/data/distribution_tree_map.html")

# Streamlit UI
st.title("Power Usage Analysis with Algorithm & Entropy")
st.plotly_chart(fig1)
st.write("\n### 🔍 Sorted Regions by Usage (Bubble Sort)")
st.dataframe(sorted_df)
st.write("\n### ⚙️ Entropy by Region")
st.dataframe(entropy_df)
st.write("\n### ✅ Optimized Score (Variational Calculus)")
st.metric("Optimized Regional Score", f"{optimized_score:.2f}")
st.write("\n### 📍 Power Distribution Map")
st.components.v1.iframe("/mnt/data/distribution_tree_map.html", height=500)
