import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from collections import deque
from math import log2
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# 경로 설정
def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# 데이터 불러오기
@st.cache_data
def load_data():
    df_power = pd.read_csv(get_path("power_by_region.csv"))
    df_temp = pd.read_csv(get_path("temperature_by_region.csv"))
    df_hourly = pd.read_csv(get_path("hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

# 1. 월별 사용량 평균 계산 후 샤논 엔트로피
month_cols = [f"{i}월" for i in range(1, 13)]
df_power["총사용량"] = df_power[month_cols].sum(axis=1)

def compute_entropy(group):
    counts = group["총사용량"].value_counts()
    prob = counts / counts.sum()
    return -np.sum([p * log2(p) for p in prob if p > 0])

entropy_df = df_power.groupby(["시도", "시군구"]).apply(compute_entropy).reset_index(name="Entropy")
st.subheader("Shannon Entropy by Region")
st.dataframe(entropy_df)

# 2. 평균기온 vs 전력 사용량 회귀
power_avg = df_power.groupby("시도")["총사용량"].mean().reset_index(name="AvgUsage")
temp_avg = df_temp.groupby("시도명")["평균기온값"].mean().reset_index()
temp_avg.rename(columns={"시도명": "시도"}, inplace=True)
df_merged = pd.merge(power_avg, temp_avg, on="시도")

fig1 = px.scatter(df_merged, x="평균기온값", y="AvgUsage", trendline="ols", title="Regression: Temperature vs Power")
st.plotly_chart(fig1)

# 3. Queue & Stack 활용
power_queue = deque(df_hourly.iloc[0, 1:25])
power_stack = list(power_queue)
power_stack.reverse()

st.subheader("Power Queue (Morning to Night)")
st.write(list(power_queue))
st.subheader("Power Stack (Night to Morning)")
st.write(power_stack)

# 4. 트리 구조 - 배전 경로
st.subheader("Power Distribution Tree")
G = nx.DiGraph()
regions = df_power[["시도", "시군구"]].drop_duplicates()
for _, row in regions.iterrows():
    G.add_edge("Korea", row["시도"])
    G.add_edge(row["시도"], row["시군구"])

fig, ax = plt.subplots(figsize=(10, 6))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, arrows=True, node_size=800, node_color="lightblue", font_size=10)
st.pyplot(fig)

# 5. 정렬 알고리즘 (버블 정렬 예시)
def bubble_sort(data):
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j][1] < data[j + 1][1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data

sorted_usage = df_power.groupby("시도")["총사용량"].sum().reset_index()
sorted_list = list(sorted_usage.to_records(index=False))
sorted_result = bubble_sort(sorted_list.copy())
st.subheader("Top Regions by Power Usage (Bubble Sort)")
st.write(sorted_result)

# 6. SDG 목표 데이터 시각화
try:
    df_sdg = pd.read_csv(get_path("7-1-1.csv"))
    fig2 = px.bar(df_sdg, x="지역", y="보급률", title="SDG 7.1.1 Electrification Rate")
    st.plotly_chart(fig2)
except:
    st.warning("SDG 7.1.1 CSV not found. Upload 7-1-1.csv in the same directory.")
