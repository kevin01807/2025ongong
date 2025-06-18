import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from math import log2
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load Data (with absolute path)
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base, 'power_by_region.csv'))
    df_temp = pd.read_csv(os.path.join(base, 'temperature_by_region.csv'))
    df_hourly = pd.read_csv(os.path.join(base, 'hourly_power.csv'))
    df_sdg = pd.read_csv(os.path.join(base, '7-1-1.csv'))
    return df_power, df_temp, df_hourly, df_sdg

df_power, df_temp, df_hourly, df_sdg = load_data()
st.title("Power Consumption Analysis & Optimization")

# 2. Shannon Entropy
def compute_entropy(group):
    counts = group['사용량(kWh)'].value_counts()
    prob = counts / counts.sum()
    return -np.sum([p * log2(p) for p in prob if p > 0])

entropy_df = df_power.groupby(['시도', '시군구']).apply(compute_entropy).reset_index(name='Entropy')
st.subheader("Shannon Entropy by Region")
st.dataframe(entropy_df)

# 3. Regression: Temperature vs Power
df_temp['평균기온값'] = df_temp['평균기온값'].astype(float)
df_power_avg = df_power.groupby('시군구')['사용량(kWh)'].mean().reset_index(name='AvgUsage')
df_merged = pd.merge(df_temp, df_power_avg, on='시군구')
fig1 = px.scatter(df_merged, x='평균기온값', y='AvgUsage', trendline='ols', title='Regression: Temperature vs Power')
st.plotly_chart(fig1)

# 4. Queue and Stack (선형 구조)
queue = deque(df_hourly['전력사용량(MWh)'][:10])
stack = list(df_hourly['전력사용량(MWh)'][:10])[::-1]
st.subheader("Queue (Latest 10)")
st.write(list(queue))
st.subheader("Stack (Latest 10 Reversed)")
st.write(stack)

# 5. Sorting - Bubble Sort
def bubble_sort(data):
    arr = data.copy()
    for i in range(len(arr)):
        for j in range(len(arr)-1-i):
            if arr[j][1] < arr[j+1][1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

sorted_data = df_power.groupby('시군구')['사용량(kWh)'].sum().reset_index()
sorted_list = bubble_sort(sorted_data.values.tolist())
st.subheader("Top Usage Regions (Bubble Sort)")
st.write(sorted_list[:10])

# 6. Binary Search
def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid][1] == target:
            return arr[mid]
        elif arr[mid][1] < target:
            high = mid - 1
        else:
            low = mid + 1
    return None

target = sorted_list[3][1]
result = binary_search(sorted_list, target)
st.subheader("Binary Search Result")
st.write(f"Search for usage = {target}: {result}")

# 7. Tree Graph for Distribution Hierarchy
G = nx.DiGraph()
for _, row in df_power.iterrows():
    G.add_edge(row['시도'], row['시군구'])

fig2, ax2 = plt.subplots(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', arrows=True, ax=ax2)
st.subheader("Electricity Distribution Tree")
st.pyplot(fig2)

# 8. Variational Optimization
values = df_power.groupby('시군구')['사용량(kWh)'].sum().values
min_diff = np.min([abs(a - b) for i, a in enumerate(values) for b in values[i+1:]])
st.subheader("Minimum Usage Gap (Variational Logic)")
st.write(f"Minimum kWh difference: {min_diff:.2f}")

# 9. SDG 7.1.1 Bar Chart
df_sdg['총에너지소비량'] = df_sdg['총에너지소비량'].astype(float)
fig3 = px.bar(df_sdg, x='시도', y='총에너지소비량', title='SDG 7.1.1 Total Energy Consumption')
st.subheader("SDG 7.1.1 Report")
st.plotly_chart(fig3)
