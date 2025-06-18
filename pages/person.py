import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from math import log2
from sklearn.linear_model import LinearRegression

# === File Paths ===
BASE_DIR = os.path.dirname(__file__)
file_power = os.path.join(BASE_DIR, 'power_by_region.csv')
file_temp = os.path.join(BASE_DIR, 'temperature_by_region.csv')
file_hourly = os.path.join(BASE_DIR, 'hourly_power.csv')
file_sdg = os.path.join(BASE_DIR, '7-1-1.csv')

# === Load Data ===
df_power = pd.read_csv(file_power)
df_temp = pd.read_csv(file_temp)
df_hourly = pd.read_csv(file_hourly)
df_sdg = pd.read_csv(file_sdg)

# === Shannon Entropy Function ===
def compute_entropy(row):
    months = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월']
    values = row[months].values
    probs = values / np.sum(values)
    entropy = -np.sum([p * log2(p) for p in probs if p > 0])
    return entropy

df_power['Entropy'] = df_power.apply(compute_entropy, axis=1)

# === Linear Regression: Temperature vs Power ===
df_temp_mean = df_temp[['시도명','평균기온값']].groupby('시도명').mean().reset_index()
df_power_mean = df_power.groupby('시도')[['1월','2월','3월']].mean().mean(axis=1).reset_index(name='AvgUsage')
df_merged = pd.merge(df_temp_mean, df_power_mean, left_on='시도명', right_on='시도')

model = LinearRegression()
model.fit(df_merged[['평균기온값']], df_merged['AvgUsage'])
df_merged['Prediction'] = model.predict(df_merged[['평균기온값']])

# === Bubble Sort Example ===
def bubble_sort(arr):
    arr = arr.copy()
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j][1] < arr[j+1][1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

region_usage = df_power.groupby('시도')[['1월','2월','3월']].mean().mean(axis=1).reset_index(name='Usage')
sorted_usage = bubble_sort(region_usage.values.tolist())
sorted_df = pd.DataFrame(sorted_usage, columns=['Region', 'Usage'])

# === Queue/Stack Example ===
queue = deque(df_hourly.iloc[0,1:].tolist())
stack = list(df_hourly.iloc[1,1:].tolist())[::-1]

# === Tree Visualization (Simplified) ===
tree_data = df_power.groupby(['시도','시군구']).size().reset_index(name='Count')
fig_tree = px.sunburst(tree_data, path=['시도','시군구'], values='Count', title='Regional Power Distribution Tree')

# === Variational Optimization (Simple Example) ===
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * x**2
min_idx = np.argmin(y)

# === Streamlit Interface ===
st.title("SDG 7.1.1 Energy Analysis Dashboard")

st.subheader("🔢 Shannon Entropy by Region")
st.dataframe(df_power[['시도','시군구','Entropy']])

st.subheader("📈 Regression: Temperature vs Power Usage")
fig1 = px.scatter(df_merged, x='평균기온값', y='AvgUsage', trendline='ols', title='Regression: Temperature vs Power')
st.plotly_chart(fig1)

st.subheader("🧠 Sorted Region by Power Consumption")
fig2 = px.bar(sorted_df, x='Region', y='Usage', title='Power Usage (Sorted by Bubble Sort)')
st.plotly_chart(fig2)

st.subheader("📊 SDG 7.1.1 Indicator Visualization")
fig3 = px.line(df_sdg, x='Year', y='Value', color='거주지역별', title='SDG 7.1.1 Progress by Region')
st.plotly_chart(fig3)

st.subheader("🌳 Power Distribution Tree")
st.plotly_chart(fig_tree)

st.subheader("📉 Variational Optimization Curve")
fig4, ax = plt.subplots()
ax.plot(x, y, label='Objective Function')
ax.plot(x[min_idx], y[min_idx], 'ro', label='Minimum')
ax.set_title('Variational Optimization')
ax.legend()
st.pyplot(fig4)

st.subheader("🧱 Queue & Stack Simulation")
st.write("Queue (from earliest hour):", list(queue))
st.write("Stack (from latest hour):", stack)
