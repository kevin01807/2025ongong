import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
from collections import deque
from math import log2

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

st.title("⚡ Regional Power Usage and Environmental Analysis")

# -----------------------------
# Shannon Entropy Function
# -----------------------------
def compute_entropy(row):
    values = row[['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']].values
    prob = values / np.sum(values) if np.sum(values) > 0 else np.zeros_like(values)
    return -np.sum([p * log2(p) for p in prob if p > 0])

# -----------------------------
# Apply Entropy
# -----------------------------
df_entropy = df_power.groupby(['시도', '시군구']).apply(compute_entropy).reset_index()
df_entropy.columns = ['Region', 'Subregion', 'Entropy']
st.subheader("🔀 Monthly Usage Entropy")
st.dataframe(df_entropy)
fig1 = px.bar(df_entropy, x='Region', y='Entropy', color='Subregion', title="Shannon Entropy by Region")
st.plotly_chart(fig1)

# -----------------------------
# Queue (최근 시간별 소비량)
# -----------------------------
latest_hourly = df_hourly.iloc[-1, 1:].astype(float).values
power_queue = deque(latest_hourly)
st.subheader("🕐 Hourly Power Queue")
st.write("Current Queue: ", list(power_queue))

# -----------------------------
# Stack (가장 많이 소비한 시간대 우선)
# -----------------------------
power_stack = sorted([(i+1, v) for i, v in enumerate(latest_hourly)], key=lambda x: x[1], reverse=True)
st.subheader("📊 Stack: Peak Hours")
st.write("Most Demanding Hours (Top 5): ", power_stack[:5])

# -----------------------------
# Binary Search on Sorted Peak Hours
# -----------------------------
def binary_search(data, target):
    left, right = 0, len(data) - 1
    while left <= right:
        mid = (left + right) // 2
        if data[mid][1] == target:
            return data[mid][0]  # Hour
        elif data[mid][1] < target:
            right = mid - 1
        else:
            left = mid + 1
    return None

st.subheader("🔍 Binary Search on Peak Hours")
target = st.number_input("Enter power usage to search (kW)", min_value=0.0)
if target > 0:
    hour = binary_search(power_stack, target)
    if hour:
        st.success(f"Found usage at hour {hour}")
    else:
        st.warning("Not found in current data")

# -----------------------------
# Temperature and Power Correlation
# -----------------------------
st.subheader("🌡️ Temperature vs Power Usage (By Region)")
df_temp_avg = df_temp.groupby('시도명').agg({'평균기온값': 'mean'}).reset_index()
df_power_avg = df_power.groupby('시도').agg({str(m): 'mean' for m in range(1, 13)}).reset_index()
df_power_avg['avg_usage'] = df_power_avg[[str(m) for m in range(1, 13)]].mean(axis=1)
df_merge = pd.merge(df_temp_avg, df_power_avg, left_on='시도명', right_on='시도')
fig2 = px.scatter(df_merge, x='평균기온값', y='avg_usage', text='시도명',
                 title="Temperature vs Average Monthly Power Usage")
st.plotly_chart(fig2)

# -----------------------------
# Variational Analysis (Simple)
# -----------------------------
# 목적함수: L = (du/dx)^2, 최소화
st.subheader("📐 Variational Principle (Usage Smoothing)")
def variational_solution(data):
    x = np.linspace(0, 1, len(data))
    y = np.array(data)
    dy = np.gradient(y, x)
    L = np.sum(dy**2)
    return round(L, 3)

monthly_usage = df_power[[str(m) + "월" for m in range(1, 13)]].mean().values
vp_score = variational_solution(monthly_usage)
st.write(f"Variational Smoothness Score: {vp_score}")
fig3 = plt.figure()
plt.plot(range(1, 13), monthly_usage, marker='o')
plt.title("Average Monthly Power Usage")
plt.xlabel("Month")
plt.ylabel("kW")
st.pyplot(fig3)

st.caption("Data: KEPCO, KMA, SGIS / Analysis by Engineering Data Student")

