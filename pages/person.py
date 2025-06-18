# energy_sdg_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
from math import log2
from collections import deque

# ========== 1. Load Data ==========
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    df_sdg711 = pd.read_csv(os.path.join(base_dir, "7-1-1.csv"))
    return df_power, df_temp, df_hourly, df_sdg711

# ========== 2. Shannon Entropy ==========
def compute_entropy(group):
    total = group.sum()
    prob = group / total
    return -np.sum([p * log2(p) for p in prob if p > 0])

# ========== 3. Variational Cost ==========
def compute_variational_cost(values):
    return np.sum(np.diff(values)**2)

# ========== 4. Regression ==========
def predict_power_by_temp(df_power, df_temp):
    df = pd.merge(df_power, df_temp, on=['시도', '시군구'])
    from sklearn.linear_model import LinearRegression
    X = df[['평균기온']]
    y = df['사용량']
    model = LinearRegression().fit(X, y)
    df['예측사용량'] = model.predict(X)
    return df

# ========== 5. Sorting ==========
def bubble_sort(df, column):
    data = df.copy()
    arr = data[column].values.tolist()
    idx = list(range(len(arr)))
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] < arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                idx[j], idx[j+1] = idx[j+1], idx[j]
    return data.iloc[idx]

# ========== 6. Tree Representation ==========
def display_power_tree(df):
    tree = {}
    for _, row in df.iterrows():
        sido = row['시도']
        sigungu = row['시군구']
        tree.setdefault(sido, set()).add(sigungu)
    return tree

# ========== 7. Regional Imbalance Score ==========
def imbalance_score(entropy, cost):
    return np.round(entropy / (cost + 1e-6), 3)

# ========== Main ==========
st.set_page_config(layout="wide")
st.title("🔌 SDG 7.1.1 Regional Power Usage Explorer")

# Load
df_power, df_temp, df_hourly, df_sdg711 = load_data()

# Entropy & Variational Cost
entropy_df = df_power.groupby(['시도', '시군구'])['사용량'].apply(compute_entropy).reset_index(name='Entropy')
cost_df = df_power.groupby(['시도', '시군구'])['사용량'].apply(compute_variational_cost).reset_index(name='Cost')
merged = pd.merge(entropy_df, cost_df, on=['시도', '시군구'])
merged['ImbalanceScore'] = merged.apply(lambda row: imbalance_score(row['Entropy'], row['Cost']), axis=1)

# Regression
reg_df = predict_power_by_temp(df_power, df_temp)

# Sorting (Bubble sort)
sorted_df = bubble_sort(df_power.groupby(['시도', '시군구'])['사용량'].sum().reset_index(), '사용량')

# Queue - Hourly Usage
power_queue = deque(df_hourly['load(MWh)'][:10])

# Stack - Top Temp Regions
df_temp['평균기온'] = pd.to_numeric(df_temp['평균기온'], errors='coerce')
top_temp = df_temp.sort_values(by='평균기온', ascending=False).head(5)
temp_stack = list(top_temp['시군구'])

# Tree Visualization
power_tree = display_power_tree(df_power)
st.subheader("Power Distribution Tree")
for sido, sigungus in power_tree.items():
    st.markdown(f"**{sido}**")
    st.markdown(", ".join(sorted(list(sigungus))))

# Plot - Entropy
fig1, ax1 = plt.subplots()
ax1.bar(merged['시군구'], merged['Entropy'])
plt.xticks(rotation=90)
plt.title("Average Entropy per District")
st.pyplot(fig1)

# Plot - Variational Cost
fig2, ax2 = plt.subplots()
ax2.bar(merged['시군구'], merged['Cost'])
plt.xticks(rotation=90)
plt.title("Variational Cost per District")
st.pyplot(fig2)

# SDG 7.1.1 Visualization
st.subheader("SDG 7.1.1 Access Performance")
fig3 = px.bar(df_sdg711, x="지역", y="보급률(%)", title="Access to Electricity (%)")
st.plotly_chart(fig3)

# Queue View
st.subheader("Hourly Power Load (Queue)")
st.write(list(power_queue))

# Stack View
st.subheader("Top 5 Hottest Regions (Stack)")
st.write(temp_stack)

# Sorted Region Usage
st.subheader("Sorted Region Power Usage (Bubble Sort)")
st.dataframe(sorted_df)

# Imbalance Score Map
st.subheader("Regional Imbalance Score Map")
map_df = df_temp.merge(merged, on=['시도', '시군구'])
fig_map = px.scatter_mapbox(map_df,
                            lat='위도', lon='경도',
                            size='ImbalanceScore',
                            color='ImbalanceScore',
                            mapbox_style="carto-positron",
                            zoom=5,
                            hover_name='시군구')
st.plotly_chart(fig_map)
