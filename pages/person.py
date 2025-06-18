import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt
from math import log2

# Load data using os.path.join for compatibility
base_path = os.path.dirname(__file__)
df_power = pd.read_csv(os.path.join(base_path, "power_by_region.csv"))
df_temp = pd.read_csv(os.path.join(base_path, "temperature_by_region.csv"))
df_hourly = pd.read_csv(os.path.join(base_path, "hourly_power.csv"))

st.title("⚡ Regional Power Data Analysis with Algorithms")

# --- Shannon Entropy Calculation ---
def compute_entropy(row):
    monthly_values = row[['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월']].values
    probs = monthly_values / np.sum(monthly_values) if np.sum(monthly_values) > 0 else np.zeros_like(monthly_values)
    entropy = -np.sum([p * log2(p) for p in probs if p > 0])
    return entropy

df_power['Entropy'] = df_power.apply(compute_entropy, axis=1)
st.subheader("Shannon Entropy by Region")
st.dataframe(df_power[['시도','시군구','Entropy']])

# --- Variational Optimization (minimize total power variance between months) ---
def variational_cost(row):
    monthly_values = row[['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월']].values
    gradient = np.diff(monthly_values)
    return np.sum(gradient**2)

df_power['Variational Cost'] = df_power.apply(variational_cost, axis=1)

# --- Sorting Algorithm (Descending by Entropy) ---
sorted_df = df_power.sort_values(by='Entropy', ascending=False)
st.subheader("Sorted by Entropy")
st.dataframe(sorted_df[['시도','시군구','Entropy']])

# --- Queue Simulation (first 10 hourly power values) ---
power_deque = deque(df_hourly.iloc[0][1:].values[:10])
st.subheader("Queue Simulation: First 10 Hours")
st.write(list(power_deque))

# --- Stack Simulation (top 5 temperature areas) ---
temp_stack = list(df_temp.sort_values(by='평균기온값', ascending=False)['관측소명'][:5])
st.subheader("Stack Simulation: Top 5 Hot Areas")
st.write(temp_stack[::-1])  # LIFO order

# --- Tree Structure Visualization (City to District) ---
st.subheader("Tree Simulation: City to District")
region_tree = {}
for _, row in df_power.iterrows():
    city = row['시도']
    district = row['시군구']
    if city not in region_tree:
        region_tree[city] = []
    if district not in region_tree[city]:
        region_tree[city].append(district)

for city, districts in region_tree.items():
    st.markdown(f"**{city}**")
    for dist in districts:
        st.markdown(f"- {dist}")

# --- Graph: Entropy by Region ---
st.subheader("Entropy Visualization")
fig, ax = plt.subplots(figsize=(10,5))
entropy_plot = df_power.groupby('시도')['Entropy'].mean().sort_values(ascending=False)
entropy_plot.plot(kind='bar', ax=ax)
plt.title("Average Entropy per City")
plt.xlabel("City")
plt.ylabel("Entropy")
st.pyplot(fig)

# --- Graph: Variational Cost ---
st.subheader("Variational Energy Cost")
fig2, ax2 = plt.subplots(figsize=(10,5))
v_cost_plot = df_power.groupby('시도')['Variational Cost'].mean().sort_values(ascending=False)
v_cost_plot.plot(kind='bar', ax=ax2)
plt.title("Average Variational Cost per City")
plt.xlabel("City")
plt.ylabel("Cost")
st.pyplot(fig2)

st.success("✅ All data processed with queue, stack, sorting, searching, tree, entropy, and variational methods!")
