import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import deque
from math import log2
import heapq

# ✅ 데이터 경로 설정
BASE_DIR = os.path.dirname(__file__)
power_path = os.path.join(BASE_DIR, "power_by_region.csv")
temp_path = os.path.join(BASE_DIR, "temperature_by_region.csv")
hourly_path = os.path.join(BASE_DIR, "hourly_power.csv")

# ✅ 데이터 불러오기
@st.cache_data
def load_data():
    df_power = pd.read_csv(power_path)
    df_temp = pd.read_csv(temp_path)
    df_hourly = pd.read_csv(hourly_path)
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

# ✅ 컬럼명 정제
df_power.columns = df_power.columns.str.strip()
df_temp.columns = df_temp.columns.str.strip()
df_hourly.columns = df_hourly.columns.str.strip()

st.title("Electricity Usage Analysis")

# ✅ Shannon Entropy 계산
def compute_entropy(group):
    prob = group / group.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

if '시도' in df_power.columns and '시군구' in df_power.columns and '사용량' in df_power.columns:
    df_entropy = df_power.groupby(['시도', '시군구'])['사용량'].apply(compute_entropy).reset_index(name='entropy')
    st.subheader("Regional Energy Entropy")
    st.dataframe(df_entropy)
    st.bar_chart(df_entropy.set_index('시군구')['entropy'])

# ✅ Queue: 전력 수요량 최근 10개
if '수요량(MWh)' in df_hourly.columns:
    st.subheader("Queue - Recent Power Demand")
    power_queue = deque(df_hourly['수요량(MWh)'][:10])
    st.write(list(power_queue))

# ✅ Stack: 최고 기온 최근 10개
if '최고기온' in df_temp.columns:
    st.subheader("Stack - Max Temperature (Last 10)")
    temp_stack = list(df_temp['최고기온'].tail(10))
    st.write(temp_stack[::-1])  # Stack 특성상 역순

# ✅ Binary Search
def binary_search(arr, target):
    arr = sorted(arr)
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

if '수요량(MWh)' in df_hourly.columns:
    search_list = df_hourly['수요량(MWh)'][:50].tolist()
    target = int(np.mean(search_list))
    index = binary_search(search_list, target)
    st.subheader("Binary Search on Power Demand")
    st.write(f"Target value: {target}")
    st.write(f"Index (in sorted list): {index}")

# ✅ Heap Sort (Min-Heap)
if '수요량(MWh)' in df_hourly.columns:
    st.subheader("Heap Sort - Power Demand")
    min_heap = []
    for val in df_hourly['수요량(MWh)'][:50]:
        heapq.heappush(min_heap, val)
    sorted_power = [heapq.heappop(min_heap) for _ in range(len(min_heap))]
    fig, ax = plt.subplots()
    ax.plot(sorted_power)
    ax.set_title("Heap Sorted Power Demand")
    st.pyplot(fig)
