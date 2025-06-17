import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from collections import deque
import heapq
import os

st.set_page_config(layout="wide")
st.title("⚡ 지역별 전력 사용 분석 및 자료구조 알고리즘 탐색")

# 1. 데이터 불러오기 (업로드된 파일 기반)
@st.cache_data

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

df_power, df_temp, df_hourly = load_data()

# 2. 전처리 - 월별 합계 열 추가
df_power['사용량'] = df_power[[f"{i}월" for i in range(1,13)]].sum(axis=1)

# 3. 샤논 엔트로피 계산 함수
def compute_entropy(x):
    p = x / x.sum()
    return -(p * np.log2(p)).sum()

entropy_df = df_power.groupby("시군구")["사용량"].apply(compute_entropy).reset_index(name="샤논 엔트로피")

# 4. 변분법 기반 최적 사용량 분석
def variational_optimization(values):
    gradients = np.gradient(values)
    return np.argmin(np.abs(gradients))

hourly_mean = df_hourly.iloc[:, 1:].mean()
optimal_hour = variational_optimization(hourly_mean)

# 5. 큐 / 스택 시뮬레이션 (전력 수요 우선순위)
power_queue = deque(df_hourly.iloc[0, 1:].tolist())
power_stack = list(df_hourly.iloc[-1, 1:].tolist())

# 6. 정렬 + 탐색 알고리즘 (우선순위 지역)
usage_by_region = df_power.groupby("시군구")["사용량"].sum().reset_index()
usage_sorted = usage_by_region.sort_values(by="사용량", ascending=False).reset_index(drop=True)

def binary_search(region):
    sorted_names = usage_sorted["시군구"].tolist()
    left, right = 0, len(sorted_names) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_names[mid] == region:
            return mid
        elif sorted_names[mid] < region:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 7. 시각화
tab1, tab2, tab3 = st.tabs(["📊 전력 사용량 분석", "🌡 기온 + 전력 상관관계", "📚 자료구조 알고리즘"])

with tab1:
    fig = px.bar(usage_sorted.head(10), x="시군구", y="사용량", title="전력 사용량 상위 10개 지역")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    merged = pd.merge(df_power, df_temp, left_on="시도", right_on="시도명")
    fig2 = px.scatter(merged, x="평균기온값", y="사용량", color="시도명", title="평균기온 vs 전력 사용량")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📌 큐 구조: 시간순 전력 수요")
    st.write(list(power_queue))
    st.subheader("📌 스택 구조: 최근 전력 수요")
    st.write(list(power_stack[::-1]))

    st.subheader("🔍 이진 탐색으로 지역 찾기")
    search_target = st.text_input("검색할 시군구 입력:")
    if search_target:
        idx = binary_search(search_target)
        if idx >= 0:
            st.success(f"{search_target} 지역은 사용량 순위 {idx + 1}위입니다.")
        else:
            st.error("해당 지역은 데이터에 없습니다.")

    st.subheader("📈 시간대별 전력 수요 평균")
    fig3, ax = plt.subplots()
    ax.plot(hourly_mean.index, hourly_mean.values)
    ax.axvline(optimal_hour + 1, color='red', linestyle='--', label='최적 사용 시간')
    ax.set_title("평균 전력 수요 (시간별)")
    ax.set_xlabel("시간대")
    ax.set_ylabel("kWh")
    ax.legend()
    st.pyplot(fig3)

st.caption("Made for SDGs 7.1 / 9.4 | Shannon Entropy + Variational Method + 자료구조 실습")

