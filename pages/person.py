import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from collections import deque
from math import log2
from scipy.optimize import minimize_scalar

st.set_page_config(layout="wide")
st.title("📊 지역 전력 사용 분석 및 최적 경로 시각화")

# ✅ 안전한 경로 지정 함수
def load_data():
    base_dir = os.path.dirname(__file__)
    df_power = pd.read_csv(os.path.join(base_dir, "power_by_region.csv"))
    df_temp = pd.read_csv(os.path.join(base_dir, "temperature_by_region.csv"))
    df_hourly = pd.read_csv(os.path.join(base_dir, "hourly_power.csv"))
    return df_power, df_temp, df_hourly

@st.cache_data
def compute_entropy(region_usage):
    probs = region_usage / region_usage.sum()
    return -sum(p * log2(p) for p in probs if p > 0)

# 🔽 데이터 불러오기
try:
    df_power, df_temp, df_hourly = load_data()

    # ✅ 월별 사용량 합산 후 "사용량" 컬럼 생성
    df_power["사용량"] = df_power.loc[:, "1월":"12월"].sum(axis=1)

    # ✅ 샤논 엔트로피 계산
    entropy_df = df_power.groupby("시군구")["사용량"].apply(
        lambda x: compute_entropy(x.values)
    ).reset_index(name="샤논 엔트로피")

    st.subheader("1️⃣ 지역별 전력 사용 샤논 엔트로피")
    fig_entropy = px.bar(entropy_df.sort_values("샤논 엔트로피", ascending=False),
                         x="시군구", y="샤논 엔트로피", color="샤논 엔트로피")
    st.plotly_chart(fig_entropy, use_container_width=True)

    # ✅ 큐와 스택 구조 활용
    st.subheader("2️⃣ 자료구조 적용 예시: 큐/스택")

    power_queue = deque(df_hourly["수요량"].head(10))
    stack_peak = []

    for v in power_queue:
        if not stack_peak or v > stack_peak[-1]:
            stack_peak.append(v)

    st.write(f"최근 10개 시간 수요량 (큐): {list(power_queue)}")
    st.write(f"점점 증가한 고점 수요량 (스택): {stack_peak}")

    # ✅ 탐색 및 정렬 알고리즘
    st.subheader("3️⃣ 탐색/정렬: 이진탐색 & 정렬")
    sorted_power = sorted(df_hourly["수요량"].dropna())

    def binary_search(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    search_val = st.slider("탐색할 수요량 값:", int(min(sorted_power)), int(max(sorted_power)), step=1)
    result = binary_search(sorted_power, search_val)
    st.write(f"🔍 이진 탐색 결과: {f'{result}번째 위치' if result != -1 else '찾을 수 없음'}")

    # ✅ 변분법 기반 최적화 예시 (모의 목적함수)
    st.subheader("4️⃣ 변분법 최적화 예시")

    def mock_cost(x):
        return (x - 50)**2 + 10*np.sin(x / 5)

    res = minimize_scalar(mock_cost, bounds=(0, 100), method='bounded')
    x_vals = np.linspace(0, 100, 300)
    y_vals = mock_cost(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="비용 함수")
    ax.plot(res.x, res.fun, 'ro', label=f"최소값: {res.x:.2f}")
    ax.set_title("에너지 분배 최적화 (모의 시나리오)")
    ax.legend()
    st.pyplot(fig)

    # ✅ 지도 시각화 (온도 데이터 예시)
    st.subheader("5️⃣ 지도 시각화 (기온 기반)")

    if {'위도', '경도', '기온'}.issubset(df_temp.columns):
        st.map(df_temp.rename(columns={'위도': 'latitude', '경도': 'longitude'}))
    else:
        st.warning("기온 데이터에 '위도', '경도', '기온' 컬럼이 필요합니다.")

except FileNotFoundError as e:
    st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
except KeyError as e:
    st.error(f"❌ 잘못된 컬럼 이름: {e}")
except Exception as e:
    st.error(f"❌ 예기치 못한 오류 발생: {e}")
