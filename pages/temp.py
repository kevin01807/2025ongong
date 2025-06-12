# streamlit_app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smart Temperature Finder", layout="wide")

st.title("🌡️ 실시간 온도 분석 및 기준 온도 탐색기")
st.markdown("기상 데이터를 정렬하고, 기준 온도에 가장 가까운 시간을 이진 탐색으로 찾습니다.")

# 데이터 로드
df = pd.read_csv("hourly_temperature_seoul.csv")

# 디버깅 해결된 버전: 에러 원인 분석 - CSV 파일에 '℃'가 포함된 경우 float 변환 오류
try:
    df["temperature"] = df["temperature"].astype(float)  # ← 실제 상황에서는 replace도 필요할 수 있음
except ValueError:
    df["temperature"] = df["temperature"].str.replace("℃", "").astype(float)

# 오름차순 정렬
df_sorted = df.sort_values("temperature").reset_index(drop=True)

# 기준 온도 슬라이더
target = st.slider("📏 기준 온도 설정", min_value=15.0, max_value=30.0, step=0.1, value=25.0)

# 이진 탐색 함수
def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return low  # 가장 가까운 상한값 index

index = binary_search(df_sorted["temperature"].tolist(), target)

# 결과 표시
st.success(f"가장 가까운 측정 온도: **{df_sorted.iloc[index]['temperature']}°C**")
st.write("해당 시각:", df_sorted.iloc[index]["datetime"])

# 시각화
st.line_chart(df_sorted["temperature"])
