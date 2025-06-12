import streamlit as st
import pandas as pd
import os

# 파일 경로 안전 체크
file_path = "hourly_temperature_seoul.csv"
if not os.path.exists(file_path):
    st.error("CSV 파일이 현재 디렉터리에 존재하지 않습니다. 파일을 업로드하거나 경로를 확인하세요.")
else:
    df = pd.read_csv(file_path)
    df["temperature"] = df["temperature"].astype(float)
    df_sorted = df.sort_values("temperature").reset_index(drop=True)

    target = st.slider("기준 온도 선택", min_value=15.0, max_value=30.0, step=0.1, value=25.0)

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
        return low

    idx = binary_search(df_sorted["temperature"].tolist(), target)

    st.success(f"기준 온도에 가장 가까운 값: {df_sorted.iloc[idx]['temperature']} °C")
    st.write("측정 시간:", df_sorted.iloc[idx]["datetime"])
    st.line_chart(df_sorted["temperature"])

