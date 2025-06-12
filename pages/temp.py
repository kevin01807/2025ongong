import streamlit as st
import pandas as pd
import plotly.express as px
import heapq

# 데이터 로드
df = pd.read_csv("ulsan_energy.csv")

# 예시: '연도', '석탄', '석유', '가스', '태양광', '풍력', '기타' 컬럼 포함
years = sorted(df['연도'].unique())
year = st.selectbox("연도 선택", years)
row = df[df['연도'] == year].iloc[0]

# 트리 구조로 계층화 (딕셔너리)
energy_tree = {
    '화석연료': {'석탄': row['석탄'], '석유': row['석유'], '가스': row['가스']},
    '재생에너지': {'태양광': row['태양광'], '풍력': row['풍력'], '기타': row['기타']}
}

st.write("## 🌳 에너지 소비 계층 트리")
st.json(energy_tree)

# 정렬: 소비량 내림차순
amounts = [(k, row[k]) for k in ['석탄','석유','가스','태양광','풍력','기타']]
sorted_amts = sorted(amounts, key=lambda x: x[1], reverse=True)
st.write("## 🔥 상위 소비 에너지원")
st.write(sorted_amts)

# 이진 탐색 함수
def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high)//2
        if arr[mid][1] == target:
            return mid
        elif arr[mid][1] < target:
            high = mid - 1
        else:
            low = mid + 1
    return low

target = st.number_input("소비량 기준값 입력", min_value=0.0, value=sorted_amts[0][1])
idx = binary_search(sorted_amts, target)
st.write("입력한 소비량과 가장 근접한 에너지원:", sorted_amts[idx])

# 힙: 최대 소비 에너지원 빠르게
max_heap = [(-val, key) for key, val in amounts]
heapq.heapify(max_heap)
top1 = heapq.heappop(max_heap)
st.success(f"최대 소비 에너지원: {top1[1]} ({-top1[0]})")

# 시계열 시각화
df_long = df.melt(id_vars='연도', var_name='에너지원', value_name='소비량')
fig = px.line(df_long, x='연도', y='소비량', color='에너지원', title="에너지원별 소비 추이")
st.plotly_chart(fig)

