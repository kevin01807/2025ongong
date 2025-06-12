import os, streamlit as st, pandas as pd, plotly.express as px
from collections import deque
import heapq

st.title("🏢 다중이용시설 실내공기질 분석")

@st.cache_data
def load_data():
    return pd.read_csv("incheon_junggu_indoor_air_quality.csv")

df = load_data()

# 1. 트리 구조: 시설별 주요 지표
tree = {
    r["시설명"]: {
        "연면적": r["연면적(m2)"],
        "PM2.5": r.get("PM25", None),
        "PM10": r.get("PM10", None)
    } for _, r in df.iterrows()
}
st.subheader("🌲 시설별 실내 공기질 트리 구조")
st.json(tree)

# 2. 큐: 연면적 기준 FIFO 분석
q = deque(df.sort_values("연면적(m2)").to_dict("records"))
total_area = 0
while q:
    r = q.popleft()
    total_area += r["연면적(m2)"]
st.write(f"📋 전체 시설 연면적 합계 (FIFO 처리): {total_area:.1f} m²")

# 3. 스택: 최근 위험 높은 시설 Top5 (LIFO)
stack = [(r["시설명"], r.get("PM25", 0)) for _, r in df.iterrows() if r.get("PM25",0) > 35]
latest5 = stack[-5:][::-1]
st.write("📌 최근 위험 수준 높은 시설 Top5:", latest5)

# 4. 힙: PM2.5 기준 우선 순위
heap = [(-r.get("PM25",0), r["시설명"]) for _, r in df.iterrows()]
heapq.heapify(heap)
top3 = [heapq.heappop(heap) for _ in range(min(3, len(heap)))]
st.write("🏥 PM2.5 기준 상위 위험 시설 Top3:", [(name, -pm) for pm, name in top3])

# 5. 정렬 + 이진탐색: 기준 PM2.5 초과 탐색
lst = sorted([(r["시설명"], r.get("PM25",0)) for _, r in df.iterrows()], key=lambda x: x[1])
def bsearch(a, target):
    l, h = 0, len(a)-1
    while l <= h:
        m = (l+h)//2
        if a[m][1] == target: return m
        elif a[m][1] < target: l = m+1
        else: h = m-1
    return l
idx = bsearch(lst, 25.0)
st.write("🔍 PM2.5 =25µg/m³ 초과 시설 예시:", lst[idx:idx+5])

# 6. 시각화
fig = px.bar(df, x="시설명", y="PM25", title="시설별 PM2.5 수치 분포")
st.plotly_chart(fig, use_container_width=True)
