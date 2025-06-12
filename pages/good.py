import pandas as pd
import streamlit as st
import plotly.express as px
import os
import heapq

# ✅ 파일 경로 설정
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "population.csv")
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='cp949')

# ✅ Heap 자료구조 기반 경보 우선순위 큐
class AlertPriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, year, level, count):
        priority = 0 if level == '경보' else 1
        heapq.heappush(self.queue, (priority, -count, year, level))

    def pop(self):
        return heapq.heappop(self.queue)

    def is_empty(self):
        return len(self.queue) == 0

# ✅ Streamlit 앱 시작
def main():
    st.title("💨 미세먼지 경보 대응 시스템 - 공학적 접근")
    st.markdown("사회적 약자를 위한 고위험 대기질 예측 및 대응 알고리즘")

    df = load_data()

    st.subheader("1. 데이터 미리보기")
    st.dataframe(df)

    st.subheader("2. 연도별 경보/주의보 발령 시각화")
    fig = px.bar(df, x="연도", y="발령횟수", color="단계", barmode="group", title="연도별 발령 횟수 비교")
    st.plotly_chart(fig)

    st.subheader("3. 자료구조 알고리즘 적용: 우선 대응 대상 선정 (Heap)")
    pq = AlertPriorityQueue()
    for _, row in df.iterrows():
        pq.push(row["연도"], row["단계"], row["발령횟수"])

    result = []
    while not pq.is_empty():
        item = pq.pop()
        result.append({
            "우선순위": "높음" if item[0] == 0 else "낮음",
            "연도": item[2],
            "단계": item[3],
            "발령횟수": -item[1]
        })

    st.write("🚨 우선 대응 순위")
    st.dataframe(pd.DataFrame(result))

    st.caption("📁 데이터 출처: 공공데이터포털 (https://www.data.go.kr)")

if __name__ == "__main__":
    main()
