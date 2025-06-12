import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
from collections import deque

st.set_page_config(page_title="재난 경보 시뮬레이션", layout="wide")
st.title("🌪️ 재난 대응 시뮬레이션: 서울시 대피소 최적화")

def safe_read_csv(path_or_buffer):
    """utf-8, cp949, euc-kr 인코딩을 순서대로 시도"""
    encs = ['utf-8', 'cp949', 'euc-kr']
    for e in encs:
        try:
            return pd.read_csv(path_or_buffer, encoding=e)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return None

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    shelters = safe_read_csv(os.path.join(base, "seoul_shelters.csv"))
    warnings = safe_read_csv(os.path.join(base, "kma_warnings_sample.csv"))
    return shelters, warnings

# 자동 로드 시도
shelters, warnings = load_data()

# 업로더 폴백
if shelters is None or warnings is None:
    st.warning("CSV 파일을 찾지 못했습니다. 아래에서 직접 업로드해주세요.")
    up1 = st.file_uploader("서울 대피소 CSV 업로드", type=["csv"])
    up2 = st.file_uploader("기상특보 CSV 업로드", type=["csv"])
    if up1 and up2:
        shelters = safe_read_csv(up1)
        warnings = safe_read_csv(up2)

# 파일 로드 실패 시 중단
if shelters is None or warnings is None:
    st.error("데이터가 없습니다. 앱을 종료합니다.")
    st.stop()

# 컬럼 공백 제거 후 확인 (옵션)
shelters.columns = shelters.columns.str.strip()
warnings.columns = warnings.columns.str.strip()

# 1) 큐: 최신 3개 특보 (FIFO) - 컬럼명 'TM', 'ALERT_TYPE'
warnings = warnings.sort_values("TM", ascending=False)
recent_alerts = deque(warnings.head(3)["ALERT_TYPE"])
st.subheader("📋 최근 기상특보 (최신 3개)")
st.write(list(recent_alerts))

# 2) 트리: 자치구별 대피소 분류
tree = {"대피소": {}}
for _, r in shelters.iterrows():
    gu = r["자치구별"]
    tree["대피소"].setdefault(gu, []).append(r["시설명"])
st.subheader("🌲 대피소 분류 트리")
st.json(tree)

# 3) 그래프 + BFS: 자치구↔대피소 연결망, 최단 경로 탐색
G = nx.Graph()
for _, r in shelters.iterrows():
    G.add_edge(r["자치구별"], r["시설명"])

st.subheader("🗺️ 대피소 연결 네트워크")
pos = nx.spring_layout(G, seed=42)
fig_net = px.scatter(
    x=[pos[n][0] for n in G.nodes()],
    y=[pos[n][1] for n in G.nodes()],
    text=list(G.nodes()),
    title="대피소 연결망 (노드 레이아웃)"
)
st.plotly_chart(fig_net, use_container_width=True)

start = st.selectbox("출발 자치구 선택", sorted(shelters["자치구별"].unique()))
paths = nx.single_source_shortest_path(G, start)
dest = st.selectbox("도착 대피소 선택", list(paths.keys()))
st.write(f"✅ BFS 최단 경로: {start} → {dest}", paths[dest])

# 4) 스택: 최근 특보 Top5 (LIFO)
stack_alerts = list(warnings.head(5)["ALERT_TYPE"])
st.subheader("📌 최근 특보 Top5 (LIFO)")
st.write(stack_alerts)

# 5) 정렬: 자치구별 대피소 수 내림차순
counts = shelters["자치구별"].value_counts().to_dict()
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
st.subheader("🎯 대피소 수 많은 자치구 Top5")
st.write(sorted_counts[:5])

# 6) 선형 탐색: 자치구 이름으로 대피소 검색
query = st.text_input("🧭 자치구 이름으로 대피소 검색 (선형 탐색)", "")
if query:
    found = [n for n in shelters[shelters["자치구별"] == query]["시설명"]]
    st.write("🔍 검색 결과:", found or "해당 자치구에 대피소 없음")

st.caption("📁 데이터 출처: 서울시 공공데이터, 기상자료개방포털")
