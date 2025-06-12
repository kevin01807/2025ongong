import os, streamlit as st, pandas as pd, networkx as nx
import plotly.express as px
from collections import deque

st.set_page_config(page_title="재난 대응 시뮬레이션", layout="wide")
st.title("🌪️ 재난 시뮬레이션: 서울시 대피소 최적화")

# 데이터 로드
@st.cache_data
def load_data():
    shelters = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "seoul_shelters.csv"), encoding='cp949'
    )
    warnings = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "FCT_WRN_20250612234617.csv"), encoding='utf-8'
    )
    return shelters, warnings

shelters, warnings = load_data()

# 큐: 최신 3개 특보 (FIFO)
q = deque(warnings.sort_values("발표시각", ascending=False).head(3)["특보종류"])
st.subheader("📋 최근 기상특보 (최신 3개)")
st.write(list(q))

# 트리: 자치구별 대피소 분류
tree = {"대피소": {}}
for _, r in shelters.iterrows():
    tree["대피소"].setdefault(r["자치구별"], []).append(r["시설명"])
st.subheader("🌲 대피소 분류 트리")
st.json(tree)

# 그래프: 자치구↔대피소 연결망 구축
G = nx.Graph()
for _, r in shelters.iterrows():
    G.add_edge(r["자치구별"], r["시설명"])

st.subheader("🗺️ 대피소 연결 네트워크")
pos = nx.spring_layout(G, seed=42)
fig_net = px.scatter(
    x=[pos[n][0] for n in G.nodes()],
    y=[pos[n][1] for n in G.nodes()],
    text=list(G.nodes()), title="대피소 연결망 (노드 레이아웃)")
st.plotly_chart(fig_net)

# BFS 탐색: 자치구별 대피 경로 출력
start = st.selectbox("출발 자치구 선택", shelters["자치구별"].unique())
bfs_paths = nx.single_source_shortest_path(G, start)
dest = st.selectbox("목적 대피소 선택", list(bfs_paths.keys()))
st.write(f"✅ BFS 경로: {start} → {dest}", bfs_paths[dest])

# 스택: 최근 상위 5개 특보 (LIFO)
stack = list(warnings.sort_values("발표시각", ascending=False)["특보종류"].head(5))
st.subheader("📌 최근 특보 Top5 (LIFO)")
st.write(stack)

# 정렬: 자치구별 대피소 수 내림차순
counts = shelters["자치구별"].value_counts().to_dict()
sorted_list = sorted(counts.items(), key=lambda x: x[1], reverse=True)
st.subheader("🎯 대피소 수 많은 자치구 Top5")
st.write(sorted_list[:5])

# 선형탐색 예시: 특정 자치구의 대피소 존재 여부
query = st.text_input("🧭 자치구 이름으로 대피소 존재 검색 (선형탐색)", "")
if query:
    found = [name for name in shelters[shelters["자치구별"] == query]["시설명"]]
    st.write("🔍 검색 결과:", found or "해당 자치구에 대피소 없음")

st.caption("📁 데이터 출처: 서울시 공공데이터, 기상청초단기특보")
