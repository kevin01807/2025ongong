import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
from collections import deque

st.set_page_config(page_title="재난 경보 시뮬레이션", layout="wide")
st.title("🌪️ 재난 대응 시뮬레이션: 서울시 대피소 최적화")

def safe_read_csv(path_or_buffer):
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
    s = safe_read_csv(os.path.join(base, "seoul_shelters.csv"))
    w = safe_read_csv(os.path.join(base, "kma_warnings_sample.csv"))
    return s, w

# 1) 데이터 로드 (자동 + 업로더 폴백)
shelters, warnings = load_data()
if shelters is None or warnings is None:
    st.warning("CSV 파일을 찾지 못했습니다. 아래에 업로드해주세요.")
    up1 = st.file_uploader("서울 대피소 CSV", type=["csv"])
    up2 = st.file_uploader("기상특보 CSV", type=["csv"])
    if up1 and up2:
        shelters = safe_read_csv(up1)
        warnings = safe_read_csv(up2)
if shelters is None or warnings is None:
    st.error("데이터가 없습니다. 앱을 종료합니다.")
    st.stop()

# 2) 컬럼명 자동 매핑
def find_col(cols, keywords):
    for k in keywords:
        for c in cols:
            if k in c:
                return c
    return None

# shelters 데이터
s_cols = shelters.columns.str.strip().tolist()
col_gu      = find_col(s_cols, ["자치구", "시군구", "구별"])
col_shelter = find_col(s_cols, ["시설", "이름", "명"])
if not col_gu or not col_shelter:
    st.error(f"예상 컬럼명을 찾을 수 없습니다.\n시트 컬럼: {s_cols}")
    st.stop()

# warnings 데이터
w_columns = warnings.columns.str.strip().tolist()
col_time  = find_col(w_columns, ["TM", "시간", "발표"])
col_type  = find_col(w_columns, ["ALERT", "특보", "종류"])
if not col_time or not col_type:
    st.error(f"예상 컬럼명을 찾을 수 없습니다.\n경보 컬럼: {w_columns}")
    st.stop()

# 3) 최근 3개 특보 (FIFO)
warnings = warnings.sort_values(col_time, ascending=False)
recent = deque(warnings.head(3)[col_type])
st.subheader("📋 최근 기상특보 (최신 3개)")
st.write(list(recent))

# 4) 트리: 자치구 → 대피소
tree = {"대피소": {}}
for _, r in shelters.iterrows():
    gu = r[col_gu]
    tree["대피소"].setdefault(gu, []).append(r[col_shelter])
st.subheader("🌲 대피소 분류 트리")
st.json(tree)

# 5) 그래프+ BFS: 자치구↔대피소 연결망 & 경로 찾기
G = nx.Graph()
for _, r in shelters.iterrows():
    G.add_edge(r[col_gu], r[col_shelter])

st.subheader("🗺️ 대피소 연결 네트워크")
pos = nx.spring_layout(G, seed=42)
fig = px.scatter(
    x=[pos[n][0] for n in G.nodes()],
    y=[pos[n][1] for n in G.nodes()],
    text=list(G.nodes()),
    title="대피소 연결망"
)
st.plotly_chart(fig, use_container_width=True)

start = st.selectbox("출발 자치구 선택", sorted(shelters[col_gu].unique()))
paths = nx.single_source_shortest_path(G, start)
dest_cols = list(paths.keys())
dest = st.selectbox("도착 대피소 선택", dest_cols)
st.write(f"✅ BFS 최단 경로: {start} → {dest}", paths[dest])

# 6) 스택: 최근 특보 Top5 (LIFO)
stack5 = list(warnings.head(5)[col_type])
st.subheader("📌 최근 특보 Top5 (LIFO)")
st.write(stack5)

# 7) 정렬: 자치구별 대피소 수 Top5
counts = shelters[col_gu].value_counts().to_dict()
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
st.subheader("🎯 대피소 수 많은 자치구 Top5")
st.write(sorted_counts[:5])

# 8) 선형 탐색: 자치구 이름으로 대피소 조회
query = st.text_input("🧭 자치구 이름으로 대피소 조회", "")
if query:
    found = shelters[shelters[col_gu] == query][col_shelter].tolist()
    st.write("🔍 검색 결과:", found or "해당 자치구에 대피소 없음")

st.caption("📁 데이터 출처: 서울시 공공데이터, 기상자료개방포털")
