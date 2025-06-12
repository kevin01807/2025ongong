import os
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree

st.set_page_config(page_title="당뇨병 환자 위험도 관리 시스템", layout="wide")
st.title("🩺 당뇨병 환자 위험도 스트림 처리 & 우선순위 관리")

def safe_read_csv(path_or_buffer):
    for enc in ('utf-8', 'cp949', 'euc-kr'):
        try:
            return pd.read_csv(path_or_buffer, encoding=enc)
        except Exception:
            continue
    return None

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = safe_read_csv(os.path.join(base, "diabetes_data_upload.csv"))
    return df

# 1) 데이터 로드 및 전처리
df = load_data()
if df is None:
    st.error("❌ diabetes_data_upload.csv 파일을 로드할 수 없습니다.")
    st.stop()

# 매핑
df["class"] = df["class"].map({"Positive":1, "Negative":0})
binary_cols = df.columns.drop(["Age","Gender","class"])
for col in binary_cols:
    df[col] = df[col].map({"Yes":1,"No":0})
df["Gender"] = df["Gender"].map({"Male":1,"Female":0})

# 간단한 모델 학습 (랜덤포레스트) -> risk_score
X = df.drop(columns="class")
y = df["class"]
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)
df["risk_score"] = rf.predict_proba(X)[:,1]

# 2) 큐: 신규 환자 스트림 (FIFO)
queue = deque(df.to_dict("records"))
st.subheader("▶ 신규 환자 대기열 (FIFO)")
st.write(f"전체 대기 환자 수: {len(queue)}")

# 처리할 환자 수 선택
n = st.sidebar.slider("처리할 환자 수", 1, min(20, len(queue)), 5)
processed = []
stack = []
for _ in range(n):
    rec = queue.popleft()
    processed.append(rec)
    # 3) 스택: 고위험 환자 히스토리 (risk_score > 0.5)
    if rec["risk_score"] > 0.5:
        stack.append(rec)

st.write("처리된 환자 (처리 순서):")
st.dataframe(pd.DataFrame(processed)[["Age","Gender","risk_score"]])

# 4) 스택(LIFO)에서 가장 최근 위험 사례 Top5
st.subheader("⚠️ 고위험 환자 히스토리 Top5 (LIFO)")
recent_high = stack[-5:][::-1]
st.dataframe(pd.DataFrame(recent_high)[["Age","Gender","risk_score"]])

# 5) 정렬: 남은 대기열 위험도 순 정렬
st.subheader("🔢 남은 대기열 환자 위험도 순 정렬")
remaining = list(queue)
sorted_remaining = sorted(remaining, key=lambda r: r["risk_score"], reverse=True)
st.dataframe(pd.DataFrame(sorted_remaining)[["Age","Gender","risk_score"]].head(10))

# 6) 선형 탐색: 나이로 환자 찾기
st.subheader("🔍 선형 탐색: 나이로 환자 조회")
age_query = st.number_input("찾을 환자의 나이 입력", int(df["Age"].min()), int(df["Age"].max()))
found = [r for r in remaining if r["Age"] == age_query]
st.write(found or "해당 나이 환자가 대기열에 없습니다.")

# 7) 의사결정트리 학습 & 구조화
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# 노드 구조 추출
tree_ = dt.tree_
feature_names = X.columns.tolist()

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right = left, right
        self.value = value

def build_tree(node_id=0):
    if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
        name = feature_names[tree_.feature[node_id]]
        thr = tree_.threshold[node_id]
        left = build_tree(tree_.children_left[node_id])
        right = build_tree(tree_.children_right[node_id])
        return Node(feature=name, threshold=thr, left=left, right=right)
    else:
        return Node(value=tree_.value[node_id])

root = build_tree()

# 트리 순회
def preorder(node, out):
    if node is None: return
    out.append(f"{node.feature}<={node.threshold}" if node.feature else f"leaf:{node.value}")
    preorder(node.left, out)
    preorder(node.right, out)

def inorder(node, out):
    if node is None: return
    inorder(node.left, out)
    out.append(f"{node.feature}<={node.threshold}" if node.feature else f"leaf:{node.value}")
    inorder(node.right, out)

def postorder(node, out):
    if node is None: return
    postorder(node.left, out)
    postorder(node.right, out)
    out.append(f"{node.feature}<={node.threshold}" if node.feature else f"leaf:{node.value}")

pre, ino, post = [], [], []
preorder(root, pre)
inorder(root, ino)
postorder(root, post)

st.subheader("🌳 Decision Tree 순회 결과")
st.markdown(f"- **전위 순회(Pre-order)**: {pre}")
st.markdown(f"- **중위 순회(In-order)**: {ino}")
st.markdown(f"- **후위 순회(Post-order)**: {post}")
