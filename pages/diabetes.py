import os
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree

st.set_page_config(page_title="당뇨 환자 위험 관리", layout="wide")
st.title("🩺 당뇨병 환자 위험도 스트림 처리 & 자료구조 시연")

# 1) CSV 불러오기
file_path = os.path.join(os.path.dirname(__file__), "diabetes_data_upload.csv")
df = pd.read_csv(file_path)

# 2) 전처리
df["class"] = df["class"].map({"Positive":1,"Negative":0})
bin_cols = df.columns.drop(["Age","Gender","class"])
for c in bin_cols:
    df[c] = df[c].map({"Yes":1,"No":0})
df["Gender"] = df["Gender"].map({"Male":1,"Female":0})

# 3) 모델 학습 & risk_score
X = df.drop(columns="class")
y = df["class"]
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X,y)
df["risk_score"] = rf.predict_proba(X)[:,1]

# 4) 큐: FIFO
queue = deque(df.to_dict("records"))
st.subheader("▶ 큐 시연: 신규 환자 대기열 (FIFO)")
st.write(f"현재 대기열 환자 수: {len(queue)}")

# FIFO 처리
stack = []
n = st.sidebar.slider("처리할 환자 수", 1, min(20,len(queue)), 5)
processed = []
for _ in range(n):
    rec = queue.popleft()
    processed.append(rec)
    if rec["risk_score"] > 0.5:
        stack.append(rec)

st.dataframe(pd.DataFrame(processed)[["Age","Gender","risk_score"]])

# 5) 스택: LIFO
st.subheader("⚠️ 스택 시연: 고위험 환자 최근 Top5 (LIFO)")
recent_high = stack[-5:][::-1]
st.dataframe(pd.DataFrame(recent_high)[["Age","Gender","risk_score"]])

# 6) 정렬: 위험도 순
st.subheader("🔢 정렬 시연: 남은 대기열 위험도 순서")
remaining = list(queue)
sorted_rem = sorted(remaining, key=lambda r: r["risk_score"], reverse=True)
st.dataframe(pd.DataFrame(sorted_rem)[["Age","Gender","risk_score"]].head(10))

# 7) 이진 탐색: 위험도 기준 초과 첫 위치
threshold = st.number_input("⚠️ 기준 위험도 입력", 0.0, 1.0, 0.5, 0.01)
asc = sorted_rem[::-1]
scores = [r["risk_score"] for r in asc]
lo, hi, idx = 0, len(scores)-1, len(scores)
while lo <= hi:
    mid = (lo+hi)//2
    if scores[mid] >= threshold:
        idx = mid
        hi = mid-1
    else:
        lo = mid+1
first_idx = len(scores)-1-idx
st.write(f"이진 탐색으로 위험도 ≥{threshold} 첫 위치: {first_idx}")

# 8) 트리: 의사결정트리 + 사람이 읽기 좋은 순회
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)
tree_ = dt.tree_
feature_names = X.columns.tolist()

# 사람 친화적 트리 빌드
def build_readable(node_id=0):
    if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
        name = feature_names[tree_.feature[node_id]]
        thr  = round(tree_.threshold[node_id], 2)
        return {
            "feature": name,
            "threshold": thr,
            "left":  build_readable(tree_.children_left[node_id]),
            "right": build_readable(tree_.children_right[node_id])
        }
    else:
        counts = tree_.value[node_id][0]
        total = counts.sum()
        pos_ratio = round(counts[1]/total, 2)
        pred = "Positive" if pos_ratio > 0.5 else "Negative"
        return {"leaf_pred": pred, "pos_prob": pos_ratio}

root = build_readable()

# 순회 함수
def preorder_readable(node, out):
    if "feature" in node:
        out.append(f"{node['feature']} ≤ {node['threshold']}")
        preorder_readable(node["left"], out)
        preorder_readable(node["right"], out)
    else:
        out.append(f"▶ {node['leaf_pred']} ({node['pos_prob']*100:.0f}%)")

def inorder_readable(node, out):
    if "feature" in node:
        inorder_readable(node["left"], out)
        out.append(f"{node['feature']} ≤ {node['threshold']}")
        inorder_readable(node["right"], out)
    else:
        out.append(f"▶ {node['leaf_pred']} ({node['pos_prob']*100:.0f}%)")

def postorder_readable(node, out):
    if "feature" in node:
        postorder_readable(node["left"], out)
        postorder_readable(node["right"], out)
        out.append(f"{node['feature']} ≤ {node['threshold']}")
    else:
        out.append(f"▶ {node['leaf_pred']} ({node['pos_prob']*100:.0f}%)")

pre, ino, post = [], [], []
preorder_readable(root, pre)
inorder_readable(root, ino)
postorder_readable(root, post)

st.subheader("🌳 트리 순회 결과 (사람 친화적)")
st.markdown(f"- **전위 순회**: {pre}")
st.markdown(f"- **중위 순회**: {ino}")
st.markdown(f"- **후위 순회**: {post}")

# 9) 기존 예측 폼
st.subheader("🧪 내 증상으로 당뇨병 예측해보기")
with st.form("predict_form"):
    age = st.slider("나이", 10, 100, 45)
    gender = st.radio("성별", ["남성","여성"])
    input_data = {"Age": age, "Gender": 1 if gender=="남성" else 0}
    for col in X.columns:
        if col not in ["Age","Gender"]:
            input_data[col] = st.radio(f"{col}", ["아님","있음"])=="있음"
    submitted = st.form_submit_button("예측하기")

if submitted:
    input_df = pd.DataFrame([input_data])
    pred = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0][pred]
    if pred == 1:
        st.error(f"⚠️ 당뇨병 위험 있음 ({prob*100:.2f}%)")
    else:
        st.success(f"✅ 위험 낮음 ({prob*100:.2f}%)")

