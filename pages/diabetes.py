import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import deque

st.set_page_config(page_title="당뇨병 예측 시스템", layout="wide")
st.title("🩺 당뇨병 예측 시스템")

# ✅ 현재 파일 기준으로 csv 경로 지정
file_path = os.path.join(os.path.dirname(__file__), "diabetes_data_upload.csv")

# ✅ 데이터 불러오기 및 전처리
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["class"] = df["class"].map({"Positive": 1, "Negative": 0})
    binary_cols = df.columns.drop(["Age", "Gender", "class"])
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    return df

try:
    df = load_data(file_path)
except FileNotFoundError:
    st.error("❌ CSV 파일이 현재 디렉터리에 없습니다. 'diabetes_data_upload.csv'를 동일 폴더에 넣어주세요.")
    st.stop()

# 📊 데이터 시각화
st.subheader("📈 나이대별 당뇨병 분포")
fig = px.histogram(df, x="Age", color="class", barmode="group",
                   color_discrete_map={1: "red", 0: "blue"},
                   labels={"class": "당뇨병 여부"})
st.plotly_chart(fig, use_container_width=True)

# 🔍 머신러닝 예측 모델 학습
X = df.drop(columns=["class"])
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
st.success(f"✅ 랜덤 포레스트 정확도: {acc * 100:.2f}%")

# --- 추가: 자료구조 & 알고리즘 시연 ---

# 1) 큐 (Queue): 환자 데이터 FIFO 대기열
queue = deque(df.to_dict("records"))
st.subheader("▶ 큐 시연: 신규 환자 대기열 (FIFO)")
st.write(f"현재 대기열 환자 수: {len(queue)}")

# 2) 스택 (Stack): 고위험 환자 히스토리 (LIFO)
stack = []
n = st.sidebar.slider("FIFO로 처리할 환자 수", 1, min(20, len(queue)), 5)
processed = []
for _ in range(n):
    rec = queue.popleft()
    processed.append(rec)
    # 위험 기준: Positive 판정
    if rec["class"] == 1:
        stack.append(rec)

st.write("처리된 환자 목록:")
st.dataframe(pd.DataFrame(processed)[["Age", "Gender", "class"]])

st.subheader("⚠️ 스택 시연: 고위험 환자 최근 Top5 (LIFO)")
recent_high = stack[-5:][::-1]
st.dataframe(pd.DataFrame(recent_high)[["Age", "Gender", "class"]])

# 3) 정렬 (Sorting): 남은 대기열 환자 class 내림차순 정렬
st.subheader("🔢 정렬 시연: 남은 대기열 위험도 순 정렬")
remaining = list(queue)
sorted_rem = sorted(remaining, key=lambda r: r["class"], reverse=True)
st.dataframe(pd.DataFrame(sorted_rem)[["Age", "Gender", "class"]].head(10))

# 4) 이진 탐색 (Binary Search): 특정 위험도(class) 환자 첫 위치 찾기
threshold = st.number_input("⚠️ 탐색할 class 기준값 (0 또는 1)", 0, 1, 1)
# 리스트는 class 내림차순. 오름차순으로 뒤집어 탐색
asc = sorted_rem[::-1]
classes = [r["class"] for r in asc]
lo, hi, idx = 0, len(classes)-1, len(classes)
while lo <= hi:
    mid = (lo + hi) // 2
    if classes[mid] >= threshold:
        idx = mid
        hi = mid - 1
    else:
        lo = mid + 1
first_idx = len(classes) - 1 - idx
st.write(f"이진 탐색으로 class ≥ {threshold} 첫 위치: {first_idx}")

# 5) 트리 (Decision Tree): 전위·중위·후위 순회
from sklearn.tree import DecisionTreeClassifier, _tree

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)
tree_ = dt.tree_
feature_names = X.columns.tolist()

# 트리 빌드
def build(node_id=0):
    if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
        name = feature_names[tree_.feature[node_id]]
        thr = tree_.threshold[node_id]
        left = build(tree_.children_left[node_id])
        right = build(tree_.children_right[node_id])
        return {"feature": name, "threshold": thr, "left": left, "right": right}
    else:
        return {"value": tree_.value[node_id].tolist()}

root = build()

# 순회 함수
def preorder(node, out):
    if "feature" in node:
        out.append(f"{node['feature']}≤{node['threshold']:.2f}")
        preorder(node["left"], out); preorder(node["right"], out)
    else:
        out.append(f"leaf:{node['value']}")

def inorder(node, out):
    if "feature" in node:
        inorder(node["left"], out)
        out.append(f"{node['feature']}≤{node['threshold']:.2f}")
        inorder(node["right"], out)
    else:
        out.append(f"leaf:{node['value']}")

def postorder(node, out):
    if "feature" in node:
        postorder(node["left"], out); postorder(node["right"], out)
        out.append(f"{node['feature']}≤{node['threshold']:.2f}")
    else:
        out.append(f"leaf:{node['value']}")

pre, inord, post = [], [], []
preorder(root, pre); inorder(root, inord); postorder(root, post)

st.subheader("🌳 트리 순회 결과")
st.markdown(f"- **전위 순회**: {pre}")
st.markdown(f"- **중위 순회**: {inord}")
st.markdown(f"- **후위 순회**: {post}")

# --- 기존 사용자 입력 예측 폼 ---

st.subheader("🧪 내 증상으로 당뇨병 예측해보기")
with st.form("predict_form"):
    age = st.slider("나이", 10, 100, 45)
    gender = st.radio("성별", ["남성", "여성"])
    input_data = {"Age": age, "Gender": 1 if gender == "남성" else 0}
    for col in X.columns:
        if col not in ["Age", "Gender"]:
            input_data[col] = st.radio(f"{col}", ["아님", "있음"]) == "있음"
    submitted = st.form_submit_button("예측하기")

if submitted:
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]
    if prediction == 1:
        st.error(f"⚠️ 당뇨병 위험 있음 (예측 확률 {prob*100:.2f}%)")
    else:
        st.success(f"✅ 당뇨병 위험 낮음 (예측 확률 {prob*100:.2f}%)")
