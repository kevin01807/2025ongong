import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import deque

# 1. 기본 설정
st.title("ICT 역량 분류 및 격차 분석")
base_dir = "/mnt/data"
main_data_path = os.path.join(base_dir, "4-4-1.csv")
queue_path = os.path.join(base_dir, "queue_data.csv")
stack_path = os.path.join(base_dir, "stack_data.csv")

# 2. 데이터 불러오기
df = pd.read_csv(main_data_path)
queue_df = pd.read_csv(queue_path)
stack_df = pd.read_csv(stack_path)

st.write(f"데이터 경로 확인: {main_data_path}")

# 3. 기술 유형 선택
st.header("기술 유형별 ICT 활용 격차")
skill_types = df["기술유형"].unique().tolist()
selected_skill = st.selectbox("기술을 선택하세요", skill_types)

# 4. 필터링 및 시각화
filtered = df[df["기술유형"] == selected_skill]
if filtered.empty:
    st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        sns.barplot(data=filtered, x="Year", y="Value", hue="성별", ax=ax)
        ax.set_title(f"{selected_skill} 기술 활용도 (성별 비교)")
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"시각화 중 오류 발생: {e}")

# 5. 큐 시뮬레이션
st.header("ICT 접근 대기열 시뮬레이션 (Queue)")
try:
    queue_items = queue_df["Skill"].dropna().tolist()
    q = deque(queue_items)
    st.markdown(f"**초기 대기열:** {list(q)}")
    if st.button("ICT 기술 1건 처리 (Dequeue)"):
        if q:
            removed = q.popleft()
            st.success(f"처리 완료: {removed}")
        else:
            st.warning("대기열이 비어 있습니다.")
    st.markdown(f"**현재 대기열:** {list(q)}")
except Exception as e:
    st.error(f"큐 처리 중 오류 발생: {e}")

# 6. 스택 시뮬레이션
st.header("ICT 학습 이력 시뮬레이션 (Stack)")
try:
    stack_items = stack_df["Skill"].dropna().tolist()
    s = list(stack_items)
    st.markdown(f"**초기 학습 이력:** {s}")
    if st.button("최근 학습 기술 제거 (Pop)"):
        if s:
            popped = s.pop()
            st.success(f"제거된 기술: {popped}")
        else:
            st.warning("스택이 비어 있습니다.")
    st.markdown(f"**현재 학습 이력:** {s}")
except Exception as e:
    st.error(f"스택 처리 중 오류 발생: {e}")

# 7. 기술 우선순위 정렬 (Value 기준)
st.header("ICT 기술 우선순위 정렬")
try:
    sort_data = df[df["성별"] == "전체"].groupby("기술유형")["Value"].mean().reset_index()
    sorted_df = sort_data.sort_values(by="Value", ascending=False)
    st.dataframe(sorted_df)
    st.markdown("※ Value 기준 평균 ICT 활용도가 높은 순으로 기술을 정렬한 결과입니다.")
except Exception as e:
    st.error(f"정렬 중 오류 발생: {e}")
