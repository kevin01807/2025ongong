import streamlit as st
import plotly.graph_objects as go

# 페이지 설정이요
st.set_page_config(page_title="자료구조 시각화", page_icon="📊")

st.title("📚 자료구조 시각화: 스택과 큐")
st.write("아래에서 자료구조를 선택하면 시각적으로 개념을 이해할 수 있어요!")

# 선택
structure = st.sidebar.selectbox("자료구조 선택", ["스택 (Stack)", "큐 (Queue)"])

# 공통 요소
elements = ["A", "B", "C", "D"]

def draw_stack(elements):
    fig = go.Figure()
    for i, val in enumerate(reversed(elements)):
        fig.add_shape(
            type="rect",
            x0=0, x1=2,
            y0=i, y1=i+1,
            line=dict(color="blue"),
            fillcolor="lightblue"
        )
        fig.add_trace(go.Scatter(
            x=[1],
            y=[i + 0.5],
            text=[val],
            mode="text",
            showlegend=False
        ))
    fig.update_layout(
        height=400,
        width=300,
        title="📦 스택 구조 (LIFO)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def draw_queue(elements):
    fig = go.Figure()
    for i, val in enumerate(elements):
        fig.add_shape(
            type="rect",
            x0=i, x1=i+1,
            y0=0, y1=1,
            line=dict(color="green"),
            fillcolor="lightgreen"
        )
        fig.add_trace(go.Scatter(
            x=[i + 0.5],
            y=[0.5],
            text=[val],
            mode="text",
            showlegend=False
        ))
    fig.update_layout(
        height=300,
        width=500,
        title="🚶 큐 구조 (FIFO)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

if structure.startswith("스택"):
    st.subheader("🧱 스택 (Stack)")
    st.write("""
    - **정의**: 스택은 LIFO (Last-In, First-Out) 구조로, 나중에 들어온 데이터가 먼저 나가요.
    - **주요 연산**:
        - `push()`: 데이터를 스택 위에 추가
        - `pop()`: 스택의 가장 위 데이터를 제거하고 반환
    - **활용 예시**:
        - 함수 호출 스택
        - 웹 브라우저의 뒤로 가기 기록
    """)
    st.plotly_chart(draw_stack(elements))

elif structure.startswith("큐"):
    st.subheader("🚏 큐 (Queue)")
    st.write("""
    - **정의**: 큐는 FIFO (First-In, First-Out) 구조로, 먼저 들어온 데이터가 먼저 나가요.
    - **주요 연산**:
        - `enqueue()`: 데이터를 큐의 뒤에 추가
        - `dequeue()`: 큐의 앞에서 데이터를 제거하고 반환
    - **활용 예시**:
        - 프린터 작업 처리
        - 운영체제의 프로세스 스케줄링
    """)
    st.plotly_chart(draw_queue(elements))

st.markdown("---")
st.caption("자료구조는 알고리즘을 이해하는 핵심입니다. 시각화를 통해 더 쉽게 배워보세요!")
