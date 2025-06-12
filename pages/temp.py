# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import heapq, os, re

st.set_page_config(page_title="울산 에너지 소비 분석", layout="wide")
st.title("🔋 울산광역시 연도별 에너지원 소비 분석 · 최적 믹스 추천")

# ------------------------------------------------------------------
# 1) 기본 데이터 경로 (동일 폴더에 있을 때 자동 로드)
DEFAULT_PATH = "울산광역시_울산광역시_연도별 에너지원별 소비현황_20240726.csv"

def load_csv(path: str) -> pd.DataFrame:
    """CSV 로드 + 컬럼명 정규화('태양광(kWh)' → '태양광')"""
    df = pd.read_csv(path)
    # 한글+영문+숫자 이외 문자 제거
    df.columns = [re.sub(r"[^\w]", "", col) for col in df.columns]
    return df

df = None
if os.path.exists(DEFAULT_PATH):
    try:
        df = load_csv(DEFAULT_PATH)
        st.success(f"✅ 기본 데이터 파일 자동 로드: {DEFAULT_PATH}")
    except Exception as e:
        st.warning(f"기본 파일 로드 실패: {e}")

# ------------------------------------------------------------------
# 2) 사용자 파일 업로드 (옵션)
uploaded = st.file_uploader("📂 CSV 파일을 업로드하세요 (선택)", type="csv")
if uploaded:
    df = load_csv(uploaded)
    st.success("✅ 업로드한 파일을 사용합니다.")

# ------------------------------------------------------------------
# 3) 데이터 확인 & 분석 로직
if df is None:
    st.error("CSV 파일을 찾을 수 없습니다. 기본 파일 위치를 확인하거나 업로드해 주세요.")
    st.stop()

# 연도 목록 & 선택
years = sorted(df['연도'].unique())
year = st.selectbox("분석할 연도 선택", years)
row = df[df['연도'] == year].iloc[0]

# ---------------- 트리(계층 구조) ----------------
energy_tree = {
    '화석연료': {
        '석탄': row['석탄'],
        '석유': row['석유'],
        '가스': row['가스']
    },
    '재생에너지': {
        '태양광': row['태양광'],
        '풍력': row['풍력'],
        '기타': row['기타']
    }
}
st.subheader("🌲 에너지 소비 계층 트리")
st.json(energy_tree)

# ---------------- 정렬 ----------------
amounts = [(k, row[k]) for k in ['석탄','석유','가스','태양광','풍력','기타']]
sorted_amts = sorted(amounts, key=lambda x: x[1], reverse=True)

st.subheader("🔢 소비량 순위(내림차순)")
for i, (k, v) in enumerate(sorted_amts, 1):
    st.write(f"{i}. **{k}**: {v:,.0f}")

# ---------------- 이진 탐색 ----------------
def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high)//2
        if arr[mid][1] == target:
            return mid
        elif arr[mid][1] < target:
            high = mid - 1
        else:
            low = mid + 1
    return low                 # 가장 근접한 상한 index

default_target = float(sorted_amts[0][1])
target_val = st.number_input("🔍 기준 소비량 입력", value=default_target, step=100.0)
idx = binary_search(sorted_amts, target_val)
st.info(f"기준값과 가장 가까운 에너지원 → **{sorted_amts[idx][0]}** ({sorted_amts[idx][1]:,.0f})")

# ---------------- 힙(최대 소비) ----------------
max_heap = [(-val, key) for key, val in amounts]
heapq.heapify(max_heap)
top_val, top_key = heapq.heappop(max_heap)
st.success(f"💡 가장 많이 소비한 에너지원: **{top_key}** ({-top_val:,.0f})")

# ---------------- 시계열 시각화 ----------------
st.subheader("📈 연도별 에너지원 소비 추이")
df_long = df.melt(id_vars='연도', var_name='에너지원', value_name='소비량')
fig = px.line(df_long, x='연도', y='소비량', color='에너지원',
              markers=True, title="울산광역시 에너지원별 소비 변화")
st.plotly_chart(fig, use_container_width=True)

# ---------------- 데이터 테이블 (옵션) ----------------
with st.expander("🔍 원본 데이터 확인"):
    st.dataframe(df)
