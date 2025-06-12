import streamlit as st
import pandas as pd
import plotly.express as px
import heapq

st.set_page_config(page_title="울산 에너지 소비 분석", layout="wide")
st.title("⚡ 울산 에너지 소비 분석 및 최적 에너지 믹스 추천 시스템")

# 파일 업로드 함수
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding='cp949')

# 업로드
uploaded_file = st.file_uploader("📎 CSV 파일 업로드 (공공데이터포털에서 받은 울산 에너지 소비 파일)", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file)

    # 울산만 필터링
    df = df[df["지역"] == "울산"]

    # 에너지원 매핑
    energy_columns = {
        "석탄": "석탄사용량(천토)",
        "석유": "석유사용량(천토)",
        "가스": "천연 및 도시가스사용량(천토)",
        "전력": "전력사용량(천토)",
        "열에너지": "열에너지사용량(천토)",
        "신재생": "신재생사용량(천토)"
    }

    # 연도 선택
    years = df["연도"].unique()
    year = st.selectbox("📅 분석할 연도 선택", sorted(years))
    row = df[df["연도"] == year].iloc[0]

    # 트리 구조 출력
    energy_tree = {
        "화석에너지": {
            "석탄": row[energy_columns["석탄"]],
            "석유": row[energy_columns["석유"]],
            "가스": row[energy_columns["가스"]],
        },
        "기타에너지": {
            "전력": row[energy_columns["전력"]],
            "열에너지": row[energy_columns["열에너지"]],
            "신재생": row[energy_columns["신재생"]],
        }
    }

    st.subheader("🌲 울산시 에너지 소비 트리 (천톤 기준)")
    st.json(energy_tree)

    # 정렬
    consumption_list = [(k, row[v]) for k, v in energy_columns.items()]
    sorted_energy = sorted(consumption_list, key=lambda x: x[1] if pd.notnull(x[1]) else 0, reverse=True)

    st.subheader("📊 소비량 정렬 (내림차순)")
    for i, (k, v) in enumerate(sorted_energy, 1):
        st.write(f"{i}. {k}: {v} 천톤")

    # 이진 탐색
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid][1] == target:
                return mid
            elif arr[mid][1] < target:
                high = mid - 1
            else:
                low = mid + 1
        return low

    st.subheader("🔍 기준 소비량으로 탐색")
    target = st.number_input("기준 소비량 입력 (천톤)", min_value=0.0, value=sorted_energy[0][1])
    idx = binary_search(sorted_energy, target)
    st.info(f"기준과 가장 가까운 소비량: {sorted_energy[idx][0]} → {sorted_energy[idx][1]} 천톤")

    # 힙 기반 최대 소비
    heap = [(-v if pd.notnull(v) else 0, k) for k, v in consumption_list]
    heapq.heapify(heap)
    top = heapq.heappop(heap)
    st.success(f"🔥 최다 소비 에너지원: {top[1]} → {-top[0]} 천톤")

    # 시계열 그래프
    df_long = df.melt(id_vars="연도", value_vars=list(energy_columns.values()),
                      var_name="에너지원", value_name="소비량")
    df_long["에너지원"] = df_long["에너지원"].replace({v: k for k, v in energy_columns.items()})

    st.subheader("📈 연도별 에너지원 소비 추이")
    fig = px.line(df_long, x="연도", y="소비량", color="에너지원", markers=True,
                  title="연도별 울산시 에너지원별 소비량")
    st.plotly_chart(fig)

else:
    st.warning("❗ 먼저 울산광역시 에너지 소비 CSV 파일을 업로드해주세요.")
