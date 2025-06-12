import streamlit as st
import pandas as pd
import plotly.express as px
import heapq

st.set_page_config(page_title="울산 에너지 소비 분석", layout="wide")
st.title("⚡ 울산 에너지 소비 분석 및 최적 에너지 믹스 추천")

# ✅ CSV 불러오기 함수
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding='cp949')
    except pd.errors.EmptyDataError:
        st.error("❌ CSV 파일에 읽을 수 있는 데이터가 없습니다.")
        return None

# ✅ 파일 업로드
uploaded_file = st.file_uploader("📎 울산 에너지 소비 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file)

    if df is not None:
        # ✅ 울산시 데이터 필터링
        df = df[df["지역"] == "울산"].copy()

        # ✅ 사용할 에너지원 매핑
        energy_columns = {
            "석탄": "석탄사용량(천토)",
            "석유": "석유사용량(천토)",
            "가스": "천연 및 도시가스사용량(천토)",
            "전력": "전력사용량(천토)",
            "열에너지": "열에너지사용량(천토)",
            "신재생": "신재생사용량(천토)"
        }

        # ✅ 연도 선택
        year = st.selectbox("🔎 분석할 연도 선택", sorted(df["연도"].unique()))
        row = df[df["연도"] == year].iloc[0]

        # ✅ 트리 구조 출력
        tree = {
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
        st.subheader("🌲 에너지 소비 트리 구조 (천톤 기준)")
        st.json(tree)

        # ✅ 정렬
        sorted_data = sorted(
            [(k, row[v]) for k, v in energy_columns.items()],
            key=lambda x: x[1] if pd.notnull(x[1]) else 0,
            reverse=True
        )
        st.subheader("📊 에너지원 소비량 정렬 (내림차순)")
        for i, (k, v) in enumerate(sorted_data, 1):
            st.write(f"{i}. {k}: {v} 천톤")

        # ✅ 기준 소비량 입력 및 이진 탐색
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
        target = st.number_input("기준 소비량 입력 (천톤)", min_value=0.0, value=sorted_data[0][1])
        idx = binary_search(sorted_data, target)
        st.info(f"입력값과 가장 가까운 에너지원: {sorted_data[idx][0]} → {sorted_data[idx][1]} 천톤")

        # ✅ 힙으로 최다 소비 에너지원 탐색
        heap = [(-v if pd.notnull(v) else 0, k) for k, v in [(k, row[v]) for k, v in energy_columns.items()]]
        heapq.heapify(heap)
        top = heapq.heappop(heap)
        st.success(f"🔥 최다 소비 에너지원: {top[1]} ({-top[0]} 천톤)")

        # ✅ 시계열 그래프 (연도별 전체 소비 추이)
        df_long = df.melt(id_vars="연도", value_vars=list(energy_columns.values()),
                          var_name="에너지원", value_name="소비량")
        df_long["에너지원"] = df_long["에너지원"].replace({v: k for k, v in energy_columns.items()})

        st.subheader("📈 연도별 울산시 에너지원 소비 추이")
        fig = px.line(df_long, x="연도", y="소비량", color="에너지원", markers=True)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("📌 울산광역시 에너지 소비 CSV 파일을 먼저 업로드해주세요.")
