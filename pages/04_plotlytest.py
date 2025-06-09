import pandas as pd
import streamlit as st
import plotly.express as px

# CSV 파일 불러오기 (CP949 인코딩)
df_sum = pd.read_csv("people_sum.csv", encoding="cp949")
df_gender = pd.read_csv("people_gender.csv", encoding="cp949")

# Streamlit 앱 기본 설정
st.set_page_config(page_title="인구 시각화 대시보드", layout="wide")
st.title("📊 지역별 인구 피라미드 시각화")

# 지역 선택
region = st.selectbox("지역을 선택하세요", df_sum["행정구역"].unique())

# 연령 슬라이더
age_range = st.slider("연령대 범위 선택", 0, 100, (0, 100))

# 해당 지역 필터링
df_selected = df_gender[df_gender["행정구역"] == region]

# 연령별 컬럼 필터링
age_columns_male = [col for col in df_selected.columns if "남_" in col and "세" in col]
age_columns_female = [col for col in df_selected.columns if "여_" in col and "세" in col]

# 연령 필터 함수
def filter_ages(cols, age_range):
    result = []
    for col in cols:
        age = col.split("_")[-1].replace("세", "").replace("이상", "")
        if age.isdigit():
            age = int(age)
            if age_range[0] <= age <= age_range[1]:
                result.append(col)
    return result

filtered_male_cols = filter_ages(age_columns_male, age_range)
filtered_female_cols = filter_ages(age_columns_female, age_range)

# 문자열 숫자 안전 변환
def parse_number(val):
    try:
        return int(str(val).replace(",", ""))
    except:
        return 0

# 데이터가 없을 경우 예외 처리
if len(filtered_male_cols) == 0 or len(filtered_female_cols) == 0:
    st.warning("해당 연령 구간에 데이터가 없습니다. 다른 연령대를 선택해주세요.")
else:
    # 인구 수 시리즈 생성
    male_series = df_selected[filtered_male_cols].iloc[0].apply(parse_number)
    female_series = df_selected[filtered_female_cols].iloc[0].apply(parse_number)

    # 연령 라벨
    male_ages = [col.split("_")[-1] for col in filtered_male_cols]
    female_ages = [col.split("_")[-1] for col in filtered_female_cols]

    # 최소 길이에 맞추기
    min_len = min(len(male_ages), len(female_ages), len(male_series), len(female_series))
    if min_len == 0:
        st.warning("선택한 연령대에 유효한 인구 데이터가 없습니다.")
    else:
        age_labels = male_ages[:min_len]
        male_counts = male_series[:min_len] * -1
        female_counts = female_series[:min_len]

        # 데이터프레임 구성
        df_plot = pd.DataFrame({
            "연령": age_labels,
            "남성": male_counts,
            "여성": female_counts
        })

        # Melt 형식으로 변환
        df_melted = df_plot.melt(id_vars="연령", var_name="성별", value_name="인구수")

        # Plotly 시각화
        fig = px.bar(
            df_melted,
            x="인구수",
            y="연령",
            color="성별",
            orientation="h",
            title=f"{region} 인구 피라미드 (연령대: {age_range[0]}세 ~ {age_range[1]}세)",
            height=700
        )

        # 시각화 출력
        st.plotly_chart(fig, use_container_width=True)
