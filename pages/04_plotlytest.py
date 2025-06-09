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

# 연령 범위 선택 슬라이더
age_range = st.slider("연령대 범위 선택", 0, 100, (0, 100))

# 선택한 지역 데이터 필터링
df_selected = df_gender[df_gender["행정구역"] == region]

# 남/여 연령별 컬럼 추출
age_columns_male = [col for col in df_selected.columns if "남_" in col and "세" in col]
age_columns_female = [col for col in df_selected.columns if "여_" in col and "세" in col]

# 연령 필터링 함수
def filter_ages(cols, age_range):
    filtered = []
    for col in cols:
        age = col.split("_")[-1].replace("세", "").replace("이상", "")
        if age.isdigit():
            age = int(age)
            if age_range[0] <= age <= age_range[1]:
                filtered.append(col)
    return filtered

# 연령 필터 적용
filtered_male_cols = filter_ages(age_columns_male, age_range)
filtered_female_cols = filter_ages(age_columns_female, age_range)

# 문자열 숫자 처리 함수
def parse_number(val):
    try:
        return int(str(val).replace(",", ""))
    except:
        return 0

# 데이터 전처리
male_counts = df_selected[filtered_male_cols].iloc[0].apply(parse_number) * -1
female_counts = df_selected[filtered_female_cols].iloc[0].apply(parse_number)
age_labels = [col.split("_")[-1] for col in filtered_male_cols]

# 리스트 길이 정렬 (불일치 방지)
min_len = min(len(age_labels), len(male_counts), len(female_counts))
age_labels = age_labels[:min_len]
male_counts = male_counts[:min_len]
female_counts = female_counts[:min_len]

# 시각화용 데이터프레임 생성
df_plot = pd.DataFrame({
    "연령": age_labels,
    "남성": male_counts,
    "여성": female_counts
})

# Long-form 변환
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

# 차트 출력
st.plotly_chart(fig, use_container_width=True)
