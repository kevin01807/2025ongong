# app.py
import pandas as pd
import streamlit as st
import plotly.express as px

# CP949로 인코딩된 CSV 파일 불러오기
df_sum = pd.read_csv("people_sum.csv", encoding="cp949")
df_gender = pd.read_csv("people_gender.csv", encoding="cp949")

# Streamlit 앱 설정
st.set_page_config(page_title="인구 시각화 대시보드", layout="wide")

# 지역 선택
region = st.selectbox("지역을 선택하세요", df_sum["행정구역"].unique())

# 연령대 슬라이더 (0세 ~ 100세 이상)
age_range = st.slider("연령대 범위 선택", 0, 100, (0, 100))

# 선택한 지역 필터링
df_selected = df_gender[df_gender["행정구역"] == region]

# 연령별 컬럼 구분
age_columns_male = [col for col in df_selected.columns if "남_" in col and "세" in col]
age_columns_female = [col for col in df_selected.columns if "여_" in col and "세" in col]

# 필터링 함수
def filter_ages(cols, age_range):
    filtered = []
    for col in cols:
        age = col.split("_")[-1].replace("세", "").replace("이상", "")
        if age.isdigit():
            age = int(age)
            if age_range[0] <= age <= age_range[1]:
                filtered.append(col)
    return filtered

# 적용
filtered_male_cols = filter_ages(age_columns_male, age_range)
filtered_female_cols = filter_ages(age_columns_female, age_range)

# 데이터 정제
age_labels = [col.split("_")[-1] for col in filtered_male_cols]
male_counts = df_selected[filtered_male_cols].iloc[0].str.replace(",", "").astype(int) * -1
female_counts = df_selected[filtered_female_cols].iloc[0].str.replace(",", "").astype(int)

# 시각화용 데이터
df_plot = pd.DataFrame({
    "연령": age_labels,
    "남성": male_counts,
    "여성": female_counts
})

df_melted = df_plot.melt(id_vars="연령", var_name="성별", value_name="인구수")

# Plotly 인구 피라미드
fig = px.bar(
    df_melted,
    x="인구수",
    y="연령",
    color="성별",
    orientation="h",
    title=f"{region} 인구 피라미드 (연령대: {age_range[0]}세 ~ {age_range[1]}세)",
    height=700
)

# 표시
st.plotly_chart(fig, use_container_width=True)
