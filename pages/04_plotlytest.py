import pandas as pd
import streamlit as st
import plotly.express as px

# 데이터 불러오기
df_sum = pd.read_csv("people_sum.csv", encoding="cp949")
df_gender = pd.read_csv("people_gender.csv", encoding="cp949")

st.set_page_config(page_title="인구 시각화 대시보드", layout="wide")
st.title("📊 지역별 인구 피라미드 시각화")

# 사용자 입력
region = st.selectbox("지역을 선택하세요", df_sum["행정구역"].unique())
age_range = st.slider("연령대 범위 선택", 0, 100, (0, 100))
df_selected = df_gender[df_gender["행정구역"] == region]

# 연령 필터링
def filter_ages(cols, age_range):
    result = []
    for col in cols:
        age = col.split("_")[-1].replace("세", "").replace("이상", "")
        if age.isdigit():
            age = int(age)
            if age_range[0] <= age <= age_range[1]:
                result.append(col)
    return result

male_cols = filter_ages([col for col in df_selected.columns if "남_" in col and "세" in col], age_range)
female_cols = filter_ages([col for col in df_selected.columns if "여_" in col and "세" in col], age_range)

# 숫자 파싱 함수
def parse_number(val):
    try:
        return int(str(val).replace(",", ""))
    except:
        return 0

# 데이터 없을 때 처리
if not male_cols or not female_cols:
    st.warning("해당 연령대에 유효한 데이터가 없습니다.")
    st.stop()

# 값 변환
male_series = df_selected[male_cols].iloc[0].apply(parse_number)
female_series = df_selected[female_cols].iloc[0].apply(parse_number)
age_labels_male = [col.split("_")[-1] for col in male_cols]
age_labels_female = [col.split("_")[-1] for col in female_cols]

# 최소 길이
min_len = min(len(male_series), len(female_series), len(age_labels_male), len(age_labels_female))
if min_len == 0:
    st.warning("데이터가 부족합니다.")
    st.stop()

# 안전하게 자르기
age_labels = list(age_labels_male)[:min_len]
male_counts = list(male_series)[:min_len]
female_counts = list(female_series)[:min_len]

# NaN 방지
if any(pd.isnull(age_labels)) or any(pd.isnull(male_counts)) or any(pd.isnull(female_counts)):
    st.warning("누락된 값이 존재합니다. 다른 조건을 선택해 주세요.")
    st.stop()

# 인구 피라미드용 데이터프레임
df_plot = pd.DataFrame({
    "연령": age_labels,
    "남성": [-x for x in male_counts],
    "여성": female_counts
})
df_melted = df_plot.melt(id_vars="연령", var_name="성별", value_name="인구수")

# 시각화
fig = px.bar(
    df_melted,
    x="인구수",
    y="연령",
    color="성별",
    orientation="h",
    title=f"{region} 인구 피라미드 (연령대: {age_range[0]}세 ~ {age_range[1]}세)",
    height=700
)

st.plotly_chart(fig, use_container_width=True)
