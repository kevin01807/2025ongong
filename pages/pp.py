# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# 샤논 엔트로피 계산 함수
def shannon_entropy(data):
    p_data = data.value_counts(normalize=True)
    return -np.sum(p_data * np.log2(p_data + 1e-9))

# 변분법 최적화 함수
def energy_distribution_cost(path):
    return np.sum(np.diff(path)**2)

def optimal_path(start, end, num_points=10):
    initial_path = np.linspace(start, end, num_points)
    result = minimize(energy_distribution_cost, initial_path, method='L-BFGS-B')
    return result.x

# Streamlit 앱
def main():
    st.title("⚡ SDGs 기반 지역 전력 분석과 최적 배전 시뮬레이션")
    st.caption("주제: 지역 간 전력 소비 예측 불확실성과 배전 경로 최적화 (SDGs 7.1 + 9.4)")

    # 파일 업로드
    usage_file = st.file_uploader("📂 지역별 전력사용량 데이터 업로드", type=["csv"])
    weather_file = st.file_uploader("📂 기상 데이터 업로드", type=["csv"])
    sdgs_file = st.file_uploader("📂 SDGs 7.1.1 지표 업로드", type=["csv"])

    if usage_file and weather_file and sdgs_file:
        usage_df = pd.read_csv(usage_file, encoding='utf-8')
        weather_df = pd.read_csv(weather_file, encoding='utf-8')
        sdgs_df = pd.read_csv(sdgs_file, encoding='utf-8')

        st.subheader("① 샤논 엔트로피 기반 지역 전력 불확실성 분석")
        region = st.selectbox("지역 선택", usage_df['지역'].unique())
        selected = usage_df[usage_df['지역'] == region]
        ent = shannon_entropy(selected['전력사용량(합계)'])
        st.metric("샤논 엔트로피 (정보량)", f"{ent:.3f}")
        st.line_chart(selected[['전력사용량(합계)']])

        st.subheader("② 기온에 따른 전력 사용량 분석")
        merged = pd.merge(weather_df, usage_df, left_on='시군구명', right_on='지역', how='inner')
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=merged['평균기온'], y=merged['전력사용량(합계)'], hue=merged['지역'], ax=ax1)
        ax1.set_xlabel("평균기온(℃)")
        ax1.set_ylabel("전력사용량")
        ax1.set_title("기온 vs 전력사용량")
        st.pyplot(fig1)

        st.subheader("③ 변분법 기반 최적 배전 경로 시뮬레이션")
        start = st.slider("시작 부하 (kWh)", 0.0, 100.0, 10.0)
        end = st.slider("종료 부하 (kWh)", 0.0, 100.0, 90.0)
        points = st.slider("경로 내 노드 수", 5, 30, 10)
        path = optimal_path(start, end, points)
        st.line_chart(path)

        st.subheader("④ SDGs 7.1.1 지표 시각화")
        sdgs_df.columns = ['국가', '연도', '보급률']
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=sdgs_df, x='연도', y='보급률', hue='국가', ax=ax2)
        ax2.set_title("전력 접근성 보급률 추이")
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
