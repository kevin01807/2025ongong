import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'

def clean_unicode(text):
    return ''.join(c for c in str(text) if c.isprintable())

st.set_page_config(page_title=clean_unicode("ICT 역량 분류 및 격차 분석"), layout="wide")

@st.cache_data
def load_data():
    base_dir = os.getcwd()
    main_file = os.path.join(base_dir, "data", "4-4-1.csv")
    queue_file = os.path.join(base_dir, "data", "queue_data.csv")
    stack_file = os.path.join(base_dir, "data", "stack_data.csv")
    gap_file = os.path.join(base_dir, "data", "일반국민_대비_취약계층_디지털정보화종합수준_20250620115549.csv")

    df = pd.read_csv(main_file, encoding="utf-8")
    queue_df = pd.read_csv(queue_file, encoding="utf-8")
    stack_df = pd.read_csv(stack_file, encoding="utf-8")
    gap_df = pd.read_csv(gap_file, encoding="cp949")

    return df, queue_df, stack_df, gap_df

df, queue_df, stack_df, gap_df = load_data()

st.title(clean_unicode("ICT 역량 분류 및 격차 분석"))
st.header(clean_unicode("기술 유형별 ICT 활용 격차"))

if '기술유형' in df.columns and '성별' in df.columns and 'Year' in df.columns and 'Value' in df.columns:
    df['기술유형'] = df['기술유형'].astype(str)
    selected_skill = st.selectbox("기술을 선택하세요", df['기술유형'].unique())
    filtered = df[df['기술유형'] == selected_skill]

    if not filtered.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered, x='Year', y='Value', hue='성별', ax=ax)
        ax.set_title(clean_unicode(f"{selected_skill} 기술 활용도 (성별 비교)"))
        st.pyplot(fig)
    else:
        st.warning("선택한 기술에 해당하는 데이터가 없습니다.")
else:
    st.error("데이터셋에 필요한 컬럼이 없습니다.")

# ----------------------
st.subheader("나이브 베이즈 분류기를 활용한 예측")
try:
    numeric_df = df[['Year', 'Value', '성별', '기술유형']].copy()
    numeric_df['Gender_Code'] = numeric_df['성별'].map({'남자': 0, '여자': 1, '전체': 2})
    numeric_df['Skill_Code'] = numeric_df['기술유형'].astype('category').cat.codes

    imputer = SimpleImputer(strategy='mean')
    X = numeric_df[['Year', 'Gender_Code', 'Skill_Code']]
    X_imputed = imputer.fit_transform(X)
    y = numeric_df['Value'] > numeric_df['Value'].mean()

    if len(X_imputed) < 10:
        st.warning("📉 학습에 사용할 데이터가 부족합니다.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.text("📌 나이브 베이즈 분류 보고서")
        st.text(classification_report(y_test, y_pred))
except Exception as e:
    st.error(f"나이브 베이즈 실행 중 오류 발생: {e}")

st.markdown("#### 📊 나이브 베이즈 분류기의 주제 연계 설명")
st.markdown("""
ICT 기술 활용 격차를 줄이기 위해서는, 어느 **기술 유형**이 **여성** 또는 **연도**에서 상대적으로 활용도가 낮은지 빠른 파악이 중요합니다.

나이브 베이즈 분류기를 활용하면 주어진 데이터(성별, 연도, 기술유형)를 기반으로 **ICT 활용도가 평균 이상인지 여부**를 예측할 수 있습니다.
이를 통해 정책 입안자는 특정 그룹(예: 2018년 여성의 'ub514지털 기초 기술')의 활용 격차를 조금이라도 비정보하고, **선제적인 교육 자원 배분 또는 지원 정책**을 구설할 수 있습니다.

또한, 이 분류기는 데이터의 누락(NaN)이나 소수의 허리에도 **단순하고 빠른 의상결정**을 할 수 있는 기반을 제공함으로써 **ICT 역량의 공정한 분배와 격차 해소**에 기여할 수 있습니다.
""")

# -------------------
# 3. 큐/스택 시뮬레이션 (수정된 부분)
# -------------------
st.subheader("자료구조 시뮬레이션: 큐와 스택")
st.markdown("#### 📥 ICT 요청 처리 구조: Queue(선착순) vs Stack(긴급처리)")

# 새로운 데이터 로드
df_vul = pd.read_csv("data/일반국민_대비_취약계층_디지털정보화종합수준_20250620115549.csv", encoding="cp949")

# 연도 데이터만 숫자로 추출
years = [col for col in df_vul.columns if col != '계층별']
df_vul['2023점수'] = df_vul['2023']
df_vul['점수증가량'] = df_vul['2023'] - df_vul['2015']

# 큐: 점수가 낮은 순으로 선착순 처리
queue_sorted = df_vul.sort_values(by='2023점수').reset_index(drop=True)
stack_sorted = df_vul.sort_values(by='점수증가량').reset_index(drop=True)

tab1, tab2 = st.tabs(["📥 큐 (Queue)", "📦 스택 (Stack)"])

with tab1:
    st.dataframe(queue_sorted[['계층별', '2023점수']])
    st.markdown("""
    **Queue(선착순 ICT 지원 정책 시뮬레이션)**  
    매년 정보화 수준이 낮은 계층을 먼저 지원하는 방식입니다.  
    연도별로 점수가 낮은 계층을 우선 처리 → 선형 구조로 순차 지원.  
    예: 고령층 → 농어민 → 장애인 순으로 디지털 교육 예산 배분.
    """)

with tab2:
    st.dataframe(stack_sorted[['계층별', '점수증가량']])
    st.markdown("""
    **Stack(최근 정보 격차 악화 계층 우선 개입)**  
    최근 연도(2023)에 격차가 급격히 커진 계층을 우선 처리합니다.  
    후입선출 구조에 따라, 최근 악화된 계층부터 긴급 대응.  
    예: 최근 점수 상승 폭이 정체된 고령층을 가장 먼저 투입.
    """)


