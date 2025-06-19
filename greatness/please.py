import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import deque
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'

# 유니코드 깨짐 방지용 텍스트 정리 함수
def clean_unicode(text):
    return ''.join(c for c in str(text) if c.isprintable() and (ord(c) < 55296 or ord(c) > 57343))

st.set_page_config(page_title=clean_unicode("ICT 역량 분류 및 격차 분석"), layout="wide")

@st.cache_data
def load_data():
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "4-4-1.csv")
    st.write("데이터 경로 확인:", file_path)
    df = pd.read_csv(file_path, encoding="utf-8")

    df.rename(columns={
        '기술유형': 'Skill_Type',
        '성별': 'Gender',
        'Year': 'Year',
        'Value': 'Value'
    }, inplace=True)

    skill_map = {
        'ARSP': '문서 편집',
        'EMAIL': '이메일 사용',
        'COPY': '파일 복사',
        'SEND': '파일 전송',
        'INST': 'SW 설치',
        'COMM': '온라인 커뮤니케이션',
        'BUY': '온라인 구매',
        'BANK': '온라인 뱅킹',
        'USEC': '보안 설정'
    }
    df['Skill_KR'] = df['Skill_Type'].map(skill_map)
    df['Gender'] = df['Gender'].fillna('전체')
    return df

df = load_data()

st.title(clean_unicode("ICT 역량 분류 및 격차 분석"))
st.header(clean_unicode("기술 유형별 ICT 활용 격차"))

selected_skill = st.selectbox("기술을 선택하세요", df['Skill_KR'].dropna().unique())
filtered = df[df['Skill_KR'] == selected_skill].dropna(subset=['Year', 'Value', 'Gender'])

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=filtered, x='Year', y='Value', hue='Gender', ax=ax)
    ax.set_title(clean_unicode(f"{selected_skill} 기술 활용도 (성별 비교)"))
    st.pyplot(fig)
except ValueError as e:
    st.error(f"시각화 중 오류 발생: {e}")
