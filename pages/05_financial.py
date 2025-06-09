import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# 기본 설정
st.set_page_config(page_title="Top10 주가 분석", layout="wide")
st.title("🌍 글로벌 시가총액 Top10 기업 주가 분석")

# 기업 목록 및 티커
company_dict = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Berkshire Hathaway": "BRK-B",
    "TSMC": "TSM",
    "Eli Lilly": "LLY",
    "Tesla": "TSLA"
}

company_names = list(company_dict.keys())
selected_names = st.multiselect("✅ 기업 선택", company_names, default=company_names[:5])
selected_tickers = [company_dict[name] for name in selected_names]

# 날짜 설정
today = datetime.today()
one_year_ago = today - timedelta(days=365)

# 데이터 불러오기
with st.spinner("📥 데이터를 불러오는 중입니다..."):
    data = yf.download(selected_tickers, start=one_year_ago, end=today)

# 데이터 정리
if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.colum
