import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="📈 글로벌 주식 트렌드", layout="wide")

st.title("📈 글로벌 시가총액 TOP10 기업 주가 추이")
st.markdown("💹 **최근 1년 간 주가와 누적 수익률을 시각화합니다.**")

# 시가총액 기준 상위 10개 기업 정보 (2025 기준, yfinance 호환 티커 사용)
company_info = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Nvidia': 'NVDA',
    'Amazon': 'AMZN',
    'Alphabet (Google)': 'GOOGL',
    'Berkshire Hathaway': 'BRK.B',  # yfinance용 표기법
    'Meta': 'META',
    'Eli Lilly': 'LLY',
    'TSMC': 'TSM',
    'Visa': 'V'
}

# 사용자 선택
selected_companies = st.multiselect(
    "🔎 비교할 기업을 선택하세요",
    list(company_info.keys()),
    default=['Apple', 'Microsoft', 'Nvidia']
)

if not selected_companies:
    st.warning("⚠️ 최소 하나 이상의 회사를 선택해주세요.")
    st.stop()

# 티커 리스트 추출
tickers = [company_info[comp] for comp in selected_companies]

# 기간 설정
end_date = datetime.today()
start_date = end_date - timedel
