import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# 시가총액 기준 Top 10 글로벌 기업 티커
TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
    "Meta (Facebook)": "META",
    "Berkshire Hathaway": "BRK-B",
    "Tesla": "TSLA",
    "TSMC": "TSM",
    "Johnson & Johnson": "JNJ"
}

# Streamlit 페이지 설정
st.set_page_config(page_title="글로벌 Top 10 기업 주가 시각화", layout="wide")
st.title("📈 글로벌 시가총액 Top 10 기업 - 최근 1년 주가 및 누적 수익률")

# 기업 선택
selected = st.multiselect("기업 선택", list(TICKERS.keys()), default=["Apple", "Microsoft"])

if not selected:
    st.warning("적어도 하나의 기업을 선택해주세요.")
    st.stop()

# 날짜 범위 설정
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# yfinance 데이터 가져오기
@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

tickers = [TICKERS[name] for name in selected]
data = fetch_data(tickers, start_date, end_date)

# 단일 선택 시 DataFrame 형식 보장
if isinstance(data, pd.Series):
    data = data.to_frame()

# 시각화 - 주가
st.subheader("📊 주가 (Adjusted Close)")
fig_price = px.line(data, labels={"value": "주가", "Date": "날짜", "variable": "기업"})
fig_price.update_layout(legend_title_text="기업", height=500)
st.plotly_chart(fig_price, use_container_width=True)

# 누적 수익률 계산
returns = (data / data.iloc[0]) - 1
returns = returns * 100  # 백분율

# 시각화 - 누적 수익률
st.subheader("📈 누적 수익률 (%)")
fig_return = px.line(returns, labels={"value": "누적 수익률 (%)", "Date": "날짜", "variable": "기업"})
fig_return.update_layout(legend_title_text="기업", height=500)
st.plotly_chart(fig_return, use_container_width=True)
