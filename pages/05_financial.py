import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# ✅ 글로벌 시가총액 TOP 10 기업 티커
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

# ✅ Streamlit 기본 설정
st.set_page_config(page_title="글로벌 주가 시각화", layout="wide")
st.title("📈 글로벌 시가총액 Top 10 기업 - 최근 1년 주가 및 누적 수익률")

# ✅ 기업 선택
selected = st.multiselect("기업 선택", list(TICKERS.keys()), default=["Apple", "Microsoft"])

if not selected:
    st.warning("적어도 하나의 기업을 선택해주세요.")
    st.stop()

# ✅ 날짜 범위
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# ✅ 주가 데이터 가져오기 함수
@st.cache_data
def fetch_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end)
    if len(tickers) == 1:
        # 단일 종목일 경우: DataFrame 평탄화 및 컬럼 이름 통일
        df = raw["Adj Close"].to_frame()
        df.columns = [tickers[0]]
    else:
        # 다중 종목일 경우: 다중 인덱스에서 'Adj Close' 추출
        df = raw["Adj Close"]
    return df

# ✅ 티커 목록으로 변환
ticker_list = [TICKERS[name] for name in selected]

# ✅ 데이터 가져오기
price_df = fetch_prices(ticker_list, start_date, end_date)

# ✅ 선 그래프: 주가
st.subheader("📊 주가 (Adjusted Close)")
fig_price = px.line(price_df, labels={"value": "주가", "index": "날짜", "variable": "기업"})
fig_price.update_layout(legend_title_text="기업", height=500)
st.plotly_chart(fig_price, use_container_width=True)

# ✅ 누적 수익률 계산
returns_df = (price_df / price_df.iloc[0] - 1) * 100

# ✅ 선 그래프: 누적 수익률
st.subheader("📈 누적 수익률 (%)")
fig_return = px.line(returns_df, labels={"value": "누적 수익률 (%)", "index": "날짜", "variable": "기업"})
fig_return.update_layout(legend_title_text="기업", height=500)
st.plotly_chart(fig_return, use_container_width=True)
