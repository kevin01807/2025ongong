import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# ✅ 정확한 티커 사용 (Yahoo Finance 기준)
TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
    "Meta (Facebook)": "META",
    "Berkshire Hathaway": "BRK.B",  # ← 중요! BRK-B → BRK.B
    "Tesla": "TSLA",
    "TSMC": "TSM",
    "Johnson & Johnson": "JNJ"
}

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

# ✅ 데이터 가져오기
@st.cache_data
def fetch_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)

    if raw.empty:
        return pd.DataFrame()  # 다운로드 실패

    # 단일 종목
    if len(tickers) == 1:
        ticker = tickers[0]
        if "Adj Close" in raw.columns:
            return raw[["Adj Close"]].rename(columns={"Adj Close": ticker})
        elif (ticker, "Adj Close") in raw.columns:
            return raw.loc[:, (ticker, "Adj Close")].to_frame(name=ticker)
        else:
            return pd.DataFrame()

    # 다중 종목
    adj_close = pd.DataFrame()
    for ticker in tickers:
        try:
            adj_close[ticker] = raw[ticker]["Adj Close"]
        except KeyError:
            continue
    return adj_close

# ✅ 변환
ticker_list = [TICKERS[name] for name in selected]
price_df = fetch_prices(ticker_list, start_date, end_date)

if price_df.empty:
    st.error("❌ 주가 데이터를 불러올 수 없습니다. 네트워크 상태나 티커명을 확인하세요.")
    st.stop()

# ✅ 주가 시각화
st.subheader("📊 주가 (Adjusted Close)")
fig_price = px.line(price_df, labels={"value": "주가", "index": "날짜", "variable": "기업"})
fig_price.update_layout(legend_title_text="기업", height=500)
st.plotly_chart(fig_price, use_container_width=True)

# ✅ 누적 수익률 계산
returns_df = (price_df / price_df.iloc[0] - 1) * 100

# ✅ 누적 수익률 시각화
st.subheader("📈 누적 수익률 (%)")
fig_return = px.line(returns_df, labels={"value": "누적 수익률 (%)", "index": "날짜", "variable": "기업"})
fig_return.update_layout(legend_title_text="기업", height=500)
st.plotly_chart(fig_return, use_container_width=True)
