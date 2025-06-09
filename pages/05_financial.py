import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🌐 글로벌 시가총액 Top 10 기업 주가 및 누적 수익률 시각화")

top10_tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Alphabet Class A (GOOGL)": "GOOGL",
    "Alphabet Class C (GOOG)": "GOOG",
    "Tesla (TSLA)": "TSLA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "NVIDIA (NVDA)": "NVDA",
    "Meta Platforms (META)": "META",
    "Visa (V)": "V",
}

selected_companies = st.multiselect(
    "관심 있는 기업을 선택하세요 (최소 1개 이상)",
    options=list(top10_tickers.keys()),
    default=["Apple (AAPL)", "Microsoft (MSFT)"]
)

if not selected_companies:
    st.warning("최소 한 개 이상의 기업을 선택해주세요.")
    st.stop()

end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end):
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        group_by='ticker',
        auto_adjust=False
    )
    return data

tickers = [top10_tickers[c] for c in selected_companies]
data = fetch_data(tickers, start_date, end_date)

price_dfs = {}

if len(tickers) == 1:
    # 단일 티커일 경우
    ticker = tickers[0]
    # 'Adj Close' 컬럼 존재하는지 확인
    if 'Adj Close' in data.columns:
        price_dfs[ticker] = data['Adj Close'].dropna()
    else:
        st.error(f"{ticker} 데이터에서 'Adj Close' 컬럼을 찾을 수 없습니다.")
        st.stop()
else:
    # 멀티 티커일 경우
    # 컬럼 멀티 인덱스 확인
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                if 'Adj Close' in data[ticker].columns:
                    price_dfs[ticker] = data[ticker]['Adj Close'].dropna()
                else:
                    st.warning(f"{ticker} 데이터에 'Adj Close' 컬럼이 없습니다.")
            else:
                st.warning(f"{ticker} 데이터가 존재하지 않습니다.")
    else:
        # 멀티 티커인데 단일 인덱스 컬럼일 경우 (가끔 발생)
        for ticker in tickers:
            col_name = f'Adj Close'
            if col_name in data.columns:
                price_dfs[ticker] = data[col_name].dropna()
            else:
                st.warning(f"{ticker} 데이터에서 'Adj Close' 컬럼을 찾을 수 없습니다.")

if not price_dfs:
    st.error("적절한 데이터가 없어 시각화를 진행할 수 없습니다.")
    st.stop()

# 1) 주가 선 그래프
fig_price = go.Figure()
for ticker, series in price_dfs.items():
    fig_price.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=ticker
        )
    )

fig_price.update_layout(
    title="최근 1년 주가 추이 (Adjusted Close)",
    xaxis_title="날짜",
    yaxis_title="주가(USD)",
    hovermode="x unified"
)
st.plotly_chart(fig_price, use_container_width=True)

# 2) 누적 수익률 계산 및 그래프
cumulative_returns = pd.DataFrame()
for ticker, series in price_dfs.items():
    returns = series.pct_change().fillna(0)
    cumulative_returns[ticker] = (1 + returns).cumprod() - 1

fig_return = go.Figure()
for ticker in cumulative_returns.columns:
    fig_return.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[ticker],
            mode='lines',
            name=ticker
        )
    )

fig_return.update_layout(
    title="최근 1년 누적 수익률",
    xaxis_title="날짜",
    yaxis_title="누적 수익률",
    hovermode="x unified",
    yaxis_tickformat=".2%"
)
st.plotly_chart(fig_return, use_container_width=True)
