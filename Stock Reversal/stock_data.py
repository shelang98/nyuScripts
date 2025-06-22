import pandas as pd
import requests
import time
import os
import yfinance as yf
from datetime import datetime, timedelta
from ib_insync import IB, Stock, util

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = [t.replace('.', '-') for t in sp500['Symbol']]


# 设置时间范围
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# 连接到 IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=3)

# 存所有数据的列表
all_data = []

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] 正在下载 {ticker} ...")
    try:
        contract = Stock(ticker, 'SMART', 'USD')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='365 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            keepUpToDate=False
        )

        df = util.df(bars)
        if df.empty:
            print(f"⚠️ {ticker} 没有数据")
            continue

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['ticker'] = ticker
        all_data.append(df[['open', 'high', 'low', 'close', 'volume', 'ticker']])

        time.sleep(1.5)

    except Exception as e:
        print(f"❌ {ticker} 下载失败: {e}")
        continue

ib.disconnect()

# 合并为一个大的 DataFrame，使用 MultiIndex
if all_data:
    big_df = pd.concat(all_data)
    big_df.set_index('ticker', append=True, inplace=True)
    big_df = big_df.reorder_levels(['date', 'ticker']).sort_index()

    # 保存为一个 CSV 文件
    big_df.to_csv('Stock Reversal/sp500_ibkr_365d_multiindex.parquet')
    print("✅ 所有数据已保存为 multi-index CSV 文件。")
else:
    print("❌ 没有成功下载任何数据。")