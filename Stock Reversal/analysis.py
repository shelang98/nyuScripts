import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import backtrader as bt
import matplotlib.pyplot as plt

# 读取 CSV 文件
df =  pd.read_parquet('Stock Reversal/sp500_ibkr_365d.parquet')

df_wide = df.pivot_table(index='date', columns = 'ticker', 
                         values = ['open','high','low','close','volume'])
df_wide = df_wide.swaplevel(axis=1)

tickers = df_wide.columns.get_level_values(0).unique().to_list()

# 得到股票每日回报数据
close_df = df_wide.xs('close', level=1, axis=1)
returns = close_df.pct_change().round(4).dropna()


#尝试利用一些特性构架一个指标衡量股票的的弹性 - 大跌之后容易回归
# 收益分布偏态+峰度， 可以衡量对利空/利多更敏感的股票， 以及容易出现极端波动的股票
skewness = returns.apply(skew) # 左偏，更容易出现极端下跌
print(skewness.sort_values().head(10))
kurt = returns.apply(kurtosis) # 
print(kurt.sort_values(ascending=False).head(10))

# 测量股票回到均线的平均天数，越短回归越快？
def calculate_half_life(series):
    lagged = series.shift(1)
    delta = series - lagged
    beta = np.polyfit(lagged.dropna(), delta.dropna(), 1)[0]
    halflife = -np.log(2) / beta
    return halflife

## 统计过去一年时间内某股票反弹超过n%后，次日上涨的频率？平均反弹的幅度？

def Nextday(returns: pd.DataFrame, ticker: str, floor: float):
    temp = returns[ticker]

    # 找出跌幅超过阈值的日期
    drop_days = temp[temp < -floor]
    
    # 第二天的收益情况
    next_day_returns = temp.shift(-1).loc[drop_days.index]

    # 平均反弹幅度
    avg_performance = next_day_returns.mean()

    # 反弹天数（即第二天收益为正的天数）
    rebound_days = (next_day_returns > 0).sum()

    # 总次数
    total = len(drop_days)

    return {
        'ticker': ticker,
        'count': total,
        'avg_return_next_day': avg_performance,
        'rebound_rate': round(rebound_days / total,2) if total > 0 else None
    }

dates = returns.index.sort_values()
split_idx = int(len(dates) * 0.75)

train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_ret = returns.loc[train_dates]
test_ret = returns.loc[test_dates]

result = {}
for tk in tickers:
    if tk not in train_ret.columns:
        continue
    result[tk] = Nextday(train_ret, tk, 0.03)

select_tk = [tk for tk in result.keys() 
            if (result[tk]['count'] > 1) and 
            (result[tk]['rebound_rate'] >= 0.55) and 
            (result[tk]['avg_return_next_day'] >= 0.01)]

### backtest

commission_fee = 0.002

class ReboundStrategy(bt.Strategy):
    params = (
        ('threshold', -0.03),        # 跌幅阈值
        ('max_positions', 10),       # 最多同时买入几只股票
        ('cash_ratio', 0.1),         # 每日最多总投资比例
        ('fee', 0.0013),             # 手续费（买入+卖出）
        ('result_dict', result),       # 存储 rebound_rate 排序字典
    )

    def __init__(self):
        self.order_refs = []

    def next(self):
        # 平掉所有持仓
        for data in self.datas:
            if self.getposition(data).size:
                self.close(data)

        # 找前一日跌幅超标的
        candidates = []
        for data in self.datas:
            if len(data) < 2:
                continue
            ret = (data.close[-1] - data.close[-2]) / data.close[-2]
            if ret <= self.params.threshold:
                tk = data._name
                rr = self.params.result_dict.get(tk, {}).get('rebound_rate', 0)
                if rr is None:
                    continue  # 跳过这个ticker
                candidates.append((tk, data, rr))

        # 排序选前 max_positions 个
        candidates = sorted(candidates, key=lambda x: -x[2])[:self.params.max_positions]

        if candidates:
            total_cash = self.broker.get_cash() * self.params.cash_ratio
            each_cash = total_cash / len(candidates)

            for _, data, _ in candidates:
                price = data.open[0]
                if price > 0:
                    size = int(each_cash / price)
                    if size > 0:
                        self.buy(data=data, size=size)

cerebro = bt.Cerebro()
cerebro.broker.set_cash(1_000_000)
cerebro.broker.setcommission(commission= commission_fee)

for ticker in df_wide.columns.levels[0]:
    df = df_wide[ticker].copy()
    # 防止列名仍是 MultiIndex（即使只有一只 ticker）
    df.columns.name = None
    df.columns = df.columns.tolist()

    # 索引命名 & 转换为 datetime
    df.index.name = 'datetime'
    df.index = pd.to_datetime(df.index)

    # 丢掉缺失
    df = df.dropna()

    # 添加数据源
    data = bt.feeds.PandasData(dataname=df, name=ticker)

    cerebro.adddata(data)
cerebro.addstrategy(ReboundStrategy, result_dict=result)

# 运行
results = cerebro.run()
strategy_instance = results[0]

# 查看账户最终资金
print("Final Portfolio Value:", cerebro.broker.getvalue())

cerebro.plot()
plt.savefig('backtest_result.png', dpi=300)