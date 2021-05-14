import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqfactor_analyzer import analyze_factor
from alphalens.utils import get_clean_factor_and_forward_returns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from basic_funcs.basic_function import *
from jqdatasdk import get_index_stocks, auth, get_query_count,get_extras
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

import warnings
warnings.filterwarnings('ignore')

# 资金流
top_10_net_buy = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_net_buy.xlsx')


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300 = hs300[hs300.index >= '2018-01-01']
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

hold_share = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/share.csv', index_col='Unnamed: 0', date_parser=dateparse)
hold_ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)
hold_value = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/value.csv', index_col='Unnamed: 0', date_parser=dateparse)

market_cap = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/market_cap.xlsx',index_col='Unnamed: 0')

# excess hold_value
model = LinearRegression()
hold_value_coef = pd.Series(index=hold_value.index)
hold_value_marketcap = pd.DataFrame(np.nan, index=hold_value.index, columns=hold_value.columns)


model.fit(market_cap.fillna(0).values, hold_value.fillna(0).values)
z = model.coef_

for i in range(hold_value.shape[0]):
    model.fit(market_cap.fillna(0).values[i].reshape(-1,1), hold_value.fillna(0).values[i].reshape(-1,1))
    hold_value_coef[i] = model.coef_
    hold_value_marketcap.iloc[i, ] = market_cap.values[i] - model.coef_*market_cap.values[i]

holds = []
for i in range(hold_value_marketcap.shape[0]):
    holds.append(hold_value_marketcap.iloc[i,].sort_values(ascending=False).index[0])


margin_buy_value = pd.read_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_buy_value.xlsx')
margin_sell_value= pd.read_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_sell_value.xlsx')

stock_list = list(hold_value.columns)
# stock_list = get_index_stocks('000985.XSHG')

close = read_excel_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.xlsx', start_date='2018-01-01', end_date='2021-04-16', stocks= stock_list)
low = read_excel_select('/Users/caichaohong/Desktop/Zenki/price/daily/low.xlsx',start_date='2018-01-01', end_date='2021-04-16', stocks= stock_list)
high_limit = read_excel_select('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.xlsx', start_date='2018-01-01', end_date='2021-04-16', stocks=stock_list)
volume = read_excel_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.xlsx', start_date='2018-01-01', end_date='2021-04-16', stocks=  stock_list)


close = clean_close(close, low, high_limit)  # 新股一字板
close = clean_st_exit(close)  # 退市和ST
close_rts = close.pct_change(1)
# excess_rts = close_rts.sub(hs300['rts'],axis=0)

low = low[close.columns]
high_limit = high_limit[close.columns]
volume = volume[close.columns]
volume = volume*10**(-8)


#


# 历史新高performance


# 抗跌因子选股 roe 净利润 定义 跑赢市场

#  sharpe_df
# sharp_df = (close_rts.sub(hs300['rts'], axis=0))/(close_rts.rolling(20).std())
zzz = hold_value_growth.apply(lambda x: x/close_rts)


value_rts, holds = get_top_value_factor_rts(factor=hold_value_growth, rts=close_rts, top_number=3, hold_time=1, weight='avg',return_holdings_list=True)



comm_fee = 0.003
hold_time = 1
out_df = pd.DataFrame(index=close.index)
# out_df['holds'] = holds['holdings']
out_df['rts'] = value_rts.fillna(0) # 没有收益的为0
out_df['rts'][::hold_time] = out_df['rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
out_df['net_value'] = (1 + out_df['rts']).cumprod()
na_num = out_df['net_value'].isna().sum()
position_max_draw = list(np.zeros(na_num))
out_df['nv_max_draw'] = position_max_draw + list(MaxDrawdown(list(out_df['net_value'].dropna())).reshape(-1))
out_df['benchmark_rts'] = hs300['close'].pct_change(1).fillna(0)
out_df['benchmark_net_value'] = (1+out_df['benchmark_rts']).cumprod()
plot_hold_position(data=out_df, risk_free_rate=0.03)



# 大盘大跌或新高时，买南北向最大的；

hs300_factor = pd.DataFrame(index=hs300.index)

# 多少股票在20线以上
hs300_stocks = get_index_stocks('000300.XSHG')
close_20 = close[hs300_stocks].rolling(20).mean()
abv_20 = (close[hs300_stocks] / close_20 - 1).mean(axis=1)

# 昨日涨停
# hs300_factor['zt_yesterday'] = zt_yesterday(close, high_limit)






