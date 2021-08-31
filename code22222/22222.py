import pandas as pd
import numpy as np
from basic_funcs.basic_function import *

import warnings
warnings.filterwarnings('ignore')


def get_short_ma_order(close, n1, n2, n3):
    ma1 = close.rolling(n1).mean()
    ma2 = close.rolling(n2).mean()
    ma3 = close.rolling(n3).mean()
    return (ma1 < ma2) & (ma2 < ma3) & (ma1 < ma3)


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['net_value'] = (1+hs300['rts_1']).cumprod()

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1) # 某列全NA
close_rts_1 = close.pct_change(1)
# close_max_5 = close.rolling(5).max()
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0',date_parser=dateparse)
open = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0',date_parser=dateparse)

# 上市大于N天的,一共1816天
ipo_days = close.shape[0] - close.isna().sum()
stock_list_days = list(close.isna().sum()[close.isna().sum() <= 244].index) # 大于一年的
close = close[stock_list_days]

high = high[close.columns]
low = low[close.columns]
open = open[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)

#RPS
# 一二三个月
rps_n1 = 120
rps_n2 = 250


def get_rps_df(rps_n):
    close_n_min = close.rolling(rps_n).min()
    close_n_max = close.rolling(rps_n).max()
    rps = ((close - close_n_min)/(close_n_max - close_n_min)) * 100
    return rps

rps1 = get_rps_df(rps_n1)
rps2 = get_rps_df(rps_n2)

tmp = pd.DataFrame(columns=['120rps_5', '250rps_5','120rps_10', '250rps_10'])
tmp['120rps_5'] = rps1.rolling(5).mean().iloc[-1,]
tmp['250rps_5'] = rps2.rolling(5).mean().iloc[-1,]
tmp['120rps_10'] = rps1.rolling(10).mean().iloc[-1,]
tmp['250rps_10'] = rps2.rolling(10).mean().iloc[-1,]
tmp.to_csv('/Users/caichaohong/Desktop/rps123')

stock_120_1m = list(rps1.iloc[-10:,].mean()[rps1.iloc[-10:,].mean() >= 85].index)
stock_250_1m = list(rps2.iloc[-10:,].mean()[rps2.iloc[-10:,].mean() >= 85].index)

# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)

stock_120_1m = list(set(stock_120_1m).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))
stock_250_1m = list(set(stock_250_1m).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))

stock_len = max(len(stock_120_1m), len(stock_250_1m))
# 停牌的
rps_df = pd.DataFrame(index= list(range(stock_len)))
rps_df['rps120_1month'] = np.nan
rps_df['1month_120_rts'] = np.nan

rps_df['rps250_1month'] = np.nan
rps_df['1month_250_rts'] = np.nan


rps_df['rps120_1month'][:len(stock_120_1m)] = list(all_name['short_name'][stock_120_1m])
rps_df['1month_120_rts'][:len(stock_120_1m)] = list(close_rts_1.iloc[-1][stock_120_1m])

rps_df['rps250_1month'][:len(stock_250_1m)] = list(all_name['short_name'][stock_250_1m])
rps_df['1month_250_rts'][:len(stock_250_1m)] = list(close_rts_1.iloc[-1][stock_250_1m])


rps_df.to_excel('/Users/caichaohong/Desktop/rps_df_10.xlsx')







