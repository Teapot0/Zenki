import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks, get_bars, bond
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from basic_funcs.basic_function import *
import talib

auth('15951961478', '961478')
get_query_count()

import warnings

warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx', index_col='Unnamed: 0')
all_name.index = all_name['code']

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['net_value'] = (1 + hs300['rts_1']).cumprod()
# 择时
hs300['short_ma'] = get_short_ma_order(hs300['close'], n1=5, n2=90, n3=180)

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
close = close.dropna(how='all', axis=1)  # 某列全NA
close_rts_1 = close.pct_change(1)

high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0',
                   date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0',
                     date_parser=dateparse)
high = high[close.columns]
low = low[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)
money_20 = money.rolling(20).mean()

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0',
                         date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')  # 2924个
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0',
                 date_parser=dateparse)  # 2924个
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv',
                         index_col='statDate')  # 2924个
market_cap = market_cap[close.columns]
pe = pe[close.columns]

# 每天财务选股
stock_list_panel = {}
for i in range(market_cap.shape[0]):
    date = market_cap.index[i]
    # 市值大于100 pe大于25
    mc_100 = list(market_cap.iloc[i, :][market_cap.iloc[i, :] > 100].index)
    # 成交额大于1000万
    money_list = list(money_20.iloc[i, :][money_20.iloc[i, :] > 0.5].index)
    tmp_list = list(set(mc_100).intersection(set(money_list)))
    stock_list_panel[date] = tmp_list

# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0', date_parser=dateparse)
st_list = {}
for date in tqdm(close_rts_1.index):
    st_list[date] = list(st_df.loc[date][st_df.loc[date] == True].index)

# 财务+停牌+st
stock_list = {}
for date in tqdm(close_rts_1.index):
    stock_list[date] = list(set(stock_list_panel[date]).difference(set(pause_list.loc[date]), set(st_list[date])))

# new factor
f1_factor = 244
f1 = (close_rts_1.rolling(f1_factor).mean() / close_rts_1.rolling(f1_factor).std()) * sqrt(f1_factor)
sharpe_df = f1.rolling(40).mean()

z = sharpe_df.loc[close.index[-5]][stock_list[close.index[-5]]].sort_values(ascending=False)

weight_n1 = 10
weight1 = -2


vol_n1 = 40
vol_f1 = 1/abs(volume.rolling(vol_n1).mean().pct_change(vol_n1))

# rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)

weight_sharpe = 0.6
weight_vol = 0.4
weight = weight_sharpe * sharpe_df.rank(axis=1) + weight_vol * vol_f1.rank(axis=1)


def stra_t5(close, hs300, top_number=10, comm_fee=0.003, max_down=0.1):
    max_N = 500

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])
    buy_list = []
    initial_cost = [1]

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        tmp_week = date.week
        week1 = date1.week

        if tmp_week != week1:  # 每周五
            buy_list = list(weight.loc[date][stock_list[date]].sort_values(ascending=False).index)[:top_number]
            initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -max_down].index)
        buy_list = list(set(buy_list).difference(set(sell_list)))  #

        if hs300['short_ma'].loc[date] == True:
            buy_list = []
            out_df['hold_daily'].loc[date1] = []
            out_df['daily_rts'].loc[date1] = 0
        else:
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if tmp_week != week1:  # 每周一调仓
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean() - comm_fee
            else:
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = stra_t5(close, hs300, top_number=10, comm_fee=0.003, max_down=0.1)
plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)

# z = weight.loc[close.index[-9]][stock_list[close.index[-9]]].sort_values(ascending=False)
# z_1 = weight.loc[close.index[-9]]['600460.XSHG']
