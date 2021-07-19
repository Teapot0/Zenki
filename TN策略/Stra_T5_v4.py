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
# auth('13382017213', 'Aasd120120')
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
hs300['rts_5'] = hs300['close'].pct_change(5)
hs300['net_value'] = (1 + hs300['rts_1']).cumprod()
hs300['rps_close'] = get_rps(hs300['close'], rps_n=122)
hs300['rps_5'] = hs300['rps_close'].rolling(5).mean()
hs300['rps_vol'] = get_rps(hs300['volume'].rolling(5).sum(), rps_n=61)
hs300['rps_vol_1'] = hs300['rps_vol']*(hs300['rps_5'] >= 90)*1
hs300['rps_5'] = hs300['rps_5'].mask(hs300['rps_5'] >= 90, hs300['rps_5']*0.01*hs300['rps_vol_1'])
hs300['short_ma'] = get_short_ma_order(hs300['close'], n1=5, n2=90, n3=180)
# rps

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1)  # 某列全NA
close_rts_1 = close.pct_change(1)
close_rts_5 = close.pct_change(5)
# 周收益
rps_df = get_rps(close,rps_n=61)

high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = high[close.columns]
low = low[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)
vol_rps = get_rps(volume,rps_n=61)

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')  # 2924个
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)  # 2924个
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv', index_col='statDate')  # 2924个
market_cap = market_cap[close.columns]
pe = pe[close.columns]

net_pct_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_xl.csv', index_col='Unnamed: 0')
net_pct_xl.index = pd.to_datetime(net_pct_xl.index)
net_xl_rts = net_pct_xl.rolling(5).mean().pct_change(5)


# 每天财务选股

# # 5年ROE
roe_5 = roe_yeayly.rolling(5, min_periods=1).mean()
# 每天财务、均线选股
stock_list_panel = get_financial_stock_list(market_cap,roe_5, pe, money,
                                            roe_mean=12, mc_min=100, pe_min=20, money_min=0.2)
panel_list = [stock_list_panel[d] for d in close_rts_1.index]
all_stock = set.union(*map(set,panel_list))

# 超额收益
std_list = get_std_list(close_rts_1, std_n_list=[10,60],std_list=[0.2,0.2])
# 超额收益
rts_list = get_alpha_list(close,hs300,rts_n_list=[10,40,60,120,250,500], rts_list=[-0.1,-0.1,-0.1,-0.12, -0.15, -0.3])

# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)


# 3个合并，每天的股票
stock_list = {}
for date in tqdm(close_rts_1.index):
    tmp_list = list(set(rts_list[date]).intersection(std_list[date], stock_list_panel[date]))
    stock_list[date] = list(set(tmp_list).difference(set(pause_list.loc[date]),set(st_df.loc[date])))


f1_factor = 244
f1 = (close_rts_1.rolling(f1_factor).mean() / close_rts_1.rolling(f1_factor).std()) * sqrt(f1_factor)
sharpe_df = f1.rolling(40).mean()

ex_rts1 = close_rts_1.sub(hs300['rts_1'],axis=0)
ex_rts5 = ex_rts1.rolling(5).sum()
up_list = list(hs300[hs300['rts_1'] > 0].index)
ex_rts1_down = ex_rts1.copy(deep=True)
ex_rts1_down.loc[up_list] = 0
ex_rts1_down40 = ex_rts1_down.rolling(40).sum()

sharpe_rts_df = pd.DataFrame(index=sharpe_df.index,columns=['up_rts','down_rts'])
for i in tqdm(range(1,sharpe_rts_df.shape[0])):
    date = sharpe_rts_df.index[i-1]
    date1 = sharpe_rts_df.index[i]
    tmp_list = list(sharpe_df[stock_list[date]].loc[date].sort_values(ascending=False).index)[:30]
    tmp_up_list = list(close_rts_5.loc[date][tmp_list].sort_values(ascending=False).index)[:10]
    tmp_down_list = list(close_rts_5.loc[date][tmp_list].sort_values(ascending=True).index)[:10]
    up_list = list(set(tmp_up_list).intersection(stock_list[date]))
    down_list = list(set(tmp_down_list).intersection(stock_list[date]))
    sharpe_rts_df['up_rts'].loc[date1] = round(close_rts_1.loc[date1][up_list].mean()*100,3)
    sharpe_rts_df['down_rts'].loc[date1] = round(close_rts_1.loc[date1][down_list].mean()*100,3)

sharep_rts_20 = sharpe_rts_df.rolling(20).sum()
sharep_rts_20['diff'] = sharep_rts_20['up_rts'] - sharep_rts_20['down_rts']


ex_rts_df = pd.DataFrame(index=ex_rts5.index,columns=['up_rts','down_rts'])
for i in tqdm(range(1,ex_rts_df.shape[0])):
    date = sharpe_rts_df.index[i-1]
    date1 = sharpe_rts_df.index[i]
    tmp_list = list(ex_rts1_down40[stock_list[date]].loc[date].sort_values(ascending=False).index)[:30]
    tmp_up_list = list(close_rts_5.loc[date][tmp_list].sort_values(ascending=False).index)[:10]
    tmp_down_list = list(close_rts_5.loc[date][tmp_list].sort_values(ascending=True).index)[:10]
    up_list = list(set(tmp_up_list).intersection(stock_list[date]))
    down_list = list(set(tmp_down_list).intersection(stock_list[date]))
    sharpe_rts_df['up_rts'].loc[date1] = round(close_rts_1.loc[date1][up_list].mean()*100,3)
    sharpe_rts_df['down_rts'].loc[date1] = round(close_rts_1.loc[date1][down_list].mean()*100,3)



def std_rts_select_dp_zs(close, hs300,
                         top_number=10, comm_fee=0.002, max_down=0.1):
    max_N = 500

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])

    buy_list = []
    initial_cost = [1]

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        tmp_week = date.week
        week1 = date1.week

        stocklist_financial = stock_list[date]
        if tmp_week != week1:  # 每周五
            if sharep_rts_20['diff'].loc[date] > 1:
                tmp_list = list(sharpe_df[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
            elif sharep_rts_20['diff'].loc[date] < -1:
                tmp_list = list(sharpe_df[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
            stocks = tmp_list
            buy_list = stocks

            else:
                buy_list = []

            initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -max_down].index)
        buy_list = list(set(buy_list).difference(set(sell_list)))  #

        out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

        if tmp_week != week1:  # 每周一调仓
            out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean() - comm_fee
        else:
            out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = std_rts_select_dp_zs(close, hs300, top_number=10, comm_fee=0.003, max_down=0.1)
daily_rts['alpha'] = (daily_rts['daily_rts'] - hs300['rts_1'])*100
plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)

plt.plot((1+daily_rts['daily_rts'].dropna()).cumprod())






