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

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1)  # 某列全NA
close_rts_1 = close.pct_change(1)

high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = high[close.columns]
low = low[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')  # 2924个
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)  # 2924个
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv', index_col='statDate')  # 2924个
market_cap = market_cap[close.columns]
pe = pe[close.columns]

# 股息率
df_bank = finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.code == '801780'))
# 回购
df_bond = bond.run_query(query(bond.REPO_DAILY_PRICE).filter(bond.REPO_DAILY_PRICE.name == 'GC182').limit(2000))
df_t1 = pd.merge(df_bond, df_bank, on='date')
df_t1 = df_t1[['date', 'close', 'dividend_ratio']]
df_t1.index = pd.to_datetime(df_t1['date'])
# 当风险偏好<0不持股
df_t1['licha'] = (df_t1['close'].rolling(60).mean() - df_t1['dividend_ratio'].rolling(60).mean()).diff(1)
df_t1['licha'] = df_t1['licha'].fillna(method='ffill')
df_t1['hs300'] = hs300['net_value']

# 回购和价格行情日期不同
df_repo = pd.DataFrame(columns=['licha'], index=close.index)
datelist = list(set(df_t1.index).intersection(set(df_repo.index)))
df_repo['licha'].loc[datelist] = df_t1['licha'].loc[datelist]
df_repo = df_repo.fillna(method='ffill')


# 每天财务、均线选股
roe_5 = roe_yeayly.rolling(5, min_periods=1).mean()
stock_list_panel = get_financial_stock_list(market_cap,roe_5, pe, money,
                                            roe_mean=12, mc_min=300, pe_min=20, money_min=1)
# 波动率
std_list = get_std_list(close_rts_1, std_n_list=[10,60],std_list=[0.2,0.2])
# 超额收益
rts_list = get_alpha_list(close,hs300,rts_n_list=[10,40,60,120,250,500], rts_list=[-0.1,-0.1,-0.1,-0.12, -0.15, -0.3])

# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)


# 异常下跌
down_list = {}
tmp_down_list = []
for i in tqdm(range(close.shape[0])):
    if i == close.shape[0]-1:
        down_list[close.index[-1]] = []
    else:
        date = close.index[i]
        date1 = close.index[i+1]
        tmp_week = date.week
        week1 = date1.week
        if hs300['rts_1'].loc[date] > 0.005:
            cp = list(close_rts_1.loc[date][close_rts_1.loc[date] <= -0.02].index)
            tmp_down_list = tmp_down_list + cp
            down_list[date] = cp
        else:
            down_list[date] = []

        if tmp_week != week1:
            down_list[date] = tmp_down_list
            tmp_down_list = []


# 3个合并，每天的股票
stock_list = {}
for date in tqdm(close_rts_1.index):
    tmp_list = list(set(std_list[date]).intersection(rts_list[date], stock_list_panel[date]))
    stock_list[date] = list(set(tmp_list).difference(set(pause_list.loc[date]),set(st_df.loc[date]), set(down_list[date])))

# 大小市值切换
small_big = pd.DataFrame(index=close.index,columns=['big', 'small','big_stock', 'small_stock'])
for date in tqdm(close_rts_1.index):
    tmp_list = list(market_cap[stock_list[date]].loc[date].sort_values(ascending=True).dropna().index)
    small_cap_list = tmp_list[:5]
    big_cap_list = tmp_list[-5:]
    small_big.loc[date]['small'] = close_rts_1.loc[date][small_cap_list].mean()
    small_big.loc[date]['big'] = close_rts_1.loc[date][big_cap_list].mean()
    small_big.loc[date]['small_stock'] = small_cap_list
    small_big.loc[date]['big_stock'] = big_cap_list

plt.plot((1+small_big['big']).cumprod(), label='big')
plt.plot((1+small_big['small']).cumprod())
plt.legend()


weight_n1 = 10
weight_n2 = 180
weight_n3 = 250
weight1 = -2
weight2 = 2
weight3 = 4

rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
rts_f2 = close.pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
rts_f3 = close.pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)

weight = weight1 * rts_f1.rank(axis=1) + weight2 * rts_f2.rank(axis=1) + \
         weight3 * rts_f3.rank(axis=1)


def std_rts_select_dp_zs(close, hs300,
                         top_number=10, comm_fee=0.002, max_down=0.1):
    max_N = 500

    while len(rts_list) != len(std_list):
        print('std list and rts list same length')
        break

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])

    buy_list = []
    initial_cost = [1]

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        tmp_week = date.week
        week1 = date1.week

        stocklist_financial = stock_list[date]
        stocklist_weighted = list(weight[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]

        if tmp_week != week1:  # 每周五
            stocks = list(set(stocklist_weighted))

            if len(stocks) <= 3:
                buy_list = []
            else:
                buy_list = stocks
            initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -max_down].index)
        buy_list = list(set(buy_list).difference(set(sell_list)))  #

        if (hs300['short_ma'].loc[date] == True) & (df_repo['licha'].loc[date] < 0):
            buy_list = []
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if len(out_df['hold_daily'].loc[date]) > 0:  # 当天持仓，扣手续费
                out_df['daily_rts'].loc[date1] = -comm_fee
            else:
                out_df['daily_rts'].loc[date1] = 0
        else:
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if tmp_week != week1:  # 每周一调仓
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean() - comm_fee
            else:
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = std_rts_select_dp_zs(close, hs300, top_number=10, comm_fee=0.003, max_down=0.1)
plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)



# z = weight.loc[close.index[-9]][stock_list[close.index[-9]]].sort_values(ascending=False)
# z_1 = weight.loc[close.index[-9]]['600460.XSHG']




