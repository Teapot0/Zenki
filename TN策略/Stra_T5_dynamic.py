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

open = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)
open = open[open.columns]
high = high[close.columns]
low = low[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0',date_parser=dateparse)
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

# 每天财务选股

# # 5年ROE
roe_5 = roe_yeayly.rolling(7, min_periods=1).mean()

# 每天财务、均线选股
stock_list_panel = get_financial_stock_list(market_cap,roe_5, pe, money, roe_mean=12, mc_min=100, pe_min=20, money_min=0.1)
# 波动率
std_list = get_std_list(close_rts_1, std_n_list=[10,60],std_list=[0.2,0.2])
# 超额收益
rts_list = get_alpha_list(close,hs300,rts_n_list=[10,40,60,120,250,500], rts_list=[-0.1,-0.1,-0.1,-0.12, -0.15, -0.3])
# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# 3个合并，每天的股票
stock_list = {}
for date in tqdm(close_rts_1.index):
    tmp_list = list(set(std_list[date]).intersection(rts_list[date], stock_list_panel[date]))
    stock_list[date] = list(set(tmp_list).difference(set(pause_list.loc[date])))

all_stock_list = [stock_list[d] for d in close.index]
all_stock = list(set.union(*map(set,all_stock_list)))
# 成交量
vol_n1 = 5
vol_n2 = 10
vol_n3 = 20
# vol停牌为0，需要改
tmp_vol = volume.replace(0, np.nan)
tmp_vol = tmp_vol.fillna(method='ffill')[all_stock]
excess_vol1 = tmp_vol.rolling(vol_n1).sum().pct_change(vol_n1).sub(hs300['volume'].rolling(vol_n1).sum().pct_change(vol_n1), axis=0)
excess_vol2 = tmp_vol.rolling(vol_n2).sum().pct_change(vol_n2).sub(hs300['volume'].rolling(vol_n2).sum().pct_change(vol_n2), axis=0)
excess_vol3 = tmp_vol.rolling(vol_n3).sum().pct_change(vol_n3).sub(hs300['volume'].rolling(vol_n3).sum().pct_change(vol_n3), axis=0)

# 短期波动率
excess_std1 = close[all_stock].rolling(5).std().diff(5).sub(hs300['close'].rolling(5).std().diff(5), axis=0)
excess_std2 = close[all_stock].rolling(10).std().diff(5).sub(hs300['close'].rolling(10).std().diff(5), axis=0)
excess_std3 = close[all_stock].rolling(20).std().diff(5).sub(hs300['close'].rolling(20).std().diff(5), axis=0)

#  超额收益
weight_n1 = 5
weight_n2 = 10
weight_n3 = 20

excess_rts1 = close[all_stock].pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
excess_rts2 = close[all_stock].pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
excess_rts3 = close[all_stock].pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)



# excess rts, vol, std
test_rts = close / open.shift(5) - 1
factor_df = pd.DataFrame(
    columns=['ex_rts1', 'ex_rts2', 'ex_rts3', 'ex_std1', 'ex_std2', 'ex_std3', 'ex_vol1', 'ex_vol2', 'ex_vol3'])
for date in tqdm(close.index):
    tmp_list = stock_list[date]
    nn = min(len(tmp_list), 10)  # 不足10个

    tmp_rts1_u = test_rts.loc[date][list(excess_rts1.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_rts1_d = test_rts.loc[date][list(excess_rts1.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_rts1 = tmp_rts1_u
    else:
        tmp_rts1 = tmp_rts1_u - tmp_rts1_d

    tmp_rts2_u = test_rts.loc[date][list(excess_rts2.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_rts2_d = test_rts.loc[date][list(excess_rts2.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_rts2 = tmp_rts2_u
    else:
        tmp_rts2 = tmp_rts2_u - tmp_rts2_d

    tmp_rts3_u = test_rts.loc[date][list(excess_rts3.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_rts3_d = test_rts.loc[date][list(excess_rts3.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_rts3 = tmp_rts3_u
    else:
        tmp_rts3 = tmp_rts3_u - tmp_rts3_d

    tmp_std1_u = test_rts.loc[date][list(excess_std1.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_std1_d = test_rts.loc[date][list(excess_std1.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_std1 = tmp_std1_u
    else:
        tmp_std1 = tmp_std1_u - tmp_std1_d

    tmp_std2_u = test_rts.loc[date][list(excess_std2.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_std2_d = test_rts.loc[date][list(excess_std2.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_std2 = tmp_std2_u
    else:
        tmp_std2 = tmp_std2_u - tmp_std2_d

    tmp_std3_u = test_rts.loc[date][list(excess_std3.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_std3_d = test_rts.loc[date][list(excess_std3.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_std3 = tmp_std3_u
    else:
        tmp_std3 = tmp_std3_u - tmp_std3_d

    tmp_vol1_u = test_rts.loc[date][list(excess_vol1.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_vol1_d = test_rts.loc[date][list(excess_vol1.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_vol1 = tmp_vol1_u
    else:
        tmp_vol1 = tmp_vol1_u - tmp_vol1_d

    tmp_vol2_u = test_rts.loc[date][list(excess_vol2.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_vol2_d = test_rts.loc[date][list(excess_vol2.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_vol2 = tmp_vol2_u
    else:
        tmp_vol2 = tmp_vol2_u - tmp_vol2_d

    tmp_vol3_u = test_rts.loc[date][list(excess_vol3.loc[date][tmp_list].sort_values(ascending=False)[:nn].index)].mean()
    tmp_vol3_d = test_rts.loc[date][list(excess_vol3.loc[date][tmp_list].sort_values(ascending=False)[-nn:].index)].mean()
    if len(tmp_list) < 30:
        tmp_vol3 = tmp_vol3_u
    else:
        tmp_vol3 = tmp_vol3_u - tmp_vol3_d

    factor_df.loc[date] = [tmp_rts1, tmp_rts2, tmp_rts3, tmp_std1, tmp_std2, tmp_std3, tmp_vol1, tmp_vol2, tmp_vol3]

factor_df = 100*factor_df
factor_df_new = factor_df.rolling(5).mean().pct_change(5)


# 所有权重
weight = excess_rts1.rank(axis=1).mul(factor_df_new['ex_rts1'], axis=0) + \
         excess_rts2.rank(axis=1).mul(factor_df_new['ex_rts2'], axis=0) + \
         excess_rts3.rank(axis=1).mul(factor_df_new['ex_rts3'], axis=0) + \
         excess_std1.rank(axis=1).mul(factor_df_new['ex_std1'], axis=0) + \
         excess_std2.rank(axis=1).mul(factor_df_new['ex_std2'], axis=0) + \
         excess_std3.rank(axis=1).mul(factor_df_new['ex_std3'], axis=0) + \
         excess_vol1.rank(axis=1).mul(factor_df_new['ex_vol1'], axis=0) + \
         excess_vol2.rank(axis=1).mul(factor_df_new['ex_vol2'], axis=0) + \
         excess_vol3.rank(axis=1).mul(factor_df_new['ex_vol3'], axis=0)


# 策略
def std_rts_select_dp_zs(top_number=10, comm_fee=0.002, max_down=0.3):
    max_N = 488

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

        stocklist_weighted = list(weight[stock_list[date]].loc[date].sort_values(ascending=False).index)[:top_number]

        if tmp_week != week1:  # 每周一调仓
            if len(stocklist_weighted) <= 3:
                buy_list = []
            else:
                buy_list = stocklist_weighted
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


daily_rts = std_rts_select_dp_zs(top_number=10, comm_fee=0.003, max_down=0.1)

plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)




