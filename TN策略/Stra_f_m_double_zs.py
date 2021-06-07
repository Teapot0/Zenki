import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks, get_bars, bond
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *

auth('15951961478', '961478')
get_query_count()

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['net_value'] = (1+hs300['rts_1']).cumprod()

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
close_rts_1 = close.pct_change(1)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0',
                     date_parser=dateparse)
money = close * volume * 10 ** (-8)
market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0',
                         date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0',
                 date_parser=dateparse)
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv', index_col='statDate')

#股息率
df_bank = finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.code=='801780'))
# 回购
df_bond = bond.run_query(query(bond.REPO_DAILY_PRICE).filter(bond.REPO_DAILY_PRICE.name=='GC182').limit(2000))
df_t1 = pd.merge(df_bond, df_bank, on='date')
df_t1 = df_t1[['date','close','dividend_ratio']]
df_t1.index = pd.to_datetime(df_t1['date'])
# 当风险偏好<0不持股
df_t1['licha'] = (df_t1['close'].rolling(60).mean() - df_t1['dividend_ratio'].rolling(60).mean()).diff(1)
df_t1['licha'] = df_t1['licha'].fillna(method='ffill')
df_t1['hs300'] = hs300['net_value']

# 回购和价格行情日期不同
df_repo = pd.DataFrame(columns=['licha'],index = close.index)
datelist = list(set(df_t1.index).intersection(set(df_repo.index)))
df_repo['licha'].loc[datelist] = df_t1['licha'].loc[datelist]
df_repo = df_repo.fillna(method='ffill')


# 每天财务选股

# # 5年ROE
roe_5 = roe_yeayly.rolling(5).mean()

# 每天财务选股
stock_list_panel = {}
all_stock = set()
for i in range(market_cap.shape[0]):
    date = market_cap.index[i]
    # 5年平均roe大于12%
    tmp_year = '{}-12-31'.format(market_cap.index[i].year - 1)
    roe_list = list((roe_5.loc[tmp_year][roe_5.loc[tmp_year] > 12]).index)
    # 市值大于100 pe大于25
    mc_100 = list(market_cap.iloc[i, :][market_cap.iloc[i, :] > 100].index)
    pe_25 = list(pe.iloc[i, :][pe.iloc[i, :] > 25].index)
    # 成交额大于1000万
    money_list = list(money.iloc[i, :][money.iloc[i, :] > 0.1].index)
    tmp_list = list(set(roe_list).intersection(set(mc_100), set(pe_25), set(money_list)))
    stock_list_panel[date] = tmp_list
    all_stock = all_stock.union(stock_list_panel[date])

tmp_close = close[all_stock]


# 择时
def std_rts_select_dp_zs(close, hs300, std_n1=20, std_n2=90, std1=0.2, std2=0.3,
                         rts_n1=5, rts_n2=20, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                         rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                         weight_n1=60, weight_n2=90, weight_n3=180, weight_n4=250,
                         weight1=0.1, weight2=0.2, weight3=0.3, weight4=0.4, top_number=10, hold_time=5,comm_fee=0.002):
    hs300['rts_1'] = hs300['close'].pct_change(1)
    max_N = max(std_n2, rts_n6, weight_n4)

    std_list = {}
    close_rts_1 = close.pct_change(1)
    std_l1 = close_rts_1.rolling(std_n1).std() * sqrt(std_n1)
    std_l2 = close_rts_1.rolling(std_n2).std() * sqrt(std_n2)
    for date in close_rts_1.index:
        tmp1 = list(std_l1.loc[date][std_l1.loc[date] < std1].index)
        tmp2 = list(std_l2.loc[date][std_l2.loc[date] < std2].index)
        std_list[date] = list(set(tmp1).intersection(tmp2))  # 每一天股票池

    rts_list = {}
    close_rts1 = close.pct_change(rts_n1).sub(hs300['close'].pct_change(rts_n1), axis=0)
    close_rts2 = close.pct_change(rts_n2).sub(hs300['close'].pct_change(rts_n2), axis=0)
    close_rts3 = close.pct_change(rts_n3).sub(hs300['close'].pct_change(rts_n3), axis=0)
    close_rts4 = close.pct_change(rts_n4).sub(hs300['close'].pct_change(rts_n4), axis=0)
    close_rts5 = close.pct_change(rts_n5).sub(hs300['close'].pct_change(rts_n5), axis=0)
    close_rts6 = close.pct_change(rts_n6).sub(hs300['close'].pct_change(rts_n6), axis=0)

    for date in close_rts_1.index:  # 去掉NA
        r1 = list(close_rts1.loc[date][close_rts1.loc[date] > rts1].index)
        r2 = list(close_rts2.loc[date][close_rts2.loc[date] > rts2].index)
        r3 = list(close_rts3.loc[date][close_rts3.loc[date] > rts3].index)
        r4 = list(close_rts4.loc[date][close_rts4.loc[date] > rts4].index)
        r5 = list(close_rts5.loc[date][close_rts5.loc[date] > rts5].index)
        r6 = list(close_rts6.loc[date][close_rts6.loc[date] > rts6].index)
        rts_list[date] = list(set(r1).intersection(r2, r3, r4, r5, r6))  # 每一天股票池

    while len(rts_list) != len(std_list):
        print('std list and rts list same length')
        break

    rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
    rts_f2 = close.pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
    rts_f3 = close.pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)
    rts_f4 = close.pct_change(weight_n4).sub(hs300['close'].pct_change(weight_n4), axis=0)
    weight = weight1 * rts_f1 + weight2 * rts_f2 + weight3 * rts_f3 + weight4 * rts_f4

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]

        if (i - max_N - 1) % hold_time == 0:
            stocklist_financial = list(set(std_list[date]).intersection(rts_list[date], stock_list_panel[date]))
            stocklist_weighted = list(
                weight[stocklist_financial].loc[date].sort_values(ascending=False).index)  # 买入的股票
            if (top_number == 'full') | (len(stocklist_weighted) <= top_number):
                buy_list = stocklist_weighted
            else:
                buy_list = stocklist_weighted[:top_number]
            initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -0.025].index)
        hold_list = list(set(buy_list).difference(set(sell_list)))  #

        if (i - max_N - 1) % hold_time == 0:
            if df_repo['licha'].loc[date] > 0:
                out_df['hold_daily'].loc[date1] = hold_list
                out_df['daily_rts'].loc[date1] = close_rts_1[hold_list].loc[date1].mean() - comm_fee
            else:
                out_df['hold_daily'].loc[date1] = []
                out_df['daily_rts'].loc[date1] = 0

        else:
            if df_repo['licha'].loc[date] > 0:
                out_df['hold_daily'].loc[date1] = hold_list
                out_df['daily_rts'].loc[date1] = close_rts_1[hold_list].loc[date1].mean()
            else:
                out_df['hold_daily'].loc[date1] = []
                out_df['daily_rts'].loc[date1] = 0

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = std_rts_select_dp_zs(tmp_close, hs300, std_n1=10, std_n2=60, std1=0.2, std2=0.2,
                                 rts_n1=10, rts_n2=40, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                                 rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                                 weight_n1=10, weight_n2=90, weight_n3=180, weight_n4=250,
                                 weight1=0.1, weight2=0.2, weight3=0.3, weight4=0.4, top_number=10, hold_time=5, comm_fee=0.002)

plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)





