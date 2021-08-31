import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss,get_money_flow
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

factor = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_008.csv', index_col='Unnamed: 0')

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
low_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv',index_col='Unnamed: 0')
highlimit_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv',index_col='Unnamed: 0')
close_daily = clean_close(close_daily,low_daily, highlimit_daily)

vol_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv',index_col='Unnamed: 0')
daily_rts = close_daily.pct_change(1)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# 每天停牌的
pause_list = vol_daily.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('~/Desktop/舆情/is_st.csv', index_col='Unnamed: 0')

# 去掉ST，停牌
stock_exclude_list = {}
for date in tqdm(close_daily.index):
    stock_exclude_list[date] = list(set(pause_list.loc[date]).union(set(st_df.loc[date])))

for date in tqdm(factor.index):
    ex_list = list(set(stock_exclude_list).intersection(factor.columns))
    factor.loc[date][ex_list] = np.nan


z_df = close_daily.isna().replace(True,0)
z_df = z_df.replace(False,1)
factor = factor * z_df


date_1 = factor.index
date_5 = factor.index[::5]
date_10 = factor.index[::10]


def get_ic_table(factor, rts, buy_date_list):
    # buy list 是换仓日
    out = pd.DataFrame(index=buy_date_list,columns=['ic', 'rank_ic'])
    for i in tqdm(range(1,len(buy_date_list))):
        date = buy_date_list[i]
        date1 = buy_date_list[i-1]
        tmp = pd.concat([factor.loc[date1],rts.loc[date]],axis=1)
        tmp.columns=['date1', 'date']
        out['ic'].loc[date] = tmp.corr().iloc[0,1]
        out['rank_ic'].loc[date] = tmp.rank().corr().iloc[0,1]
    return out


z1 = get_ic_table(factor=factor, rts=daily_rts, buy_date_list=date_10)


plt.plot(z1['ic'].values)
plt.title('IC')

z1['ic'].mean()

z = quantile_factor_test_plot(factor=factor, rts=daily_rts, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=5, plot_title=False, weight="avg",comm_fee=0.003)





