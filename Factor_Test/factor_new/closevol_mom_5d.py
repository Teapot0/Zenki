import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
from jqdatasdk import auth, get_query_count,get_industries,get_industry_stocks, get_index_stocks
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *
auth('13382017213', 'Aasd120120')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

industry_reverse = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/industry_reverse.csv',index_col='Unnamed: 0')
zz_list = get_index_stocks('000852.XSHG', date = '2021-08-12')

close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
low_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
mon_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
mon_daily = mon_daily * 10**(-4)
high_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
highlimit_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv',start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)
close_daily = clean_close(close_daily,low_daily, highlimit_daily)
open_rts = open_daily.pct_change(1)

net_amount_main = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv', start_time='2018-01-01', end_time='2021-08-26', stock_list=zz_list)

# hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300 = read_excel_select('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx',start_date='2018-01-01', end_date='2021-08-26')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# 每天停牌的
pause_list = vol_daily.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')
st_df = st_df[zz_list]

# 去掉ST，停牌
for date in tqdm(close_daily.index):
    ex_list = list(set(pause_list.loc[date]).union(set(st_df.loc[date][st_df.loc[date]==True].index)))
    close_daily.loc[date][ex_list] = np.nan

daily_rts = close_daily.pct_change(1)

z_df = close_daily.isna().replace(True,0)
z_df = z_df.replace(False,1)

industry_reverse = industry_reverse[close_daily.columns]

close_open_rts = close_daily / open_daily - 1

vol_5 = vol_daily.rolling(5).mean()
a = vol_daily / vol_5
b = close_open_rts.sub(hs300['rts'],axis=0)
b_f = (b < 0.25)*1

big_ratio = net_amount_main/mon_daily.rolling(60).sum()
big_ratio_rank = big_ratio.rank(ascending=False,axis=1)
# big_ratio = big_ratio.sub(big_ratio.mean(axis=1), axis=0)

c = (big_ratio_rank-500) / abs(big_ratio_rank).rolling(5).mean()

factor = a*b*c
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/closevol_momentum_5d.csv')

date_1 = factor.index
date_5 = factor.index[::5]
date_10 = factor.index[::10]

date_list = [date_1, date_5, date_10]
for i in range(len(date_list)):
    d = date_list[i]
    z1 = get_ic_table_open(factor=factor, open_rts=open_rts, buy_date_list=d)
    plt.plot(z1['ic'].values)
    plt.title('IC')
    plt.savefig('/Users/caichaohong/Desktop/{}.png'.format(i+1))
    plt.close()

    plt.plot(z1['ic'].cumsum().values)
    plt.title('IC_CUMSUM')
    plt.savefig('/Users/caichaohong/Desktop/{}_CUMSUM.png'.format(i + 1))
    plt.close()

    print ('IC={}, IC_STD={}'.format(z1['ic'].mean(), z1['ic'].std()))


z = quantile_factor_test_plot_open(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
                                   hold_time=3, plot_title=False, weight="avg", comm_fee=0.003)














