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
from scipy.stats import rankdata
from basic_funcs.update_data_funcs import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')
low_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv',index_col='Unnamed: 0')
high_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv',index_col='Unnamed: 0')
highlimit_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv',index_col='Unnamed: 0')
vol_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv',index_col='Unnamed: 0')
mon_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv',index_col='Unnamed: 0')
close_daily = clean_close(close_daily,low_daily, highlimit_daily)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# 每天停牌的
pause_list = vol_daily.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')

# 去掉ST，停牌
for date in tqdm(close_daily.index):
    ex_list = list(set(pause_list.loc[date]).union(set(st_df.loc[date][st_df.loc[date]==True].index)))
    close_daily.loc[date][ex_list] = np.nan

daily_rts = close_daily.pct_change(1)

z_df = close_daily.isna().replace(True,0)
z_df = z_df.replace(False,1)



def scale(factor):
    return (factor.div(abs(factor).sum(axis=1), axis=0)).clip(lower=-1, upper=1)


def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin) + 1


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1


def wmav(x):
    n = len(x)
    w = np.arange(1, n+1, 1)
    w = w / np.sum(w)
    return (x * w).sum()


def decay_linear(factor,N=10):
    return factor.rolling(N).apply(lambda x:wmav(x))

# f1
a = np.sqrt(mon_daily).diff(1)
factor = a.rolling(10).sum()
factor.index = [x.split(' ')[0] for x in factor.index]

# f2
f2 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_002.csv',index_col='Unnamed: 0')
factor = f2.diff(1)

# f3
tmp = np.sqrt(abs(mon_daily)).diff(1)
factor = decay_linear(tmp,N=10)

# f4
factor = scale(f2).diff(1).rolling(10).sum()

# f5
factor= scale(f2.rolling(10).sum())

# f6
a = f2.diff(1).rolling(10).max()
tmp= (a.rolling(10).max()).rolling(10).max()
factor = ts_argmin(tmp)

# f7
a = f2.diff(1).rolling(10).max()
tmp= (a.rolling(10).max()).rolling(10).max()
factor = ts_argmax(ts_argmin(tmp))

date_1 = factor.index
date_5 = factor.index[::5]
date_10 = factor.index[::10]

date_list = [date_1, date_5, date_10]
for i in range(len(date_list)):
    d = date_list[i]
    z1 = get_ic_table(factor=factor, rts=daily_rts, buy_date_list=d)
    plt.plot(z1['ic'].values)
    plt.title('IC')
    plt.savefig('/Users/caichaohong/Desktop/{}.png'.format(i+1))
    plt.close()

    plt.plot(z1['ic'].cumsum().values)
    plt.title('IC_CUMSUM')
    plt.savefig('/Users/caichaohong/Desktop/{}_CUMSUM.png'.format(i + 1))
    plt.close()

    print ('IC={}, IC_STD={}'.format(z1['ic'].mean(), z1['ic'].std()))

hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]

z = quantile_factor_test_plot(factor=factor, rts=daily_rts, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=20, plot_title=False, weight="avg",comm_fee=0.003)





