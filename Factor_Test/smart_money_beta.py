import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
from jqdatasdk import auth, get_query_count,get_industries,get_industry_stocks
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

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')
vol_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv',index_col='Unnamed: 0')
open_rts = open_daily.pct_change(1)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
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


close_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv', index_col='Unnamed: 0')
vol_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv', index_col='Unnamed: 0')
low_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv', index_col='Unnamed: 0')

close_1m_rts = close_1m.pct_change(1)

# close_open_rts = close_1m / open_1m - 1
close_low_rts = close_1m / low_1m -1
s1 = 1 / (abs(close_low_rts) / (((vol_1m)**0.1)))

# factor
date_list = close_daily.index[(close_daily.index>='2021-01-01') & (close_daily.index<='2021-04-30')]


def smart_money_test(type,interval_n):
    # 10天interval_n = 2399
    factor = pd.DataFrame(index= date_list, columns=close_1m.columns)

    for i in tqdm(range(interval_n,s1.shape[0], 240)):
        tmp_end = s1.index[i]
        tmp_date = tmp_end.split(' ')[0]
        tmp_df = s1.iloc[i-interval_n:i,:]
        temp_vol = vol_1m.iloc[i-interval_n:i,:]
        temp_close = close_1m.iloc[i-interval_n:i, :]

        tmp_rank = tmp_df.rank(axis=0, pct=True)
        smart = pd.DataFrame(tmp_rank.values > 0.7, index=tmp_rank.index, columns=tmp_rank.columns)
        smart_money = smart[smart]
        smart_money = smart_money.fillna(0)

        if type == 'vwap':
            smart_vol = temp_vol * smart_money

            temp_w = temp_vol / temp_vol.sum()
            temp_total_f = (temp_close * temp_w).sum()  #VWAP

            smart_w = smart_vol / smart_vol.sum()
            smart_f = (temp_close * smart_w).sum()

        elif type == 'kurt':
            temp_total_f = temp_close.kurt(axis=0)
            smart_f = (temp_close * smart_money).kurt(axis=0)

        elif type == 'skew':
            temp_total_f = temp_close.skew(axis=0)
            smart_f = (temp_close * smart_money).skew(axis=0)

        elif type == 'mean_standard':
            temp_total_f = (temp_close - temp_close.mean()) / temp_close.std()
            smart_f  = (temp_close*smart_money - (temp_close*smart_money).mean()) / (smart_money*temp_close).std()

        temp_q = round(smart_f / temp_total_f, 4)
        factor.loc[tmp_date][temp_q.index] = temp_q

    factor = factor.dropna(how='all', axis=0)
    factor = factor.astype('float64')
    return factor
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/smart_money.csv')


factor = smart_money_test(type='vwap', interval_n=1199)


def ic_test(index_pool,factor):
    date_1 = factor.index
    date_5 = factor.index[::5]
    date_10 = factor.index[::10]

    date_list = [date_1, date_5, date_10]
    for i in range(len(date_list)):
        d = date_list[i]
        z1 = get_ic_table_open_index(factor=factor, open_rts=open_rts, buy_date_list=d, index_pool=index_pool)
        plt.plot(z1['ic'].values)
        plt.title('IC')
        plt.savefig('/Users/caichaohong/Desktop/{}.png'.format(i+1))
        plt.close()

        plt.plot(z1['ic'].cumsum().values)
        plt.title('IC_CUMSUM')
        plt.savefig('/Users/caichaohong/Desktop/{}_CUMSUM.png'.format(i + 1))
        plt.close()

        print ('{}: IC={}, IC_STD={}'.format(i,z1['ic'].mean(), z1['ic'].std()))


ic_test(index_pool='hs300', factor=factor)
ic_test(index_pool='zz500', factor=factor)
ic_test(index_pool='zz1000', factor=factor)


z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
                                         hold_time=5, plot_title=False, weight="avg",index_pool='zz1000', comm_fee=0.003)












