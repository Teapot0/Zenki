import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/close_30m.csv', index_col='Unnamed: 0')
vol_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volume_30m.csv', index_col='Unnamed: 0')

close_30m_rts = close_30m.pct_change(1)

stocks = list(close_30m.columns)
# daily
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time='2017-01-01', end_time='2021-09-07',stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time='2017-01-01', end_time='2021-09-07',stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time='2017-01-01', end_time='2021-09-07',stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)

# 30分钟占比
vol_daily_8 = vol_daily.append(7*[vol_daily])
vol_daily_8 = vol_daily_8.sort_index()
vol_daily_8.index = close_30m.index

rts_daily_8 = close_rts.append(7*[close_rts])
rts_daily_8 = rts_daily_8.sort_index()
rts_daily_8.index = close_30m.index

vol_daily_pct = vol_30m / vol_daily_8
rts_daily_pct = close_30m_rts / rts_daily_8

# ----------
col_name = [x.split(' ')[1] for x in rts_daily_pct.iloc[:8,].index]
corr_300 = pd.DataFrame(0,index=close_daily.index, columns=col_name)
corr_500 = pd.DataFrame(0,index=close_daily.index, columns=col_name)
corr_1000 = pd.DataFrame(0,index=close_daily.index, columns=col_name)

for i in tqdm(range(0,8)):
    tmp_name = col_name[i]
    tmp_df = rts_daily_pct.iloc[i::8]
    tmp_name = tmp_df.index[0].split(' ')[1]
    tmp_df.index = [x.split(' ')[0] for x in tmp_df.index]
    z1 = get_ic_table_open_index(factor=tmp_df, open_rts=open_rts, buy_date_list=tmp_df.index, index_pool='hs300')
    z2 = get_ic_table_open_index(factor=tmp_df, open_rts=open_rts, buy_date_list=tmp_df.index, index_pool='zz500')
    z3 = get_ic_table_open_index(factor=tmp_df, open_rts=open_rts, buy_date_list=tmp_df.index, index_pool='zz1000')
    corr_300[tmp_name] = z1['ic']
    corr_500[tmp_name] = z2['ic']
    corr_1000[tmp_name] = z3['ic']


rts_corr_300 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/rtspct_corr_30m_300.csv')
rts_corr_500 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/rtspct_corr_30m_500.csv')
rts_corr_1000 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/rtspct_corr_30m_1000.csv')
vol_corr_300 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volpct_corr_30m_300.csv')
vol_corr_500 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volpct_corr_30m_500.csv')
vol_corr_1000 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volpct_corr_30m_1000.csv')

# corr_300.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/rtspct_corr_30m_300.csv')
# corr_500.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/rtspct_corr_30m_500.csv')
# corr_1000.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/rtspct_corr_30m_1000.csv')

corr_1000.mean().sort_values()


def ic_test(index_pool,factor):
    date_1 = factor.index
    date_5 = factor.index[::5]
    date_10 = factor.index[::10]

    date_list = [date_1, date_5, date_10]
    print(index_pool + ' : - - - - - - - - - - -')
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
                                         hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)












