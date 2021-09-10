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

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv', index_col='Unnamed: 0')
vol_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv', index_col='Unnamed: 0')
low_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv', index_col='Unnamed: 0')
high_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv', index_col='Unnamed: 0')

close_5m_rts = close_5m.pct_change(1)

stocks = list(close_5m.columns)

close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time='2018-01-01', end_time='2021-07-30',stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time='2018-01-01', end_time='2021-07-30',stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time='2018-01-01', end_time='2021-07-30',stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


vol_daily_5m = vol_daily.append(47*[vol_daily])
vol_daily_5m = vol_daily_5m.sort_index()
vol_daily_5m.index = close_5m.index

vol_daily_pct = vol_5m / vol_daily_5m


vol_corr_300 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_500.csv',index_col='Unnamed: 0')
vol_corr_300 = vol_corr_300.dropna(how='all', axis=0)

v1 = vol_corr_300.shift(2)
v1_5 = v1.rolling(5).mean()


out = pd.DataFrame(index=v1_5.index, columns=['big', 'small'] )
for i in tqdm(range(v1_5.shape[0]-1)):
    tmp = v1_5.iloc[i,:].sort_values(ascending=False)
    big_time = tmp.index[0]
    small_time = tmp.index[-1]  # IC值最大最小
    out.iloc[i+1,0] = v1.iloc[i+1,:][big_time]
    out.iloc[i+1,1] = v1.iloc[i+1,:][small_time]

# plt.plot(out['small'].cumsum().values)



tt_index = pd.DataFrame(index=v1_5.columns, columns=['ii'])
tt_index['ii'] = list(range(48))


factor = pd.DataFrame(index=close_daily.index, columns=close_daily.columns)

for i in tqdm(range(7,factor.shape[0])):
    tmp = v1_5.iloc[i, :].sort_values(ascending=False)
    big_time = tmp.index[-1]
    # small_time = tmp.index[-1]  # IC值最大最小
    tmp_5m_vol_pct = vol_daily_pct.iloc[48*i:48*(i+1),]

    tmp_tt_index = tt_index['ii'].loc[big_time]
    tmp_df = tmp_5m_vol_pct.iloc[tmp_tt_index, :]
    factor.iloc[i,] = tmp_df


factor = factor.dropna(how='all', axis=0)


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











