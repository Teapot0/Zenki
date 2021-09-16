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

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

vol_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv', index_col='Unnamed: 0', date_parser=dateparse)
low_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv', index_col='Unnamed: 0', date_parser=dateparse)
close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv', index_col='Unnamed: 0', date_parser=dateparse)

close_5m_rts = close_5m.pct_change(1)
stocks = list(close_5m.columns)

start = close_5m.index[0].strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]
end = close_5m.index[-1].strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


tmp_vol_15 = vol_5m.resample('15T', closed='right', label='right').sum()
vol_15 = tmp_vol_15.ix[~(tmp_vol_15==0).all(axis=1), :]

h1 = high_5m.resample('15T', closed='right', label='right').max()
l1 = low_5m.resample('15T', closed='right', label='right').min()
al_temp = (h1 / l1).dropna(how='all')

tmp_std1 = vol_15.rolling(16).std()
tmp_std2 = al_temp.rolling(16).std()
tmp_factor = tmp_std1 * tmp_std2

factor = tmp_factor.iloc[15::16,]
factor.index = [x.strftime('%Y-%m-%d %H:%M:%S').split(' ')[0] for x in factor.index]


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






