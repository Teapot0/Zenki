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

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv', index_col='Unnamed: 0', date_parser=dateparse)
vol_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv', index_col='Unnamed: 0', date_parser=dateparse)

close_5m_rts = close_5m.pct_change(1)
stocks = list(close_5m.columns)

start = close_5m.index[0].strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]
end = close_5m.index[-1].strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


vol_10 = vol_5m.resample('10T', closed='right', label='right').sum()
vol_10 = vol_10.ix[~(vol_10==0).all(axis=1), :]
close_10 = close_5m.resample('10T', closed='right', label='right').last()
close_10 = close_10.dropna(how='all')
close_10_rts = close_10.pct_change(1)


a = close_10_rts.rolling(24, min_periods=1).quantile(0.5)
b = vol_10[close_10_rts > a].rolling(24, min_periods=1).std()
factor = b.iloc[23::24,]
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







