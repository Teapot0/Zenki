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

close_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv', index_col='Unnamed: 0')
vol_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv', index_col='Unnamed: 0')
high_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/high_1m.csv', index_col='Unnamed: 0')
low_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv', index_col='Unnamed: 0')


close_1m_rts = close_1m.pct_change(1)
hl_1m_rts = (high_1m / low_1m)-1
cl_1m_rts = (close_1m / low_1m) -1
stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]

hs300_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/510300_1m.csv', index_col='Unnamed: 0')
hs300_1m['rts'] = hs300_1m['close'].pct_change(1)
hs300_1m['hc_rts'] = (hs300_1m['close'] / hs300_1m['low'])-1


# ex_rts = cl_1m_rts.sub(hs300_1m['hc_rts'],axis=0)
ex_rts = cl_1m_rts.sub(cl_1m_rts.mean(axis=1),axis=0)

a_big = vol_1m.rolling(240, min_periods=1).quantile(0.8)
# a_small = vol_1m.rolling(240, min_periods=1).quantile(0.1)

b = ex_rts[vol_1m > a_big]
# b2 = ex_rts[vol_1m < a_small].rolling(240, min_periods=1).sum()

c1 = abs(b)[b > 0].rolling(240, min_periods=1).sum()
c2 = abs(b)[b < 0].rolling(240, min_periods=1).sum()

factor = (c1/c2).iloc[239::240]
factor = factor.rolling(10).mean()
factor.index = [x.split(' ')[0] for x in factor.index]



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


factor = factor.dropna(how='all')
z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
                                         hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)




