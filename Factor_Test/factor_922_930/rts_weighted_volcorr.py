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
money_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/money_1m.csv', index_col='Unnamed: 0')
volume_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv', index_col='Unnamed: 0')

close_1m_rts = close_1m.pct_change(1)
stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)

vwap_1m = money_1m / volume_1m
ret = vwap_1m.pct_change(5)

v1 = vol_daily.append([vol_daily]*239)
v1 = v1.sort_index()
v1.index = volume_1m.index
vol_mean = v1 / 240

filter1 = volume_1m > vol_mean

tmp1 = (ret * volume_1m).rolling(5, min_periods=1).sum()
tmp_df1 = tmp1[filter1]
tmp_df2 = volume_1m[filter1]

tmp_factor = tmp_df1.rolling(240,min_periods=1).corr(tmp_df2)

factor = tmp_factor.iloc[239::240]
factor.index = [x.split(' ')[0] for x in factor.index]
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






