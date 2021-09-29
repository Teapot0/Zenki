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
open_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/open_1m.csv', index_col='Unnamed: 0')
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

vol_sum = vol_daily.append([vol_daily]*239)
vol_sum = vol_sum.sort_index()
vol_sum.index = volume_1m.index

weight = volume_1m / vol_sum


tmp1 = ((close_1m - open_1m) * (weight-1)).rolling(240, min_periods=1).skew()

factor = tmp1.iloc[239::240]
factor.index = [x.split(' ')[0] for x in factor.index]
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






