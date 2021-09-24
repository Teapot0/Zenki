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

stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
high_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv' , start_time=start, end_time = end,stock_list=stocks)
low_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


high_d1 = high_daily.append([high_daily]*239 )
high_d1 = high_d1.sort_index()
high_d1.index = close_1m.index

low_d1 = low_daily.append([low_daily]*239 )
low_d1 = low_d1.sort_index()
low_d1.index = close_1m.index

RPP = (close_1m - high_d1) / (low_d1 - high_d1)
filter1 = (RPP<0.3).rolling(240).mean()

factor = filter1.iloc[239::240]
factor.index = [x.split(' ')[0] for x in factor.index]
factor = factor.rolling(5).max()
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






