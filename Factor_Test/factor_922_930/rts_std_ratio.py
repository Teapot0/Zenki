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

close_1m_rts = close_1m.pct_change(1)
stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


# row_n = [237+240*i+j  for i in range(int(close_1m_rts.shape[0]/240)) for j in range(3)]

ret5 = close_1m_rts.rolling(5).mean()
ret10_min = close_1m_rts.rolling(10).min()

pt1 = close_1m_rts - ret5

pt1_threshold = pt1.rolling(237).quantile(0.5)
pt1_df = pt1_threshold.iloc[236::240,]

pt1_df2 = pt1_df.append([pt1_df]*239)
pt1_df2 = pt1_df2.sort_index()
pt1_df2.index = close_1m_rts.index
std_df = pt1[pt1 <= pt1_df2].rolling(240,min_periods=1).std()
pt1_std = std_df.iloc[239::240]

pt2 = close_1m_rts - ret10_min
pt2_std = pt2.rolling(240,min_periods=1).std().iloc[239::240]



factor = pt1_std / pt2_std
factor.index = [x.split(' ')[0] for x in factor.index]
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






