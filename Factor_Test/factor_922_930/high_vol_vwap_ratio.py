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
money_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/money_1m.csv', index_col='Unnamed: 0')

close_1m_rts = close_1m.pct_change(1)
stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
mon_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


daily_vwap = mon_daily / vol_daily

vol_high = vol_1m.rolling(240).quantile(0.5)
ex_index = list(set(vol_1m.index).difference(set(vol_high.index[239::240])))
vol_high.loc[ex_index] = np.nan

vol_high = vol_high.fillna(method='bfill')
filter_df = vol_1m < vol_high

qvwap = money_1m[filter_df].rolling(240, min_periods=1).sum() / vol_1m[filter_df].rolling(240, min_periods=1).sum()

tmp_factor = qvwap.iloc[239::240,]
tmp_factor.index = [x.split(' ')[0] for x in tmp_factor.index]

factor = tmp_factor / daily_vwap

# factor = factor.rolling(5, min_periods=1).mean()
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






