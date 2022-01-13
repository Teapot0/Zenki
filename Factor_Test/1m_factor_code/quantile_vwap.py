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

close_1m = pd.read_parquet('./data/1m_data_agg/close_1m.parquet')
close_1m_rts = pd.read_parquet('./data/1m_data_agg/close_1m_rts.parquet')
vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
high_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv' , start_time= start, end_time = end,stock_list=stocks)
low_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(vol_1m.index[239::240]) +list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

#
a1 = pd.read_parquet('./data/close_1m_rts_daily_q85.parquet')
a2 = pd.read_parquet('./data/close_1m_rts_daily_q15.parquet')

c1 = close_1m[close_1m_rts >= a1]
c2 = close_1m[close_1m_rts <= a2]

c1.index = [x.strftime("%Y-%m-%d") for x in c1.index]
c2.index = [x.strftime("%Y-%m-%d") for x in c2.index]

'''
c1 = close_1m[close_1m_rts >= a1]
v1 = vol_1m[close_1m_rts <= a2]

c1.index = [x.strftime("%Y-%m-%d") for x in c1.index]
v1.index = [x.strftime("%Y-%m-%d") for x in v1.index]

f1 = pd.DataFrame(index = open_daily.index, columns=c1.columns)

for d in tqdm(f1.index):
    tmp_c1 = c1.loc[d]
    tmp_v1 = v1.loc[d]
    tmp_w = tmp_v1 / tmp_v1.sum()
    tmp_vwap = (tmp_w * tmp_c1).sum()
    f1.loc[d] = tmp_vwap
'''


f1 = pd.read_parquet('./data/vwap_big85rts.parquet')
f2 = pd.read_parquet('./data/vwap_small15rts.parquet')
#
# vwap = pd.read_parquet('./data/vwap.parquet')
# vwap = vwap.replace(0,np.nan)


vwap1 = c1.rolling(237,min_periods=1).mean().iloc[236::237,]
vwap2 = c2.rolling(237,min_periods=1).mean().iloc[236::237,]

factor1 = f1/high_daily
factor2 = f2/low_daily


f1_ma5 = factor1.rolling(10).mean()
f2_ma5 = factor2.rolling(10).mean()

f1_max = factor1.rolling(10).max()
f2_max = factor2.rolling(10).max()

f1_std = factor1.rolling(10).std()
f2_std = factor2.rolling(10).std()

ic_test(index_pool='hs300', factor = f1 / f2, open_rts=open_rts)
ic_test(index_pool='zz500', factor = f1 / f2, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= f1 / f2, open_rts=open_rts)




