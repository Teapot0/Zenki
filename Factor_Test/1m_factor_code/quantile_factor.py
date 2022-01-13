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

#
drop_list = list(vol_1m.index[239::240]) +list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)
close_1m_rts.drop(index=drop_list,inplace=True)


a1 = pd.read_parquet('./data/close_1m_rts_daily_q85.parquet')
a2 = pd.read_parquet('./data/close_1m_rts_daily_q15.parquet')
a3 = pd.read_parquet('./data/vol_1m_rts_daily_q85.parquet')
a4 = pd.read_parquet('./data/vol_1m_rts_daily_q15.parquet')

f1 = (vol_1m**0.1)[close_1m_rts >= a1].rolling(237,min_periods=1).std()
f2 = (vol_1m**0.1)[close_1m_rts <= a2].rolling(237,min_periods=1).std()

x1 = f1.iloc[236::237,]
x1.index = [x.strftime('%Y-%m-%d') for x in x1.index]
x2 = f2.iloc[236::237,]
x2.index = [x.strftime('%Y-%m-%d') for x in x2.index]

f3 = close_1m_rts[vol_1m >= a3].rolling(237,min_periods=1).std()
f4 = close_1m_rts[vol_1m <= a4].rolling(237,min_periods=1).std()

x3 = f3.iloc[236::237,]
x3.index = [x.strftime('%Y-%m-%d') for x in x3.index]
x4 = f4.iloc[236::237,]
x4.index = [x.strftime('%Y-%m-%d') for x in x4.index]


stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

z = x1 * x3
z.to_parquet('./factor_test/factor_value/quantile_x1_mul_x3.parquet')

x1.to_parquet('./factor_test/factor_value/quantile_x1.parquet')
x2.to_parquet('./factor_test/factor_value/quantile_x2.parquet')

ic_test(index_pool='hs300', factor=x1, open_rts=open_rts)
ic_test(index_pool='zz500', factor=x1, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=x1, open_rts=open_rts)







