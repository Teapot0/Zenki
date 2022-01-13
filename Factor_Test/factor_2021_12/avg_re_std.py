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
open_1m = pd.read_parquet('./data/1m_data_agg/open_1m.parquet')

drop_list = list(vol_1m.index[239::240]) + list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)
open_1m.drop(index=drop_list,inplace=True)

avg = (open_1m + close_1m) / 2
delta = (avg - avg.shift(1))

N = 237
delta_std = delta.rolling(N).std()
delta_sum = abs(delta).rolling(N).sum()
res = delta.iloc[236::237,] / delta_sum.iloc[236::237,]

res.index = [x.strftime('%Y-%m-%d') for x in res.index]
res.to_parquet('./data/1m_data_agg/avg_res_std.parquet')

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

ic_test(index_pool='hs300', factor=res, open_rts=open_rts)
ic_test(index_pool='zz500', factor=res, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=res, open_rts=open_rts)






