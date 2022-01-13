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

close_30m = pd.read_parquet('./data/30m_data_agg/close_30m.parquet')
vol_30m = pd.read_parquet('./data/30m_data_agg/volume_30m.parquet')
open_30m = pd.read_parquet('./data/30m_data_agg/open_30m.parquet')


stocks = list(close_30m.columns)
start = close_30m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_30m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

N = 8
close_30m_rts = close_30m / open_30m

res1 = close_30m.rolling(8, min_periods=1).corr(vol_30m)
res2 = close_30m.rolling(16, min_periods=1).corr(vol_30m)
res3 = close_30m.rolling(24, min_periods=1).corr(vol_30m)
res4 = close_30m.rolling(32, min_periods=1).corr(vol_30m)


# res2 = close_30m_rts.rolling(N, min_periods=1).corr(vol_30m)
# res3 = close_30m_rts.rolling(N,min_periods=1).corr(vol_30m.shift(30))
# res4 = vol_30m.rolling(N,min_periods=1).corr(vol_30m.shift(30))


f1 = res1.iloc[7::8,]
f2 = res2.iloc[7::8,]
f3 = res2.iloc[7::8,]
f4 = res3.iloc[7::8,]
f1.index = [x.strftime('%Y-%m-%d') for x in f1.index]
f2.index = [x.strftime('%Y-%m-%d') for x in f2.index]
f3.index = [x.strftime('%Y-%m-%d') for x in f3.index]
f4.index = [x.strftime('%Y-%m-%d') for x in f4.index]


f = f2 - f1


ic_test(index_pool='hs300', factor = f1 + (f2-f3), open_rts=open_rts)
ic_test(index_pool='zz500', factor = f, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= f, open_rts=open_rts)


# f1.to_parquet('./factor_test/factor_value/30m_cv_corr.parquet')

