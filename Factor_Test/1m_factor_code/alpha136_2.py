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
vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')
close_1m_rts = close_1m.pct_change(1)

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(close_1m.index[239::240]) + list(close_1m.index[238::240])+ list(close_1m.index[237::240])
close_1m.drop(index=drop_list,inplace=True)
vol_1m.drop(index=drop_list,inplace=True)


close_r_th = close_1m_rts.rolling(237,min_periods=1).quantile(0.7)
vol_r_th = vol_1m.rolling(237,min_periods=1).quantile(0.7)
na_rows = list(set(close_1m.index).difference(set(close_1m.index[236::237])))

close_r_th.loc[na_rows] = np.nan
vol_r_th.loc[na_rows] = np.nan
close_r_th = close_r_th.fillna(method='bfill')  # 先向后找值，再填充
vol_r_th = vol_r_th.fillna(method='bfill')  # 先向后找值，再填充

close_up = close_1m[close_1m_rts > close_r_th]
vol_up = close_1m[vol_1m > vol_r_th]

res = close_up.rolling(237,min_periods=1).mean() / close_1m.rolling(237,min_periods=1).mean()
res2 = vol_up.rolling(237,min_periods=1).mean() / close_1m.rolling(237,min_periods=1).mean()

f1 = res.iloc[236::237,]
f2 = res2.iloc[236::237,]
f1.index = [x.strftime('%Y-%m-%d') for x in f1.index]
f2.index = [x.strftime('%Y-%m-%d') for x in f2.index]

f = f1.rolling(2).min() - f2.rolling(2).max()
ic_test(index_pool='hs300', factor = f, open_rts=open_rts)
ic_test(index_pool='zz500', factor = f, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= f, open_rts=open_rts)






