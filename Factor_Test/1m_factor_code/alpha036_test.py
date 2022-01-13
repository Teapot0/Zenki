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

vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')
close_1m = pd.read_parquet('./data/1m_data_agg/close_1m.parquet')

drop_list = list(vol_1m.index[239::240]) + list(vol_1m.index[238::240]) + list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

stocks = list(vol_1m.columns)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)
close_rts = close_daily.pct_change(1)

ret1 = (close_1m - close_1m.shift(1)) / close_1m.shift(1)
ret2 = (vol_1m - vol_1m.shift(1)) / vol_1m.shift(1)

ret1.replace([np.inf, -np.inf], 0, inplace=True)
ret2.replace([np.inf, -np.inf], 0, inplace=True)

ex_ret1 = ret1.sub(ret1.mean(axis=1), axis=0)
ex_ret2 = ret2.sub(ret2.mean(axis=1), axis=0)

df_filter1 = (ex_ret1 > 0)
df_filter2 = (ex_ret2 > 0)

first_30 = [list(vol_1m.index[x::237]) for x in range(30)]
last_30 = [list(vol_1m.index[x::237]) for x in range(210,237)] # 好像应该是237....参考comb_alpha1的
f30 = []
for l in first_30:
    f30 += l
l30 = []
for l in last_30:
    l30 += l

tmp_df1_f = df_filter1.loc[f30].sort_index()
tmp_df1_l = df_filter1.loc[l30].sort_index()
tmp_df2_f = df_filter2.loc[f30].sort_index()
tmp_df2_l = df_filter2.loc[l30].sort_index()

tmp_factor1_f = tmp_df1_f.rolling(30,min_periods=1).sum()
tmp_factor1_l = tmp_df1_l.rolling(26,min_periods=1).sum()
tmp_factor2_f = tmp_df2_f.rolling(30,min_periods=1).sum()
tmp_factor2_l = tmp_df2_l.rolling(26,min_periods=1).sum()

f1_f30 = tmp_factor1_f.iloc[29::30]
f1_l30 = tmp_factor1_l.iloc[26::27]
f2_f30 = tmp_factor2_f.iloc[29::30]
f2_l30 = tmp_factor2_l.iloc[26::27]

f1_f30.index = [x.strftime('%Y-%m-%d') for x in f1_f30.index]
f1_l30.index = [x.strftime('%Y-%m-%d') for x in f1_l30.index]
f2_f30.index = [x.strftime('%Y-%m-%d') for x in f2_f30.index]
f2_l30.index = [x.strftime('%Y-%m-%d') for x in f2_l30.index]


z1 = close_1m.loc[close_1m.index[209::236]]
z2 = close_1m.loc[close_1m.index[236::237]]
z1.index = [x.strftime('%Y-%m-%d') for x in z1.index]
z2.index = [x.strftime('%Y-%m-%d') for x in z2.index]
l30_rts = z2 / z1 - 1
f3 = (l30_rts / f1_l30) * -1
f33 = (l30_rts / f2_l30) * -1
f33[np.isinf(f33)] = 0


z1 = close_1m.loc[close_1m.index[0::237]]
z2 = close_1m.loc[close_1m.index[29::237]]
z1.index = [x.strftime('%Y-%m-%d') for x in z1.index]
z2.index = [x.strftime('%Y-%m-%d') for x in z2.index]
f30_rts = z2 / z1 - 1
f4 = f30_rts / (f2_f30 + 1)

f5 = f3.rank(axis=1) - f33.rank(axis=1)
ic_test(index_pool='hs300', factor=f5, open_rts=open_rts)
ic_test(index_pool='zz500', factor=f5, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=f5, open_rts=open_rts)

f3.to_parquet('./factor_test/factor_value/f3.parquet')

# factor = factor.dropna(how='all')
# z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
#                                          hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)











