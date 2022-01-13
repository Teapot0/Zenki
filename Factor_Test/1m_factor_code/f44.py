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

drop_list = list(vol_1m.index[239::240]) +list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

last_30 = [list(vol_1m.index[x::237]) for x in range(210,237)]
l30 = []
for l in last_30:
    l30 += l

first_30 = [list(vol_1m.index[x::237]) for x in range(30)]
f30 = []
for l in first_30:
    f30 += l



close_1m_pct = (close_1m - close_1m.shift(1)) / close_1m.shift(1)
close_1m_pct_l30 = close_1m_pct.loc[l30].sort_index()
tmp_pct_abs = abs(close_1m_pct_l30)
tmp_pct_abs_sum = tmp_pct_abs.rolling(27).sum()
tmp_sum_l30 = tmp_pct_abs_sum.iloc[26::27,]
tmp_sum_l30.index = [x.strftime('%Y-%m-%d') for x in tmp_sum_l30.index]

close_1m_pct_f30 = close_1m_pct.loc[f30].sort_index()
tmp_pct_abs = abs(close_1m_pct_f30)
tmp_pct_abs_sum = tmp_pct_abs.rolling(30).sum()
tmp_sum_f30 = tmp_pct_abs_sum.iloc[29::30,]
tmp_sum_f30.index = [x.strftime('%Y-%m-%d') for x in tmp_sum_f30.index]

start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

stocks = list(close_1m.columns)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)


tmpf1_l30 = pd.read_parquet('./factor_test/factor_value/minus_last30_1m_close_vol_corr.parquet')
tmpf1_f30 = pd.read_parquet('./factor_test/factor_value/minus_first30_1m_close_vol_corr.parquet')


tmp_sum_rank_l30 = tmp_sum_l30.rank(axis=1, pct=True)
f44 = tmpf1_l30 * tmp_sum_rank_l30
# factor[np.isinf(factor)] = 0
# f44.to_parquet('./factor_test/factor_value/f44.parquet')

tmp_sum_rank_f30 = tmp_sum_f30.rank(axis=1, ascending=False,pct=True)
factor = tmpf1_f30 * tmp_sum_rank_f30

ic_test(index_pool='hs300', factor=f44, open_rts=open_rts)
ic_test(index_pool='zz500', factor=f44, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=f44, open_rts=open_rts)



# factor = factor.dropna(how='all')
# z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
#                                          hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)


