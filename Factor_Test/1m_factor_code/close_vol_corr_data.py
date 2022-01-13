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


start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

stocks = list(close_1m.columns)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

corr_PV = close_1m.rolling(237).corr(vol_1m)
tmpf1 = corr_PV.iloc[236::237,]
tmpf1.index = [x.strftime('%Y-%m-%d') for x in tmpf1.index]

first_30 = [list(vol_1m.index[x::237]) for x in range(30)]
last_30 = [list(vol_1m.index[x::237]) for x in range(210,237)]
f30 = []
for l in first_30:
    f30 += l
l30 = []
for l in last_30:
    l30 += l

close_1m_f30 = close_1m.loc[f30].sort_index()
close_1m_l30 = close_1m.loc[l30].sort_index()
vol_1m_f30 = vol_1m.loc[f30].sort_index()
vol_1m_l30 = vol_1m.loc[l30].sort_index()

corr_PV_f30 = close_1m_f30.rolling(30).corr(vol_1m_f30)
corr_PV_l30 = close_1m_l30.rolling(27).corr(vol_1m_l30)

tmpf1_f30 = corr_PV_f30.iloc[29::30,]
tmpf1_l30 = corr_PV_l30.iloc[26::27,]
tmpf1_f30.index = [x.strftime('%Y-%m-%d') for x in tmpf1_f30.index]
tmpf1_l30.index = [x.strftime('%Y-%m-%d') for x in tmpf1_l30.index]

ic_test(index_pool='hs300', factor=tmpf1_f30, open_rts=open_rts)
ic_test(index_pool='zz500', factor=tmpf1_f30, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=tmpf1_f30, open_rts=open_rts)

tmpf1 = tmpf1 * -1
tmpf1_f30 = tmpf1_f30 * -1
tmpf1_l30 = tmpf1_l30 * -1
tmpf1.to_parquet('./factor_test/factor_value/minus_daily_1m_close_vol_corr.parquet') # daily 的 corr * -1
tmpf1_f30.to_parquet('./factor_test/factor_value/minus_first30_1m_close_vol_corr.parquet')
tmpf1_l30.to_parquet('./factor_test/factor_value/minus_last30_1m_close_vol_corr.parquet')

# factor = factor.dropna(how='all')
# z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
#                                          hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)











