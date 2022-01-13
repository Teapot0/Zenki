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

high_1m = pd.read_parquet('./data/1m_data_agg/high_1m.parquet')
low_1m = pd.read_parquet('./data/1m_data_agg/low_1m.parquet')
close_1m = pd.read_parquet('./data/1m_data_agg/close_1m.parquet')

drop_list = list(high_1m.index[239::240]) +list(high_1m.index[238::240])+ list(high_1m.index[237::240])
high_1m.drop(index=drop_list,inplace=True)
low_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)




stocks = list(high_1m.columns)
start = high_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = high_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time= start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)

tmp1 = (high_1m / close_1m)
tmp2 = (close_1m / low_1m)

tmp_f1 = tmp1.rolling(237).sum()
tmp_f2 = tmp1.rolling(237).sum()

f51 = tmp_f1.iloc[236::237,]
f52 = tmp_f2.iloc[236::237,]
f51.index = [x.strftime('%Y-%m-%d') for x in f51.index]
f52.index = [x.strftime('%Y-%m-%d') for x in f52.index]

f51_corr = f51.rolling(5).corr(close_rts)
ic_test(index_pool='hs300', factor=f51_corr, open_rts=open_rts)
ic_test(index_pool='zz500', factor=f51_corr, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=f51_corr, open_rts=open_rts)



# factor = factor.dropna(how='all')
# z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
#                                          hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)


