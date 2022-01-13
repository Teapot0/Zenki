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

close_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')

drop_list = list(close_1m.index[239::240]) +list(close_1m.index[238::240])+ list(close_1m.index[237::240])+ list(close_1m.index[236::240])
close_1m.drop(index=drop_list,inplace=True)

start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

stocks = list(close_1m.columns)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300 = hs300.loc[(hs300.index>='2018-01-01') & (hs300.index <='2021-06-29') ]
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
# hs300_1m = pd.read_parquet('./data/1m_data_agg/510300_1m.parquet', index_col='Unnamed: 0')
# hs300_1m['rts'] = hs300_1m['close'].pct_change(1)
# hs300_1m['hc_rts'] = (hs300_1m['close'] / hs300_1m['low'])-1


diff = close_1m - close_1m.shift(1)
close_1m_rts = diff/close_1m

log_upper = close_1m_rts[close_1m_rts > 0]
log_down = close_1m_rts[close_1m_rts < 0]

s1 = log_upper.rolling(236,min_periods=1).std()
s2 = log_down.rolling(236,min_periods=1).std()


factor = s1.iloc[235::236] / s2.iloc[235::236]
factor.index = [x.strftime('%Y-%m-%d') for x in factor.index]
factor = factor.fillna(0)

# factor.to_parquet('./factor_test/factor_value/.parquet')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)


#
# factor = factor.dropna(how='all')
# z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
#                                          hold_time=5, plot_title=False, weight="avg",index_pool='zz500', comm_fee=0.003)




