import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close_30m = pd.read_parquet('./data/30m_data_agg/close_30m.parquet')

close_120m_am = close_30m.iloc[3::8,]
close_120m_pm = close_30m.iloc[7::8,]
close_am_rts = close_120m_am.pct_change(1)
close_pm_rts = close_120m_pm.pct_change(1)

stocks = list(close_30m.columns)
start = close_30m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_30m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)
close_rts = close_daily.pct_change(1)


bench_120m = pd.read_parquet('./data/120m/512100.XSHG.parquet')
bench_am = bench_120m.iloc[::2,]
bench_pm = bench_120m.iloc[1::2,]
bench_am_rts = bench_am.pct_change(1)
bench_pm_rts = bench_pm.pct_change(1)


slope_am=pd.read_parquet('./data/apm_slope_am.parquet')
slope_pm=pd.read_parquet('./data/apm_slope_pm.parquet')
intercept_am=pd.read_parquet('./data/apm_intercept_am.parquet')
intercept_pm=pd.read_parquet('./data/apm_intercept_pm.parquet')


x_am = slope_am.mul(bench_am_rts['close'], axis=0) + intercept_am
x_pm = slope_pm.mul(bench_pm_rts['close'], axis=0) + intercept_pm

y_am = close_am_rts - x_am
y_pm = close_pm_rts - x_pm
y_am.index = [x.strftime('%Y-%m-%d') for x in y_am.index]
y_pm.index = [x.strftime('%Y-%m-%d') for x in y_pm.index]


delta = y_am - y_pm

# b1 = delta.rolling(10).corr(close_rts)
# c1 = delta / b1

N = 5
mu = delta.rolling(N).mean()
sigma = delta.rolling(N).std()
stat = sqrt(N) * mu / sigma

b2 = stat.rolling(10).corr(close_rts)

apm = stat - b2*close_rts


z = apm.rolling(5).max()

ic_test(index_pool='hs300', factor=b2 , open_rts=open_rts)
ic_test(index_pool='zz500', factor=apm, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=apm , open_rts=open_rts)






