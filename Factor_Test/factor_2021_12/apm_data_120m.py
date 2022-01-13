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
open_rts = open_daily.pct_change(1)


bench_120m = pd.read_parquet('./data/120m/512100.XSHG.parquet')
bench_am = bench_120m.iloc[::2,]
bench_pm = bench_120m.iloc[1::2,]
bench_am_rts = bench_am.pct_change(1)
bench_pm_rts = bench_pm.pct_change(1)


slope_am = pd.DataFrame(index= close_am_rts.index, columns=close_am_rts.columns)
slope_pm = pd.DataFrame(index= close_pm_rts.index, columns=close_pm_rts.columns)
intercept_am = pd.DataFrame(index= close_am_rts.index, columns=close_am_rts.columns)
intercept_pm = pd.DataFrame(index= close_pm_rts.index, columns=close_pm_rts.columns)

for i in tqdm(range(21,close_am_rts.shape[0])):
    d_am = close_am_rts.index[i]
    d_pm = close_pm_rts.index[i]
    tmp_am = close_am_rts.iloc[i-20:i,]
    tmp_pm = close_pm_rts.iloc[i-20:i,]

    z = tmp_am.apply(lambda x: linregress(bench_am_rts.iloc[i-20:i,]['close'], x), result_type='expand').rename(
        index={0: 'slope', 1:'intercept'})

    slope_am.loc[d_am] = z.loc['slope']
    intercept_am.loc[d_am] = z.loc['intercept']

    z = tmp_pm.apply(lambda x: linregress(bench_pm_rts.iloc[i-20:i,]['close'], x), result_type='expand').rename(
        index={0: 'slope', 1:'intercept'})

    slope_pm.loc[d_pm] = z.loc['slope']
    intercept_pm.loc[d_pm] = z.loc['intercept']

slope_am.to_parquet('./data/apm_slope_am.parquet')
slope_pm.to_parquet('./data/apm_slope_pm.parquet')
intercept_am.to_parquet('./data/apm_intercept_am.parquet')
intercept_pm.to_parquet('./data/apm_intercept_pm.parquet')












