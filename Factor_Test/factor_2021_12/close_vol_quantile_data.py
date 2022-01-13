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

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# high_1m = pd.read_parquet('./data/1m_data_agg/high_1m.parquet')
# low_1m = pd.read_parquet('./data/1m_data_agg/low_1m.parquet')
close_1m = pd.read_parquet('./data/1m_data_agg/close_1m.parquet')
close_1m_rts = pd.read_parquet('./data/1m_data_agg/close_1m_rts.parquet')
vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')

#
drop_list = list(vol_1m.index[239::240]) +list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

stocks = list(vol_1m.columns)
start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time= start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)


na_rows = list(set(close_1m.index).difference(set(close_1m.index[236::237])))

a1 = close_1m_rts.rolling(237, min_periods=1).quantile(0.15)
a1.loc[na_rows] = np.nan
a1 = a1.fillna(method='bfill')
a1.to_parquet('./data/close_1m_rts_daily_q15.parquet')


z = close_1m_rts[close_1m_rts < a1]
tmp_f1 = z.rolling(237,min_periods=1).std()

z.iloc[:237,1].dropna().std()


a3 = vol_1m.rolling(237, min_periods=1).quantile(0.15)
a3.loc[na_rows] = np.nan
a3 = a3.fillna(method='bfill')
a3.to_parquet('./data/vol_1m_rts_daily_q15.parquet') # 名字错了，数据没错，是vol不是vol_rts

a4 = vol_1m.rolling(237, min_periods=1).quantile(0.15)




