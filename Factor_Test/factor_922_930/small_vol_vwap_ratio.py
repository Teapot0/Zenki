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
mon_1m = pd.read_parquet('./data/1m_data_agg/money_1m.parquet')

stocks = list(vol_1m.columns)
start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
mon_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv' , start_time=start, end_time = end,stock_list=stocks)

open_rts = open_daily.pct_change(1)


daily_vwap = mon_daily / vol_daily

vol_high = vol_1m.rolling(1200).quantile(0.5)

vol_high.loc[vol_high.index[239::240]] = np.nan
vol_high.loc[vol_high.index[238::240]] = np.nan
vol_high.loc[vol_high.index[237::240]] = np.nan

# vol_high = vol_high.fillna(method='bfill')
filter_df = vol_1m < vol_high

qvwap = mon_1m[filter_df].rolling(240, min_periods=1).sum() / vol_1m[filter_df].rolling(240, min_periods=1).sum()

tmp_factor = qvwap.iloc[239::240,]
tmp_factor.index = [x.strftime('%Y-%m-%d').split(' ')[0] for x in tmp_factor.index]
tmp_factor = tmp_factor * 10**(4)

factor = tmp_factor / daily_vwap

factor.to_parquet('./factor_test/factor_value/small_vol_vwap_ratio.parquet')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






