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
close_1m_rts = pd.read_parquet('./data/1m_data_agg/close_1m_rts.parquet')
vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(vol_1m.index[239::240]) + list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

close_1m.index = [x.strftime('%Y-%m-%d') for x in close_1m.index]
close_1m_rts.index = [x.strftime('%Y-%m-%d') for x in close_1m_rts.index]
vol_1m.index = [x.strftime('%Y-%m-%d') for x in vol_1m.index]


cv_corr = close_1m.rolling(237,min_periods=1).corr(vol_1m).iloc[236::237,]
cv_corr.to_parquet('./data/close_vol_corr.parquet')
rv_corr = close_1m_rts.rolling(237,min_periods=1).corr(vol_1m).iloc[236::237,]
rv_corr.to_parquet('./data/close_rts_vol_corr.parquet')

cc = close_1m.rolling(474,min_periods=1).corr(vol_1m).iloc[236::237,]
cc.to_parquet('./data/cv_corr_2d.parquet')

cc = close_1m_rts.rolling(474,min_periods=1).corr(vol_1m).iloc[236::237,]
cc.to_parquet('./data/rv_corr_2d.parquet')



cv_corr_5d = close_1m.rolling(1185,min_periods=1).corr(vol_1m).iloc[236::237,]
cv_corr_5d.to_parquet('./data/close_vol_corr_5d.parquet')

rv_corr_5d = close_1m_rts.rolling(1185,min_periods=1).corr(vol_1m).iloc[236::237,]
rv_corr_5d.to_parquet('./data/close_rts_vol_corr_5d.parquet')

#
rv_corr_5d = close_1m.shift(237).rolling(237,min_periods=1).corr(vol_1m).iloc[236::237,]
rv_corr_5d.to_parquet('./data/close_vol_corr_shift1.parquet')

cv_corr_shift1 = close_1m.shift(237).rolling(237,min_periods=1).corr(vol_1m).iloc[236::237,]
cv_corr_shift1.to_parquet('./data/close_vol_corr_shift1.parquet')

rv_corr_shift1 = close_1m_rts.shift(237).rolling(237,min_periods=1).corr(vol_1m).iloc[236::237,]
rv_corr_shift1.to_parquet('./data/close_rts_vol_corr_shift1.parquet')




