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
vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(vol_1m.index[239::240]) + list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)



cv_corr = pd.read_parquet('./data/close_vol_corr.parquet')
cv_corr_2d = pd.read_parquet('./data/cv_corr_2d.parquet')
cv_corr_5d = pd.read_parquet('./data/close_vol_corr_5d.parquet')
rv_corr = pd.read_parquet('./data/close_rts_vol_corr.parquet')
rv_corr_2d = pd.read_parquet('./data/rv_corr_2d.parquet')
rv_corr_5d = pd.read_parquet('./data/close_rts_vol_corr_5d.parquet')

# cv_corr_shift1 = pd.read_parquet('./data/close_vol_corr_shift1.parquet')
# rv_corr_shift1 = pd.read_parquet('./data/close_rts_vol_corr_shift1.parquet')



ic_test(index_pool='hs300', factor = cv_corr_5d - cv_corr_2d, open_rts=open_rts)
ic_test(index_pool='zz500', factor = cv_corr_5d - cv_corr_2d, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= cv_corr_5d - cv_corr_2d, open_rts=open_rts)




