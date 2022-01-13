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
# close_1m_rts = pd.read_parquet('./data/1m_data_agg/close_1m_rts.parquet')
vol_1m = pd.read_parquet('./data/1m_data_agg/volume_1m.parquet')

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(vol_1m.index[239::240]) +list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

# vol_w = (mon_daily.div(mon_daily.sum(axis=1),axis=0)).rank(axis=1,pct=True,ascending=False)

f1 = pd.read_parquet('./data/vwap_big85rts.parquet')
f2 = pd.read_parquet('./data/vwap_small15rts.parquet')
f3 = pd.read_parquet('./data/vwap_big85vol.parquet')
f4 = pd.read_parquet('./data/vwap_small15vol.parquet')

vwap = pd.read_parquet('./data/vwap.parquet')
vwap = vwap.replace(0,np.nan)
v_rts = vwap.pct_change(1)


f1_rts = f1.pct_change(1)
f2_rts = f2.pct_change(1)
f3_rts = f3.pct_change(1)
f4_rts = f4.pct_change(1)

ic_test(index_pool='hs300', factor = (f4_rts - f3_rts).rolling(5).sum(), open_rts=open_rts)
ic_test(index_pool='zz500', factor = (f4_rts - f3_rts).rolling(5).sum(), open_rts=open_rts)
ic_test(index_pool='zz1000', factor= (f4_rts - f3_rts).rolling(5).sum(), open_rts=open_rts)








