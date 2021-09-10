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

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv', index_col='Unnamed: 0')
vol_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv', index_col='Unnamed: 0')

close_5m_rts = close_5m.pct_change(1)

stocks = list(close_5m.columns)
# daily
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time='2018-01-01', end_time='2021-07-30',stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time='2018-01-01', end_time='2021-07-30',stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time='2018-01-01', end_time='2021-07-30',stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)

# 30分钟占比
vol_daily_8 = vol_daily.append(47*[vol_daily])
vol_daily_8 = vol_daily_8.sort_index()
vol_daily_8.index = close_5m.index

rts_daily_8 = close_rts.append(47*[close_rts])
rts_daily_8 = rts_daily_8.sort_index()
rts_daily_8.index = close_5m.index

vol_daily_pct = vol_5m / vol_daily_8
rts_daily_pct = close_5m_rts / rts_daily_8

# ----------
col_name = [x.split(' ')[1] for x in vol_daily_pct.iloc[:48,].index]
corr_300 = pd.DataFrame(0,index=close_daily.index, columns=col_name)
corr_500 = pd.DataFrame(0,index=close_daily.index, columns=col_name)
corr_1000 = pd.DataFrame(0,index=close_daily.index, columns=col_name)

for i in tqdm(range(0,48)):
    tmp_name = col_name[i]
    tmp_df = vol_daily_pct.iloc[i::48]
    tmp_name = tmp_df.index[0].split(' ')[1]
    tmp_df.index = [x.split(' ')[0] for x in tmp_df.index]
    z1 = get_ic_table_open_index(factor=tmp_df, open_rts=open_rts, buy_date_list=tmp_df.index, index_pool='hs300')
    z2 = get_ic_table_open_index(factor=tmp_df, open_rts=open_rts, buy_date_list=tmp_df.index, index_pool='zz500')
    z3 = get_ic_table_open_index(factor=tmp_df, open_rts=open_rts, buy_date_list=tmp_df.index, index_pool='zz1000')
    corr_300[tmp_name] = z1['ic']
    corr_500[tmp_name] = z2['ic']
    corr_1000[tmp_name] = z3['ic']


vol_corr_300 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_300.csv',index_col='Unnamed: 0')
vol_corr_500 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_500.csv',index_col='Unnamed: 0')
vol_corr_1000 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_1000.csv',index_col='Unnamed: 0')

corr_300.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_300.csv')
corr_500.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_500.csv')
corr_1000.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volpct_corr_5m_1000.csv')






