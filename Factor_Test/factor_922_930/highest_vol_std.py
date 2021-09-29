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

close_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv', index_col='Unnamed: 0')
volume_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv', index_col='Unnamed: 0')

close_1m_rts = close_1m.pct_change(1)
stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)

tmp_factor = pd.DataFrame(index=close_daily.index, columns=close_daily.columns)

for i in tqdm(range(0,close_1m_rts.shape[0],240)):
    tmp_df = close_1m_rts.iloc[i:i+240,]
    tmp_index = tmp_df.index[0].split(' ')[0]

    max_id = tmp_df.cumsum().idxmax()
    max_df = pd.get_dummies(max_id)

    #
    date_ex = list(set(tmp_df.index).difference(max_df.columns))
    for dd in date_ex:
        max_df[dd] = 0
    max_df = max_df.sort_index(axis=1)

    tmp_vol = volume_1m.iloc[i:i+240,]
    tmp_vol_p = tmp_vol.rolling(5,min_periods=1).sum()[max_df.T == 1] / tmp_vol.sum()

    tmp_ff = tmp_vol_p.fillna(method='ffill')
    tmp_ff = tmp_ff.fillna(method='bfill')


    tmp = tmp_ff.iloc[0,].fillna(0)

    tmp_factor.loc[tmp_index] = tmp

factor = tmp_factor.rolling(3).std()

# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






