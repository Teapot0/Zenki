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
close_1m = pd.read_parquet('./data/1m_data_agg/close_1m.parquet')

# 去掉 58 59 集合竞价
drop_list = list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

vol_1m.index = [x.strftime('%Y-%m-%d') for x in vol_1m.index]
close_1m.index = [x.strftime('%Y-%m-%d') for x in close_1m.index]

start = vol_1m.index[0]
end = vol_1m.index[-1]
stocks = list(vol_1m.columns)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time= start, end_time = end,stock_list=stocks)


vwap_l30 = pd.DataFrame(index=vol_daily.index, columns=vol_daily.columns)
for d in tqdm(vol_daily.index):
    tmp_vol = vol_1m.loc[d].iloc[-28:,] # f30是.iloc :30, l30 是.iloc -28:
    tmp_close = close_1m.loc[d].iloc[-28:]
    tmp_w = tmp_vol / tmp_vol.sum()
    new_close = tmp_close * tmp_w
    tmp_vwap = np.round(new_close.sum(), 2)
    vwap_l30.loc[d] = tmp_vwap

vwap_l30.to_parquet('./data/vwap_l30.parquet')


