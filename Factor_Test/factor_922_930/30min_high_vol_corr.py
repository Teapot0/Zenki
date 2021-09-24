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

high_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/high_30m.csv', index_col='Unnamed: 0')
vol_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volume_30m.csv', index_col='Unnamed: 0')
close_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/close_30m.csv', index_col='Unnamed: 0')


close_30m_rts = close_30m.pct_change(1)
stocks = list(close_30m.columns)

start = close_30m.index[0].split(' ')[0]
end = close_30m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


tmp_factor = high_30m.rolling(8).corr(vol_30m)

f1 = tmp_factor.iloc[7::8,]

factor = -1 * (f1.diff(5).rolling(5).mean())
factor.index = [x.split(' ')[0] for x in factor.index]
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/30min_neg_rts_volstd.csv')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






