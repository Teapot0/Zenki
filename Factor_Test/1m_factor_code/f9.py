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

drop_list = list(close_1m.index[239::240]) + list(close_1m.index[238::240])+ list(close_1m.index[237::240])
close_1m.drop(index=drop_list,inplace=True)
vol_1m.drop(index=drop_list,inplace=True)


x1 = close_1m.rolling(120,min_periods=1).corr(vol_1m)
x2 = close_1m.rolling(117,min_periods=1).corr(vol_1m)


f1 = x1.iloc[119::237,]
f1.to_parquet('./data/first_half_cv_corr.parquet')
f2 = x2.iloc[236::237,]
f2.to_parquet('./data/second_half_cv_corr.parquet')


f1.index = [x.strftime('%Y-%m-%d') for x in f1.index]
f2.index = [x.strftime('%Y-%m-%d') for x in f2.index]


ic_test(index_pool='hs300', factor = f2 - f1, open_rts=open_rts)
ic_test(index_pool='zz500', factor = f2 - f1, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= f2 - f1, open_rts=open_rts)


f.to_parquet('./factor_test/factor_value/.parquet')

