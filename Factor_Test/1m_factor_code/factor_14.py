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

low_1m = pd.read_parquet('./data/1m_data_agg/low_1m.parquet')
high_1m = pd.read_parquet('./data/1m_data_agg/high_1m.parquet')


stocks = list(high_1m.columns)
start = high_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = high_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(high_1m.index[239::240]) + list(high_1m.index[238::240])+ list(high_1m.index[237::240])
high_1m.drop(index=drop_list,inplace=True)
low_1m.drop(index=drop_list,inplace=True)


low_rts = low_1m.diff(1)
high_rts = high_1m.diff(1)

res = low_rts.rolling(237, min_periods=1).sum()
res.index = [x.strftime('%Y-%m-%d') for x in res.index]

# z = res = (low_1m - high_1m).rolling(30, min_periods=1).std()
# z.index = [x.strftime('%Y-%m-%d') for x in z.index]

f = res.iloc[236::237]

ic_test(index_pool='hs300', factor = f, open_rts=open_rts)
ic_test(index_pool='zz500', factor = f, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= f, open_rts=open_rts)






