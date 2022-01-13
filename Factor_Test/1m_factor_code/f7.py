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
open_1m = pd.read_parquet('./data/1m_data_agg/open_1m.parquet')
ww = pd.read_parquet('./data/1m_data_agg/vol_1m_weight.parquet') # 已经drop了最后3分钟

drop_list = list(close_1m.index[239::240]) +list(close_1m.index[238::240])+ list(close_1m.index[237::240])
close_1m.drop(index=drop_list,inplace=True)
open_1m.drop(index=drop_list,inplace=True)

#
stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)



diff = close_1m - open_1m

f7 = diff * ww
f7_sk = (f7.rolling(237).std()).iloc[236::237]
diff_skew = (diff.rolling(237).std()).iloc[236::237]

f7_sk.index = [x.strftime('%Y-%m-%d') for x in f7_sk.index]
diff_skew.index = [x.strftime('%Y-%m-%d') for x in diff_skew.index]
f77 = f7_sk / diff_skew

z = f77 * -1

ic_test(index_pool='hs300', factor=z, open_rts=open_rts)
ic_test(index_pool='zz500', factor=z, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=z, open_rts=open_rts)





