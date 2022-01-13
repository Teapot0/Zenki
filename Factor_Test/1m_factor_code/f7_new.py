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

f7_sk_am = (f7.rolling(120).std()).iloc[119::237]
f7_sk_pm = (f7.rolling(117).std()).iloc[236::237]

diff_sk_am = (diff.rolling(120).std()).iloc[119::237]
diff_sk_pm = (diff.rolling(117).std()).iloc[236::237]

f7_sk_am.index = [x.strftime('%Y-%m-%d') for x in f7_sk_am.index]
f7_sk_pm.index = [x.strftime('%Y-%m-%d') for x in f7_sk_pm.index]

diff_sk_am.index = [x.strftime('%Y-%m-%d') for x in diff_sk_am.index]
diff_sk_pm.index = [x.strftime('%Y-%m-%d') for x in diff_sk_pm.index]

f77 = (f7_sk_pm / diff_sk_pm) * -1
f77.to_parquet('./factor_test/factor_value/f77.parquet')

f777 = (f7_sk_am / diff_sk_am) * -1

# y = diff_sk_am.rolling(5).kurt()
# y2 = f7_sk_am.rolling(5).skew()


ic_test(index_pool='hs300', factor=f77, open_rts=open_rts)
ic_test(index_pool='zz500', factor=f77, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=f77, open_rts=open_rts)


f44 = pd.read_parquet('./factor_test/factor_value/f44.parquet')


diff1 = f77.rolling(5).std()
s1 = f777.rolling(5).std()
s2 = f777.rolling(10).std()


ic_test(index_pool='hs300', factor = s1 , open_rts=open_rts)
ic_test(index_pool='zz500', factor = s2, open_rts=open_rts)

ic_test(index_pool='zz1000', factor = diff1, open_rts=open_rts)
















