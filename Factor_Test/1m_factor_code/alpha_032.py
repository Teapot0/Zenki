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
mon_1m = pd.read_parquet('./data/1m_data_agg/money_1m.parquet')
close_1m_rts = close_1m.diff(1)

stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

drop_list = list(mon_1m.index[239::240]) + list(mon_1m.index[238::240])+ list(mon_1m.index[237::240])
mon_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)

# mon2 = mon_1m ** 0.1

log_amt = np.log(mon_1m).replace({np.inf:np.nan}).replace({-np.inf:np.nan})

upper = log_amt[close_1m_rts > 0]
down = log_amt[close_1m_rts < 0]

us = (upper.rolling(237,min_periods=1).std()).iloc[236::237,]
ds = (down.rolling(237,min_periods=1).std()).iloc[236::237,]

res = us / ds
res.index = [x.strftime('%Y-%m-%d') for x in res.index]


ic_test(index_pool='hs300', factor = res, open_rts=open_rts)
ic_test(index_pool='zz500', factor = res, open_rts=open_rts)
ic_test(index_pool='zz1000', factor= res, open_rts=open_rts)




