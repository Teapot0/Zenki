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
close_1m_rts = close_1m.pct_change(1)

stocks = list(vol_1m.columns)
start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)


r2 = close_1m_rts**2

rv = (r2.rolling(240, min_periods=1).sum()).iloc[239::240,]
rv_pos = ((r2[close_1m_rts > 0]).rolling(240,min_periods=1).sum()).iloc[239::240,]
rv_neg = ((r2[close_1m_rts < 0]).rolling(240,min_periods=1).sum()).iloc[239::240,]

rsj = (rv_pos - rv_neg)/rv

factor = rsj.rolling(5).mean()
factor.index = [x.strftime('%Y-%m-%d').split(' ')[0] for x in factor.index]
factor.to_parquet('./factor_test/factor_value/RSJ_1min_ma5.parquet')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)






