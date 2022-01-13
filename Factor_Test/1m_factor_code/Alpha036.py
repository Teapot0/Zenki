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

drop_list = list(vol_1m.index[239::240]) +list(vol_1m.index[238::240])+ list(vol_1m.index[237::240])+ list(vol_1m.index[236::240])
vol_1m.drop(index=drop_list,inplace=True)
close_1m.drop(index=drop_list,inplace=True)


start = vol_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = vol_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]

stocks = list(vol_1m.columns)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)

ret1 = close_1m - close_1m.shift(1)
ret2 = vol_1m - vol_1m.shift(1)
df_filter1 = (ret1 > 0) & (ret2 > 0)
df_filter2 = (ret1 < 0) & (ret2 < 0)
close_1 = close_1m[df_filter1]
volume_1 = vol_1m[df_filter1]
close_2 = close_1m[df_filter2]
volume_2 = vol_1m[df_filter2]

tmp_factor1 = close_1.rolling(236,min_periods=1).corr(volume_1)
tmp_factor2 = close_2.rolling(236,min_periods=1).corr(volume_2)

f1 = tmp_factor1.iloc[235::236]
f2 = tmp_factor2.iloc[235::236]

factor = f1.rank(axis=1,ascending=False) + f2.rank(axis=1, ascending=False)
factor = factor.fillna(0)
factor.index = [x.strftime('%Y-%m-%d') for x in factor.index]
# factor1 = factor.rolling(5).mean()
# factor.to_parquet('./factor_test/factor_value/.parquet')

ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)


factor.to_parquet('./factor_test/factor_value/alpha036_rank_sum.parquet')


