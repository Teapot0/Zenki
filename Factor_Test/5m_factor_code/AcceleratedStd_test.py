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

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv', index_col='Unnamed: 0')
close_5m = close_5m.loc[(close_5m.index >= '2018-01-01') & (close_5m.index <= '2021-06-30')]
close_5m.drop(index=close_5m.index[47::48], inplace=True)

close_5m_rts = close_5m.pct_change(1)
ex_rts = abs(close_5m_rts.sub(close_5m_rts.median(axis=1),axis=0))

res = ex_rts.rolling(47).std() / ex_rts.rolling(47).sum()

stocks = list(close_5m.columns)

close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time='2018-01-01', end_time='2021-06-30',stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time='2018-01-01', end_time='2021-06-30',stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time='2018-01-01', end_time='2021-06-30',stock_list=stocks)
open_rts = open_daily.pct_change(1)


factor = res.iloc[46::47,]
factor.index = [x.split(' ')[0] for x in factor.index]


st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')
st_df = st_df.loc[factor.index]
st_df = st_df.replace(True, np.nan)
st_df = st_df.replace(False,1)

factor = -1 * factor * st_df
# factor.to_parquet('./factor_test/factor_value/AcceleratedStd.parquet')

# factor1 = factor.rolling(10).mean()
ic_test(index_pool='hs300', factor=factor)
ic_test(index_pool='zz500', factor=factor)
ic_test(index_pool='zz1000', factor=factor)











