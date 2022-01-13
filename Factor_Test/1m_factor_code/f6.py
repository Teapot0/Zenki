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

drop_list = list(close_1m.index[239::240]) +list(close_1m.index[238::240])+ list(close_1m.index[237::240])
close_1m.drop(index=drop_list,inplace=True)
vol_1m.drop(index=drop_list,inplace=True)


close_1m_copy = close_1m.copy(deep=True)
close_1m_copy.index = [x.strftime('%Y-%m-%d') for x in close_1m_copy.index]
vol_1m.index = [x.strftime('%Y-%m-%d') for x in vol_1m.index]

close_1m_rts = (close_1m - close_1m.shift(1)) / close_1m.shift(1)
close_1m_rts.index = [x.strftime('%Y-%m-%d') for x in close_1m_rts.index]

date_list = np.unique(close_1m_rts.index)

f6 = pd.DataFrame(index=date_list, columns=close_1m.columns)

for i in tqdm(range(len(date_list)-5)):
    d = date_list[i]
    d1 = date_list[i+1]
    d2 = date_list[i+2]
    d3 = date_list[i+3]
    d4 = date_list[i+4]
    tmp = close_1m_rts.loc[[d,d1,d2,d3,d4]]
    tmp_vol = vol_1m.loc[[d,d1,d2,d3,d4]]
    tmp_r = tmp.quantile(0.25)
    tmp_rdf = pd.DataFrame(tmp_r).transpose()
    # tmp_rdf.reset_index(inplace=True, drop=True)
    tmp_rdf.index = [tmp.index[0]]

    z = tmp.lt(tmp_rdf,axis=0)

    up_df = tmp_vol * z
    up_df = up_df.replace(0,np.nan)
    out = (up_df**0.1).std()
    f6.loc[d] = out


stocks = list(close_1m.columns)
start = close_1m.index[0].strftime('%Y-%m-%d').split(' ')[0]
end = close_1m.index[-1].strftime('%Y-%m-%d').split(' ')[0]
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
open_rts = open_daily.pct_change(1)



ic_test(index_pool='hs300', factor=f6, open_rts=open_rts)
ic_test(index_pool='zz500', factor=f6, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=f6, open_rts=open_rts)










