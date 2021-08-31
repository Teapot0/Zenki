import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
from jqdatasdk import auth, get_query_count,get_industries,get_industry_stocks, get_industry
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

auth('13382017213', 'Aasd120120')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')



ind_list = get_industries('sw_l1')
ind_list['name'] = ind_list['name'].astype('str')

ind_df = pd.DataFrame(np.nan,index=close_daily.index, columns = close_daily.columns)
for d in tqdm(ind_df.index):
    for ind in ind_list.index:
        tmp_ind_stocks = list(set(get_industry_stocks(ind, date=d)).intersection(set(close_daily.columns)))
        ind_df.loc[d][tmp_ind_stocks] = int(ind)

ind_df.to_csv('/Users/caichaohong/Desktop/Zenki/price/stock_swl1_industry.csv')


# 申万3级
ind_list = get_industries('sw_l3')
ind_list['name'] = ind_list['name'].astype('str')

ind_df = pd.DataFrame(np.nan,index=close_daily.index, columns = close_daily.columns)
for d in tqdm(ind_df.index):
    for ind in ind_list.index:
        tmp_ind_stocks = list(set(get_industry_stocks(ind, date=d)).intersection(set(close_daily.columns)))
        ind_df.loc[d][tmp_ind_stocks] = int(ind)

ind_df.to_csv('/Users/caichaohong/Desktop/Zenki/price/stock_swl3_industry.csv')



#
swl1 = pd.DataFrame(index = close_daily.index, columns=close_daily.columns)
swl2 = pd.DataFrame(index = close_daily.index, columns=close_daily.columns)
swl3 = pd.DataFrame(index = close_daily.index, columns=close_daily.columns)
zjw = pd.DataFrame(index = close_daily.index, columns=close_daily.columns)


stocks = list(close_daily.columns)
d = get_industry(security=stocks, date="2018-06-01")



