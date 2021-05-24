import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqfactor_analyzer import analyze_factor
from alphalens.utils import get_clean_factor_and_forward_returns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from datetime import datetime,date
from basic_funcs.basic_function import *

import warnings
warnings.filterwarnings('ignore')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')


close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
open = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
low_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)


def select_data_same_start(df_list,start_time):
    for df in df_list:
        df = df[df.index >= start_time]

select_data_same_start([close,high,low,open,high_limit],start_time='2020-01-01')

# 去掉新股一字涨停价
all_new_stock = []
for i in tqdm(range(1, close.shape[0])):
    tmp_close = close.iloc[i,]
    yesterday_close = close.iloc[i-1,]
    new_stock = list(set(tmp_close.dropna().index).difference(set(yesterday_close.dropna().index)))  # 每天新股名单
    for ss in new_stock:
        if low.iloc[i,][ss] == high_limit.iloc[i, ][ss]:
            close.iloc[i,][ss] = np.nan  # 第一天上市等于涨停价则去掉
    # 加上第一天上市涨停的
    all_new_stock = list(set(all_new_stock).union(set(close.iloc[i,][new_stock][close.iloc[i,][new_stock] == high_limit.iloc[i,][new_stock]].index)))
    # 去掉开板的
    new_stock_kai = list(low.iloc[i, ][all_new_stock][low.iloc[i, ][all_new_stock] != high_limit.iloc[i, ][all_new_stock]].index)
    # 所有未开板新股
    new_stock_not_kai = list(set(all_new_stock).difference(set(new_stock_kai)))
    # 未开板新股去掉
    close.iloc[i, ][new_stock_not_kai] = np.nan




