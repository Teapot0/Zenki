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

close = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/close.xlsx', index_col='Unnamed: 0')
high_limit = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.xlsx', index_col='Unnamed: 0')
low = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/low.xlsx', index_col='Unnamed: 0')
high = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/high.xlsx', index_col='Unnamed: 0')
open = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/open.xlsx', index_col='Unnamed: 0')

close = close[close.index >= '2020-02-01']
high = high[high.index >= '2020-02-01']
low = low[low.index >= '2020-02-01']
open = open[open.index >= '2020-02-01']
high_limit = high_limit[high_limit.index >= '2020-02-01']

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


# 20个交易日，涨幅超过100的,共有190个

stock_list = []
for col in tqdm(close.columns):
    temp = close[col]
    temp_rts = close[col].rolling(20).apply(lambda x: x[-1]/x[0]-1)
    if temp_rts.max() >= 1 :
        stock_list.append(col)


# 找相似



def find_similar(target_code, start, end,length):
    # 要用get_trade_days
    # length = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days

    corref_df = pd.DataFrame(index=close.index, columns=close.columns)
    for s in tqdm(close.columns):
        close_corr = close[s].rolling(length).apply(lambda x: round(np.corrcoef(close[target_code].loc[start:end], x)[0][1], 3))
        high_corr = high[s].rolling(length).apply(lambda x: round(np.corrcoef(high[target_code].loc[start:end], x)[0][1], 3))
        low_corr = low[s].rolling(length).apply(lambda x: round(np.corrcoef(low[target_code].loc[start:end], x)[0][1], 3))
        open_corr = open[s].rolling(length).apply(lambda x: round(np.corrcoef(open[target_code].loc[start:end], x)[0][1], 3))
        corref_df[s] = round((close_corr+high_corr+low_corr+open_corr)/4,2)
    return corref_df


z = find_similar('002284.XSHE', start='2021-04-14', end='2021-04-23',length=8)
z.to_excel('/Users/caichaohong/Desktop/Zenki/coef/002284.xlsx')
