import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, valuation, get_security_info
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

value = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/value.xlsx', index_col='Unnamed: 0')
share = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/share.xlsx', index_col='Unnamed: 0')

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0')
open = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0')
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0')
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0')
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0')
low_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.csv', index_col='Unnamed: 0')
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0')

top_10_net_buy = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_net_buy.xlsx', index_col='Unnamed: 0')

value_diff = value.diff(1)
share_diff = share.diff(1)
close_rts = close.pct_change(1)

plt.scatter(share_diff[top_10_net_buy.columns], top_10_net_buy.iloc[:-1,])


close_north = close[share.columns]
close_north = close_north.iloc[:-1, ] # close比share多一天
close_north_rts = close_north.pct_change(1)

model = LinearRegression()
model.fit(close_north_rts.fillna(0).values.reshape(-1, 1), share_diff.fillna(0).values.reshape(-1,1))
# 4208551,31479
res = share_diff - (4208551 * close_north_rts)



plt.scatter(money_est, top_10_net_buy)



money_est = close.rolling(5).mean() * share_diff.rolling(5).mean()


