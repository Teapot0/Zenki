import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler

auth('15951961478', '961478')
get_query_count()

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

hs300_list = get_index_stocks('000300.XSHG')

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['rts_interval_1'] = transform_300_rts_to_daily_intervals(hs300['rts_1'])
hs300['rts_5'] = hs300['close'].pct_change(5)
hs300['rts_10'] = hs300['close'].pct_change(10)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = clean_close(close, low, high_limit)  # 新股一字板
close = clean_st_exit(close)  # 退市和ST
close_rts_1 = close.pct_change(1)
close_rts_5 = close_rts_1.rolling(5).sum()
close_rts_10 = close_rts_1.rolling(10).sum()
close_rts_interval_1 = transform_rts_to_daily_intervals(close_rts_1)

share = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/share.csv', index_col='Unnamed: 0', date_parser=dateparse)

reverse_rts = ((close_rts_interval_1[hs300_list].sub(hs300['rts_interval_1'], axis=0)).mul(hs300['rts_interval_1']**2, axis=0))

excess_share = share.diff(1)

value_rts = get_top_value_factor_rts(factor=reverse_rts, rts=close_rts_1, top_number=5, hold_time=5)
plt.plot((1+value_rts).cumprod())

plot_rts(value_rts=value_rts['daily_rts'],benchmark_rts=hs300['rts_1'], hold_time=5)


z = get_params_out(top_number_list=[5,10,20,30], hold_time_list=[1,2,3,5,10,20],factor_df=reverse_rts, rts_df=close_rts_1)
z.to_excel()