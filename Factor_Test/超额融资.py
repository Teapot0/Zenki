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

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['rts_5'] = hs300['close'].pct_change(5)
hs300['rts_10'] = hs300['close'].pct_change(10)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close_rts_1 = close.pct_change(1)
margin_total_value = pd.read_csv('/Users/caichaohong/Desktop/Zenki/rongzi/margin_total_value.csv', index_col='Unnamed: 0',date_parser=dateparse)

# excess margin factor
hs300_list = get_index_stocks('000300.XSHG')
list_500 = get_index_stocks('000905.XSHG')
n = 10
z = close.pct_change(n).rank(axis=1) / margin_total_value.pct_change(n).rank(axis=1)
zz = abs(z.sub(z.median(axis=1),axis=0))

# pp = get_params_out(top_number_list=[10,20,30,40,50], hold_time_list=[1,5,10,20], factor_df=zz[hs300_list], rts_df=close_rts_1[hs300_list])
# pp = pp.sort_values(by='annual_rts',ascending=False)
# pp.to_excel('/Users/caichaohong/Desktop/Zenki/params.xlsx')

value_rts = get_top_value_factor_rts(factor=zz[hs300_list], rts=close_rts_1[hs300_list], top_number=10, hold_time=10)
plot_rts(value_rts=value_rts['daily_rts'], benchmark_rts=hs300['rts_1'],comm_fee=0.003, hold_time=10)

qq = quantile_factor_test_plot(factor=zz[list_500], rts=close_rts_1[list_500],benchmark_rts=hs300['rts_1'],quantiles=10, hold_time=5, comm_fee=0.003)



