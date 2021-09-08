
import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss,get_money_flow
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from jqdatasdk import alpha191
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

# auth('15951961478', '961478')
auth('13382017213', 'Aasd120120')
get_query_count()

alpha_083 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_083.csv',index_col='Unnamed: 0')
alpha_064 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_064.csv',index_col='Unnamed: 0')

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0')

new_83 = pd.DataFrame(index=close.index, columns=close.columns)
new_64 = pd.DataFrame(index=close.index, columns=close.columns)

stocks = list(close.columns)
for date in tqdm(close.index):
    tmp_83 = alpha191.alpha_083(stocks, date,fq='pre')
    new_83.loc[date] = tmp_83.values

    tmp_64 = alpha191.alpha_064(stocks, date, fq='pre')
    new_64.loc[date] = tmp_64.values

new_83.to_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_083.csv')
new_64.to_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_064.csv')

#

#
a = pd.DataFrame(index=close.index, columns=close.columns)

stocks = list(close.columns)
for date in tqdm(close.index):
    tmp = alpha191.alpha_105(stocks, date,fq='pre')
    a.loc[date] = tmp.values
a.to_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_105.csv')


