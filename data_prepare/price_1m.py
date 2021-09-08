import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss,get_money_flow
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

auth('13382017213', 'Aasd120120')
get_query_count()

hs300_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/hs300_holds.csv', index_col='Unnamed: 0')
zz500_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz500_holds.csv', index_col='Unnamed: 0')
zz1000_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz1000_holds.csv', index_col='Unnamed: 0')

hs300_s = hs300_s.dropna(how='all', axis=1)
zz500_s = zz500_s.dropna(how='all', axis=1)
zz1000_s = zz1000_s.dropna(how='all', axis=1)
all_s = list(set.union(set(hs300_s.columns), set(zz500_s.columns), set(zz1000_s.columns)))

# stock_list_days = list(close_daily.isna().sum()[close_daily.isna().sum() <= 244].index)  # 大于一年的


trade_days = get_trade_days(start_date='2021-01-01', end_date='2021-04-30')
start = trade_days[0].strftime('%Y-%m-%d 09:00:00')
end = trade_days[-1].strftime('%Y-%m-%d 15:00:00')

tmp = get_price(all_s[0], start_date=start,end_date=end,frequency='1m',
                                        fields=['open', 'close', 'high', 'low', 'volume'])

close_1m = pd.DataFrame(columns=all_s,index=tmp.index)
open_1m = pd.DataFrame(columns=all_s,index=tmp.index)
high_1m = pd.DataFrame(columns=all_s,index=tmp.index)
low_1m = pd.DataFrame(columns=all_s,index=tmp.index)
volume_1m = pd.DataFrame(columns=all_s,index=tmp.index)
money_1m = pd.DataFrame(columns=all_s,index=tmp.index)


for s in tqdm(all_s):
    tmp = get_price(s, start_date=start, end_date=end, frequency='1m',
                    fields=['open', 'close', 'high', 'low', 'volume','money'])
    close_1m[s] = tmp['close']
    open_1m[s] = tmp['open']
    high_1m[s] = tmp['high']
    low_1m[s] = tmp['low']
    volume_1m[s] = tmp['volume']
    money_1m[s] = tmp['money']


close_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv')
open_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/open_1m.csv')
high_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/high_1m.csv')
low_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv')
volume_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv')
money_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/money_1m.csv')
