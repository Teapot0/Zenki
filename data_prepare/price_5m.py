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

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']

# stock_list_days = list(close_daily.isna().sum()[close_daily.isna().sum() <= 244].index)  # 大于一年的
# close_daily = close_daily[stock_list_days]

trade_days = get_trade_days(start_date='2019-01-01', end_date='2021-07-12')
start = trade_days[0].strftime('%Y-%m-%d 09:00:00')
end = trade_days[-1].strftime('%Y-%m-%d 15:00:00')

tmp = get_price(close_daily.columns[0], start_date=start,end_date=end,frequency='5m',
                                        fields=['open', 'close', 'high', 'low', 'volume'])

close_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
open_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
high_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
low_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
volume_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
money_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)

for s in tqdm(close_daily.columns):
    tmp = get_price(s, start_date=start, end_date=end, frequency='5m',
                    fields=['open', 'close', 'high', 'low', 'volume','money'])
    close_5m[s] = tmp['close']
    open_5m[s] = tmp['open']
    high_5m[s] = tmp['high']
    low_5m[s] = tmp['low']
    volume_5m[s] = tmp['volume']
    money_5m[s] = tmp['money']


close_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv')
open_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/open_5m.csv')
high_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv')
low_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv')
volume_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv')
money_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/money_5m.csv')




# update data
close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv',index_col='Unnamed: 0')
open_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/open_5m.csv',index_col='Unnamed: 0')
high_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv',index_col='Unnamed: 0')
low_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv',index_col='Unnamed: 0')
volume_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv',index_col='Unnamed: 0')
money_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/money_5m.csv',index_col='Unnamed: 0')

# 补新票
new_stocks = list(set(close_daily.columns).difference(set(close_5m.dropna(axis=1, how='all').columns)))
old_start = close_5m.index[0]
old_end = close_5m.index[-1]
close_5m = close_5m.reindex(columns=list(close_5m.columns) + new_stocks)
close_5m = close_5m.sort_index(axis=1)

for s in tqdm(new_stocks):
    tmp = get_price(s, start_date=old_start, end_date=old_end, frequency='5m',
                    fields=['open', 'close', 'high', 'low', 'volume', 'money'])
    close_5m[s] = tmp['close']
    open_5m[s] = tmp['open']
    high_5m[s] = tmp['high']
    low_5m[s] = tmp['low']
    volume_5m[s] = tmp['volume']
    money_5m[s] = tmp['money']

close_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv')
open_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/open_5m.csv')
high_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv')
low_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv')
volume_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv')
money_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/money_5m.csv')



# 补日期
last_end_date = '2021-08-01' # 加一天，从第二天12点开始算
new_end = '2021-09-07'

tmp_close = pd.DataFrame(columns=close_daily.columns)
tmp_open = pd.DataFrame(columns=close_daily.columns)
tmp_high = pd.DataFrame(columns=close_daily.columns)
tmp_low = pd.DataFrame(columns=close_daily.columns)
tmp_volume = pd.DataFrame(columns=close_daily.columns)
tmp_money = pd.DataFrame(columns=close_daily.columns)

for s in tqdm(close_daily.columns):
    tmp = get_price(s, start_date=last_end_date, end_date=new_end, frequency='5m',
                    fields=['open', 'close', 'high', 'low', 'volume','money'])
    tmp_close[s] = tmp['close']
    tmp_open[s] = tmp['open']
    tmp_high[s] = tmp['high']
    tmp_low[s] = tmp['low']
    tmp_volume[s] = tmp['volume']
    tmp_money[s] = tmp['money']

new_close = pd.concat([close_5m,tmp_close],axis=0,join='inner')
new_open = pd.concat([open_5m,tmp_open],axis=0,join='inner')
new_high = pd.concat([high_5m,tmp_high],axis=0,join='inner')
new_low = pd.concat([low_5m,tmp_low],axis=0,join='inner')
new_volume = pd.concat([volume_5m,tmp_volume],axis=0,join='inner')
new_money = pd.concat([money_5m,tmp_money],axis=0,join='inner')

# new_close = pd.concat([tmp_close,close_5m],axis=0,join='inner')
# new_open = pd.concat([tmp_open,open_5m],axis=0,join='inner')
# new_high = pd.concat([tmp_high,high_5m],axis=0,join='inner')
# new_low = pd.concat([tmp_low,low_5m],axis=0,join='inner')
# new_volume = pd.concat([tmp_volume,volume_5m],axis=0,join='inner')
# new_money = pd.concat([tmp_money,money_5m],axis=0,join='inner')


new_close.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv')
new_open.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/open_5m.csv')
new_high.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv')
new_low.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv')
new_volume.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv')
new_money.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/money_5m.csv')



