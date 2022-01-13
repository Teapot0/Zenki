import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_bars
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

auth('13382017213', 'Aasd120120')
get_query_count()

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0')
all_s = close_daily.columns.tolist()

# stock_list_days = list(close_daily.isna().sum()[close_daily.isna().sum() <= 244].index)  # 大于一年的


trade_days = get_trade_days(start_date='2018-01-01', end_date='2021-10-28')
start = trade_days[0].strftime('%Y-%m-%d 09:00:00')
end = trade_days[-1].strftime('%Y-%m-%d 15:00:00')

tmp = get_bars(all_s,count=20,unit='1w',
                                        fields=['date','open', 'close', 'high', 'low', 'volume','money'])
tmp['code'] = [x[0]for x in tmp.index]

close = tmp.pivot(index='date', columns='code', values='close')
close.index = [x.strftime('%Y-%m-%d') for x in close.index]
close = close.loc[close.index >= '2018-01-01']
close = close.dropna()

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


# update

# update data
close_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv',index_col='Unnamed: 0')
open_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/open_1m.csv',index_col='Unnamed: 0')
high_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/high_1m.csv',index_col='Unnamed: 0')
low_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv',index_col='Unnamed: 0')
volume_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv',index_col='Unnamed: 0')
money_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/money_1m.csv',index_col='Unnamed: 0')

close_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv')
open_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/open_1m.csv')
high_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/high_1m.csv')
low_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv')
volume_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv')
money_1m.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/money_1m.csv')

# 补日期
last_end_date = '2021-05-01' # 加一天，从第二天12点开始算
new_end = '2021-09-01'

tmp_close = pd.DataFrame(columns=close_1m.columns)
tmp_open = pd.DataFrame(columns=close_1m.columns)
tmp_high = pd.DataFrame(columns=close_1m.columns)
tmp_low = pd.DataFrame(columns=close_1m.columns)
tmp_volume = pd.DataFrame(columns=close_1m.columns)
tmp_money = pd.DataFrame(columns=close_1m.columns)

for s in tqdm(close_1m.columns):
    tmp = get_price(s, start_date=last_end_date, end_date=new_end, frequency='1m',
                    fields=['open', 'close', 'high', 'low', 'volume','money'])
    tmp_close[s] = tmp['close']
    tmp_open[s] = tmp['open']
    tmp_high[s] = tmp['high']
    tmp_low[s] = tmp['low']
    tmp_volume[s] = tmp['volume']
    tmp_money[s] = tmp['money']

new_close = pd.concat([close_1m,tmp_close],axis=0,join='inner')
new_open = pd.concat([open_1m,tmp_open],axis=0,join='inner')
new_high = pd.concat([high_1m,tmp_high],axis=0,join='inner')
new_low = pd.concat([low_1m,tmp_low],axis=0,join='inner')
new_volume = pd.concat([volume_1m,tmp_volume],axis=0,join='inner')
new_money = pd.concat([money_1m,tmp_money],axis=0,join='inner')

new_close = new_close.sort_index(axis=1)
new_open = new_open.sort_index(axis=1)
new_high = new_high.sort_index(axis=1)
new_low = new_low.sort_index(axis=1)
new_volume = new_volume.sort_values(axis=1)
new_money = new_money.sort_values(axis=1)

new_close.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv')
new_open.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/open_1m.csv')
new_high.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/high_1m.csv')
new_low.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv')
new_volume.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv')
new_money.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/money_1m.csv')
