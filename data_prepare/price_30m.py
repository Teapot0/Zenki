
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

# trade_days = get_trade_days(start_date='2018-01-01', end_date='2021-09-08')
close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0')

# stock_list_days = list(close_daily.isna().sum()[close_daily.isna().sum() <= 244].index)  # 大于一年的
# close_daily = close_daily[stock_list_days]

start = '2018-01-01 15:00:00'
end = close_daily.index[-1] + ' 15:00:00'

tmp = get_price(close_daily.columns[0], start_date=start,end_date=end,frequency='30m',
                                        fields=['open', 'close', 'high', 'low', 'volume', 'money'])

close_30m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
open_30m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
high_30m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
low_30m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
volume_30m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)
money_30m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)

for s in tqdm(close_daily.columns):
    tmp = get_price(s, start_date=start, end_date=end, frequency='30m',
                    fields=['open', 'close', 'high', 'low', 'volume', 'money'])
    close_30m[s] = tmp['close']
    open_30m[s] = tmp['open']
    high_30m[s] = tmp['high']
    low_30m[s] = tmp['low']
    volume_30m[s] = tmp['volume']
    money_30m[s] = tmp['money']


close_30m.to_parquet('/Users/caichaohong/Desktop/Zenki/price/30m/close_30m.parquet')
open_30m.to_parquet('/Users/caichaohong/Desktop/Zenki/price/30m/open_30m.parquet')
high_30m.to_parquet('/Users/caichaohong/Desktop/Zenki/price/30m/high_30m.parquet')
low_30m.to_parquet('/Users/caichaohong/Desktop/Zenki/price/30m/low_30m.parquet')
volume_30m.to_parquet('/Users/caichaohong/Desktop/Zenki/price/30m/volume_30m.parquet')
money_30m.to_parquet('/Users/caichaohong/Desktop/Zenki/price/30m/money_30m.parquet')




# update data

close_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/close_30m.csv',index_col='Unnamed: 0')
open_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/open_30m.csv',index_col='Unnamed: 0')
high_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/high_30m.csv',index_col='Unnamed: 0')
low_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/low_30m.csv',index_col='Unnamed: 0')
volume_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volume_30m.csv',index_col='Unnamed: 0')
money_30m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/30m/money_30m.csv',index_col='Unnamed: 0')

last_end_date = '2021-07-24' # 加一天
new_end = '2021-09-08'

tmp_close = pd.DataFrame(columns=close_daily.columns)
tmp_open = pd.DataFrame(columns=close_daily.columns)
tmp_high = pd.DataFrame(columns=close_daily.columns)
tmp_low = pd.DataFrame(columns=close_daily.columns)
tmp_volume = pd.DataFrame(columns=close_daily.columns)
tmp_money = pd.DataFrame(columns=close_daily.columns)

for s in tqdm(close_daily.columns):
    tmp = get_price(s, start_date=last_end_date, end_date=new_end, frequency='30m',
                    fields=['open', 'close', 'high', 'low', 'volume','money'])
    tmp_close[s] = tmp['close']
    tmp_open[s] = tmp['open']
    tmp_high[s] = tmp['high']
    tmp_low[s] = tmp['low']
    tmp_volume[s] = tmp['volume']
    tmp_money[s] = tmp['money']

new_close = pd.concat([close_30m,tmp_close],axis=0,join='inner')
new_open = pd.concat([open_30m,tmp_open],axis=0,join='inner')
new_high = pd.concat([high_30m,tmp_high],axis=0,join='inner')
new_low = pd.concat([low_30m,tmp_low],axis=0,join='inner')
new_volume = pd.concat([volume_30m,tmp_volume],axis=0,join='inner')
new_money = pd.concat([money_30m,tmp_money],axis=0,join='inner')

# new_close = pd.concat([tmp_close,close_30m],axis=0,join='inner')
# new_open = pd.concat([tmp_open,open_30m],axis=0,join='inner')
# new_high = pd.concat([tmp_high,high_30m],axis=0,join='inner')
# new_low = pd.concat([tmp_low,low_30m],axis=0,join='inner')
# new_volume = pd.concat([tmp_volume,volume_30m],axis=0,join='inner')


new_close.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/close_30m.csv')
new_open.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/open_30m.csv')
new_high.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/high_30m.csv')
new_low.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/low_30m.csv')
new_volume.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/volume_30m.csv')
new_money.to_csv('/Users/caichaohong/Desktop/Zenki/price/30m/money_30m.csv')



