import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,date
from basic_funcs.basic_function import *
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss,get_money_flow

import warnings
warnings.filterwarnings('ignore')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

auth('13382017213', 'Aasd120120')
get_query_count()

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



# 大单
net_amount_main = pd.DataFrame(columns=close.columns, index=close.index)
net_pct_main = pd.DataFrame(columns=close.columns, index=close.index)
net_amount_xl = pd.DataFrame(columns=close.columns, index=close.index)
net_pct_xl = pd.DataFrame(columns=close.columns, index=close.index)
net_amount_l = pd.DataFrame(columns=close.columns, index=close.index)
net_pct_l = pd.DataFrame(columns=close.columns, index=close.index)
net_amount_m = pd.DataFrame(columns=close.columns, index=close.index)
net_pct_m = pd.DataFrame(columns=close.columns, index=close.index)
net_amount_s = pd.DataFrame(columns=close.columns, index=close.index)
net_pct_s = pd.DataFrame(columns=close.columns, index=close.index)

for s in tqdm(list(close.columns)):
    tmp = get_money_flow(s, start_date='2014-01-01', end_date='2021-07-07',
                                fields=['date','sec_code','net_amount_main','net_pct_main',
                                        'net_amount_xl','net_pct_xl',
                                        'net_amount_l','net_pct_l',
                                        'net_amount_m','net_pct_m',
                                        'net_amount_s','net_pct_s'])
    tmp.index = tmp['date']
    net_amount_main[s].loc[tmp.index] = tmp['net_amount_main']
    net_pct_main[s].loc[tmp.index] = tmp['net_pct_main']
    net_amount_xl[s].loc[tmp.index] = tmp['net_amount_xl']
    net_pct_xl[s].loc[tmp.index] = tmp['net_pct_xl']
    net_amount_l[s].loc[tmp.index] = tmp['net_amount_l']
    net_pct_l[s].loc[tmp.index] = tmp['net_pct_l']
    net_amount_m[s].loc[tmp.index] = tmp['net_amount_m']
    net_pct_m[s].loc[tmp.index] = tmp['net_pct_m']
    net_amount_s[s].loc[tmp.index] = tmp['net_amount_s']
    net_pct_s[s].loc[tmp.index] = tmp['net_pct_s']

net_amount_main.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv')
net_pct_main.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_main.csv')
net_amount_xl.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_xl.csv')
net_pct_xl.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_xl.csv')
net_amount_l.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_l.csv')
net_pct_l.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_l.csv')
net_amount_m.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_m.csv')
net_pct_m.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_m.csv')
net_amount_s.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_s.csv')
net_pct_s.to_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_s.csv')



# 金额5m
trade_days = get_trade_days(start_date='2018-01-01', end_date='2021-08-02')
close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']

stock_list_days = list(close_daily.isna().sum()[close_daily.isna().sum() <= 244].index)  # 大于一年的
close_daily = close_daily[stock_list_days]

start = trade_days[0].strftime('%Y-%m-%d 09:00:00')
end = trade_days[-1].strftime('%Y-%m-%d 15:00:00')

tmp = get_price(close_daily.columns[0], start_date=start,end_date=end,frequency='5m',
                                        fields=['money'])

money_5m = pd.DataFrame(columns=close_daily.columns,index=tmp.index)

for s in tqdm(close_daily.columns):
    tmp = get_price(s, start_date=start, end_date=end, frequency='5m',
                    fields=['money'])
    money_5m[s] = tmp['money']
money_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/money_5m.csv')



# 金额daily
close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)

start = close_daily.index[0]
end = close_daily.index[-1]

tmp = get_price(close_daily.columns[0], start_date=start,end_date=end,frequency='1d',
                                        fields=['money'])

money_daily = pd.DataFrame(columns=close_daily.columns,index=tmp.index)

for s in tqdm(close_daily.columns):
    tmp = get_price(s, start_date=start, end_date=end, frequency='1d',
                    fields=['money'])
    money_daily[s] = tmp['money']
money_daily.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv')











