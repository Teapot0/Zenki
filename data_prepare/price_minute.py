
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

trade_days = get_trade_days(start_date='2019-01-01', end_date='2021-07-12')

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)

start = trade_days[0].strftime('%Y-%m-%d 09:00:00')
end = trade_days[-1].strftime('%Y-%m-%d 15:00:00')


# 2240+1359
for s in tqdm(close.columns[3599:]):
    tmp = get_price(s, start_date=start,end_date=end,frequency='1m',
                                        fields=['open', 'close', 'high', 'low', 'volume'])
    tmp.to_csv('/Users/caichaohong/Desktop/Zenki/price/1m/{}.csv'.format(s))


tmp = get_price(close.columns[0], start_date=trade_days[-1].strftime('%Y-%m-%d 09:00:00'),end_date=trade_days[-1].strftime('%Y-%m-%d 15:00:00'), frequency='1m',
                                    fields=['open', 'close', 'high', 'low', 'volume'])


