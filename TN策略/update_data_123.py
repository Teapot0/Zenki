import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

auth(XX, XX)
get_query_count()

# hs300
p = get_price('510300.XSHG', start_date='2014-01-01', end_date='2021-06-18',
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
p.to_excel('~/Desktop/dataset/510300.XSHG.xlsx')

close = pd.read_csv('~/Desktop/dataset/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
open = pd.read_csv('~/Desktop/dataset/open.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = pd.read_csv('~/Desktop/dataset/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('~/Desktop/dataset/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('~/Desktop/dataset/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
low_limit = pd.read_csv('~/Desktop/dataset/low_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('~/Desktop/dataset/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)


# 市场行情

update_daily_prices(new_end_date='2021-06-18', new_start_date='2014-01-01', close=close, open=open, high=high, low=low,
                    high_limit=high_limit, low_limit=low_limit, volume=volume)

# market_cap

market_cap = pd.read_csv('~/Desktop/dataset/market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)

update_market_cap(new_start_date='2014-01-01',new_end_date='2021-06-17',market_cap=market_cap, close=close)


# financials pe
circulating_market_cap = pd.read_csv('~/Desktop/dataset/circulating_market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)
pe_ratio = pd.read_csv('~/Desktop/dataset/pe_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)
ps_ratio = pd.read_csv('~/Desktop/dataset/ps_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)

update_financials(new_end_date='2021-06-17', new_start_date='2014-01-01', cir_mc=circulating_market_cap,pe=pe_ratio,ps=ps_ratio)



# 所有公司名称
all_stock = finance.run_query(query(finance.STK_COMPANY_INFO.code,
                                    finance.STK_COMPANY_INFO.short_name).filter(
    finance.STK_COMPANY_INFO.code.in_(close.columns)))

all_stock.to_excel('~/Desktop/dataset/all_stock_names.xlsx')



