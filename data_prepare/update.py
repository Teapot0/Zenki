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

auth(手机, 手机后6位)
get_query_count()


all_stock = pd.read_excel('~/Desktop/Zenki/all_stock_names.xlsx')

hs300 = pd.read_excel('~/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')

# hs300
p = get_price('510300.XSHG', start_date='2014-01-01', end_date='2021-06-08',
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
p.to_excel('~/Desktop/Zenki/price/510300.XSHG.xlsx')

close = pd.read_csv('~/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
open = pd.read_csv('~/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = pd.read_csv('~/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('~/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('~/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
low_limit = pd.read_csv('~/Desktop/Zenki/price/daily/low_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('~/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)


# 市场行情

update_daily_prices(new_end_date='2021-06-08', new_start_date='2014-01-01', close=close, open=open, high=high, low=low,
                    high_limit=high_limit, low_limit=low_limit, volume=volume)


# market_cap

market_cap = pd.read_csv('~/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)

update_market_cap(new_start_date='2014-01-01',new_end_date='2021-06-08',market_cap=market_cap, close=close)


# financials pe
circulating_market_cap = pd.read_csv('~/Desktop/Zenki/financials/circulating_market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)
pe_ratio = pd.read_csv('~/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)
ps_ratio = pd.read_csv('~/Desktop/Zenki/financials/ps_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)


def update_financials(new_start_date, new_end_date, cir_mc,pe,ps):
    # share 是持股数df，换成其他df也行，用来检测股票数量是否相等

    future_trade_days = get_trade_days(start_date=pe.index[-1], end_date=new_end_date)[1:]  # 第一天重复
    old_trade_days = get_trade_days(start_date=new_start_date, end_date=pe.index[0])[:-1]  # 最后一天重复
    new_trade_days = list(future_trade_days) + list(old_trade_days)

    if len(new_trade_days) > 0:
        for date in new_trade_days:
            cir_mc.loc[date] = np.nan
            pe.loc[date] = np.nan
            ps.loc[date] = np.nan

        for date in tqdm(new_trade_days):
            df = get_fundamentals(query(valuation.code,
                                        valuation.circulating_market_cap,
                                        valuation.pe_ratio,
                                        valuation.ps_ratio).filter(valuation.code.in_(list(pe.columns))), date=date)
            cir_mc.loc[date][df['code']] = df['circulating_market_cap'].values
            pe.loc[date][df['code']] = df['pe_ratio'].values
            ps.loc[date][df['code']] = df['ps_ratio'].values
    else:
        print("No need to Update")
    cir_mc.index = pd.to_datetime(cir_mc.index)
    pe.index = pd.to_datetime(pe.index)
    ps.index = pd.to_datetime(ps.index)

    cir_mc = cir_mc.sort_index(axis=0)  # 按index排序
    pe = pe.sort_index(axis=0)  # 按index排序
    ps = ps.sort_index(axis=0)  # 按index排序

    cir_mc =cir_mc.sort_index(axis=1) # 按股票代码排序
    pe = pe.sort_index(axis=1) # 按股票代码排序
    ps = ps.sort_index(axis=1) # 按股票代码排序

    cir_mc = cir_mc.dropna(how='all',axis=0)
    pe = pe.dropna(how='all',axis=0)
    ps = ps.dropna(how='all',axis=0)

    cir_mc.to_csv('~/Desktop/Zenki/financials/circulating_market_cap.csv')
    pe.to_csv('~/Desktop/Zenki/financials/pe_ratio.csv')
    ps.to_csv('~/Desktop/Zenki/financials/ps_ratio.csv')


update_financials(new_end_date='2021-06-08', new_start_date='2014-01-01', cir_mc=circulating_market_cap,pe=pe_ratio,ps=ps_ratio)









