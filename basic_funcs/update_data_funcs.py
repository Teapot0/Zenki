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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')



def update_daily_prices(new_end_date, new_start_date, close, open, high, low, high_limit, low_limit, volume):

    # 判断是否同日期开始和结束
    if (close.index[0] == open.index[0]) & (open.index[0] == high.index[0]) & (high.index[0] == low.index[0]) & (
            low.index[0] == high_limit.index[0]) & (high_limit.index[0] == low_limit.index[0]) & (
            low_limit.index[0] == volume.index[0]):
        print('all_start_time_equal{}'.format(close.index[0]))
    else:
        print('START time NOT Equal')

    if (close.index[-1] == open.index[-1]) & (open.index[-1] == high.index[-1]) & (high.index[-1] == low.index[-1]) & (
            low.index[-1] == high_limit.index[-1]) & (high_limit.index[-1] == low_limit.index[-1]) & (
            low_limit.index[-1] == volume.index[-1]):
        print('all_end_time_equal{}'.format(close.index[-1]))
    else:
        print('END time NOT Equal')

    future_trade_days = get_trade_days(start_date=close.index[-1], end_date=new_end_date)[1:]  # 第一天重复
    old_trade_days = get_trade_days(start_date=new_start_date, end_date=close.index[0])[:-1]  # 最后一天重复
    new_trade_days = list(future_trade_days) + list(old_trade_days)
    print ('{} new trade days'.format(len(new_trade_days)))

    for date in tqdm(new_trade_days):
        close.loc[date] = np.nan
        open.loc[date] = np.nan
        high.loc[date] = np.nan
        low.loc[date] = np.nan
        volume.loc[date] = np.nan
        high_limit.loc[date] = np.nan
        low_limit.loc[date] = np.nan


    stock_list = list(close.columns)
    for s in tqdm(stock_list):
        if len(future_trade_days) > 0:
            temp = get_price(s, start_date=future_trade_days[0], end_date=future_trade_days[-1],
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
            close[s].loc[future_trade_days] = temp['close'].values
            open[s].loc[future_trade_days] = temp['open'].values
            high[s].loc[future_trade_days] = temp['high'].values
            low[s].loc[future_trade_days] = temp['low'].values
            volume[s].loc[future_trade_days] = temp['volume'].values
            high_limit[s].loc[future_trade_days] = temp['high_limit'].values
            low_limit[s].loc[future_trade_days] = temp['low_limit'].values
        else:
            print('NO future trade days to update')

        if len(old_trade_days) > 0:
            temp = get_price(s, start_date=old_trade_days[0], end_date=old_trade_days[-1],
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
            close[s].loc[old_trade_days] = temp['close'].values
            open[s].loc[old_trade_days] = temp['open'].values
            high[s].loc[old_trade_days] = temp['high'].values
            low[s].loc[old_trade_days] = temp['low'].values
            volume[s].loc[old_trade_days] = temp['volume'].values
            high_limit[s].loc[old_trade_days] = temp['high_limit'].values
            low_limit[s].loc[old_trade_days] = temp['low_limit'].values
        else:
            print('NO old trade days to update')

    close.index = pd.to_datetime(close.index)
    open.index = pd.to_datetime(open.index)
    high.index = pd.to_datetime(high.index)
    low.index = pd.to_datetime(low.index)
    high_limit.index = pd.to_datetime(high_limit.index)
    low_limit.index = pd.to_datetime(low_limit.index)
    volume.index = pd.to_datetime(volume.index)

    close = close.sort_index(axis=0)  # index时间排序
    open = open.sort_index(axis=0)  # index时间排序
    high = high.sort_index(axis=0)  # index时间排序
    low = low.sort_index(axis=0)  # index时间排序
    volume = volume.sort_index(axis=0)  # index时间排序
    high_limit = high_limit.sort_index(axis=0)  # index时间排序
    low_limit = low_limit.sort_index(axis=0)  # index时间排序

    close.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv')
    open.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv')
    high.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv')
    low.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv')
    high_limit.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv')
    low_limit.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.csv')
    volume.to_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv')



def update_market_cap(new_start_date, new_end_date, market_cap, close):
    # share 是持股数df，换成其他df也行，用来检测股票数量是否相等

    future_trade_days = get_trade_days(start_date=market_cap.index[-1], end_date=new_end_date)[1:]  # 第一天重复
    old_trade_days = get_trade_days(start_date=new_start_date, end_date=market_cap.index[0])[:-1]  # 最后一天重复
    new_trade_days = list(future_trade_days) + list(old_trade_days)

    if len(new_trade_days) > 0:
        for date in new_trade_days:
            market_cap.loc[date] = np.nan

        for date in tqdm(new_trade_days):
            df = get_fundamentals(query(valuation.code,
                                        valuation.market_cap).filter(valuation.code.in_(list(market_cap.columns))), date=date)
            market_cap.loc[date][df['code']] = df['market_cap'].values
    else:
        print("No need to Update")
    market_cap.index = pd.to_datetime(market_cap.index)
    # close 是持股数，用来检测股票数量是否相等, 新加入股票补齐

    new_stocks = list(set(close.columns).difference(set(market_cap.columns)))
    if len(new_stocks) > 0:
        print('total number of new stocks = {}'.format(len(new_stocks)))
        for s in new_stocks:
            market_cap[s] = np.nan

        for date in tqdm(list(market_cap.index)):
            df = get_fundamentals(query(valuation.code,
                                        valuation.market_cap).filter(valuation.code.in_(new_stocks)), date=datetime.date(date))
            # get_fundamentals 必须是 date格式的日期
            market_cap.loc[date][df['code']] = df['market_cap'].values

    market_cap = market_cap.sort_index(axis=0)  # 按index排序
    market_cap = market_cap.sort_index(axis=1) # 按股票代码排序
    market_cap=market_cap.dropna(how='all',axis=0)
    market_cap.to_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv')






