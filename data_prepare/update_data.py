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

auth('15951961478', '961478')
get_query_count()



all_stock = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx')

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')

# hs300
p = get_price('510300.XSHG', start_date='2014-01-01', end_date='2021-05-25',
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
p.to_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx')

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
open = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
low_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)


# 市场行情

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


update_daily_prices(new_end_date='2021-05-25', new_start_date='2014-01-01', close=close, open=open, high=high, low=low,
                    high_limit=high_limit, low_limit=low_limit, volume=volume)

#  南北向资金持仓-----------------------------

share = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/share.csv', index_col='Unnamed: 0', date_parser=dateparse)
ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)
value = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/value.csv', index_col='Unnamed: 0', date_parser=dateparse)


def update_north_data(new_end_date, new_start_date, share, ratio, value,close):
    if (share.index[0] == ratio.index[0]) & (ratio.index[0] == value.index[0]):
        print('all_start_time_equal{}'.format(share.index[0]))
    else:
        print('START time NOT equal')

    if (share.index[-1] == ratio.index[-1]) & (ratio.index[-1] == value.index[-1]):
        print('all_start_time_equal{}'.format(share.index[-1]))
    else:
        print('START time NOT equal')

    future_trade_days = get_trade_days(start_date=share.index[-1], end_date=new_end_date)[1:]  # 第一天重复
    old_trade_days = get_trade_days(start_date=new_start_date, end_date=share.index[0])[:-1]  # 最后一天重复
    new_trade_days = list(future_trade_days) + list(old_trade_days)

    for date in new_trade_days:
        share.loc[date] = np.nan
        ratio.loc[date] = np.nan
        value.loc[date] = np.nan

    df = finance.run_query(
        query(finance.STK_HK_HOLD_INFO).filter(finance.STK_HK_HOLD_INFO.link_id.in_([310001, 310002]),
                                               finance.STK_HK_HOLD_INFO.day.in_(new_trade_days)))

    new_stocks = list(set(df['code']).difference(set(share.columns)))
    for s in new_stocks:
        share[s] = np.nan
        ratio[s] = np.nan
        value[s] = np.nan

    for date in tqdm(new_trade_days):
        temp = finance.run_query(
            query(finance.STK_HK_HOLD_INFO).filter(finance.STK_HK_HOLD_INFO.link_id.in_([310001, 310002]),
                                                   finance.STK_HK_HOLD_INFO.day == date))
        share.loc[date][temp['code']] = temp['share_number'].values # 有error???????
        ratio.loc[date][temp['code']] = temp['share_ratio'].values

        tmp_share = share.loc[date]
        tmp_price = close.loc[date][share.columns]
        value.loc[date] = tmp_price * tmp_share * 10 ** (-8)

    share = share.sort_index(axis=1)
    ratio = ratio.sort_index(axis=1)
    value = value.sort_index(axis=1)
    share.index = pd.to_datetime(share.index)
    ratio.index = pd.to_datetime(ratio.index)
    value.index = pd.to_datetime(value.index)

    share.to_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/share.csv')
    ratio.to_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/ratio.csv')
    value.to_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/value.csv')


update_north_data(new_end_date='2021-05-25', new_start_date='2014-01-01', share=share, ratio=ratio, value=value, close=close)



# market_cap

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)

def update_market_cap(new_start_date, new_end_date, market_cap, share):
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
    # share 是持股数，用来检测股票数量是否相等, 新加入股票补齐

    new_stocks = list(set(share.columns).difference(set(market_cap.columns)))
    if len(new_stocks)>0:
        print ('total number of new stocks = {}'.format(len(new_stocks)))
        for s in new_stocks:
            market_cap[s] = np.nan

        for date in tqdm(list(market_cap.index)):
            df = get_fundamentals(query(valuation.code,
                                        valuation.market_cap).filter(valuation.code.in_(new_stocks)), date=datetime.date(date))
            # get_fundamentals 必须是 date格式的日期
            market_cap.loc[date][df['code']] = df['market_cap'].values

    market_cap = market_cap.sort_index(axis=0)  # 按index排序
    market_cap = market_cap.sort_index(axis=1) # 按股票代码排序
    market_cap.to_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/market_cap.csv')


update_market_cap(new_start_date='2014-01-01',new_end_date='2021-05-25',market_cap=market_cap, share=share)

# 融资融券
margin_buy_value = pd.read_csv('/Users/caichaohong/Desktop/Zenki/融资融券/margin_buy_value.csv', index_col='Unnamed: 0',date_parser=dateparse)


def update_margin_buy(new_start_date, new_end_date, margin_df):
    future_trade_days = get_trade_days(start_date=margin_df.index[-1], end_date=new_end_date)[1:]  # 第一天重复
    old_trade_days = get_trade_days(start_date=new_start_date, end_date=margin_df.index[0])[:-1]  # 最后一天重复
    new_trade_days = list(future_trade_days) + list(old_trade_days)

    if len(new_trade_days) > 0:
        for date in new_trade_days:
            margin_df.loc[date] = np.nan
        margin_buy_value.index = pd.to_datetime(margin_buy_value.index) # 最后新加的行index不是datetime

        for stock in tqdm(margin_df.columns):
            if len(future_trade_days) > 0:

                for d in future_trade_days:
                    margin_df.loc[d] = np.nan
                margin_buy_value.index = pd.to_datetime(margin_buy_value.index)  # 最后新加的行index不是datetime

                df1 = get_mtss(stock, future_trade_days[0], future_trade_days[-1], fields=['date', 'sec_code', 'fin_buy_value'])
                df1.index = pd.to_datetime(df1['date'])
                margin_df[stock].loc[df1.index] = df1['fin_buy_value'].values

            if len(old_trade_days) > 0:

                for d in old_trade_days:
                    margin_df.loc[d] = np.nan
                margin_buy_value.index = pd.to_datetime(margin_buy_value.index)  # 最后新加的行index不是datetime

                df2 = get_mtss(stock, old_trade_days[0], old_trade_days[-1], fields=['date', 'sec_code', 'fin_buy_value'])
                df2.index = pd.to_datetime(df2['date'])
                margin_df[stock].loc[df2.index] = df2['fin_buy_value'].values

        margin_df.to_csv('/Users/caichaohong/Desktop/Zenki/融资融券/margin_buy_value.csv')

    else:
        print("No need to Update")

new_start_date='2017-03-17'
new_end_date='2021-05-10'


update_margin_buy(new_start_date='2017-03-17', new_end_date='2021-05-10', margin_df=margin_buy_value)


# 前10活跃
top_10_net_buy = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_net_buy.xlsx', index_col='Unnamed: 0')
top_10_net_buy = top_10_net_buy.dropna(axis=0, how='all')
raw_df = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_raw_data.xlsx', index_col='Unnamed: 0')

market_cap = pd.read('/Users/caichaohong/Desktop/Zenki/南北向资金/market_cap.xlsx', index_col='Unnamed: 0')

margin_buy_value = pd.read_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_buy_value.xlsx')
margin_sell_value = pd.read_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_sell_value.xlsx')

# 南北总资金
north_amount = finance.run_query(query(finance.STK_ML_QUOTA).filter(finance.STK_ML_QUOTA.day>='2017-03-17',
                                                                    finance.STK_ML_QUOTA.link_id.in_([310001,310002])))
north_df = north_amount.groupby('day').sum()
north_df = north_df.drop(columns=['id','link_id', 'currency_id', 'quota_daily', 'quota_daily_balance'])
north_df.index = pd.to_datetime(north_df.index)
north_df = north_df[north_df.index<= '2021-05-10'] #比交易日少很多




# 所有公司名称
all_stock = finance.run_query(query(finance.STK_COMPANY_INFO.code,
                                    finance.STK_COMPANY_INFO.short_name).filter(
    finance.STK_COMPANY_INFO.code.in_(close.columns)))

all_stock.to_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx')


