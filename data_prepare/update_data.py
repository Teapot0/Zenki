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

# auth('15951961478', '961478')
auth('13382017213', 'Aasd120120')
get_query_count()

# hs300

New_end_date = '2021-09-10'

index_code = ['510300.XSHG','510050.XSHG', '510500.XSHG','159948.XSHE']
for code in index_code:
    p = get_price(code, start_date='2014-01-01', end_date=New_end_date,
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
    p.to_excel('/Users/caichaohong/Desktop/Zenki/price/{}.xlsx'.format(code))

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0')
open = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv', index_col='Unnamed: 0')
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0')
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0')
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0')
low_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.csv', index_col='Unnamed: 0')
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0')
money = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv', index_col='Unnamed: 0')

# 市场行情
update_daily_prices(new_end_date=New_end_date, new_start_date='2014-01-01', close=close, open=open, high=high, low=low,
                    high_limit=high_limit, low_limit=low_limit, volume=volume, money=money)


# ST
st_df = get_extras('is_st', list(close.columns), start_date='2014-01-01', end_date=New_end_date)
st_df.to_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv')

# index_holds
hs300_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/hs300_holds.csv', index_col='Unnamed: 0')
zz500_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz500_holds.csv', index_col='Unnamed: 0')
zz1000_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz1000_holds.csv', index_col='Unnamed: 0')

update_index_holds(hs300_holds=hs300_holds, zz500_holds=zz500_holds, zz1000_holds=zz1000_holds, close=close)


# 5分钟
hs300_5m = get_price('510300.XSHG', start_date='2018-01-01', end_date='2021-08-13'+' 15:00:00',frequency='5m',
                             fields=['open', 'close', 'high', 'low', 'volume'])
hs300_5m.to_csv('/Users/caichaohong/Desktop/Zenki/price/5m/510300_5m.csv')


# 大单
net_amount_main = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv', index_col='Unnamed: 0')
net_pct_main = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_main.csv', index_col='Unnamed: 0')
net_amount_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_xl.csv', index_col='Unnamed: 0')
net_pct_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_xl.csv', index_col='Unnamed: 0')
net_amount_l = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_l.csv', index_col='Unnamed: 0')
net_pct_l = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_l.csv', index_col='Unnamed: 0')
net_amount_m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_m.csv', index_col='Unnamed: 0')
net_pct_m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_m.csv', index_col='Unnamed: 0')
net_amount_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_s.csv', index_col='Unnamed: 0')
net_pct_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_s.csv', index_col='Unnamed: 0')


update_money_flow(New_end_date='2021-08-26', close=close,
                  net_amount_main=net_amount_main,net_pct_main=net_pct_main,
                  net_amount_xl=net_amount_xl,net_pct_xl=net_pct_xl,
                  net_amount_l=net_amount_l,net_pct_l=net_pct_l,
                  net_amount_m=net_amount_m, net_pct_m=net_pct_m, net_amount_s=net_amount_s,net_pct_s=net_pct_s)


# market_cap
market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0')

update_market_cap(new_start_date='2014-01-01',new_end_date='2021-09-03',market_cap=market_cap, close=close)


# financials pe
circulating_market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/circulating_market_cap.csv', index_col='Unnamed: 0')
pe_ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0')
ps_ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/ps_ratio.csv', index_col='Unnamed: 0')

update_financials(new_end_date='2021-09-03', new_start_date='2014-01-01', cir_mc=circulating_market_cap,pe=pe_ratio,ps=ps_ratio)

#  南北向资金持仓 -----------------------------
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


update_north_data(new_end_date='2021-05-26', new_start_date='2014-01-01', share=share, ratio=ratio, value=value, close=close)


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


new_start_date = '2017-03-17'
new_end_date = '2021-05-10'


update_margin_buy(new_start_date='2017-03-17', new_end_date='2021-05-10', margin_df=margin_buy_value)


# 前10活跃
top_10_net_buy = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_net_buy.xlsx', index_col='Unnamed: 0')
top_10_net_buy = top_10_net_buy.dropna(axis=0, how='all')
raw_df = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_raw_data.xlsx', index_col='Unnamed: 0')


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







