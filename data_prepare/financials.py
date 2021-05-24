import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, indicator,income
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *



dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

auth('15951961478', '961478')
get_query_count()

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = clean_close(close, low, high_limit)  # 新股一字板
close = clean_st_exit(close)  # 退市和ST
# 上市大于4年的
stock_list_4 = list(close.isna().sum()[close.isna().sum()<=40].index)
close = close[stock_list_4]

years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
years_q = [x+'q'+str(i) for x in years for i in range(1,5)]
years_q.append('2021q1')
all_stock = list(close.columns)

datelist=[get_fundamentals(query(indicator.statDate).filter(income.code == all_stock[0]),statDate=y) for y in years_q]
date_df = pd.DataFrame()
for rr in datelist:
    date_df = pd.concat([date_df, rr], join='outer')

roe = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
roa = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
total_revenue = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
total_revenue_growth = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
net_profit = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
net_profit_growth = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
gross_profit_margin = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
adjusted_profit = pd.DataFrame(columns=all_stock, index=date_df['statDate']) #扣非净利润

for i in tqdm(range(2914,len(all_stock))):
    ret = [get_fundamentals(query(indicator.statDate,
                                  indicator.roe,
                                  indicator.roa,
                                  income.total_operating_revenue,
                                  indicator.inc_total_revenue_year_on_year,
                                  income.np_parent_company_owners,
                                  indicator.inc_net_profit_to_shareholders_year_on_year,
                                  indicator.gross_profit_margin,
                                  indicator.adjusted_profit).filter(income.code == all_stock[i],),
                           statDate=y) for y in years_q]
    temp = pd.DataFrame()
    for rr in ret:
        temp = pd.concat([temp, rr], join='outer')
    temp.index = temp['statDate']

    roe[all_stock[i]].loc[temp.index] = temp['roe'].values
    roa[all_stock[i]].loc[temp.index] = temp['roa'].values
    total_revenue[all_stock[i]].loc[temp.index] = temp['total_operating_revenue'].values * 10**(-8)  # 亿元
    total_revenue_growth[all_stock[i]].loc[temp.index] = temp['inc_total_revenue_year_on_year'].values
    net_profit[all_stock[i]].loc[temp.index] = temp['np_parent_company_owners'].values * 10**(-8)
    net_profit_growth[all_stock[i]].loc[temp.index] = temp['inc_net_profit_to_shareholders_year_on_year'].values
    gross_profit_margin[all_stock[i]].loc[temp.index] = temp['gross_profit_margin'].values
    adjusted_profit[all_stock[i]].loc[temp.index] = temp['adjusted_profit'].values


roe.to_csv('/Users/caichaohong/Desktop/Zenki/financials/roe.csv')
roa.to_csv('/Users/caichaohong/Desktop/Zenki/financials/roa.csv')
total_revenue.to_csv('/Users/caichaohong/Desktop/Zenki/financials/total_revenue.csv')
total_revenue_growth.to_csv('/Users/caichaohong/Desktop/Zenki/financials/total_revenue_growth.csv')
net_profit.to_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit.csv')
net_profit_growth.to_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_growth.csv')
gross_profit_margin.to_csv('/Users/caichaohong/Desktop/Zenki/financials/gross_profit_margin.csv')
adjusted_profit.to_csv('/Users/caichaohong/Desktop/Zenki/financials/adjusted_profit.csv')


