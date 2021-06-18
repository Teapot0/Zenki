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


years = ['2010','2011','2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
years_q = [x+'q'+str(i) for x in years for i in range(1,5)]
years_q.append('2021q1')
all_stock = list(close.columns)

# 季度：get_date_index :
datelist = [get_fundamentals(query(indicator.statDate).filter(income.code == all_stock[0]),statDate=y) for y in years_q]
date_df = pd.DataFrame()
for rr in datelist:
    date_df = pd.concat([date_df, rr], join='outer')
#
roe = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
inc_roe = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
roa = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
total_revenue = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
total_revenue_growth = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
net_profit = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
net_profit_growth = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
gross_profit_margin = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
adjusted_profit = pd.DataFrame(columns=all_stock, index=date_df['statDate']) #扣非净利润

for i in tqdm(range(len(all_stock))):
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


# 年度：

datelist=[get_fundamentals(query(indicator.statDate).filter(income.code == all_stock[0]),statDate=y) for y in years]
date_df = pd.DataFrame()
for rr in datelist:
    date_df = pd.concat([date_df, rr], join='outer')
#
roe_yearly = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
inc_roe_yearly = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
roa_yearly = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
total_revenue_yearly = pd.DataFrame(columns = all_stock, index=date_df['statDate'])
total_revenue_growth_yearly = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
net_profit_yearly = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
net_profit_growth_yearly = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
gross_profit_margin_yearly = pd.DataFrame(columns=all_stock, index=date_df['statDate'])
adjusted_profit_yearly = pd.DataFrame(columns=all_stock, index=date_df['statDate']) #扣非净利润

for i in tqdm(range(len(all_stock))):
    ret = [get_fundamentals(query(indicator.statDate,
                                  indicator.roe,
                                  indicator.roa,
                                  income.total_operating_revenue,
                                  indicator.inc_total_revenue_year_on_year,
                                  income.np_parent_company_owners,
                                  indicator.inc_net_profit_to_shareholders_year_on_year,
                                  indicator.gross_profit_margin,
                                  indicator.adjusted_profit).filter(income.code == all_stock[i],),
                           statDate=y) for y in years]
    temp = pd.DataFrame()
    for rr in ret:
        temp = pd.concat([temp, rr], join='outer')
    temp.index = temp['statDate']

    roe_yearly[all_stock[i]].loc[temp.index] = temp['roe'].values
    roa_yearly[all_stock[i]].loc[temp.index] = temp['roa'].values
    total_revenue_yearly[all_stock[i]].loc[temp.index] = temp['total_operating_revenue'].values * 10**(-8)  # 亿元
    total_revenue_growth_yearly[all_stock[i]].loc[temp.index] = temp['inc_total_revenue_year_on_year'].values
    net_profit_yearly[all_stock[i]].loc[temp.index] = temp['np_parent_company_owners'].values * 10**(-8)
    net_profit_growth_yearly[all_stock[i]].loc[temp.index] = temp['inc_net_profit_to_shareholders_year_on_year'].values
    gross_profit_margin_yearly[all_stock[i]].loc[temp.index] = temp['gross_profit_margin'].values
    adjusted_profit_yearly[all_stock[i]].loc[temp.index] = temp['adjusted_profit'].values * 10**(-8)


roe_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv')
roa_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/roa_yearly.csv')
total_revenue_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/total_revenue_yearly.csv')
total_revenue_growth_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/total_revenue_growth_yearly.csv')
net_profit_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv')
net_profit_growth_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_growth_yearly.csv')
gross_profit_margin_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/gross_profit_margin_yearly.csv')
adjusted_profit_yearly.to_csv('/Users/caichaohong/Desktop/Zenki/financials/adjusted_profit_yearly.csv')


# valuation 市值数据：======================
dateList = [x.strftime('%Y-%m-%d') for x in close.index]
circulating_market_cap = pd.DataFrame(index=dateList, columns=close.columns)
pe_ratio = pd.DataFrame(index=dateList, columns=close.columns)
ps_ratio = pd.DataFrame(index=dateList, columns=close.columns)

for i in tqdm(range(len(dateList))):
    date = dateList[i]

    temp = get_fundamentals(query(valuation.code, valuation.day, valuation.market_cap, valuation.circulating_market_cap,
                                  valuation.pe_ratio, valuation.ps_ratio).filter(
        valuation.code.in_(close.columns)), date=dateList[i])

    circulating_market_cap.loc[date][temp['code']] = temp['circulating_market_cap'].values
    pe_ratio.loc[date][temp['code']] = temp['pe_ratio'].values
    ps_ratio.loc[date][temp['code']] = temp['ps_ratio'].values

circulating_market_cap.to_csv('/Users/caichaohong/Desktop/Zenki/financials/circulating_market_cap.csv')
pe_ratio.to_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv')
ps_ratio.to_csv('/Users/caichaohong/Desktop/Zenki/financials/ps_ratio.csv')




# 股息率

sw_ind1 = get_industries('sw_l1')
# 801780银行，801150医药，801160公用事业，801180房地产
df_bank=finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.code=='801780'))







