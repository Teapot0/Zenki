import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_mtss
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from jqdatasdk import auth, get_price, get_trade_days, get_margincash_stocks

auth('15951961478', '961478')
get_query_count()


start = '2017-03-17'
end = '2021-04-18'
dateList = get_trade_days(start_date=start, end_date=end)

margin_stock_list = []
for date in tqdm(dateList):
    temp = get_margincash_stocks(date=dateList[0])
    margin_stock_list = list(set(margin_stock_list).union(set(temp)))


margin_buy_value = pd.DataFrame(index=dateList, columns=margin_stock_list)
margin_total_value = pd.DataFrame(index=dateList, columns=margin_stock_list)
margin_sell_value = pd.DataFrame(index=dateList, columns=margin_stock_list)
for stock in tqdm(margin_stock_list):
    temp_df = get_mtss(stock, start_date=start, end_date=end, fields=['date','sec_code','fin_value','fin_buy_value','fin_refund_value','sec_value','sec_sell_value','sec_refund_value','fin_sec_value'])
    margin_buy_value[stock].loc[temp_df['date']] = temp_df['fin_buy_value'].values
    margin_total_value[stock].loc[temp_df['date']] = temp_df['fin_value'].values
    margin_sell_value[stock].loc[temp_df['date']] = temp_df['sec_sell_value'].values

margin_buy_value.to_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_buy_value.xlsx')
margin_total_value.to_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_total_value.xlsx')
margin_sell_value.to_excel('/Users/caichaohong/Desktop/Zenki/融资融券/margin_sell_value.xlsx')