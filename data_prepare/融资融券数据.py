import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_mtss
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from jqdatasdk import auth, get_price, get_trade_days, get_margincash_stocks

auth('13382017213', 'Aasd120120')
get_query_count()


start = '2021-01-11'
end = '2021-12-09'
dateList = get_trade_days(start_date=start, end_date=end)

margin_stock_list = []
for date in tqdm(dateList):
    temp = get_margincash_stocks(date=dateList[0])
    margin_stock_list = list(set(margin_stock_list).union(set(temp)))


rongziyue = pd.DataFrame(index=dateList, columns=margin_stock_list)
rongzimairu = pd.DataFrame(index=dateList, columns=margin_stock_list)
rongquanyue = pd.DataFrame(index=dateList, columns=margin_stock_list)
rongquanmaichu = pd.DataFrame(index=dateList, columns=margin_stock_list)
total_value = pd.DataFrame(index=dateList, columns=margin_stock_list)


for stock in tqdm(margin_stock_list):
    temp_df = get_mtss('301089.XSHE', start_date=start, end_date=end, fields=['date','sec_code',
                                                                      'fin_value','fin_buy_value','fin_refund_value',
                                                                      'sec_value','sec_sell_value','sec_refund_value',
                                                                      'fin_sec_value'])
    rongziyue[stock].loc[temp_df['date']] = temp_df['fin_value'].values
    rongzimairu[stock].loc[temp_df['date']] = temp_df['fin_buy_value'].values
    
    rongquanyue[stock].loc[temp_df['date']] = temp_df['sec_value'].values
    rongquanmaichu[stock].loc[temp_df['date']] = temp_df['sec_sell_value'].values

    total_value[stock].loc[temp_df['date']] = temp_df['fin_sec_value'].values

rongziyue = rongziyue.sort_index(axis=1)
rongzimairu = rongzimairu.sort_index(axis=1)
rongquanyue = rongquanyue.sort_index(axis=1)
rongquanmaichu = rongquanmaichu.sort_index(axis=1)
total_value = total_value.sort_index(axis=1)

rongziyue.to_parquet('./data/rongzi/rongziyue.parquet')
rongzimairu.to_parquet('./data/rongzi/rongzimairu.parquet')
rongquanyue.to_parquet('./data/rongzi/rongquanyue.parquet')
rongquanmaichu.to_parquet('./data/rongzi/rongquanmaichu.parquet')
total_value.to_parquet('./data/rongzi/rongzirongquanyue.parquet')


z = rongquanyue.pct_change(1)
z = z.iloc[:-1,]
