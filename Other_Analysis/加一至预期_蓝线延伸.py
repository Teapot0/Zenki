import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, income, balance, cash_flow, indicator, \
    get_fundamentals, get_fundamentals_continuously, finance
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

auth('15951961478', '961478')
get_query_count()

roe_15_df = pd.read_excel('/Users/caichaohong/Desktop/Zenki/roe_15.xlsx', index_col='Unnamed: 0')
expectations = pd.read_excel('/Users/caichaohong/Desktop/Zenki/expectations.xlsx', index_col='/亿元')

ttm_profit = pd.read_excel('/Users/caichaohong/Desktop/Zenki/ttm_profit.xlsx',index_col='statDate')

# 每9个图一个

for j in tqdm(range(0, len(roe_15_df['code']), 9)): #每9张图一个
    plt.figure(figsize=(10, 10))

    for i in tqdm(range(j, j + 9)):
        stock_code = roe_15_df['code'][i]

        p = get_price(stock_code, start_date='2015-01-01', end_date='2022-05-01', frequency='daily',fields=['open','close'])
        p[p.index >= '2021-04-14'] = np.nan
        p['date_str'] = [x.strftime('%Y-%m-%d') for x in p.index]
        p['year'] = p.index.year

        np_df = pd.DataFrame(ttm_profit[roe_15_df['name'][i]]).dropna()
        np_df['ttm'] = np_df[roe_15_df['name'][i]].rolling(4).sum()  # ttm净利润
        np_df = np_df.dropna()  # 2011年前三季度为NA

        p['ttm'] = np.nan
        for ii in range(np_df.shape[0]):
            p['ttm'][(p['date_str'] <= np_df.index[ii]) & (
                        p['date_str'] > np_df.index[ii - 1])] = np_df['ttm'].values[ii]

        # plot_前后时间要重叠，不然画出来断层
        p['ttm'][(p['date_str']>np_df.index[-1])&(p['date_str']<='2021-12-31')] = expectations[roe_15_df['name'][i]][2021]
        p['ttm'][(p['date_str']>'2021-12-31')&(p['date_str']<='2022-12-31')] = expectations[roe_15_df['name'][i]][2022]

        ax = plt.subplot(3, 3, i % 9 + 1)
        ax.plot(p['close'], 'black')
        ax11 = ax.twinx()

        ttm_min = p['ttm'].dropna().min()
        ax11.set_ylim(ttm_min, ttm_min*(p['close'].max()/p['close'].min()))

        ax11.plot(p['ttm'][p['date_str'] < np_df.index[-1]], 'blue')
        ax11.plot(p['ttm'][(p['date_str'] >= np_df.index[-1]) & (p['date_str']<='2021-12-31')], 'red', linestyle='--')
        ax11.plot(p['ttm'][(p['date_str'] >= '2021-12-31') & (p['date_str']<='2022-12-31')], 'red', linestyle='--')

        ax.set_title(roe_15_df['name'][i])
        plt.tight_layout()

    if i == len(roe_15_df['code'])-1:
        plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(str(i)))
        plt.close()
        break

    plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(j))
    plt.close()






