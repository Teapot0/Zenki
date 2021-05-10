import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, income, balance, cash_flow, indicator, get_fundamentals, get_fundamentals_continuously, finance
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

# 每9个图一个
ttm_profit = pd.read_excel('/Users/caichaohong/Desktop/Zenki/ttm_profit.xlsx',index_col='statDate')



for j in tqdm(range(0, len(roe_15_df['code']), 9)): #每9张图一个
    plt.figure(figsize=(10, 10))

    for i in tqdm(range(j, j + 9)):
        stock_code = roe_15_df['code'][i]

        p = get_price(stock_code, start_date='2015-01-01', end_date='2022-05-01', frequency='daily',fields=['open','close'])
        p[p.index >= '2021-04-14'] = np.nan
        p['date_str'] = [x.strftime('%Y-%m-%d') for x in p.index]
        p['year'] = p.index.year

        # q_np = query(income.np_parent_company_owners, income.statDate).filter(income.code == stock_code)
        # np_list = [get_fundamentals(q_np, statDate='20{}q'.format(y) + str(i)) for y in
        #            [14, 15, 16, 17, 18, 19, 20] for i in range(1, 5)]

        # np_df = pd.DataFrame()  # 转换为单季度 net profit dataframe
        # for q in range(len(np_list)):
        #     np_df = np_df.append(np_list[q])

        # np_df['net_profit'] = np_df['np_parent_company_owners'] * 10 ** (-8)  # 改为亿元

        # #存数据
        # ttm_profit[roe_15_df['name'][i]][np_df['statDate']] = np_df['net_profit'].values

        np_df = pd.DataFrame(ttm_profit[roe_15_df['name'][i]]).dropna()
        np_df['ttm'] = np_df[roe_15_df['name'][i]].rolling(4).sum()  # ttm净利润
        np_df = np_df.dropna()  # 2011年前三季度为NA

        p['ttm'] = np.nan
        for ii in range(np_df.shape[0]):
            p['ttm'][(p['date_str'] <= np_df.index[ii]) & (
                        p['date_str'] > np_df.index[ii - 1])] = np_df['ttm'].values[ii]

        # 把>2021.4.12的ttm变成na

        p['exp'] = np.nan
        for y in list(expectations.index):
          p['exp'][p['year'] == y] = expectations[roe_15_df['name'][i]][y]

        ax = plt.subplot(3, 3, i % 9 + 1)
        ax.plot(p['close'], 'black')
        ax11 = ax.twinx()
        ax11.plot(p['ttm'], 'blue')
        ax11.plot(p['exp'][p['date_str'] < np_df.index[-1]], 'red')
        ax11.plot(p['exp'][p['date_str'] >= np_df.index[-1]], 'red', linestyle='--')
        ax.set_title(roe_15_df['name'][i])
        plt.tight_layout()

    if i == len(roe_15_df['code'])-1:
        plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(str(i)))
        plt.close()
        break

    plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(j))
    plt.close()







