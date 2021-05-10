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

q = query(indicator.code,
          indicator.pubDate,
          indicator.statDate,
          indicator.roe)

years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
quarters = [x+'q{}'.format(i)  for x in years for i in range(1,5) ]

roe_df = [get_fundamentals(q, statDate=x) for x in quarters]

stock_list=[]
for i in range(40):
    stock_list = sorted(list(set(stock_list).union(set(roe_df[i].code))))


q_name = query(finance.STK_COMPANY_INFO).filter(finance.STK_COMPANY_INFO.code.in_(ret))

out = pd.DataFrame(index = quarters, columns=stock_list)

for i in range(40):
    tmp_stocks = roe_df[i].code
    out.loc[quarters[i]][tmp_stocks] = np.round(roe_df[i].roe.values, 3)

out.to_excel('/Users/caichaohong/Desktop/Zenki/roe_quarters.xlsx')


# quarterly price


quarter_date = ['-03-31', '-06-30', '-09-30', '-12-31'] #对应1-4季度
datelist = [y + quarter_date[i] for y in years for i in range(4)]
datelist.insert(0, '2011-01-01') # 多加一个季度算rts
# quarter_price = pd.DataFrame(columns=stock_list, index=datelist)

quarter_price = pd.read_excel('/Users/caichaohong/Desktop/Zenki/quarterly_price.xlsx')

pricelist=[]
for date in datelist:
    p = get_price('510300.XSHG', end_date=date, count=1, fields=['close'])['close']
    pricelist.append(p.values)

quarter_price_300 = pd.DataFrame(columns=['close'], index=datelist)
quarter_price_300['close'] = pricelist
quarter_price_300.to_excel('/Users/caichaohong/Desktop/Zenki/quarterly_price_300.xlsx')

for i in tqdm(range(quarter_price.shape[1])):
    stock = stock_list[i]
    pricelist = []
    for date in datelist:
        p = get_price(stock, end_date=date, count=1, fields=['close'])['close']
        pricelist.append(p.values)
    quarter_price[stock] = pricelist





#
# out['code'] = roe_df[0].sort_values(by='code')[roe_df[0]['code'].isin(stock_list)]['code'].values
# # out['name'] = finance.run_query(q_name)['short_name']
# out['2011'] = roe_df[0].sort_values(by='code')[roe_df[0]['code'].isin(stock_list)]['roe'].values



roe_15_df = pd.read_excel('/Users/caichaohong/Desktop/Zenki/roe_15.xlsx', index_col='Unnamed: 0')

# q_pub_date = query(indicator.pubDate,indicator.statDate).filter(indicator.code == stock_code)
#     pub_date = [get_fundamentals(q_pub_date, statDate=x) for x in ['2015', '2016', '2017', '2018', '2019']]


# single roe plot
for i in tqdm(range(len(roe_15_df['code']))):
    stock_code = roe_15_df['code'][i]

    # p = get_price(stock_code, start_date='2015-01-01', end_date=datetime.today().strftime('%Y-%m-%d'),frequency='daily')
    p = get_price(stock_code, start_date='2015-01-01', end_date='2019-12-31', frequency='daily')
    p['year'] = p.index.year
    p['roe'] = np.nan
    p['roe'][p['year'] == 2015] = roe_15_df[2015][roe_15_df['code'] == stock_code].values
    p['roe'][p['year'] == 2016] = roe_15_df[2016][roe_15_df['code'] == stock_code].values
    p['roe'][p['year'] == 2017] = roe_15_df[2017][roe_15_df['code'] == stock_code].values
    p['roe'][p['year'] == 2018] = roe_15_df[2018][roe_15_df['code'] == stock_code].values
    p['roe'][p['year'] == 2019] = roe_15_df[2019][roe_15_df['code'] == stock_code].values

    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ax1.plot(p['close'], 'black')
    ax2.plot(p['roe'], 'blue')
    plt.title(str(stock_code) + '\n' + roe_15_df['name'][i])
    plt.savefig('/Users/caichaohong/Desktop/Zenki/roe_plot/{}.png'.format(roe_15_df['name'][i]))
    plt.close()

# 每9个图一个

for j in tqdm(range(0, len(roe_15_df['code']), 9)): #每9张图一个
    plt.figure(figsize=(10, 10))

    for i in tqdm(range(j, j + 9)):
        stock_code = roe_15_df['code'][i]

        p = get_price(stock_code, start_date='2012-01-01', end_date='2021-05-01', frequency='daily')
        p['date_str'] = [x.strftime('%Y-%m-%d') for x in p.index]

        q_np = query(income.net_profit, income.statDate).filter(income.code == stock_code)
        np_list = [get_fundamentals(q_np, statDate='20{}q'.format(y) + str(i)) for y in
                   [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] for i in range(1, 5)]

        np_df = pd.DataFrame()  # 转换为单季度财务dataframe
        for q in range(len(np_list)):
            np_df = np_df.append(np_list[q])

        np_df['net_profit'] = np_df['net_profit'] * 10 ** (-8)  # 改为亿元
        np_df['ttm'] = np_df['net_profit'].rolling(4).sum()  # ttm净利润
        np_df = np_df.dropna()  # 2011年前三季度为NA
        np_df = np_df.reset_index(drop=True)

        p['ttm'] = np.nan
        for ii in range(np_df.shape[0]):
            if ii == np_df.shape[0] - 1:
                p['ttm'][(p['date_str'] >= np_df['statDate'].values[ii])] = np_df['ttm'].values[ii]
            else:
                p['ttm'][(p['date_str'] >= np_df['statDate'].values[ii]) & (
                            p['date_str'] < np_df['statDate'].values[ii + 1])] = np_df['ttm'].values[ii]

        ax = plt.subplot(3, 3, i % 9 + 1)
        ax.plot(p['close'], 'black')
        ax11 = ax.twinx()
        ax11.plot(p['ttm'], 'blue')
        ax.set_title(roe_15_df['name'][i])

    if i == 83:
        plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(str(i)))
        plt.close()
        break

    plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(j))
    plt.close()
