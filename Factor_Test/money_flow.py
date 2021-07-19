import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, income, balance, cash_flow, indicator, get_fundamentals, get_fundamentals_continuously, finance
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings('ignore')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

net_amount_main = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv',
                              index_col='Unnamed: 0')
net_pct_main = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_main.csv', index_col='Unnamed: 0')
net_amount_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_xl.csv',
                            index_col='Unnamed: 0')
net_pct_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_xl.csv', index_col='Unnamed: 0')
net_amount_l = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_l.csv', index_col='Unnamed: 0')
net_pct_l = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_l.csv', index_col='Unnamed: 0')
net_amount_m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_m.csv', index_col='Unnamed: 0')
net_pct_m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_m.csv', index_col='Unnamed: 0')
net_amount_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_s.csv', index_col='Unnamed: 0')
net_pct_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_s.csv', index_col='Unnamed: 0')

def quantile_factor_test_plot(factor, rts, benchmark_rts, quantiles, hold_time, plot_title, weight="avg", comm_fee=0.003):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # daily 收益率的df
    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    out.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    stock_number = factor.shape[1]

    plt.figure(figsize=(9, 9))
    plt.plot((1+benchmark_rts).cumprod().values, color='black', label='benchmark_net_value')

    for q in range(quantiles):

        start_i = q * int(stock_number / quantiles)
        if q == quantiles - 1:  # 最后一层
            end_i = stock_number
        else:
            end_i = (q + 1) * int(stock_number / quantiles)

        for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓

            date = factor.index[i]
            date_1 = factor.index[i + 1]

            temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
            if temp_ii == 0:  # 换仓日
                temp_factor = factor.loc[date].dropna().sort_values(ascending=False)  # 每天从大到小
                temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池

            temp_rts_daily = rts.loc[date_1][temp_stock_list]

            if weight == 'avg':  # 每天收益率均值
                out.loc[date_1] = temp_rts_daily.mean()

        out['daily_rts'][::hold_time] = out['daily_rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
        out['daily_rts'] = out['daily_rts'].fillna(0)
        out['net_value'] = (1 + out['daily_rts']).cumprod()
        plt.plot(out['net_value'].values, color=color_list[q], label='Quantile{}'.format(q))
    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title,  quantiles, hold_time))

    return out


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/quarterly_price_300.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = clean_close(close, low, high_limit)  # 新股一字板
close_rts = close.pct_change(1)

z = quantile_factor_test_plot(factor = net_pct_main, rts=close_rts, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=30, plot_title='Default', weight="avg",comm_fee=0.003)











