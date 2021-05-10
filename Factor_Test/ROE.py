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

auth('15951961478', '961478')
get_query_count()

quarter_roe = pd.read_excel('/Users/caichaohong/Desktop/Zenki/roe_quarters.xlsx',index_col='Unnamed: 0')

quarter_price = pd.read_excel('/Users/caichaohong/Desktop/Zenki/quarterly_price.xlsx', index_col='Unnamed: 0')
quarter_rts = quarter_price.pct_change(1)

quarter_roe.index =quarter_price.index[1:]

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
hs300.index = quarter_price.index

z = quantile_factor_test_plot(quarter_roe, quarter_rts, benchmark_rts=hs300['rts'].iloc[1:,],
                          quantiles=10, hold_time=1, plot_title='ROE因子分层回测')







