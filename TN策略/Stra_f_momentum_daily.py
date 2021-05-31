import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks, get_bars,indicator,income
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler

auth('15951961478', '961478')
get_query_count()

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0',date_parser=dateparse)
close_rts_1 = close.pct_change(1)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0',date_parser=dateparse)
money = close * volume * 10 ** (-8)
market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0',date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv', index_col='statDate')
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)


# # 5年ROE
share_nv = net_profit/(roe_yeayly*0.01)
roe_5 = net_profit.rolling(5).sum()/share_nv.rolling(5).sum() * 100

# 每天财务选股
stock_list_panel = {}
all_stock = set()
for i in range(market_cap.shape[0]):
    date = market_cap.index[i]
    # 5年平均roe大于12%
    tmp_year = '{}-12-31'.format(market_cap.index[i].year-1)
    roe_list = list((roe_5.loc[tmp_year][roe_5.loc[tmp_year] > 12]).index)
    # 市值大于100 pe大于25
    mc_100 = list(market_cap.iloc[i, :][market_cap.iloc[i, :] > 100].index)
    pe_25 = list(pe.iloc[i, :][pe.iloc[i, :] > 25].index)
    # 成交额大于1000万
    money_list = list(money.iloc[i, :][money.iloc[i, :] > 0.1].index)
    tmp_list = list(set(roe_list).intersection(set(mc_100), set(pe_25), set(money_list)))
    stock_list_panel[date] = tmp_list
    all_stock = all_stock.union(stock_list_panel[date])

tmp_close = close[all_stock]

def std_rts_select(close,hs300,std_n1=20, std_n2=90, std1=0.2, std2=0.3,
                   rts_n1=5, rts_n2=20, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                   rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                   weight_n1=60, weight_n2=90, weight_n3=180, weight_n4=250,
                   weight1=0.1, weight2=0.2, weight3=0.3, weight4=0.4, top_number=10, hold_time=5):
    hs300['rts_1'] = hs300['close'].pct_change(1)
    max_N = max(std_n2, rts_n6, weight_n4)

    std_list = {}
    close_rts_1 = close.pct_change(1)
    std_l1 = close_rts_1.rolling(std_n1).std() * sqrt(std_n1)
    std_l2 = close_rts_1.rolling(std_n2).std() * sqrt(std_n2)
    for date in close_rts_1.index:
        tmp1 = list(std_l1.loc[date][std_l1.loc[date] < std1].index)
        tmp2 = list(std_l2.loc[date][std_l2.loc[date] < std2].index)
        std_list[date] = list(set(tmp1).intersection(tmp2))  # 每一天股票池

    rts_list = {}
    close_rts1 = close.pct_change(rts_n1).sub(hs300['close'].pct_change(rts_n1), axis=0)
    close_rts2 = close.pct_change(rts_n2).sub(hs300['close'].pct_change(rts_n2), axis=0)
    close_rts3 = close.pct_change(rts_n3).sub(hs300['close'].pct_change(rts_n3), axis=0)
    close_rts4 = close.pct_change(rts_n4).sub(hs300['close'].pct_change(rts_n4), axis=0)
    close_rts5 = close.pct_change(rts_n5).sub(hs300['close'].pct_change(rts_n5), axis=0)
    close_rts6 = close.pct_change(rts_n6).sub(hs300['close'].pct_change(rts_n6), axis=0)

    for date in close_rts_1.index:  # 去掉NA
        r1 = list(close_rts1.loc[date][close_rts1.loc[date] > rts1].index)
        r2 = list(close_rts2.loc[date][close_rts2.loc[date] > rts2].index)
        r3 = list(close_rts3.loc[date][close_rts3.loc[date] > rts3].index)
        r4 = list(close_rts4.loc[date][close_rts4.loc[date] > rts4].index)
        r5 = list(close_rts5.loc[date][close_rts5.loc[date] > rts5].index)
        r6 = list(close_rts6.loc[date][close_rts6.loc[date] > rts6].index)
        rts_list[date] = list(set(r1).intersection(r2, r3, r4,r5,r6))  # 每一天股票池

    while len(rts_list) != len(std_list):
        print('std list and rts list same length')
        break

    rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
    rts_f2 = close.pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
    rts_f3 = close.pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)
    rts_f4 = close.pct_change(weight_n4).sub(hs300['close'].pct_change(weight_n4), axis=0)
    weight = weight1 * rts_f1 + weight2 * rts_f2 + weight3 * rts_f3 + weight4 * rts_f4

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N+1:])

    for i in tqdm(range(max_N+1,close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        if (i-max_N-1) % hold_time == 0:
            stocklist_financial = list(set(std_list[date]).intersection(rts_list[date], stock_list_panel[date]))
            stocklist_weighted = list(weight[stocklist_financial].loc[date].sort_values(ascending=False).index)  # 买入的股票
            if (top_number == 'full') | (len(stocklist_weighted) <= top_number):
                buy_list = stocklist_weighted
            else:
                buy_list = stocklist_weighted[:top_number]
        out_df['hold_daily'].loc[date1] = buy_list
        out_df['daily_rts'].loc[date1] = close_rts_1[buy_list].loc[date1].mean()
    out_df['net_value'] = (1+out_df['daily_rts']).cumprod()
    return out_df



daily_rts = std_rts_select(tmp_close, hs300=hs300,std_n1=10, std_n2=60, std1=0.2, std2=0.2,
                           rts_n1=10, rts_n2=40, rts_n3=60, rts_n4=120,rts_n5=250, rts_n6=500,
                           rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12,rts5=-0.15, rts6=-0.3,
                           weight_n1=10, weight_n2=90, weight_n3=180, weight_n4=500,
                           weight1=0.1, weight2=0.2, weight3=0.3, weight4=0.4, top_number=10, hold_time=5)

plot_rts(value_rts=daily_rts['daily_rts'],benchmark_df=hs300, comm_fee=0.002, hold_time=5)

# 一共3888个
stdn1_list = [10]
stdn2_list = [40]
std1_list = [0.2]
std2_list = [0.2]
rts_n1_list = [10]
rts_n2_list = [40]
rts_n3_list = [60,90]
weight_n1_list = [10,20,30]
weight_n2_list = [40,60,90]
weight_n3_list = [120,180]
weight_n4_list = [500]

params_df = pd.DataFrame(index=['std_n1', 'std_n2','std1', 'std2',
                                'rts_n1', 'rts_n2', 'rts_n3',
                                'weight_n1', 'weight_n2', 'weight_n3', 'weight_n4','annual_rts', 'sharpe'])

ii = 0
for stdn1 in stdn1_list:
    for stdn2 in stdn2_list:
        for std1 in std1_list:
            for std2 in std2_list:
                for rtsn1 in rts_n1_list:
                    for rtsn2 in rts_n2_list:
                        for rtsn3 in rts_n3_list:
                            for w1 in weight_n1_list:
                                for w2 in [x for x in weight_n2_list if x >w1]:
                                    for w3 in [x for x in weight_n3_list if x >w2]:
                                        for w4 in [x for x in weight_n4_list if x >w3]:
                                            rts = std_rts_select(close,hs300,std_n1=stdn1, std_n2=stdn2, std1=std1, std2=std2,
                                                                 rts_n1=rtsn1, rts_n2=rtsn2, rts_n3=rtsn3, rts_n4=120, rts_n5=250, rts_n6=500,
                                                                 rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                                                                 weight_n1=w1, weight_n2=w2, weight_n3=w3, weight_n4=w4,
                                                                 weight1=0.1, weight2=0.2, weight3=0.3, weight4=0.4, top_number=10, hold_time=5)
                                            rts = rts['daily_rts'].dropna()
                                            rts[::5] = rts[::5] - 0.002
                                            n_years = len(rts) / 244
                                            rts_annual = ((1+rts).cumprod()[-1]) ** (1/n_years) - 1
                                            rts_sharpe = (rts_annual-0.03)/(np.std(rts)*sqrt(244))

                                            params_df[ii] = [stdn1, stdn2, std1, std2,rtsn1,rtsn2, rtsn3, w1,w2,w3,w4,rts_annual, rts_sharpe]
                                            ii += 1
                                            print(ii)

params_df.to_excel('/Users/caichaohong/Desktop/Zenki/stra_f_momentum2.xlsx')



