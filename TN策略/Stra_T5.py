import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks, get_bars, bond
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from basic_funcs.basic_function import *
import talib

auth('15951961478', '961478')
get_query_count()

import warnings
warnings.filterwarnings('ignore')


def get_short_ma_order(close, n1, n2, n3):
    ma1 = close.rolling(n1).mean()
    ma2 = close.rolling(n2).mean()
    ma3 = close.rolling(n3).mean()
    return (ma1 < ma2) & (ma2 < ma3) & (ma1 < ma3)


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['net_value'] = (1+hs300['rts_1']).cumprod()
# 择时
hs300['short_ma'] = get_short_ma_order(hs300['close'], n1=5,n2=90,n3=180)


close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1)  # 某列全NA
close_rts_1 = close.pct_change(1)

high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0',date_parser=dateparse)
high = high[close.columns]
low = low[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0',date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')# 2924个
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0',date_parser=dateparse) # 2924个
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv', index_col='statDate')# 2924个
market_cap = market_cap[close.columns]
pe = pe[close.columns]

#股息率
df_bank = finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.code=='801780'))
# 回购
df_bond = bond.run_query(query(bond.REPO_DAILY_PRICE).filter(bond.REPO_DAILY_PRICE.name=='GC182'))
df_t1 = pd.merge(df_bond, df_bank, on='date')
df_t1 = df_t1[['date','close','dividend_ratio']]
df_t1.index = pd.to_datetime(df_t1['date'])
# 当风险偏好<0不持股
df_t1['licha'] = (df_t1['close'].rolling(60).mean() - df_t1['dividend_ratio'].rolling(60).mean()).diff(1)
df_t1['licha'] = df_t1['licha'].fillna(method='ffill')
df_t1['hs300'] = hs300['net_value']

# 回购和价格行情日期不同
df_repo = pd.DataFrame(columns=['licha'],index = close.index)
datelist = list(set(df_t1.index).intersection(set(df_repo.index)))
df_repo['licha'].loc[datelist] = df_t1['licha'].loc[datelist]
df_repo = df_repo.fillna(method='ffill')

# 每天财务选股

# # 5年ROE
roe_5 = roe_yeayly.rolling(5,min_periods=1).mean()


# 每天财务选股

stock_list_panel = get_financial_stock_list(market_cap,roe_5, pe, money,
                                            roe_mean=12, mc_min=100, pe_min=20, money_min=0.1)


# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)

# 3个合并，每天的股票
stock_list = {}
for date in tqdm(close_rts_1.index):
    stock_list[date] = list(set(stock_list_panel).difference(set(pause_list.loc[date]),set(st_df.loc[date]), set(pause_list[date])))



def std_rts_select_dp_zs(close, hs300, std_n1=20, std_n2=90, std1=0.2, std2=0.3,
                         rts_n1=5, rts_n2=20, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                         rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                         weight_n1=60, weight_n2=90, weight_n3=180, weight_n4=250,
                         weight1=0.1, weight2=0.1, weight3=0.1, weight4=0.3,
                         top_number=10, comm_fee=0.002,max_down=0.1):
    max_N = max(std_n2, rts_n6, weight_n4)

    std_list = {}

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
        rts_list[date] = list(set(r1).intersection(r2, r3, r4, r5, r6))  # 每一天股票池

    while len(rts_list) != len(std_list):
        print('std list and rts list same length')
        break

    rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
    rts_f2 = close.pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
    rts_f3 = close.pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)
    rts_f4 = close.pct_change(weight_n4).sub(hs300['close'].pct_change(weight_n4), axis=0)
    weight = weight1 * rts_f1.rank(axis=1) + weight2 * rts_f2.rank(axis=1) + \
             weight3 * rts_f3.rank(axis=1) + weight4 * rts_f4.rank(axis=1)

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])

    buy_list = []
    initial_cost = [1]

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        tmp_week = date.week
        week1 = date1.week

        stocklist_financial = list(set(std_list[date]).intersection(rts_list[date], stock_list_panel[date]))
        stocklist_weighted = list(weight[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]

        if tmp_week != week1: # 每周一调仓
            if len(stocklist_weighted) <= 3:
                buy_list = []
            else :
                buy_list = stocklist_weighted
            initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -max_down].index)
        buy_list = list(set(buy_list).difference(set(sell_list)))  #

        if (hs300['short_ma'].loc[date]==True) & (df_repo['licha'].loc[date] < 0):
            buy_list = []
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if len(out_df['hold_daily'].loc[date]) > 0: #当天持仓，扣手续费
                out_df['daily_rts'].loc[date1] = -comm_fee
            else:
                out_df['daily_rts'].loc[date1] = 0
        else:
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if tmp_week!= week1: # 每周一调仓
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()-comm_fee
            else:
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = std_rts_select_dp_zs(close, hs300, std_n1=10, std_n2=60, std1=0.2, std2=0.2,
                                 rts_n1=10, rts_n2=40, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                                 rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                                 weight_n1=10, weight_n2=20, weight_n3=180, weight_n4=250,
                                 weight1=-2, weight2=0, weight3=2, weight4=4,
                                 top_number=10, comm_fee=0.003, max_down=0.1)

plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)



daily_rts.to_excel('/Users/caichaohong/Desktop/Zenki/daily_holdings_new.xlsx')
# 换名字
daily_rts = pd.read_excel('/Users/caichaohong/Desktop/Zenki/daily_holdings.xlsx')

for i in tqdm(range(1,daily_rts.shape[0])):
    if len(daily_rts['hold_daily'][i])==0:
        daily_rts['hold_daily'][i] = 'none'
    else:
        daily_rts['hold_daily'][i] = all_name['short_name'][daily_rts['hold_daily'][i]]
daily_rts.to_excel('/Users/caichaohong/Desktop/Zenki/daily_holdings.xlsx')


'''
daily_rts['len'] = [np.nan if x is np.nan else len(x) for x in daily_rts['hold_daily']]

rsrs_param = pd.DataFrame(index=['w1','w2','w3','w4','w5','w6','shape'])
ii=0
for w1 in [-2,0,2]:
    for w2 in [0,1]:
        for w3 in [2]:
            for w4 in [2,4]:
                for w5 in [-2,0,2]:
                    for w6 in [-2,0,2]:
                        tmp = std_rts_select_dp_zs(tmp_close, hs300, std_n1=10, std_n2=60, std1=0.2, std2=0.2,
                                         rts_n1=10, rts_n2=40, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                                         rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                                         weight_n1=10, weight_n2=90, weight_n3=180, weight_n4=250,
                                         weight1=w1, weight2=w2, weight3=w3, weight4=w4,weight5=0,weight6=0,
                                         top_number=10, hold_time=5, comm_fee=0.002, max_down=0.1)
                        annual_rts = tmp['net_value'].values[-1] ** (1 / (round(tmp.shape[0] / (244), 2))) - 1
                        sharpe = (annual_rts - 0.03) / (np.std(tmp['daily_rts']) * np.sqrt(244))
                        rsrs_param[ii] = [w1,w2,w3,w4,w5,w6,sharpe]
                        print ('current i = {}'.format(ii))
                        ii+=1
rsrs_param.to_excel('/Users/caichaohong/Desktop/Zenki/new_weights_params_rts_rank.xlsx')
'''


def get_latest_holds(date=close.index[-2],rts_n1=10, rts_n2=40, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                     rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                     weight_n1=10, weight_n2=20, weight_n3=180, weight_n4=250,
                     weight1=-2, weight2=0, weight3=2, weight4=4):
    std_l1 = close_rts_1.rolling(10).std() * sqrt(10)
    std_l2 = close_rts_1.rolling(60).std() * sqrt(60)
    tmp1 = list(std_l1.loc[date][std_l1.loc[date] < 0.2].index)
    tmp2 = list(std_l2.loc[date][std_l2.loc[date] < 0.2].index)
    std_list = list(set(tmp1).intersection(tmp2))  # 每一天股票池

    close_rts1 = close.pct_change(rts_n1).sub(hs300['close'].pct_change(rts_n1), axis=0)
    close_rts2 = close.pct_change(rts_n2).sub(hs300['close'].pct_change(rts_n2), axis=0)
    close_rts3 = close.pct_change(rts_n3).sub(hs300['close'].pct_change(rts_n3), axis=0)
    close_rts4 = close.pct_change(rts_n4).sub(hs300['close'].pct_change(rts_n4), axis=0)
    close_rts5 = close.pct_change(rts_n5).sub(hs300['close'].pct_change(rts_n5), axis=0)
    close_rts6 = close.pct_change(rts_n6).sub(hs300['close'].pct_change(rts_n6), axis=0)

    r1 = list(close_rts1.loc[date][close_rts1.loc[date] > rts1].index)
    r2 = list(close_rts2.loc[date][close_rts2.loc[date] > rts2].index)
    r3 = list(close_rts3.loc[date][close_rts3.loc[date] > rts3].index)
    r4 = list(close_rts4.loc[date][close_rts4.loc[date] > rts4].index)
    r5 = list(close_rts5.loc[date][close_rts5.loc[date] > rts5].index)
    r6 = list(close_rts6.loc[date][close_rts6.loc[date] > rts6].index)
    rts_list = list(set(r1).intersection(r2, r3, r4, r5, r6))  # 每一天股票池

    rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
    rts_f2 = close.pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
    rts_f3 = close.pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)
    rts_f4 = close.pct_change(weight_n4).sub(hs300['close'].pct_change(weight_n4), axis=0)
    weight = weight1 * rts_f1.rank(axis=1) + weight2 * rts_f2.rank(axis=1) + \
             weight3 * rts_f3.rank(axis=1) + weight4 * rts_f4.rank(axis=1)

    stocklist_financial = list(set(std_list).intersection(rts_list, stock_list_panel[date]))
    stocklist_weighted = list(weight[stocklist_financial].loc[date].sort_values(ascending=False).index)[:10]  # 买入的股票

    return stocklist_weighted




new_holds = get_latest_holds(date=close.index[-2],rts_n1=10, rts_n2=40, rts_n3=60, rts_n4=120, rts_n5=250, rts_n6=500,
                     rts1=-0.1, rts2=-0.1, rts3=-0.1, rts4=-0.12, rts5=-0.15, rts6=-0.3,
                     weight_n1=10, weight_n2=20, weight_n3=180, weight_n4=250,
                     weight1=-2, weight2=0, weight3=2, weight4=4)

new_holds = all_name['short_name'][new_holds]
# new_holds.to_excel('/Users/caichaohong/Desktop/Zenki/new_holdings.xlsx')

zz = pd.read_excel('/Users/caichaohong/Desktop/Zenki/new_holdings.xlsx')

week_rts = close_rts_1.iloc[-10:,][zz['code']]
week_rts.columns = zz['short_name']
week_rts.to_excel('/Users/caichaohong/Desktop/最新持仓周报.xlsx')




# 排名

test_rts = daily_rts.sort_values(by='daily_rts')







