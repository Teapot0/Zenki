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

auth('13382017213', 'Aasd120120')
get_query_count()

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx', index_col='Unnamed: 0')
all_name.index = all_name['code']

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['rts_5'] = hs300['close'].pct_change(5)
hs300['net_value'] = (1 + hs300['rts_1']).cumprod()
hs300['short_ma'] = get_short_ma_order(hs300['close'], n1=5, n2=90, n3=180)
# rps

hs300['rps_close'] = get_rps(hs300['close'], rps_n=122)
hs300['rps_vol'] = get_rps(hs300['volume'].rolling(5).sum(), rps_n=61)
hs300['rps_5'] = hs300['rps_close'].rolling(5).mean()
hs300['rps_vol_1'] = hs300['rps_vol']*(hs300['rps_5']>=90)*1
hs300['rps_5'] = hs300['rps_5'].mask(hs300['rps_5']>=90, hs300['rps_5']*0.01*hs300['rps_vol_1'])
hs300['rps_5_rts'] = hs300['rps_5'].pct_change(5)
hs300['rps_diff'] = hs300['rps_5'] - hs300['rps_close']

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1)  # 某列全NA
close_rts_1 = close.pct_change(1)
close_rts_5 = close.pct_change(5)
# 周收益
week_rts = close.resample('W-FRI',how='last').pct_change(1)
rps_df = get_rps(close,rps_n=61)

high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)
high = high[close.columns]
low = low[close.columns]
volume = volume[close.columns]
money = close * volume * 10 ** (-8)
vol_rps = get_rps(volume,rps_n=61)

market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')  # 2924个
pe = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)  # 2924个
net_profit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/net_profit_yearly.csv', index_col='statDate')  # 2924个
market_cap = market_cap[close.columns]
pe = pe[close.columns]

net_pct_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_xl.csv', index_col='Unnamed: 0')
net_pct_xl.index = pd.to_datetime(net_pct_xl.index)
net_xl_rts = net_pct_xl.rolling(5).mean().pct_change(5)

# 股息率
df_t1 = get_licha()

# 当风险偏好<0不持股
df_t1['licha'] = (df_t1['close'].rolling(60).mean() - df_t1['dividend_ratio'].rolling(60).mean()).diff(1)
df_t1['licha'] = df_t1['licha'].fillna(method='ffill')
df_t1['hs300'] = hs300['net_value']

# 回购和价格行情日期不同
df_repo = pd.DataFrame(columns=['licha'], index=close.index)
datelist = list(set(df_t1.index).intersection(set(df_repo.index)))
df_repo['licha'].loc[datelist] = df_t1['licha'].loc[datelist]
df_repo = df_repo.fillna(method='ffill')

# 每天财务选股

# # 5年ROE
roe_5 = roe_yeayly.rolling(5, min_periods=1).mean()
# 每天财务、均线选股
stock_list_panel = get_financial_stock_list(market_cap,roe_5, pe, money,
                                            roe_mean=12, mc_min=100, pe_min=20, money_min=0.2)
panel_list = [stock_list_panel[d] for d in close_rts_1.index]
all_stock = set.union(*map(set,panel_list))


stock_list_panel2 = get_financial_stock_list(market_cap,roe_5, pe, money,
                                            roe_mean=3, mc_min=200, pe_min=30, money_min=1)
panel_list2 = [stock_list_panel2[d] for d in close_rts_1.index]
all_stock2 = set.union(*map(set,panel_list2))

# 波动率
std_list = get_std_list(close_rts_1, std_n_list=[10,60],std_list=[0.2,0.2])
# 超额收益
rts_list = get_alpha_list(close,hs300,rts_n_list=[10,40,60,120,250,500], rts_list=[-0.1,-0.1,-0.1,-0.12, -0.15, -0.3])

# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)


# 异常下跌
down_list = get_down_list(close,hs300)

# 3个合并，每天的股票
stock_list = {}
for date in tqdm(close_rts_1.index):
    tmp_list = list(set(rts_list[date]).intersection(std_list[date],stock_list_panel[date]))
    stock_list[date] = list(set(tmp_list).difference(set(pause_list.loc[date]),set(st_df.loc[date]), set(down_list[date])))


# factors
weight_n1 = 5
weight_n2 = 40
weight_n3 = 120
rts_f1 = close.pct_change(weight_n1).sub(hs300['close'].pct_change(weight_n1), axis=0)
rts_f2 = close.pct_change(weight_n2).sub(hs300['close'].pct_change(weight_n2), axis=0)
rts_f3 = close.pct_change(weight_n3).sub(hs300['close'].pct_change(weight_n3), axis=0)

ex_rts1 = close_rts_1.sub(hs300['rts_1'],axis=0)
ex_rts5 = ex_rts1.rolling(5).sum()
up_list = list(hs300[hs300['rts_1'] > 0].index)
ex_rts1_down = ex_rts1.copy(deep=True)
ex_rts1_down.loc[up_list] = 0
ex_rts1_down40 = ex_rts1_down.rolling(40).sum()


rts_f1_rank = pd.DataFrame(columns=all_stock, index=close_rts_1.index)
for date in tqdm(rts_f1_rank.index):
    tmp_list = stock_list[date]
    rts_f1_rank.loc[date][tmp_list] = rts_f1.loc[date][tmp_list].rank()

f1_factor = 244
f1 = (close_rts_1.rolling(f1_factor).mean() / close_rts_1.rolling(f1_factor).std()) * sqrt(f1_factor)
sharpe_df = f1.rolling(40).mean()
sharpe_df_rts = f1.pct_change(10)

vol_n1 = 10
vol_f1 = volume.rolling(vol_n1).mean().pct_change(vol_n1)

rr = rps_df * vol_rps * 0.01
factor_new = (1/sharpe_df) * rts_f1_rank * rr

# weights
# weight1 = 2 * pd.Series([np.sign(x) for x in bs['diff5']], index=rts_f1.index)
weight1 = -4
weight2 = 2
weight3 = 4
weight4 = 0
weight5 = 0

weight = pd.DataFrame(columns=all_stock, index=close_rts_1.index)
for date in tqdm(weight.index):
    tmp_list = stock_list[date]
    weight.loc[date][tmp_list] = weight1 * rts_f1.loc[date][tmp_list].rank() + weight2 * rts_f2.loc[date][tmp_list].rank() + \
                                          weight3 * rts_f3.loc[date][tmp_list].rank()



rts_fnew = close.pct_change(20).sub(hs300['close'].pct_change(20), axis=0)
weight_new = pd.DataFrame(columns=all_stock, index=close_rts_1.index)
for date in tqdm(weight.index):
    tmp_list = stock_list[date]
    weight_new.loc[date][tmp_list] = ( 8 * rts_f2.loc[date][tmp_list] + -4 * rts_fnew.loc[date][tmp_list])*vol_rps.loc[date][tmp_list]



def std_rts_select_dp_zs(close, hs300,
                         top_number=10, comm_fee=0.002, max_down=0.1):
    max_N = 500

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])

    buy_list = []
    initial_cost = [1]

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        tmp_week = date.week
        week1 = date1.week

        stocklist_financial = stock_list[date]
        if tmp_week != week1:  # 每周五
            if hs300['rps_diff'].loc[date] < -80:
                buy_list = []
            else:
                if hs300['rps_5'].loc[date] >= 80:
                    stocks = list(weight[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
                    buy_list = stocks
                else:  # rps_5小于80
                    if hs300['rps_5'].loc[date] > 20:
                        if hs300['rps_5'].loc[date] > 50:
                            stocks = list(net_xl_rts[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
                            buy_list = stocks
                        else:  # <50
                            # stocks = list(ex_rts1_down40[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
                            buy_list = []
                    else:
                        stocks = list(ex_rts1_down40[stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
                        buy_list = stocks
                initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -max_down].index)
        buy_list = list(set(buy_list).difference(set(sell_list)))  #

        if (hs300['short_ma'].loc[date] == True) & (df_repo['licha'].loc[date] < 0):
            buy_list = []
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if len(out_df['hold_daily'].loc[date]) > 0:  # 当天持仓，扣手续费
                out_df['daily_rts'].loc[date1] = -comm_fee
            else:
                out_df['daily_rts'].loc[date1] = 0
        else:
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if tmp_week != week1:  # 每周一调仓
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean() - comm_fee
            else:
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = std_rts_select_dp_zs(close, hs300, top_number=10, comm_fee=0.003, max_down=0.1)
daily_rts['alpha'] = (daily_rts['daily_rts'] - hs300['rts_1'])*100
plot_rts(value_rts=daily_rts['daily_rts'], benchmark_df=hs300, comm_fee=0.0, hold_time=5)




def test_rts(close, hs300,top_number=10, comm_fee=0.002, max_down=0.1):
    max_N = 500

    out_df = pd.DataFrame(columns=['daily_rts', 'hold_daily', 'net_value'], index=close_rts_1.index[max_N + 1:])

    buy_list = []
    initial_cost = [1]

    for i in tqdm(range(max_N + 1, close_rts_1.shape[0] - 1)):  # 去掉最后一天
        date = close_rts_1.index[i]
        date1 = close_rts_1.index[i + 1]
        tmp_week = date.week
        week1 = date1.week

        stocklist_financial = stock_list[date]

        if tmp_week != week1:  # 每周五
            if hs300['rps_diff'].loc[date] < -80:
                buy_list = []
            else:
                if (hs300['rps_5'].loc[date] > 80) & (hs300['rps_5'].loc[date] < 80) & (hs300['rps_5_rts'].loc[date] > 0):
                    stocks = list([stocklist_financial].loc[date].sort_values(ascending=False).index)[:top_number]
                    buy_list = stocks
                else:
                    buy_list = []
                if len(buy_list) <= 3:
                    buy_list = []
                initial_cost = close.loc[date][buy_list]  # 成本

        acc_rts = close.loc[date][buy_list] / initial_cost - 1  # 累计收益小于5% 则卖出
        sell_list = list(acc_rts[acc_rts < -max_down].index)
        buy_list = list(set(buy_list).difference(set(sell_list)))  #

        if (hs300['short_ma'].loc[date] == True) & (df_repo['licha'].loc[date] < 0):
            buy_list = []
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if len(out_df['hold_daily'].loc[date]) > 0:  # 当天持仓，扣手续费
                out_df['daily_rts'].loc[date1] = -comm_fee
            else:
                out_df['daily_rts'].loc[date1] = 0
        else:
            out_df['hold_daily'].loc[date1] = list(all_name['short_name'][buy_list])

            if tmp_week != week1:  # 每周一调仓
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean() - comm_fee
            else:
                out_df['daily_rts'].loc[date1] = close_rts_1.loc[date1][buy_list].mean()

    out_df['net_value'] = (1 + out_df['daily_rts']).cumprod()
    return out_df


daily_rts = test_rts(close, hs300, top_number=10, comm_fee=0.003, max_down=0.1)
plt.plot((1+daily_rts['daily_rts'].dropna()).cumprod())


vol5 = get_rps(volume, rps_n=21)

vol55 = vol5.rolling(5).mean().pct_change(5)

i = -1
date = week_rts.index[i]
date_yes = week_rts.index[i-1]
z = pd.DataFrame()
z['week_rts'] = week_rts[stock_list_panel2[date_yes]].loc[date]
z['rts_1'] = rts_f1.loc[date_yes][stock_list_panel2[date_yes]]
z['rts_2'] = rts_f2.loc[date_yes][stock_list_panel2[date_yes]]
z['sharpe'] = sharpe_df.loc[date_yes][stock_list_panel2[date_yes]]
z['vol_rps'] = vol55.loc[date_yes][stock_list_panel2[date_yes]]

z = z.sort_values(by='week_rts')


zz = z.dropna()



zzz = hs300['rps_5'] - hs300['rps_close']



