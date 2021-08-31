import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss,get_money_flow
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv',index_col='Unnamed: 0')
volume_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv',index_col='Unnamed: 0')

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close_daily = close_daily[close_daily.index >= '2018-01-01'][close_5m.columns]
close_daily_rts_1 = close_daily.pct_change(1)
close_daily_rts_5 = close_daily.pct_change(5)

friday_list = close_daily.index[close_daily.index.weekday == 4]

rts_5m = close_5m.pct_change(1)

corr_df = rts_5m.rolling(48*5).corr(volume_5m)

corr_df_mean = corr_df.rolling(240).mean()
corr_df_std = corr_df.rolling(240).std()

corr_mean_daily = corr_df_mean.iloc[47::48]
corr_mean_daily.index = [x.split(' ')[0] for x in corr_mean_daily.index]
corr_std_daily = corr_df_std.iloc[47::48]
corr_std_daily.index = [x.split(' ')[0] for x in corr_std_daily.index]

corr_mean_1 = corr_mean_daily.apply(lambda x: (x-x.mean())/x.std(), axis=1)
corr_std_1 = corr_std_daily.apply(lambda x: (x-x.mean())/x.std(), axis=1)
pv_corr = corr_mean_1 + corr_std_1
pv_corr.index = pd.to_datetime(pv_corr.index)



def get_ic_table(factor, rts, buy_date_list):
    # buy list 是换仓日
    out = pd.DataFrame(index=buy_date_list,columns=['ic', 'rank_ic'])
    for i in tqdm(range(1,len(buy_date_list))):
        date = buy_date_list[i]
        date1 = buy_date_list[i-1]
        tmp = pd.concat([factor.loc[date1],rts.loc[date]],axis=1)
        tmp.columns=['date1', 'date']
        out['ic'].loc[date] = tmp.corr().iloc[0,1]
        out['rank_ic'].loc[date] = tmp.rank().corr().iloc[0,1]
    return out

ic = get_ic_table(factor=pv_corr, rts = close_daily_rts_5, buy_date_list=friday_list)

plt.plot(ic['rank_ic'])

z = quantile_factor_test_plot(factor=pv_corr, rts=close_daily_rts_1, benchmark_rts=hs300['rts'], quantiles=10, hold_time=5, plot_title=False, weight="avg",
                              comm_fee=0.003)









def get_weekly_rts(close):
    for i in range(close.shape[0]):

