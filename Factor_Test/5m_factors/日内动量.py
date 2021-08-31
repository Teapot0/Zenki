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


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]
hs300_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/510300_5m.csv',index_col='Unnamed: 0')
hs300_5m_rts = hs300_5m['close'].pct_change(1)

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv',index_col='Unnamed: 0')
open_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/open_5m.csv',index_col='Unnamed: 0')
high_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv',index_col='Unnamed: 0')
low_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv',index_col='Unnamed: 0')
volume_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv',index_col='Unnamed: 0')

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close_rts_1 = close_daily.pct_change(1)
close_daily = close_daily[close_daily.index >= '2018-01-01'][close_5m.columns]

friday_list = close_daily.index[close_daily.index.weekday == 4]

close_rts_5 = close_daily.loc[friday_list].pct_change(1)

rts_5m = close_5m.pct_change(1).sub(hs300_5m_rts,axis=0)
hs300_mul = (hs300_5m_rts > 0.001)*1
rts_5m = rts_5m.mul(hs300_mul,axis=0)
# rts_5m = rts_5m.apply(lambda x: (x-x.mean())/x.std(),axis=1)

rts_5m_daily_index = rts_5m.copy(deep=True)
rts_5m_daily_index.index = [x.split(' ')[0] for x in rts_5m_daily_index.index]
rts_5m_daily_index.index = pd.to_datetime(rts_5m_daily_index.index)
# 换成每天
rts_5m_daily = rts_5m_daily_index.groupby(rts_5m_daily_index.index).sum()


rts_dict = {}
for i in tqdm(range(48)):
    tmp = rts_5m.iloc[i::48].rolling(5).sum()
    tmp.index = [x.split(' ')[0] for x in tmp.index]
    tmp.index = pd.to_datetime(tmp.index)
    rts_dict[i] = tmp.loc[friday_list]


# 上期因子值和本期收益的corr
ic_table = pd.DataFrame(index=friday_list, columns=list(range(48)))
rankic_table = pd.DataFrame(index=friday_list, columns=list(range(48)))
for minute_i in tqdm(range(48)):
    factor = rts_5m.iloc[minute_i::48,]
    factor.index = [x.split(' ')[0] for x in factor.index]
    factor.index = pd.to_datetime(factor.index)
    for i in range(1,len(friday_list)):
        date = friday_list[i]
        date1 = friday_list[i - 1]
        ic_table[minute_i].loc[date] = np.corrcoef(factor.loc[date1], close_rts_5.loc[date])[0, 1]
        rankic_table[minute_i].loc[date] = np.corrcoef(factor.loc[date1].rank(), close_rts_5.loc[date].rank())[0, 1]


rankic_diff_table = rankic_table.diff(1)
rankic_diff_rank = rankic_diff_table.rank(axis=1)
rankic_weight = rankic_diff_rank.apply(lambda x: (x >= 44)*1, axis=1)
rankic_weight = rankic_weight.loc[friday_list]


rankic_factor = 0
for i in tqdm(range(48)):
    rankic_factor += rts_dict[i].mul(rankic_weight[i],axis=0)


ic = get_ic_table(factor=rts_5m_daily, rts=close_rts_5, buy_date_list=friday_list)
plt.plot(ic['rank_ic'])


z = weekly_quantile_factor_test_plot(factor=rts_5m_daily, rts=close_rts_1, benchmark_rts=hs300['rts'], quantiles=10,
                             buy_date_list=friday_list, plot_title=False, weight="avg",comm_fee=0.003)


z = quantile_factor_test_plot(factor = rts_5m_daily, rts=close_rts_1, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=5, plot_title=False, weight="avg",comm_fee=0.003)



z0 = rts_dict[0]

