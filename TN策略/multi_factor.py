import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score
from functools import reduce
import seaborn as sns
from basic_funcs.basic_funcs_open import *

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

min_exrts_volstd = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_exrts_volstd.csv', index_col='Unnamed: 0').dropna(how='all')
min_neg_rts_volstd = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/1min_neg_rts_volstd.csv', index_col='Unnamed: 0').dropna(how='all')
smart_money = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/smart_money_vwap_b0.1_low.csv', index_col='Unnamed: 0').dropna(how='all')

# shape
min_neg_rts_volstd = min_neg_rts_volstd.loc[min_exrts_volstd.index]

amtEntropy = read_csv_select(path='/Users/caichaohong/Desktop/Zenki/factors/amtRatioEntropy.csv',
                             start_time='2021-01-15', end_time='2021-08-31', stock_list=list(min_neg_rts_volstd.columns))

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')
daily_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]


zz500 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510500.XSHG.xlsx', index_col='Unnamed: 0')
zz500.index = [x.strftime('%Y-%m-%d') for x in zz500.index]
zz500['rts'] = zz500['close'].pct_change(1)

zz1000 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/512100.XSHG.xlsx', index_col='Unnamed: 0')
zz1000.index = [x.strftime('%Y-%m-%d') for x in zz1000.index]
zz1000['rts'] = zz1000['close'].pct_change(1)


factor_df_list = [amtEntropy, min_exrts_volstd, min_neg_rts_volstd, smart_money]

multi_corr = []
for i in tqdm(range(len(factor_df_list))):
    f = factor_df_list[i]
    tmp = get_single_ic_table_open(factor=f, open_rts=open_rts)
    tmp.columns = ['factor{}'.format(i+1)]
    multi_corr.append(tmp)

multi_corr_df = pd.concat(multi_corr,axis=1, join='inner')


# def get_multifactor_ic_table(daily_rts, factor_df_list):
#     # same_index = set.intersection(set(x.index) for x in factor_df_list)
#
#     col_name = ['factor_{}'.format(i) for i in range(1,len(factor_df_list)+1)]
#     ic_table = pd.DataFrame(index=daily_rts.index, columns=col_name)
#
#     for i in tqdm(range(len(col_name))):
#         col = col_name[i]
#         factor = factor_df_list[i]
#
#         tmp_index = set.intersection(set(factor.index), set(daily_rts.index))
#         tmp_cols = set.intersection(set(factor.columns), set(daily_rts.columns))
#
#         tmp_factor = factor.loc[tmp_index][tmp_cols]
#         tmp_rts = daily_rts.loc[tmp_index][tmp_cols]
#
#         tmp_corr = get_single_ic_table(factor=tmp_factor, rts=tmp_rts)
#         ic_table[col].loc[tmp_corr.index] = tmp_corr['ic']
#     return ic_table
#
#
# multi_corr = get_multifactor_ic_table(daily_rts=daily_rts, factor_df_list=factor_df_list,)
# # multi_corr_20 = multi_corr.rolling(20).mean()


final_factor_list = []
for i in tqdm(range(len(factor_df_list))):
    tmp_factor = factor_df_list[i].loc[multi_corr_df.index]
    final_factor_list.append((tmp_factor.rank(axis=1).mul(multi_corr_df.iloc[:,i], axis=0)))

factor = reduce(lambda x, y: x + y, final_factor_list)


ic_test(index_pool='hs300', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz500', factor=factor, open_rts=open_rts)
ic_test(index_pool='zz1000', factor=factor, open_rts=open_rts)


z = quantile_factor_test_plot_open_index(factor=-factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
                                   hold_time=10,index_pool='zz500', plot_title=False, weight="avg", comm_fee=0.003)



plt.plot((z[0] - zz500['rts']).cumsum())


