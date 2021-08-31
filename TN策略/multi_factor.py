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

amtEntropy = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/amtRatioEntropy.csv',index_col='Unnamed: 0')
momentum_5d = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/closevol_momentum_5d.csv',index_col='Unnamed: 0')

# rankinglist = pd.read_csv('./rankinglist.csv',index_col='Unnamed: 0')
# Turnrankinglist = pd.read_csv('./Turnrankinglist.csv',index_col='Unnamed: 0')
#
# extremevol_std = pd.read_csv('./extremevol_std.csv',index_col='Unnamed: 0')
# alpha_083 = pd.read_csv('./191/alpha_083.csv',index_col='Unnamed: 0')
# alpha_062 = pd.read_csv('./191/alpha_062.csv',index_col='Unnamed: 0')
# alpha_064 = pd.read_csv('./191/alpha_064.csv',index_col='Unnamed: 0')
# industry_reverse = pd.read_csv('./industry_reverse.csv',index_col='Unnamed: 0')
close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')
daily_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# factor_df_list = [smallplayer_abs_turnover, amtEntropy, rankinglist, Turnrankinglist, extremevol_std,
#                   alpha_083, alpha_062, alpha_064, industry_reverse]

factor_df_list = [amtEntropy]

multi_corr = []
for i in tqdm(range(len(factor_df_list))):
    f = factor_df_list[i]
    tmp = get_single_ic_table_open(factor=f, open_rts=open_rts)
    tmp.columns = ['factor{}'.format(i+1)]
    multi_corr.append(tmp)

multi_corr_df = pd.concat(multi_corr,axis=1, join='inner')
multi_corr_df_40 = multi_corr_df.rolling(20).mean()


#
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
    final_factor_list.append((tmp_factor.rank(axis=1).mul(multi_corr_df_40.iloc[:,i], axis=0)))

factor = reduce(lambda x, y: x + y, final_factor_list)


date_1 = factor.index
date_5 = factor.index[::5]
date_10 = factor.index[::10]

date_list = [date_1, date_5, date_10]
for i in range(len(date_list)):
    d = date_list[i]
    z1 = get_ic_table_open(factor=factor, open_rts=open_rts, buy_date_list=d)
    plt.plot(z1['ic'].values)
    plt.title('IC')
    plt.savefig('/Users/caichaohong/Desktop/{}.png'.format(i+1))
    plt.close()

    plt.plot(z1['ic'].cumsum().values)
    plt.title('IC_CUMSUM')
    plt.savefig('/Users/caichaohong/Desktop/{}_CUMSUM.png'.format(i + 1))
    plt.close()

    print ('IC={}, IC_STD={}'.format(z1['ic'].mean(), z1['ic'].std()))


z = quantile_factor_test_plot_open(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
                                   hold_time=5, plot_title=False, weight="avg", comm_fee=0.003)









