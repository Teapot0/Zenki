import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
from jqdatasdk import auth, get_query_count,get_industries,get_industry_stocks
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.update_data_funcs import *

auth('13382017213', 'Aasd120120')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')
low_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv',index_col='Unnamed: 0')
high_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv',index_col='Unnamed: 0')
highlimit_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv',index_col='Unnamed: 0')
vol_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv',index_col='Unnamed: 0')
close_daily = clean_close(close_daily,low_daily, highlimit_daily)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# 每天停牌的
pause_list = vol_daily.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')

# 去掉ST，停牌
for date in tqdm(close_daily.index):
    ex_list = list(set(pause_list.loc[date]).union(set(st_df.loc[date][st_df.loc[date]==True].index)))
    close_daily.loc[date][ex_list] = np.nan

daily_rts = close_daily.pct_change(1)

z_df = close_daily.isna().replace(True,0)
z_df = z_df.replace(False,1)

# 5 minute
close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv',index_col='Unnamed: 0')
vol_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv',index_col='Unnamed: 0')
close_5m_rts = close_5m.pct_change(1)
vol_5m_rts = vol_5m.pct_change(1)
vol_5m_mean = vol_5m.rolling(48).mean().iloc[47::48]
vol_5m_mean.index = [x.split(' ')[0] for x in vol_5m_mean.index]

rts_mean = close_5m_rts.rolling(48).mean().iloc[47::48]
rts_std = close_5m_rts.rolling(48).std().iloc[47::48]

tmp_rts = rts_mean.append([rts_mean]*47).sort_index(axis=0)
tmp_rts.index = close_5m_rts.index

tmp_std = rts_std.append([rts_std]*47).sort_index(axis=0)
tmp_std.index = close_5m_rts.index

df_high = pd.DataFrame(close_5m_rts.values > tmp_rts.values + tmp_std.values, index=tmp_rts.index, columns=tmp_rts.columns)
df_low = pd.DataFrame(close_5m_rts.values < tmp_rts.values - tmp_std.values, index=tmp_rts.index, columns=tmp_rts.columns)

high_vol = vol_5m[df_high]
low_vol = vol_5m[df_low]
high_vol.index = [x.split(' ')[0] for x in high_vol.index]
low_vol.index = [x.split(' ')[0] for x in low_vol.index]

tmp_high_vol = high_vol.groupby(high_vol.index).mean()
tmp_low_vol = low_vol.groupby(low_vol.index).mean()

factor = (tmp_low_vol - tmp_high_vol) / vol_5m_mean
factor = factor * z_df.loc[factor.index]

date_1 = factor.index
date_5 = factor.index[::5]
date_10 = factor.index[::10]

date_list = [date_1, date_5, date_10]
for i in range(len(date_list)):
    d = date_list[i]
    z1 = get_ic_table(factor=factor, rts=daily_rts, buy_date_list=d)
    plt.plot(z1['ic'].values)
    plt.title('IC')
    plt.savefig('/Users/caichaohong/Desktop/{}.png'.format(i+1))
    plt.close()

    plt.plot(z1['ic'].cumsum().values)
    plt.title('IC_CUMSUM')
    plt.savefig('/Users/caichaohong/Desktop/{}_CUMSUM.png'.format(i + 1))
    plt.close()

    print ('IC={}, IC_STD={}'.format(z1['ic'].mean(), z1['ic'].std()))


hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
z = quantile_factor_test_plot(factor=factor, rts=daily_rts, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=5, plot_title=False, weight="avg",comm_fee=0.003)


















