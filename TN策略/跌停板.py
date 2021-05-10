import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqfactor_analyzer import analyze_factor
from alphalens.utils import get_clean_factor_and_forward_returns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from basic_funcs.basic_function import *

import warnings

warnings.filterwarnings('ignore')

open = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/open.xlsx', index_col='Unnamed: 0')
close = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/close.xlsx', index_col='Unnamed: 0')
high = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/high.xlsx', index_col='Unnamed: 0')
low = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/low.xlsx', index_col='Unnamed: 0')
high_limit = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.xlsx', index_col='Unnamed: 0')
low_limit = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.xlsx', index_col='Unnamed: 0')
vol = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/daily/volume.xlsx', index_col='Unnamed: 0')
price_rts = close.pct_change(1)

# 一字涨停价格的处理，只去掉新股上市后的一字板
all_new_stock = []
for i in tqdm(range(1, close.shape[0])):
    tmp_close = close.iloc[i,]
    yesterday_close = close.iloc[i-1,]
    new_stock = list(set(tmp_close.dropna().index).difference(set(yesterday_close.dropna().index)))  # 每天新股名单
    for ss in new_stock:
        if close.iloc[i,][ss] == high_limit.iloc[i, ][ss]:
            close.iloc[i,][ss] = np.nan  # 第一天上市等于涨停价则去掉
    # 加上第一天上市涨停的
    all_new_stock = list(set(all_new_stock).union(set(close.iloc[i,][new_stock][close.iloc[i,][new_stock] == high_limit.iloc[i,][new_stock]].index)))
    # 去掉开板的
    new_stock_kai = list(low.iloc[i, ][all_new_stock][low.iloc[i, ][all_new_stock] != high_limit.iloc[i, ][all_new_stock]].index)
    # 所有未开板新股
    new_stock_not_kai = list(set(all_new_stock).difference(set(new_stock_kai)))
    # 未开板新股去掉
    close.iloc[i, ][new_stock_not_kai] = np.nan

# weekly close
close_week = resample_data_weekly(close, df_type='close')
price_rts_week = close_week.pct_change(1)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)


# 转化为因子


dt_fb = pd.DataFrame(0, index=close.index, columns =close.columns)
for i in range(1, close.shape[0]-1):
    today = close.index[i]
    next_day = close.index[i+1]
    tmp_dt_list = list(close.loc[today][close.loc[today] == low_limit.loc[today]].index)
    next_day_rts = price_rts.loc[next_day][tmp_dt_list]
    next_day_buy_list = list(next_day_rts[next_day_rts >= 0.05].index)
    dt_fb.loc[next_day][next_day_buy_list] = 1


value_rts = get_top_signal_factor_rts(factor=dt_fb, rts=price_rts, hold_time=1)

comm_fee = 0.003
hold_time = 5
out_df = pd.DataFrame(index=close.index)
# out_df['holds'] = holds['holdings']
out_df['rts'] = value_rts
out_df['rts'][::hold_time] = out_df['rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
# out_df['rts'].fillna(0)  # 未交易日，收益率为0，也不扣手续费
out_df['net_value'] = (1 + out_df['rts']).cumprod()

na_num = out_df['net_value'].isna().sum()
position_max_draw = list(np.zeros(na_num))
out_df['nv_max_draw'] = position_max_draw + list(MaxDrawdown(list(out_df['net_value'].dropna())).reshape(-1))
out_df['benchmark_rts'] = hs300['close'].pct_change(1)
out_df['benchmark_net_value'] = hs300['close'] / (hs300['close'][0])

plot_hold_position(data=out_df, risk_free_rate=0.03)





# AI打板





