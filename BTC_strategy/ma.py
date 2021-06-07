import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import talib
from basic_funcs.CTA_basics import *
from basic_funcs.basic_function import *

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

price_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/BTC-USD/price_df.csv',index_col='id')
price_df['close_nv'] = price_df['close']/price_df['close'][0]
price_df['rts_1'] = price_df['close'].pct_change(1)

fake_bench = pd.DataFrame(index=price_df.index)
fake_bench['close'] = 1

# 均线
price_df['close_ma1'] = price_df['close'].rolling(2).mean()
price_df['close_ma2'] = price_df['close'].rolling(5).mean()
price_df['close_ma3'] = price_df['close'].rolling(10).mean()

# 放量上涨
# price_df['rts_1'][price_df['rts_1']>=0.001]
# 大于千1的有5.5万分钟，千2的 有1.8万分钟

price_df['vol_ma1'] = price_df['vol'].rolling(120).mean()
price_df['vol_ex'] = price_df['vol'] / price_df['vol_ma1'] - 1

#放量超过1倍的有6.17万个
# price_df['vol_ex'][price_df['vol_ex']>=1]

price_df['long_ma'] = ((price_df['close_ma1'] > price_df['close_ma2']) & (price_df['close_ma1'] > price_df['close_ma3']) & (price_df['close_ma2'] > price_df['close_ma3']))* 1
price_df['vol_ex_1'] = (price_df['vol_ex'] > 1) * 1
price_df['rts_big'] = (price_df['rts_1'] >= 0.002)*1

price_df['buy_1'] = ((price_df['rts_1'] > 0.005) & (price_df['vol_ex_1'] == 1))*1
price_df['hold_1'] = price_df['buy_1'].shift(1)
price_df['comm_fee'] = (price_df['hold_1'].diff(1) == 1) * 1 * 0.002
price_df['hold_rts'] = price_df['rts_1'] * price_df['hold1'] - price_df['comm_fee']
price_df['net_value'] = (1+price_df['hold_rts']).cumprod()
price_df['nv_max_draw'] = [np.nan] + list(MaxDrawdown(list(price_df['net_value'].dropna())).reshape(-1))

ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(price_df['net_value'].values, 'black')
ax2.plot(price_df['nv_max_draw'].values, 'red', linestyle='-.',linewidth=1, label='port_max_draw')


def plot_rts(df):
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ax1.plot(df['net_value'].values, 'black', label='port_net_value')
    # ax1.plot(df['benchmark_net_value'], 'blue', label='benchmark_net_value')
    # ax1.plot((1 + df['rts'] - df['benchmark_rts']).cumprod(), 'gold', label='cumulative alpha')  # 画超额收益 Alpha
    ax2.plot(df['nv_max_draw'].values, 'red', linestyle='-.',linewidth=1, label='port_max_draw')
    ax1.legend()
    ax2.legend()
    annual_rts = df['net_value'].values[-1] ** (1 / (round(df.shape[0] / (244), 2))) - 1
    plt.title('years_={} Max_Drawdown={} \n total_rts={} annualized rts ={}\n Sharpe={}'.format(
        round(df.shape[0] / (244), 2),
        np.round(MaxDrawdown(list(df['net_value'].dropna())).max(),4),
        np.round(df['net_value'].values[-1],2),
        np.round(annual_rts,4),
        (annual_rts - 0.03) / (np.std(df['rts_1']) * np.sqrt(244))))
    plt.show()

plot_rts(price_df)








