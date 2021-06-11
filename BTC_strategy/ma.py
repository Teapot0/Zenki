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

# 均线
price_df['close_ma1'] = price_df['close'].rolling(4).mean()
price_df['close_ma2'] = price_df['close'].rolling(16).mean()
price_df['close_ma3'] = price_df['close'].rolling(64).mean()
price_df['long_ma'] = ((price_df['close_ma1'] > price_df['close_ma2']) & (price_df['close_ma1'] > price_df['close_ma3']) & (price_df['close_ma2'] > price_df['close_ma3']))* 1

plt.plot(price_df['close'].values,'black')
plt.plot(price_df['close_ma1'].values,'blue')
plt.plot(price_df['close_ma2'].values,'green')
plt.plot(price_df['close_ma3'].values,'red')

price_df['buy_signal'] = ((price_df['long_ma'] == 1) & (price_df['close'] > price_df['close_ma1']))*1


price_df['hold_status'] = 0
hold_status = 0
for i in tqdm(range(price_df.shape[0]-1)):
    if hold_status == 0:
        if price_df['buy_signal'][i] == 1:
            hold_status = 1
            tmp_net_value = [1]
    else:  # hold status =1
        price_df['hold_status'][i] = 1
        tmp_net_value.append(tmp_net_value[-1]*(1+price_df['rts_1'][i]))
        tmp_max_draw = tmp_net_value[-1] / max(tmp_net_value) - 1
        if tmp_max_draw < -0.01:  # 最大回撤
            hold_status = 0

price_df['comm_fee'] = (price_df['hold_status'].diff(1) == 1) * 1 * 0.00
price_df['hold_rts'] = price_df['rts_1'] * price_df['hold_status'] - price_df['comm_fee']
price_df['net_value'] = (1+price_df['hold_rts']).cumprod()
price_df['nv_max_draw'] = [np.nan] + list(MaxDrawdown(list(price_df['net_value'].dropna())).reshape(-1))

ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(price_df['net_value'].values, 'black')
# ax1.plot(price_df['close_nv'].values, 'blue')
ax2.plot(price_df['nv_max_draw'].values, 'red', linestyle='-.',linewidth=1, label='port_max_draw')







