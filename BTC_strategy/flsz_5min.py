import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import talib
import seaborn as sns
from basic_funcs.CTA_basics import *
from basic_funcs.basic_function import *

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# data index是每分钟开始时间
raw_price_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/BTC-USD/price_df.csv',index_col='id')
raw_price_df.index = pd.to_datetime(raw_price_df.index)

price_df = raw_price_df.groupby(pd.Grouper(freq='5Min')).agg({"open": "first", "close": "last", "low": "min", "high": "max",
                                                       "amount":"sum", "vol":"sum", "count":"sum"})

price_df['close_nv'] = price_df['close']/price_df['close'][0]

#收益率
price_df['rts_1'] = price_df['close'].pct_change(1)

# 成交量
price_df['vol_ma1'] = price_df['vol'].rolling(10).mean()
price_df['vol_ex1'] = (price_df['vol'] / price_df['vol_ma1'] - 1)
vol_n = 1  # 放量大小
price_df['vol_ex1_big'] = (price_df['vol_ex1'] > vol_n) * 1

#  均线
price_df['close_ma1'] = price_df['close'].rolling(4).mean()
price_df['close_ma2'] = price_df['close'].rolling(16).mean()
price_df['close_ma3'] = price_df['close'].rolling(64).mean()
price_df['close_ma4'] = price_df['close'].rolling(248).mean()
price_df['close_ma5'] = price_df['close'].rolling(992).mean()
price_df['long_ma'] = ((price_df['close_ma4']>price_df['close_ma5'])&(price_df['close_ma1'] > price_df['close_ma2']) & (price_df['close_ma1'] > price_df['close_ma3']) & (price_df['close_ma2'] > price_df['close_ma3']))* 1

price_df['buy_signal'] = ((price_df['long_ma'] == 1) & (price_df['rts_1'] > 0.002) & (price_df['vol_ex1_big'] == 1)) * 1


def get_maxdraw_hold_position(price_df, max_draw):
    hold_status = 0
    price_df['hold_status'] = 0

    for i in tqdm(range(price_df.shape[0]-1)):
        if hold_status == 0: # 当天没有持仓
            if price_df['buy_signal'][i] == 1:
                hold_status = 1
                tmp_net_value = [1]

        else:  # hold status =1
            price_df['hold_status'][i] = 1
            tmp_net_value.append(tmp_net_value[-1] * (1 + price_df['rts_1'][i]))
            tmp_max_draw = tmp_net_value[-1] / max(tmp_net_value) - 1
            if tmp_max_draw < -max_draw:  # 最大回撤
                hold_status = 0
    return price_df


price_df = get_maxdraw_hold_position(price_df, max_draw=0.1)


price_df['comm_fee'] = (price_df['hold_status'].diff(1) == 1) * 1 * 0.001
price_df['hold_rts'] = price_df['rts_1'] * price_df['hold_status'] - price_df['comm_fee']
price_df['net_value'] = (1+price_df['hold_rts']).cumprod()
price_df['nv_max_draw'] = [np.nan] + list(MaxDrawdown(list(price_df['net_value'].dropna())).reshape(-1))

ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(price_df['net_value'].values, 'black')
ax1.plot(price_df['close_nv'].values,'blue')
# ax2.plot(price_df['close'].values,'red')
# ax2.plot(price_df['close_ma1'].values,'blue')
# ax2.plot(price_df['close_ma3'].values,'green')
# ax2.plot(price_df['close_nv'].values, 'blue')
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



price_df['future_10min_rts'] = price_df['close'].pct_change(10).shift(-10)
price_df['future_30min_rts'] = price_df['close'].pct_change(30).shift(-30)
z = price_df.sort_values(by='future_30min_rts',ascending=False)


raw_price_df.index = pd.to_datetime(raw_price_df.index)
price_5m = raw_price_df.groupby(pd.Grouper(freq='60Min')).agg({"open": "first", "close": "last", "low": "min", "high": "max",
                                                       "amount":"sum", "vol":"sum", "count":"sum"})
price_5m['close_rts'] = price_5m['close'].pct_change(1)
price_5m['future_1h_rts'] = price_5m['close_rts'].shift(-1)
zz = price_5m.sort_values(by='future_1h_rts', ascending=False)












