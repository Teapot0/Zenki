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
price_df['vol_ma1'] = price_df['vol'].rolling(120).mean()
price_df['vol_ex1'] = (price_df['vol'] / price_df['vol_ma1'] - 1)
vol_n = 1  # 放量大小
price_df['vol_ex1_big'] = (price_df['vol_ex1'] > vol_n) * 1

#  均线
price_df['close_ma1'] = price_df['close'].rolling(4).mean()
price_df['close_ma2'] = price_df['close'].rolling(12).mean()
price_df['close_ma3'] = price_df['close'].rolling(36).mean()
price_df['close_ma4'] = price_df['close'].rolling(108).mean()
price_df['close_ma5'] = price_df['close'].rolling(324).mean()
price_df['long_ma'] = ((price_df['close_ma4']>price_df['close_ma5'])&
                       (price_df['close_ma1'] > price_df['close_ma2']) &
                       (price_df['close_ma1'] > price_df['close_ma3']) &
                       (price_df['close_ma2'] > price_df['close_ma3'])) * 1

price_df['buy_signal'] = ((price_df['long_ma'] == 1) &
                          (price_df['rts_1'] > 0.005) &
                          (price_df['vol_ex1_big'] == 1)) * 1


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
ax2.plot(price_df['nv_max_draw'].values, 'red', linestyle='-.',linewidth=1, label='port_max_draw')




price_df['future_10min_rts'] = price_df['close'].pct_change(2).shift(-2)
price_df['future_30min_rts'] = price_df['close'].pct_change(6).shift(-6)

price_df['date'] = [x.strftime('%Y-%m-%d') for x in price_df.index]

rts_day = price_df[['hold_rts', 'date']].groupby('date').apply(lambda x: np.float(((1+x).cumprod()-1).values[-1]))
plt.boxplot(rts_day)
rts_day_sorted = rts_day.sort_values(ascending=False)

z = price_df.sort_values(by='future_10min_rts',ascending=False)


# 1月29日、30号
zz = price_df[(price_df.index >= '2021-01-19')]





'''
param_df = pd.DataFrame(index=['vol_t', 'vol_n', 'rts_min', 'max_draw_param', 'total_rts','max_draw'])


def params_test(vol_period_list, vol_number_list, rts_min_list,max_draw_list):
    ii=0
    for vol_period in vol_period_list:
        for vol_number in vol_number_list:
            for rts_min in rts_min_list:
                for max_d in max_draw_list:
                    price_df['vol_ma1'] = price_df['vol'].rolling(vol_period).mean()
                    price_df['vol_ex1'] = (price_df['vol'] / price_df['vol_ma1'] - 1)
                    price_df['vol_ex1_big'] = (price_df['vol_ex1'] > vol_number) * 1

                    price_df['buy_signal'] = ((price_df['rts_1'] > rts_min) &
                                              (price_df['vol_ex1_big'] == 1)) * 1

                    rts_df = get_maxdraw_hold_position(price_df, max_draw=max_d)
                    rts_df['comm_fee'] = (rts_df['hold_status'].diff(1) == 1) * 1 * 0.001
                    rts_df['hold_rts'] = rts_df['rts_1'] * rts_df['hold_status'] - rts_df['comm_fee']
                    rts_df['net_value'] = (1 + rts_df['hold_rts']).cumprod()
                    rts_df['nv_max_draw'] = [np.nan] + list(MaxDrawdown(list(rts_df['net_value'].dropna())).reshape(-1))
                    total_rts = ((1+rts_df['hold_rts']).cumprod()-1).
                    Max_d = rts_df['nv_max_draw'].dropna().max()
                param_df[ii] = [vol_period,vol_number,rts_min, max_d, total_rts[-1], Max_d]
                ii+=1
                print ('ii={}'.format(ii))
    return param_df


vol_period = [4,16,64,120]
vol_number = [0.5,1]
rts_min = [0,0.002,0.005,0.01]
max_d_list = [0.05,0.1]

param_df  = params_test(vol_period, vol_number,rts_min,max_d_list)
for j in range(param_df.shape[1]):
    param_df[j]['total_rts'] = param_df[j]['total_rts'][-1] - 1
param_df.to_excel('/Users/caichaohong/Desktop/Zenki/btc_params.xlsx')
'''







