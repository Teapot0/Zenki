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
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['interactive'] == True



# data index是每分钟开始时间
raw_price_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/BTC-USD/price_df.csv',index_col='id')
raw_price_df.index = pd.to_datetime(raw_price_df.index)

price_df = raw_price_df.groupby(pd.Grouper(freq='5Min')).agg({"open": "first", "close": "last", "low": "min", "high": "max",
                                                       "amount":"sum", "vol":"sum", "count":"sum"})

price_df['close_nv'] = price_df['close']/price_df['close'][0]
price_df['close_rts1'] = price_df['close'].pct_change(1)
price_df['close_rts_shift2'] = price_df['close_rts1'].shift(2)

hold_status = 0
price_df['hold_status'] = 0

# 连续下跌，出现阳线包裹阴线，
for i in tqdm(range(price_df.shape[0] - 1)):
    if hold_status == 0:  # 当天没有持仓
        if price_df['buy_signal'][i] == 1:
            hold_status = 1
            tmp_net_value = [1]

    else:  # hold status =1
        price_df['hold_status'][i] = 1
        tmp_net_value.append(tmp_net_value[-1] * (1 + price_df['rts_1'][i]))
        tmp_max_draw = tmp_net_value[-1] / max(tmp_net_value) - 1
        if tmp_max_draw < -max_draw:  # 最大回撤
            hold_status = 0


# boll
# price_df['boll_up'],price_df['boll_mid'], price_df['boll_low'] = talib.BBANDS(price_df['close'],timeperiod=60,nbdevup=2,nbdevdn=2,matype=0)
#Type0: Moving average type: simple moving average here



z = raw_price_df[(raw_price_df.index >= '2021-05-19') & (raw_price_df.index<= ' 2021-05-20')]
z['date'] = z.index
z['rts_1'] = z['close'].pct_change(1)





# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
# include candlestick with rangeselector
fig.add_trace(go.Candlestick(x=z['date'],
                open=z['open'], high=z['high'],
                low=z['low'], close=z['close'],
                increasing_line_color= 'red', decreasing_line_color= 'green'),
               secondary_y=True)
# include a go.Bar trace for volumes
fig.add_trace(go.Bar(x=z.index, y=z['vol'], marker_color = 'lightgrey'),
               secondary_y=False)
fig.layout.yaxis2.showgrid=False








