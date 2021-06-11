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


# data index是每分钟开始时间
raw_price_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/BTC-USD/price_df.csv',index_col='id')
price_df = raw_price_df.copy(deep=True)
price_df['close_nv'] = price_df['close']/price_df['close'][0]


# boll
price_df['boll_up'],price_df['boll_mid'], price_df['boll_low'] = talib.BBANDS(price_df['close'],timeperiod=60,nbdevup=2,nbdevdn=2,matype=0)
#Type0: Moving average type: simple moving average here

plt.plot(price_df['close'].values,'black')
plt.plot(price_df['boll_up'].values,'red')
plt.plot(price_df['boll_mid'].values,'blue')
plt.plot(price_df['boll_low'].values,'green')




