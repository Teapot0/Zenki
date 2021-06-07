import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import talib

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

files = os.listdir('/Users/caichaohong/Desktop/Zenki/BTC-USD/')
price_df = pd.DataFrame()
for f in tqdm(files):
    df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/BTC-USD/{}'.format(f),index_col='Unnamed: 0')
    df.index = df['id']
    df = df.drop(columns='id')
    price_df = price_df.append(df)

price_df = price_df.sort_index()
price_df.to_csv('/Users/caichaohong/Desktop/Zenki/BTC-USD/price_df.csv')


ax1=plt.subplot()
ax11 = ax1.twinx()
ax1.plot(price_df['close'].values,'black')
ax11.plot(price_df['vol'].values,'blue')





