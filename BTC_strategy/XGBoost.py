import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
price_df = raw_price_df.copy(deep=True)
price_df['close_nv'] = price_df['close']/price_df['close'][0]
feature_df = pd.DataFrame(index=raw_price_df.index)


fake_bench = pd.DataFrame(index=price_df.index)
fake_bench['close'] = 1

# 均线
price_df['close_ma1'] = price_df['close'].rolling(3).mean()
price_df['close_ma2'] = price_df['close'].rolling(6).mean()
price_df['close_ma3'] = price_df['close'].rolling(12).mean()
feature_df['long_ma1'] = ((price_df['close_ma1'] > price_df['close_ma2']) & (price_df['close_ma1'] > price_df['close_ma3']) & (price_df['close_ma2'] > price_df['close_ma3'])) * 1
feature_df['short_ma1'] = ((price_df['close_ma1'] < price_df['close_ma2']) & (price_df['close_ma1'] < price_df['close_ma3']) & (price_df['close_ma2'] < price_df['close_ma3'])) * 1

price_df['close_ma4'] = price_df['close'].rolling(30).mean()
price_df['close_ma5'] = price_df['close'].rolling(60).mean()
price_df['close_ma6'] = price_df['close'].rolling(120).mean()
feature_df['long_ma2'] = ((price_df['close_ma4'] > price_df['close_ma5']) & (price_df['close_ma4'] > price_df['close_ma6']) & (price_df['close_ma5'] > price_df['close_ma6'])) * 1
feature_df['short_ma2'] = ((price_df['close_ma4'] < price_df['close_ma5']) & (price_df['close_ma4'] < price_df['close_ma6']) & (price_df['close_ma5'] < price_df['close_ma6'])) * 1


# std
std_percent_high = 90
std_percent_low = 10
close_std_n1 = 10  # 95% 0.003 ，5% 0.00013
price_df['close_std_n1'] = price_df['close'].rolling(close_std_n1).std() / price_df['close']
feature_df['close_std_n1_big'] = (price_df['close_std_n1'] > np.percentile(price_df['close_std_n1'].dropna(), std_percent_high)) * 1
feature_df['close_std_n1_small'] = (price_df['close_std_n1'] < np.percentile(price_df['close_std_n1'].dropna(), std_percent_low)) * 1

close_std_n2 = 120  # 25 75 大部分都在[0.1%, 0.4%], 95%在0.01以上
price_df['close_std_n2'] = price_df['close'].rolling(close_std_n2).std() / price_df['close']
feature_df['close_std_n2_big'] = (price_df['close_std_n2'] > np.percentile(price_df['close_std_n2'].dropna(), std_percent_high)) * 1
feature_df['close_std_n2_small'] = (price_df['close_std_n2'] < np.percentile(price_df['close_std_n2'].dropna(), std_percent_low)) * 1

# 放量上涨
# price_df['rts_1'][price_df['rts_1']>=0.001]
# 大于千1的有5.5万分钟，千2的 有1.8万分钟

vol_percent_high = 90
vol_percent_low = 10
vol_n1 = 10
price_df['vol_ma1'] = price_df['vol'].rolling(vol_n1).mean()
price_df['vol_ex_ma1'] = price_df['vol'] / price_df['vol_ma1'] - 1
feature_df['vol1_big'] = (price_df['vol_ex_ma1'] > np.percentile(price_df['vol_ex_ma1'].dropna(), vol_percent_high)) * 1
feature_df['vol1_low'] = (price_df['vol_ex_ma1'] < np.percentile(price_df['vol_ex_ma1'].dropna(), vol_percent_low) )* 1

vol_n2 = 120
price_df['vol_ma2'] = price_df['vol'].rolling(vol_n2).mean()
price_df['vol_ex_ma2'] = price_df['vol'] / price_df['vol_ma2'] - 1
feature_df['vol2_big'] = (price_df['vol_ex_ma2'] > np.percentile(price_df['vol_ex_ma2'].dropna(), vol_percent_high)) * 1
feature_df['vol2_low'] = (price_df['vol_ex_ma2'] < np.percentile(price_df['vol_ex_ma2'].dropna(), vol_percent_low)) * 1

# 收益率
p_percent_high = 90
p_percent_low = 10

p_n1 = 1
p_n2 = 10
p_n3 = 40
p_n4 = 120
price_df['rts_n1'] = price_df['close'].pct_change(p_n1)
price_df['rts_n2'] = price_df['close'].pct_change(p_n2)
price_df['rts_n3'] = price_df['close'].pct_change(p_n3)
price_df['rts_n4'] = price_df['close'].pct_change(p_n4)
feature_df['rts_n1_big'] = (price_df['rts_n1'] >= np.percentile(price_df['rts_n1'].dropna(), p_percent_high))*1
feature_df['rts_n1_small'] = (price_df['rts_n1'] <= np.percentile(price_df['rts_n1'].dropna(), p_percent_low))*1

feature_df['rts_n2_big'] = (price_df['rts_n2'] >= np.percentile(price_df['rts_n2'].dropna(), p_percent_high))*1
feature_df['rts_n2_small'] = (price_df['rts_n2'] <= np.percentile(price_df['rts_n2'].dropna(), p_percent_low))*1

feature_df['rts_n3_big'] = (price_df['rts_n3'] >= np.percentile(price_df['rts_n3'].dropna(), p_percent_high))*1
feature_df['rts_n3_small'] = (price_df['rts_n3'] <= np.percentile(price_df['rts_n3'].dropna(), p_percent_low))*1

feature_df['rts_n4_big'] = (price_df['rts_n4'] >= np.percentile(price_df['rts_n4'].dropna(), p_percent_high))*1
feature_df['rts_n4_small'] = (price_df['rts_n4'] <= np.percentile(price_df['rts_n4'].dropna(), p_percent_low))*1



# target
target_n = 5  # 5分钟
feature_df['target_Y'] = (price_df['close'].pct_change(target_n).shift(-target_n) >= 0.005) * 1


X = feature_df.iloc[:,0:-1]
Y = feature_df.iloc[:,-1]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train, verbose=True)
# make predictions for test data
y_pred = model.predict(X_test)

pred_table = pd.DataFrame(index=X_test.index)
pred_table['y_test'] = y_test
pred_table['y_pred'] = y_pred
