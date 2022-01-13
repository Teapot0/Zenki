import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *
from jqdatasdk import bond, auth, query
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

auth('13382017213', 'Aasd120120')

#确定日期
kzz_date = '2021-08-25'


#获取可转债的当天价格
kzz_close = bond.run_query(query(bond.CONBOND_DAILY_PRICE.code,
                           bond.CONBOND_DAILY_PRICE.name,
                           bond.CONBOND_DAILY_PRICE.close,
                           bond.CONBOND_DAILY_PRICE.pre_close,
                          ).filter(bond.CONBOND_DAILY_PRICE.date==kzz_date,
                                   bond.CONBOND_DAILY_PRICE.close > 0))

kzz_close = kzz_close.sort_values(by=['close'], ascending=True)
kzz_close = kzz_close.reset_index(drop = True)
kzz_close['order_close']=kzz_close.index


kzz_zg = bond.run_query(query(bond.CONBOND_DAILY_CONVERT.code,
                                     bond.CONBOND_DAILY_CONVERT.convert_premium_rate)
                               .filter(bond.CONBOND_DAILY_CONVERT.date==kzz_date))
# 按溢价率排序
kzz_zg = kzz_zg. sort_values(by=['convert_premium_rate'], ascending=True)
kzz_zg = kzz_zg.reset_index(drop = True)
# 记录溢价率序号
kzz_zg['order_rate']=kzz_zg.index


#
kzz_df = pd.merge(kzz_close, kzz_zg, on="code")

# 计算双低排序值
kzz_df['double_low'] = kzz_df['order_close'] + kzz_df['order_rate']
kzz_l1 = kzz_df. sort_values (by=['double_low'], ascending=True)[:10]

df = bond.run_query(query(bond.CONBOND_DAILY_PRICE).filter(bond.CONBOND_DAILY_PRICE.code==kzz_close['code'][0]))