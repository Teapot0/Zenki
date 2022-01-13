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


price_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/KZZ/price.csv', index_col='Unnamed: 0')
price_df['double_low'] = price_df['close'] + 100 * price_df['convert_premium_rate']

date_list = sorted(price_df['date'].unique())

out = pd.DataFrame(index=date_list, columns=['rts'])


for i in tqdm(range(len(date_list)-1)):
    d = date_list[i]
    d1 = date_list[i+1]
    tmp_df = price_df[price_df['date'] == d].sort_values(by='double_low')
    tmp_df1 = price_df[price_df['date'] == d1].sort_values(by='double_low')

    tmp_bond = list(set(tmp_df1['code']).intersection(set(tmp_df['code'])))
    tmp_df = tmp_df[tmp_df['code'].isin (tmp_bond)]
    tmp_df1 = tmp_df1[tmp_df1['code'].isin (tmp_bond)]

    buy_bonds = tmp_df['code'][:10]

    tmp_df = tmp_df[tmp_df['code'].isin(buy_bonds)]
    tmp_df1 = tmp_df1[tmp_df1['code'].isin(buy_bonds)]

    cost_df = pd.DataFrame(columns=['d', 'd1'])
    cost_df['d'] = tmp_df['close'].values
    cost_df['d1'] = tmp_df1['close'].values
    cost_df['rts'] = cost_df['d1'] / cost_df['d'] - 1

    tmp_rts = cost_df['rts'].mean()

    out.loc[d1]['rts'] = tmp_rts - 0.01

out = out.fillna(0)
out['nv'] = (1+out['rts']).cumprod()
plt.plot(out['nv'].values)

annual_rts = out['nv'][-1]**(1/(out.shape[0] /  255)) -1
print('年化:{}'.format(annual_rts))

sharpe = ((out['rts'] - 0.03/255).mean() / out['rts'].std()) * np.sqrt(255)
print('夏普:{}'.format(sharpe))





z = price_df.sort_values(by='money')



