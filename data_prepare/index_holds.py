import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks, get_bars, bond
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
import json
from basic_funcs.basic_function import *

auth('13382017213', 'Aasd120120')
get_query_count()

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0')

hs300_holds = pd.DataFrame(index=close.index, columns=close.columns)
zz500_holds = pd.DataFrame(index=close.index, columns=close.columns)
zz1000_holds = pd.DataFrame(index=close.index, columns=close.columns)

for tmp_date in tqdm(close.index):
    hs300_list = get_index_stocks('000300.XSHG', date =tmp_date)
    zz500_list = get_index_stocks('000905.XSHG', date =tmp_date)
    zz1000_list = get_index_stocks('000852.XSHG', date =tmp_date)

    tmp_300 = list(set(hs300_list).intersection(close.columns))
    tmp_500 = list(set(zz500_list).intersection(close.columns))
    tmp_1000 = list(set(zz1000_list).intersection(close.columns))
    hs300_holds.loc[tmp_date][tmp_300] = 1
    zz500_holds.loc[tmp_date][tmp_500] =1
    zz1000_holds.loc[tmp_date][tmp_1000] =1


hs300_holds.to_csv('/Users/caichaohong/Desktop/Zenki/hs300_holds.csv')
zz500_holds.to_csv('/Users/caichaohong/Desktop/Zenki/zz500_holds.csv')
zz1000_holds.to_csv('/Users/caichaohong/Desktop/Zenki/zz1000_holds.csv')

# a_file = open('/Users/caichaohong/Desktop/Zenki/price/index_holds/hs300.json', "r")
# output = a_file.read()
