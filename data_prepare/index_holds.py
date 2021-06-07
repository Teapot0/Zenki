import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_index_stocks, get_bars, bond
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *

auth('15951961478', '961478')
get_query_count()

import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)

z = pd.DataFrame(index=close.index)
z['hs300'] = np.nan
for tmp_date in tqdm(close.index):
    hs300_list = get_index_stocks('000300.XSHG', date =tmp_date)
    z['hs300'].loc[tmp_date] = hs300_list
z.to_csv('/Users/caichaohong/Desktop/Zenki/index_holdings.csv')