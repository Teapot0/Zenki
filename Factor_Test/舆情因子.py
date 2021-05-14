import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqfactor_analyzer import analyze_factor
from alphalens.utils import get_clean_factor_and_forward_returns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from basic_funcs.basic_function import *
from jqdatasdk import get_index_stocks, auth, get_query_count,get_extras

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings('ignore')

out_0 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor0.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_1 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor1.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_2 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor2.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
# out_012 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor012.csv', index_col='Unnamed: 0',
#                       date_parser=dateparse)

out_0.index = pd.to_datetime(out_0.index)
out_1.index = pd.to_datetime(out_1.index)
out_2.index = pd.to_datetime(out_2.index)

out_0_diff1 = out_0.diff(1)
out_0_diff5 = out_0.diff(5)
out_0_diff10 = out_0.diff(10)

out_1_diff1 = out_1.diff(1)
out_1_diff5 = out_1.diff(5)
out_1_diff10 = out_1.diff(10)

out_2_diff1 = out_2.diff(1)
out_2_diff5 = out_2.diff(5)
out_2_diff10 = out_2.diff(10)


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0', date_parser=dateparse)

close_rts = close.pct_change(1)


z = quantile_factor_test_plot(factor = (out_2-out_1).diff(1).loc[close_rts.index], rts=close_rts, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=30, plot_title=False, weight="avg",comm_fee=0.003)




