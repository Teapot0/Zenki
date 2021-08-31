import numpy as np
from numpy import sqrt, pi, e
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqdatasdk import get_industries, get_industry_stocks,get_extras, finance, query, bond
from sklearn.linear_model import LinearRegression
import talib

color_list = ['grey', 'rosybrown', 'saddlebrown', 'orange', 'goldenrod',
              'olive', 'yellow', 'darkolivegreen', 'lime', 'lightseagreen',
              'cyan', 'cadetblue', 'deepskyblue', 'steelblue', 'lightslategrey',
              'navy', 'slateblue', 'darkviolet', 'thistle', 'orchid',
              'deeppink', 'lightpink']


import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')

# ss = ['600519.XSHG', '601012.XSHG', '000001.XSHE', '002140.XSHE', '300750.XSHE', '002714.XSHE']
ss = ['600519.XSHG']
close_daily = close_daily[ss]
open_daily = open_daily[ss]

open_close_rts = open_daily / (close_daily.shift(1)) - 1
close_open_rts = close_daily / open_daily - 1

circulating_market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/circulating_market_cap.csv', index_col='Unnamed: 0')
circulating_market_cap = circulating_market_cap[ss]

# 大单
net_amount_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv', index_col='Unnamed: 0')
net_amount_s = net_amount_s[ss]


factor = 1/(abs(net_amount_s) / circulating_market_cap)
factor = net_amount_s/ circulating_market_cap
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/smallplayer_abs_turnover.csv')

# factor = factor.rolling(5).mean()

date_1 = factor.index
date_5 = factor.index[::5]
date_10 = factor.index[::10]

quantiles=10
buy_date_list=date_1
hold_time_title='1D'
plot_title=False
comm_fee=0.003

#################################################
open_close_rts = open_daily/(close_daily.shift(1)) - 1
close_open_rts = close_daily/open_daily - 1
factor_date = list(factor.index)
is_date_buy = pd.DataFrame(0,columns=['is_buy'],index=factor_date)
is_date_buy['is_buy'].loc[buy_date_list] = 1

factor = factor.dropna(how='all', axis=1)

plt.figure(figsize=(9, 9))
plt.plot((1 + benchmark_rts[factor.index]).cumprod(), color='black', label='benchmark_net_value')

quantile_df = pd.DataFrame(np.nan, index=factor.index, columns=range(quantiles))

hold_stock = {}  # 每天开盘价买入
for q in range(quantiles):
    hold_stock[q] = {}

# 每天开盘买入名单
temp_stock_list = []

for q in range(quantiles):
    hold_stock[q][factor_date[0]] = []  # 第一天开盘无买入

for i in range(len(factor_date)-1):
    date = factor_date[i]  # t时刻
    date_1 = factor_date[i + 1]  # t+1时刻

    tmp_ii = is_date_buy['is_buy'].loc[date]

    if tmp_ii == 1:  # 换仓日
        temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
        stock_number = len(temp_factor)
        if len(temp_factor) > 0:  # 若无股票，则持仓不变
            for q in range(quantiles):
                start_i = q * int(stock_number / quantiles)
                if q == quantiles - 1:  # 最后一层
                    end_i = stock_number
                else:
                    end_i = (q + 1) * int(stock_number / quantiles)
                temp_stock_list = list(temp_factor.index[start_i:end_i])  # 下一个buy_list_date持仓股票
                hold_stock[q][date_1] = temp_stock_list  # t+1，当天以开盘价买入temp_stock_list
    else:
        hold_stock[q][date_1] = temp_stock_list

print ('buy_stocks_added')

# 收益率
for i in range(1,len(factor_date)):
    date_yes = factor_date[i-1]  # 昨天
    date = factor_date[i]  # t时刻

    for q in range(quantiles):
        stock_list_yes = hold_stock[q][date_yes]
        stock_list = hold_stock[q][date]

        tmp_open_rts = open_close_rts.loc[date][stock_list_yes].mean()  # 昨日持仓 今天开盘卖出收益
        tmp_close_rts = close_open_rts.loc[date][stock_list].mean()
        tmp_rts = tmp_open_rts+ tmp_close_rts - comm_fee*2
        quantile_df[q].loc[date] = tmp_rts

print ('quantile rts calculated')
quantile_df = quantile_df.fillna(0)

for q in range(quantiles):
    net_value = (1 + quantile_df[q]).cumprod()
    plt.plot(net_value, color=color_list[q], label='Quantile{}'.format(q))

plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
if plot_title == False:
    plt.title('quantiles={}\n持股时间={}'.format(quantiles, hold_time_title))
else :
    plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title, quantiles, hold_time_title))



plt.plot((1+z[0]).cumprod().values)



