import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, indicator
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler

auth('15951961478', '961478')
get_query_count()

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

stock_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx', index_col='Unnamed: 0')
stock_name.index = stock_name['code']

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['net_value'] = hs300['close'] / hs300['close'][0]
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['rts_5'] = hs300['close'].pct_change(5)
hs300['rts_10'] = hs300['close'].pct_change(10)
hs300['rts_20'] = hs300['close'].pct_change(20)
hs300['rts_30'] = hs300['close'].pct_change(30)
hs300['rts_60'] = hs300['close'].pct_change(60)

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0',
                   date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0',
                         date_parser=dateparse)

close_rts_1 = close.pct_change(1)
close_rts_1['hs300'] = hs300['rts_1']
close_rts_5 = close.pct_change(5)
close_rts_5['hs300'] = hs300['rts_5']
close_rts_10 = close.pct_change(10)
close_rts_10['hs300'] = hs300['rts_10']
close_rts_20 = close.pct_change(20)
close_rts_20['hs300'] = hs300['rts_20']
close_rts_30 = close.pct_change(30)
close_rts_30['hs300'] = hs300['rts_30']
close_rts_60 = close.pct_change(60)
close_rts_60['hs300'] = hs300['rts_60']

df = list(get_fundamentals(query(valuation.code).filter(valuation.market_cap > 100, valuation.pe_ratio>25))['code'])  # 市值大于100的股票池
roe_yeayly = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/roe_yearly.csv', index_col='statDate')

# N日波动率
rts_std_20 = close_rts_1.std()
rts_std_90 = close_rts_1.std()

# 年度：
annual_index = [x for x in roe_yeayly.index if x.endswith('12-31')]

# 选股：
rev_growth_value = 10
np_growth_value = 10
roe_value = 2
stock_list = list(set(rev_growth.mean()[rev_growth.mean() > rev_growth_value].index).intersection(
    set(np_growth.mean()[np_growth.mean() > np_growth_value].index), set(roe.min()[roe.min() > roe_value].index),
    set(df)))
# 不加市值大于100共73个；市值大于100的57个。
stock_list.insert(0, 'hs300')


out = pd.DataFrame(columns=['过去1天', '过去5天', '过去10天', '过去20天', '过去30天', '过去60天'],
                   index=stock_list)
out['过去1天'] = close_rts_1[stock_list].iloc[-1, :]
out['过去5天'] = close_rts_5[stock_list].iloc[-1, :]
out['过去10天'] = close_rts_10[stock_list].iloc[-1, :]
out['过去20天'] = close_rts_20[stock_list].iloc[-1, :]
out['过去30天'] = close_rts_30[stock_list].iloc[-1, :]
out['过去60天'] = close_rts_60[stock_list].iloc[-1, :]
out.index = ['沪深300'] + list(stock_name['short_name'].loc[stock_list[1:]])
out.to_excel('/Users/caichaohong/Desktop/Zenki/5.21_table.xlsx')


# plot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#  大盘大跌时间窗口：
# 2021年2月18日-2021年3月9日

# 2020年1月21日-2020年2月3日


# 2020年3月6日 - 2020年3月23日


# 2019年4月22日 -2019年5月9日

def kangdie_plot(start, end, stock_list):
    # start 是最高点前一天，重置net value=1
    hs300_rts = hs300['rts_1'][start:end]
    hs300_rts.loc[start] = 0
    hs300_net_value = (1 + hs300_rts).cumprod()

    plot_df = pd.DataFrame(index=hs300[start:end].index)
    plot_df['hs300'] = hs300_net_value

    for j in tqdm(range(0, len(stock_list), 9)):  # 每9张图一个
        plt.figure(figsize=(10, 10))

        for i in range(j, min(j + 9, len(stock_list))):
            stock_code = stock_list[i]
            tmp_rts = close_rts_1[stock_code].loc[start:end]
            tmp_rts.loc[start] = 0
            tmp_net_value = (1 + tmp_rts).cumprod()
            plot_df[stock_code] = tmp_net_value

            ax = plt.subplot(3, 3, i % 9 + 1)
            ax.plot(plot_df[stock_code], 'black', label=stock_code)
            ax.plot(plot_df['hs300'], 'blue', label='hs300')
            ax.set_title('{} vs 沪深300'.format(stock_name['short_name'].loc[stock_code]))
            ax.legend()
            ax.set_xticks(list(plot_df.index[::int(plot_df.shape[0]/3)]))
        plt.tight_layout()

        plt.savefig('/Users/caichaohong/Desktop/Zenki/{}.png'.format(j))
        plt.close()
        print('i={} j = {}'.format(i, j))


kangdie_plot(start='2021-02-10', end='2021-03-09', stock_list=stock_list)


close['hs300'] = hs300['close']
close_index_list = [x.strftime('%Y-%m-%d') for x in close.index]


def get_kangdie_table(start, end, stock_list):
    # start 不需要像kangdie_plot那样往前+1天
    stocklist = stock_list.insert(0,'hs300')
    end_index = close_index_list.index()
    out = pd.DataFrame(columns=['{} to {}跌幅'.format(start,end) , '未来1天', '未来5天', '未来10天', '未来20天', '未来30天', '未来60天'], index=stock_list)
    out['{} to {}跌幅'.format(start,end)] = np.round(close[stocklist].loc[start]/close[stocklist].loc[end] -1, 4)
    out['未来1天'] = close_rts_1[stock_list].iloc[-1, :]
    out['未来5天'] = close_rts_5[stock_list].iloc[-1, :]
    out['未来10天'] = close_rts_10[stock_list].iloc[-1, :]
    out['未来20天'] = close_rts_20[stock_list].iloc[-1, :]
    out['未来30天'] = close_rts_30[stock_list].iloc[-1, :]
    out['未来60天'] = close_rts_60[stock_list].iloc[-1, :]
    out.index = ['沪深300'] + list(stock_name['short_name'].loc[stock_list[1:]])
    out.to_excel('/Users/caichaohong/Desktop/Zenki/5.21_table.xlsx')





