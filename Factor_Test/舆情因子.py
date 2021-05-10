import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqfactor_analyzer import analyze_factor
from alphalens.utils import get_clean_factor_and_forward_returns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from basic_funcs.basic_function import *
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def transform_ticker(x):
    if x.startswith('6'):
        return x + '.XSHG'
    else:
        return x + '.XSHE'


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = (1+hs300['rts'].fillna(0)).cumprod()

price = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/price.xlsx', index_col='Unnamed: 0')

# 转换数据
''' 
senti_data = pd.DataFrame()
for i in range(6):
    data_prepare = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor{}.csv'.format(i+1))
    if senti_data.empty:
        senti_data = data_prepare.copy()
    else:
        senti_data = pd.concat([senti_data, data_prepare], axis=0)
senti_data = senti_data.drop_duplicates() # 去重
senti_data['secID'] = senti_data['secID'].apply(lambda x: '0'*(6-len(str(x)))+str(x))
senti_data = senti_data.rename(columns = {'date':'tradeDate', 'secID':'ticker'})

senti_data = senti_data.reset_index(drop=True)
senti_data = senti_data.rename(columns = {'charts_0':'中性新闻', 'charts_1':'正面新闻', 'charts_2':'负面新闻', 'charts_012': '所有新闻'})
senti_data.index = senti_data['tradeDate']

senti_data['code'] = [transform_ticker(x) for x in senti_data['ticker']]
stock_list = senti_data['code'].unique()

out_0 = pd.DataFrame()
out_1 = pd.DataFrame()
out_2 = pd.DataFrame()
out_012 = pd.DataFrame()

for s in tqdm(stock_list):
    temp = senti_data[senti_data['code']==s][['中性新闻', '正面新闻', '负面新闻', '所有新闻']]
    temp_0 = temp[['中性新闻']].rename(columns={'中性新闻': s})
    temp_1 = temp[['正面新闻']].rename(columns={'正面新闻': s})
    temp_2 = temp[['负面新闻']].rename(columns={'负面新闻': s})
    temp_012 = temp[['所有新闻']].rename(columns={'所有新闻': s})

    out_0 = pd.concat([out_0, temp_0],axis=1)
    out_1 = pd.concat([out_1, temp_1],axis=1)
    out_2 = pd.concat([out_2, temp_2],axis=1)
    out_012 = pd.concat([out_012, temp_012],axis=1)

out_0.to_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor0.xlsx')
out_1.to_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor1.xlsx')
out_2.to_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor2.xlsx')
out_012.to_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor012.xlsx')


stock_baseinfo = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/stockbaseinfo_daily_HS300.csv')



close = stock_baseinfo[['secID', 'tradeDate', 'closePrice']]

out_close = pd.DataFrame()
for s in tqdm(close['secID'].unique()):
    temp = close[close['secID'] == s][['tradeDate', 'closePrice']]
    temp.index = temp['tradeDate']
    temp = temp['closePrice']
    temp = temp.rename(s)
    out_close = pd.concat([out_close, temp],axis=1)

out_close.to_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/closePrice.xlsx')


'''

out_0 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor0.xlsx', index_col='Unnamed: 0')
out_1 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor1.xlsx', index_col='Unnamed: 0')
out_2 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor2.xlsx', index_col='Unnamed: 0')
out_012 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor012.xlsx', index_col='Unnamed: 0')
out_0 = out_0.sort_index(axis=1)
out_1 = out_1.sort_index(axis=1)
out_2 = out_2.sort_index(axis=1)
out_012 = out_012.sort_index(axis=1)

out_0 = out_0[out_0.index >= '2020-01-01']
out_1 = out_1[out_1.index >= '2020-01-01']
out_2 = out_2[out_2.index >= '2020-01-01']
out_012 = out_012[out_012.index >= '2020-01-01']


out_0 = out_0.pct_change(1)
out_1 = out_1.pct_change(1)
out_2 = out_2.pct_change(1)
out_012 = out_012.pct_change(1)


out_close = pd.read_excel('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/closePrice.xlsx', index_col='tradeDate')


def out_date_transform(factor, price):
    # 舆情数据每天都有，周一的舆情数据等于周末相加
    stock_list = sorted(list(set(factor.columns).intersection(set(price.columns))))
    factor_df = factor[stock_list]
    price_df = price[stock_list]

    out = pd.DataFrame(index=price_df.index, columns=stock_list)
    trade_date = list(price_df.index)
    temp_holiday = pd.DataFrame(columns=stock_list)

    for i in tqdm(range(factor_df.shape[0])):
        temp_date = factor_df.index[i]
        if (temp_date in trade_date) & (temp_holiday.shape[0] == 0):  # 交易日正常复制
            out.loc[temp_date] = factor_df.loc[temp_date]

        elif temp_date not in trade_date:  # 非交易日
            temp_holiday = temp_holiday.append(factor_df.loc[temp_date])

        elif (temp_date in trade_date) & (temp_holiday.shape[0] != 0):  # 假日后第一个交易日
            temp_holiday = temp_holiday.append(factor_df.loc[temp_date])  # 当天的也算，求和
            out.loc[temp_date] = temp_holiday.sum()
            temp_holiday = pd.DataFrame(columns=stock_list)  # 重置
    return out


out0_trade_date = out_date_transform(out_0, out_close)
out1_trade_date = out_date_transform(out_1, out_close)
out2_trade_date = out_date_transform(out_2, out_close)
out012_trade_date = out_date_transform(out_012, out_close)

price_rts = out_close.pct_change(1)
get_top_factor_rts(factor=out0_trade_date, rts=price_rts, top_number=10, hold_time=30, weight='avg')


def quantile_factor_test_plot(factor, rts, benchmark_rts, quantiles, hold_time, plot_title, weight="avg", comm_fee=0.003):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # daily 收益率的df
    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    out.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    stock_number = factor.shape[1]

    plt.figure(figsize=(9, 9))
    plt.plot((1+benchmark_rts).cumprod().values, color='black', label='benchmark_net_value')

    for q in range(quantiles):

        start_i = q * int(stock_number / quantiles)
        if q == quantiles - 1:  # 最后一层
            end_i = stock_number
        else:
            end_i = (q + 1) * int(stock_number / quantiles)

        for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓

            date = factor.index[i]
            date_1 = factor.index[i + 1]

            temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
            if temp_ii == 0:  # 换仓日
                temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
                temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池

            temp_rts_daily = rts.loc[date_1][temp_stock_list]

            if weight == 'avg':  # 每天收益率均值
                out.loc[date_1] = temp_rts_daily.mean()

        out['daily_rts'][::hold_time] = out['daily_rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
        out['daily_rts'] = out['daily_rts'].fillna(0)
        out['net_value'] = (1 + out['daily_rts']).cumprod()
        plt.plot(out['net_value'].values, color=color_list[q], label='Quantile{}'.format(q))
    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title,  quantiles, hold_time))

    return out


z = quantile_factor_test_plot(factor=out_2.loc[out_close.index], rts=price_rts, benchmark_rts=hs300['rts'][hs300.index>='2020-01-01'],
                          quantiles=10, hold_time=30, plot_title='所有新闻')









