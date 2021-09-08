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


def get_ic_table_open_index(factor, open_rts, buy_date_list, index_pool):
    # buy list 是换仓日
    # index_pool 是制定代码池，目前支持300,500,1000
    if index_pool == 'hs300':
        index_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/hs300_holds.csv', index_col='Unnamed: 0')

    elif index_pool == 'zz500':
        index_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz500_holds.csv', index_col='Unnamed: 0')

    elif index_pool == 'zz1000':
        index_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz1000_holds.csv', index_col='Unnamed: 0')

    rts = open_rts.shift(-1)
    out = pd.DataFrame(np.nan,index=buy_date_list,columns=['ic', 'rank_ic'])
    for i in range(len(buy_date_list)-1):
        date = buy_date_list[i]
        date1 = buy_date_list[i+1]
        tmp_hold = list(index_holds.loc[date].dropna().index)
        tmp_mean = rts.loc[date1][tmp_hold].mean()
        tmp = pd.concat([factor.loc[date][tmp_hold],rts.loc[date1][tmp_hold] - tmp_mean],axis=1)
        tmp.columns = ['date', 'date1']
        tmp['date1'] = tmp['date1'] - tmp['date1'].mean()
        out['ic'].loc[date] = tmp.dropna().corr().iloc[0,1]
        out['rank_ic'].loc[date] = tmp.rank().corr().iloc[0,1]
    return out


def get_single_ic_table_open(factor, open_rts):
    rts = open_rts.shift(-1)
    out = pd.DataFrame(np.nan,index=factor.index,columns=['ic'])
    for i in tqdm(range(factor.shape[0]-1)):
        date = factor.index[i]
        date1 = factor.index[i+1]
        tmp = pd.concat([factor.loc[date],rts.loc[date1]],axis=1)
        tmp.columns = ['date', 'date1']
        tmp['date1'] = tmp['date1'] - tmp['date1'].mean()
        out['ic'].loc[date] = tmp.dropna().corr().iloc[0,1]
    return out


def quantile_factor_test_plot_open_index(factor, open_rts, benchmark_rts, quantiles, hold_time,
                                   index_pool, plot_title=False, weight="avg", comm_fee=0.003):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel
    if index_pool == 'hs300':
        index_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/hs300_holds.csv', index_col='Unnamed: 0')

    elif index_pool == 'zz500':
        index_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz500_holds.csv', index_col='Unnamed: 0')

    elif index_pool == 'zz1000':
        index_holds = pd.read_csv('/Users/caichaohong/Desktop/Zenki/zz1000_holds.csv', index_col='Unnamed: 0')

    rts = open_rts.shift(-1)
    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA

    plt.figure(figsize=(9, 9))
    plt.plot((1 + benchmark_rts[factor.index]).cumprod(), color='black', label='benchmark_net_value')

    quantile_df = pd.DataFrame(np.nan,index = factor.index, columns=range(quantiles))
    hold_stock = {}
    for q in range(quantiles):
        hold_stock[q] = []

    for i in tqdm(range(NA_rows, len(factor.index) - 2)):  # 每hold_time换仓

        date = factor.index[i]
        date_2 = factor.index[i + 2]
        tmp_hold = list(index_holds.loc[date].dropna().index)

        temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
        if temp_ii == 0:  # 换仓日
            temp_factor = factor.loc[date][tmp_hold].sort_values(ascending=False).dropna()  # 每天从大到小
            stock_number = len(temp_factor)  # 一直都na去掉
            if len(temp_factor) > 0:  # 若无股票，则持仓不变
                for q in range(quantiles):
                    start_i = q * int(stock_number / quantiles)
                    if q == quantiles - 1:  # 最后一层
                        end_i = stock_number
                    else:
                        end_i = (q + 1) * int(stock_number / quantiles)
                    temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池
                    hold_stock[q] = temp_stock_list

        for q in range(quantiles):
            temp_rts_daily = rts.loc[date_2][hold_stock[q]]
            if weight == 'avg':  # 每天收益率均值
                quantile_df[q].loc[date_2] = temp_rts_daily.mean()

    quantile_df.iloc[::hold_time] = quantile_df.iloc[::hold_time] - comm_fee
    quantile_df = quantile_df.fillna(0)

    for q in range(quantiles):
        net_value = (1 + quantile_df[q]).cumprod()
        plt.plot(net_value, color=color_list[q], label='Quantile{}'.format(q))

    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    if plot_title == False:
        plt.title('quantiles={}\n持股时间={}'.format(quantiles, hold_time))
    else :
        plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title, quantiles, hold_time))

    return quantile_df


# def quantile_factor_test_plot_open(factor, close,open, benchmark_rts, quantiles, buy_date_list, hold_time_title,
#                                    plot_title=False, comm_fee=0.003):
#
#     open_close_rts = open/(close.shift(1)) - 1
#     close_open_rts = close/open - 1
#     factor_date = list(factor.index)
#     is_date_buy = pd.DataFrame(0,columns=['is_buy'],index=factor_date)
#     is_date_buy['is_buy'].loc[buy_date_list] = 1
#
#     factor = factor.dropna(how='all', axis=1)
#
#     plt.figure(figsize=(9, 9))
#     plt.plot((1 + benchmark_rts[factor.index]).cumprod(), color='black', label='benchmark_net_value')
#
#     quantile_df = pd.DataFrame(np.nan, index=factor.index, columns=range(quantiles))
#
#     hold_stock = {}  # 每天开盘价买入
#     for q in range(quantiles):
#         hold_stock[q] = {}
#
#     # 每天开盘买入名单
#     temp_stock_list = []
#
#     for q in range(quantiles):
#         hold_stock[q][factor_date[0]] = []  # 第一天开盘无买入
#
#     for i in range(len(factor_date)-1):
#         date = factor_date[i]  # t时刻
#         date_1 = factor_date[i + 1]  # t+1时刻
#
#         tmp_ii = is_date_buy['is_buy'].loc[date]
#
#         if tmp_ii == 1:  # 换仓日
#             temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
#             stock_number = len(temp_factor)
#             if len(temp_factor) > 0:  # 若无股票，则持仓不变
#                 for q in range(quantiles):
#                     start_i = q * int(stock_number / quantiles)
#                     if q == quantiles - 1:  # 最后一层
#                         end_i = stock_number
#                     else:
#                         end_i = (q + 1) * int(stock_number / quantiles)
#                     temp_stock_list = list(temp_factor.index[start_i:end_i])  # 下一个buy_list_date持仓股票
#                     hold_stock[q][date_1] = temp_stock_list  # t+1，当天以开盘价买入temp_stock_list
#         else:
#             hold_stock[q][date_1] = temp_stock_list
#
#     print ('buy_stocks_added')
#
#     # 收益率
#     for i in range(1,len(factor_date)):
#         date_yes = factor_date[i-1]  # 昨天
#         date = factor_date[i]  # t时刻
#
#         for q in range(quantiles):
#             stock_list_yes = hold_stock[q][date_yes]
#             stock_list = hold_stock[q][date]
#
#             tmp_open_rts = open_close_rts.loc[date][stock_list_yes].mean()  # 昨日持仓 今天开盘卖出收益
#             tmp_close_rts = close_open_rts.loc[date][stock_list].mean()
#             tmp_rts = tmp_open_rts + tmp_close_rts - comm_fee*2
#             quantile_df[q].loc[date] = tmp_rts
#
#     print ('quantile rts calculated')
#     quantile_df = quantile_df.fillna(0)
#
#     for q in range(quantiles):
#         net_value = (1 + quantile_df[q]).cumprod()
#         plt.plot(net_value, color=color_list[q], label='Quantile{}'.format(q))
#
#     plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
#     if plot_title == False:
#         plt.title('quantiles={}\n持股时间={}'.format(quantiles, hold_time_title))
#     else :
#         plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title, quantiles, hold_time_title))
#
#     return quantile_df




