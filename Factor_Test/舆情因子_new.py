import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings('ignore')

color_list = ['grey', 'rosybrown', 'saddlebrown', 'orange', 'goldenrod',
              'olive', 'yellow', 'darkolivegreen', 'lime', 'lightseagreen',
              'cyan', 'cadetblue', 'deepskyblue', 'steelblue', 'lightslategrey',
              'navy', 'slateblue', 'darkviolet', 'thistle', 'orchid',
              'deeppink', 'lightpink']


def quantile_factor_test_plot(factor, rts, benchmark_rts, quantiles, hold_time, plot_title=False, weight="avg",
                              comm_fee=0.003):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # daily 收益率的df
    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    out.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    stock_number = factor.dropna(axis=1, how='all').shape[1] # 一直都na去掉

    plt.figure(figsize=(9, 9))
    plt.plot((1 + benchmark_rts[factor.index]).cumprod(), color='black', label='benchmark_net_value')

    for q in tqdm(range(quantiles)):
        # 默认q取0-9，共10层
        start_i = q * int(stock_number / quantiles)
        if q == quantiles - 1:  # 最后一层
            end_i = stock_number
        else:
            end_i = (q + 1) * int(stock_number / quantiles)

        for i in range(NA_rows, len(factor.index) - 1):  # 每hold_time换仓

            date = factor.index[i]
            date_1 = factor.index[i + 1]

            temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
            if temp_ii == 0:  # 换仓日
                temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
                if len(temp_factor) > 0:  # 若无股票，则持仓不变
                    temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池
            temp_rts_daily = rts.loc[date_1][temp_stock_list]

            if weight == 'avg':  # 每天收益率均值
                out.loc[date_1] = temp_rts_daily.mean()

        out['daily_rts'][::hold_time] = out['daily_rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
        out['daily_rts'] = out['daily_rts'].fillna(0)
        out['net_value'] = (1 + out['daily_rts']).cumprod()
        plt.plot(out['net_value'], color=color_list[q], label='Quantile{}'.format(q))
    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    if plot_title == False:
        plt.title('quantiles={}\n持股时间={}'.format(quantiles, hold_time))
    else :
        plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title, quantiles, hold_time))

    return out

out_0 = pd.read_csv('~/Desktop/舆情/sentimentfactor0.csv', index_col='Unnamed: 0')
out_1 = pd.read_csv('~/Desktop/舆情/sentimentfactor1.csv', index_col='Unnamed: 0')
out_2 = pd.read_csv('~/Desktop/舆情/sentimentfactor2.csv', index_col='Unnamed: 0')

# 大盘数据
hs300 = pd.read_excel('~/Desktop/舆情/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# 行情数据
close = pd.read_csv('~/Desktop/舆情/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('~/Desktop/舆情/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)


factor = (out_2-out_1).diff(1)
factor.index = pd.to_datetime(factor.index)
# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('~/Desktop/舆情/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)

# 去掉ST，停牌
stock_exclude_list = {}
for date in tqdm(close.index):
    stock_exclude_list[date] = list(set(pause_list.loc[date]).union(set(st_df.loc[date])))


for date in tqdm(factor.index):
    ex_list = list(set(stock_exclude_list).intersection(factor.columns))
    factor.loc[date][ex_list] = np.nan


close_rts = close.pct_change(1)
close_rts.columns = [x.split('.')[0] for x in close_rts.columns]

test_dates = list(sorted(set(factor.index).intersection(set(close_rts.index))))
z = quantile_factor_test_plot(factor = factor.loc[test_dates], rts=close_rts.loc[test_dates], benchmark_rts=hs300['rts'].loc[test_dates],
                              quantiles=10, hold_time=30, plot_title=False, weight="avg",comm_fee=0.003)




