import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from basic_funcs.basic_function import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings('ignore')

out_0 = pd.read_csv('/Users/caichaohong/Desktop/舆情/sentimentfactor0.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_1 = pd.read_csv('/Users/caichaohong/Desktop/舆情/sentimentfactor1.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_2 = pd.read_csv('/Users/caichaohong/Desktop/舆情/sentimentfactor2.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_0.index = pd.to_datetime(out_0.index)
out_1.index = pd.to_datetime(out_1.index)
out_2.index = pd.to_datetime(out_2.index)


data1 = pd.read_excel('/Users/caichaohong/Desktop/舆情/data_1.xlsx',converters={'股票代码':str})
data2 = pd.read_excel('/Users/caichaohong/Desktop/舆情/data_2.xlsx',converters={'股票代码':str})
data3 = pd.read_excel('/Users/caichaohong/Desktop/舆情/data_3.xlsx',converters={'股票代码':str})



def transform_miaodong(df_list):
    date_list = df_list[0]['日期'].unique()
    df_0 = pd.DataFrame(index=date_list)
    df_1 = pd.DataFrame(index=date_list)
    df_2 = pd.DataFrame(index=date_list)
    df_all = pd.DataFrame(index=date_list)

    for df in df_list:
        tmp_stock = list(df['股票代码'].unique())
        for s in tqdm(tmp_stock):
            df_0[s] = df['0'][df['股票代码']==s].values
            df_1[s] = df['1'][df['股票代码']==s].values
            df_2[s] = df['2'][df['股票代码']==s].values
            df_all[s] = df['all'][df['股票代码']==s].values

    return df_0, df_1, df_2, df_all


tmp_out_0, tmp_out_1, tmp_out_2, tmp_out_3 = transform_miaodong([data1,data2,data3])

out_0.columns = [x.split('.')[0] for x in out_0.columns]
out_1.columns = [x.split('.')[0] for x in out_1.columns]
out_2.columns = [x.split('.')[0] for x in out_2.columns]

inter_stock = list(set(out_0.columns).intersection(set(tmp_out_0.columns)))

# out_0_diff1 = out_0.diff(1)
# out_0_diff5 = out_0.diff(5)
# out_0_diff10 = out_0.diff(10)
#
# out_1_diff1 = out_1.diff(1)
# out_1_diff5 = out_1.diff(5)
# out_1_diff10 = out_1.diff(10)
#
# out_2_diff1 = out_2.diff(1)
# out_2_diff5 = out_2.diff(5)
# out_2_diff10 = out_2.diff(10)

new_out_0 = pd.concat([out_0[inter_stock], tmp_out_0[inter_stock]])
new_out_1 = pd.concat([out_1[inter_stock], tmp_out_1[inter_stock]])
new_out_2 = pd.concat([out_2[inter_stock], tmp_out_2[inter_stock]])

# 大盘数据
hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

# 行情数据
close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
volume = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv', index_col='Unnamed: 0', date_parser=dateparse)


factor = (out_2-out_1).diff(1)
# 每天停牌的
pause_list = volume.apply(lambda x: list(x[x == 0].index), axis=1)
# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)

# 去掉ST，停牌
stock_exclude_list = {}
for date in tqdm(close.index):
    stock_exclude_list[date] = list(set(pause_list.loc[date]).union(set(st_df.loc[date])))


for date in tqdm(factor.index):
    ex_list = list(set(stock_exclude_list).intersection(factor.columns))
    factor.loc[date][ex_list] = np.nan


close_rts = close.pct_change(1)
close_rts.columns = [x.split('.')[0] for x in close_rts.columns]

z = quantile_factor_test_plot(factor = factor.loc[close_rts.index], rts=close_rts, benchmark_rts=hs300['rts'], quantiles=10,
                             hold_time=30, plot_title=False, weight="avg",comm_fee=0.003)




