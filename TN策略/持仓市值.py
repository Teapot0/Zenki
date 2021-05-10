import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqfactor_analyzer import analyze_factor
from alphalens.utils import get_clean_factor_and_forward_returns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from basic_funcs.basic_function import *

import warnings
warnings.filterwarnings('ignore')


share = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/share.xlsx', time='2018-01-01')
high = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/high.xlsx', time='2018-01-01')
Open = read_excel_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.xlsx', time='2018-01-01')
Open = Open[share.columns][Open.index <= '2021-04-16'] # 只留下北向资金股票池
close = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/price.xlsx', time='2018-01-01')
hold_value = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/hold_value.xlsx', time='2018-01-01')
market_cap = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/market_cap.xlsx', time='2018-01-01')
low = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/low.xlsx', time='2018-01-01')
high_limit = read_excel_select('/Users/caichaohong/Desktop/Zenki/南北向资金/涨停价.xlsx', time='2018-01-01')
# hold value ffill补齐NA值,否则get_top会出=NA
hold_value = hold_value.fillna(method='ffill')

close_open_df=pd.concat([close,Open],keys=range(2)).groupby(level=1) #

shangying = (high - close_open_df.max()) / (close_open_df.max())
xiaying = (close_open_df.min()-low) /  close_open_df.min()

shangying_fb =0 # 反包因子




# 去掉一字涨停价
all_new_stock = []
for i in tqdm(range(1, close.shape[0])):
    tmp_close = close.iloc[i,]
    yesterday_close = close.iloc[i-1,]
    new_stock = list(set(tmp_close.dropna().index).difference(set(yesterday_close.dropna().index)))  # 每天新股名单
    for ss in new_stock:
        if close.iloc[i,][ss] == high_limit.iloc[i, ][ss]:
            close.iloc[i,][ss] = np.nan  # 第一天上市等于涨停价则去掉
    # 加上第一天上市涨停的
    all_new_stock = list(set(all_new_stock).union(set(close.iloc[i,][new_stock][close.iloc[i,][new_stock] == high_limit.iloc[i,][new_stock]].index)))
    # 去掉开板的
    new_stock_kai = list(low.iloc[i, ][all_new_stock][low.iloc[i, ][all_new_stock] != high_limit.iloc[i, ][all_new_stock]].index)
    # 所有未开板新股
    new_stock_not_kai = list(set(all_new_stock).difference(set(new_stock_kai)))
    # 未开板新股去掉
    close.iloc[i, ][new_stock_not_kai] = np.nan

price_rts = close.pct_change(1)  # 每日收益

# money = money*10**(-8)
hold_value_growth = hold_value.pct_change(1)

hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['ma_110'] = hs300['close'].rolling(110).mean()


# 其他因子挖掘：

ma5 = close.rolling(5).mean()
ma20 = close.rolling(20).mean()
ma60 = close.rolling(60).mean()
ma_order = (ma5>ma20) & (ma20>ma60)




#  按因子值每日平均收益
value_rts, holds = get_top_value_factor_rts(factor=hold_value, rts=price_rts, top_number=10, hold_time=30,
                               weight='avg', return_holdings_list=True)


# value_rts = get_signal_factor_rts(factor=, rts=price_rts, hold_time=3)

comm_fee = 0.003
hold_time = 30
out_df = pd.DataFrame(index=close.index)
# out_df['holds'] = holds['holdings']
out_df['rts'] = value_rts
out_df['rts'][::hold_time] = out_df['rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动

# 则时添加：
# out_df['ma_100'] = ((hs300['close'] > hs300['ma_110']).rolling(3).sum() == 3)*1 # 连续3天才交易，否则空仓
# out_df['rts'][out_df['ma_100']!=1] = 0
# out_df['rts'].fillna(0)  # 未交易日，收益率为0，也不扣手续费

out_df['net_value'] = (1 + out_df['rts']).cumprod()

na_num = out_df['net_value'].isna().sum()
position_max_draw = list(np.zeros(na_num))
out_df['nv_max_draw'] = position_max_draw + list(MaxDrawdown(list(out_df['net_value'].dropna())).reshape(-1))
out_df['benchmark_rts'] = hs300['close'].pct_change(1)
out_df['benchmark_net_value'] = (1+out_df['benchmark_rts']).cumprod()

plot_hold_position(data=out_df, risk_free_rate=0.03)



# model = LinearRegression()
# model.fit(X=hold_value_growth, y=price_rts)
# hold_value_growth_rts_res = 0

params = get_params_out(top_number_list=[5,10,20,30,50,100,200],
                              hold_time_list=[1,2,3,5,10,20,30,60],
                              factor_df = hold_value, rts_df=price_rts)

params=params.sort_values(by='annual_rts',ascending=False)

params.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/params/hold_value_growth_params.xlsx')








