import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

close_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/close_1m.csv', index_col='Unnamed: 0')
vol_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/volume_1m.csv', index_col='Unnamed: 0')
low_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/low_1m.csv', index_col='Unnamed: 0')
open_1m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/1m/open_1m.csv', index_col='Unnamed: 0')

close_1m_rts = close_1m.pct_change(1)
stocks = list(close_1m.columns)

start = close_1m.index[0].split(' ')[0]
end = close_1m.index[-1].split(' ')[0]
close_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv' , start_time=start, end_time = end,stock_list=stocks)
open_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv' , start_time= start, end_time = end,stock_list=stocks)
vol_daily = read_csv_select('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv' , start_time=start, end_time = end,stock_list=stocks)
close_rts = close_daily.pct_change(1)
open_rts = open_daily.pct_change(1)


hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

rts_crt = 0.002
c = (vol_1m**0.1)[abs(close_1m_rts) < rts_crt].rolling(240, min_periods=1).std()
tmp_factor1 = c.iloc[239::240,]
tmp_factor1.index = [x.split(' ')[0] for x in tmp_factor1.index]


close_open_rts = close_1m / open_1m -1
s1 = 1 / (abs(close_open_rts) / (((vol_1m)**0.1)))
# factor
date_list = close_daily.index[(close_daily.index>='2021-01-01') & (close_daily.index <= '2021-08-31')]


def smart_money_test(type,interval_n):
    # 10天interval_n = 2399
    factor = pd.DataFrame(index= date_list, columns=close_1m.columns)

    for i in tqdm(range(interval_n,s1.shape[0], 240)):
        tmp_end = s1.index[i]
        tmp_date = tmp_end.split(' ')[0]
        tmp_df = s1.iloc[i-interval_n:i,:]
        temp_vol = vol_1m.iloc[i-interval_n:i,:]
        temp_close = close_1m.iloc[i-interval_n:i, :]

        tmp_rank = tmp_df.rank(axis=0, pct=True)
        smart = pd.DataFrame(tmp_rank.values > 0.7, index=tmp_rank.index, columns=tmp_rank.columns)
        smart_money = smart[smart]
        smart_money = smart_money.fillna(0)

        if type == 'vwap':
            smart_vol = temp_vol * smart_money

            temp_w = temp_vol / temp_vol.sum()
            temp_total_f = (temp_close * temp_w).sum()  #VWAP

            smart_w = smart_vol / smart_vol.sum()
            smart_f = (temp_close * smart_w).sum()

        elif type == 'kurt':
            temp_total_f = temp_close.kurt(axis=0)
            smart_f = (temp_close * smart_money).kurt(axis=0)

        elif type == 'skew':
            temp_total_f = temp_close.skew(axis=0)
            smart_f = (temp_close * smart_money).skew(axis=0)

        elif type == 'mean_standard':
            temp_total_f = (temp_close - temp_close.mean()) / temp_close.std()
            smart_f  = (temp_close*smart_money - (temp_close*smart_money).mean()) / (smart_money*temp_close).std()

        temp_q = round(smart_f / temp_total_f, 4)
        factor.loc[tmp_date][temp_q.index] = temp_q

    factor = factor.dropna(how='all', axis=0)
    factor = factor.astype('float64')
    return factor
# factor.to_csv('/Users/caichaohong/Desktop/Zenki/factors/smart_money.csv')


tmp_factor2 = smart_money_test(type='vwap', interval_n=2399)


vol210 = vol_1m.rolling(210).sum()
q = vol_1m.iloc[239::240]
q.index = [x.split(' ')[0] for x in q.index]
n = vol210.iloc[209::240]
n.index = [x.split(' ')[0] for x in n.index]

tmp_factor3 = q/n

factor = 1/tmp_factor2 * tmp_factor1 * tmp_factor3.rolling(5).mean().rank(axis=1, pct=True, ascending=False)


def ic_test(index_pool,factor):
    date_1 = factor.index
    date_5 = factor.index[::5]
    date_10 = factor.index[::10]

    date_list = [date_1, date_5, date_10]
    print(index_pool + ' : - - - - - - - - - - -')
    for i in range(len(date_list)):
        d = date_list[i]
        z1 = get_ic_table_open_index(factor=factor, open_rts=open_rts, buy_date_list=d, index_pool=index_pool)
        plt.plot(z1['ic'].values)
        plt.title('IC')
        plt.savefig('/Users/caichaohong/Desktop/{}.png'.format(i+1))
        plt.close()

        plt.plot(z1['ic'].cumsum().values)
        plt.title('IC_CUMSUM')
        plt.savefig('/Users/caichaohong/Desktop/{}_CUMSUM.png'.format(i + 1))
        plt.close()

        print ('{}: IC={}, IC_STD={}'.format(i,z1['ic'].mean(), z1['ic'].std()))


ic_test(index_pool='hs300', factor=factor)
ic_test(index_pool='zz500', factor=factor)
ic_test(index_pool='zz1000', factor=factor)

z = quantile_factor_test_plot_open_index(factor=factor, open_rts=open_rts, benchmark_rts=hs300['rts'], quantiles=10,
                                         hold_time=1, plot_title=False, weight="avg",index_pool='hs300', comm_fee=0.003)






