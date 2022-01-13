import glob
import os
import joblib
from joblib import Parallel,delayed
import pandas as pd
import time
from tqdm import tqdm
from jqdatasdk import get_price,auth,get_query_count

import warnings
warnings.filterwarnings("ignore")

file_list = glob.glob('./data/30m_data/*.parquet')

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
close_daily = close_daily.loc[(close_daily.index >= '2018-01-01') & (close_daily.index < '2021-06-30')]
close_daily.index = pd.to_datetime(close_daily.index)


st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')
st_df = st_df.loc[(st_df.index >= '2018-01-01') & (st_df.index < '2021-06-30')]
st_sum = st_df.sum()
all_st_stock = list(st_sum[st_sum == st_df.shape[0]].index)


# 上市大于N天的,一共1816天
ipo_days = close_daily.shape[0] - close_daily.isna().sum()
tmp_stock_list = list(ipo_days[ipo_days >= 255].index) # 大于一年的

data_list = [x for x in file_list if (x.split('.parquet')[0]).split('df30m_')[1] not in all_st_stock]
data_list = [x for x in data_list if (x.split('.parquet')[0]).split('df30m_')[1] in tmp_stock_list]


close_30m = pd.DataFrame()
open_30m = pd.DataFrame()
high_30m = pd.DataFrame()
low_30m = pd.DataFrame()
volume_30m = pd.DataFrame()
money_30m = pd.DataFrame()

for path in tqdm(data_list):
    tmp_code = (path.split('.parquet')[0]).split('df30m_')[1]
    tmp = pd.read_parquet(path)
    # tmp['money'] = tmp['money'] * 10 ** (-4)
    close_30m[tmp_code] = tmp['close']
    open_30m[tmp_code] = tmp['open']
    high_30m[tmp_code] = tmp['high']
    low_30m[tmp_code] = tmp['low']
    volume_30m[tmp_code] = tmp['volume']
    money_30m[tmp_code] = tmp['money']

close_30m.to_parquet('./data/30m_data_agg/close_30m.parquet')
open_30m.to_parquet('./data/30m_data_agg/open_30m.parquet')
high_30m.to_parquet('./data/30m_data_agg/high_30m.parquet')
low_30m.to_parquet('./data/30m_data_agg/low_30m.parquet')
volume_30m.to_parquet('./data/30m_data_agg/volume_30m.parquet')
money_30m.to_parquet('./data/30m_data_agg/money_30m.parquet')


# tmp = pd.read_parquet(data_list[0])
# tmp['money'] = tmp['money'] * 10**(-4)
# close_1m = pd.read_parquet('./data/1m_data_agg/low_1m.parquet')









