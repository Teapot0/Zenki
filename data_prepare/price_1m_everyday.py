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

file_list = glob.glob('./data/1m_data/*.parquet')

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
close_daily = close_daily.loc[(close_daily.index >= '2018-01-01') & (close_daily.index < '2021-06-30')]
close_daily.index = pd.to_datetime(close_daily.index)


st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')
st_df = st_df.loc[(st_df.index >= '2018-01-01') & (st_df.index < '2021-06-30')]
st_sum = st_df.sum()
all_st_stock = list(st_sum[st_sum == st_df.shape[0]].index)


def single_date_df(path, date):
    zz = path.split('.parquet')[0]
    tmp_stock_id = zz.split('df1m_')[1]
    tmp_start = date + ' 09:30'
    tmp_end = date + ' 15:00'

    df = pd.read_parquet(path)
    tmp_df = df.loc[(df.index >= tmp_start) & (df.index <= tmp_end)].copy(deep=True)
    tmp_df['stock_id'] = tmp_stock_id
    tmp_df['date'] = tmp_df.index
    return tmp_df




date_list = [x.strftime('%Y-%m-%d') for x in close_daily.index]


# 上市大于N天的,一共1816天
ipo_days = close_daily.shape[0] - close_daily.isna().sum()
tmp_stock_list = list(ipo_days[ipo_days >= 255].index) # 大于一年的

data_list = [x for x in file_list if (x.split('.parquet')[0]).split('df1m_')[1] not in all_st_stock]
data_list = [x for x in data_list if (x.split('.parquet')[0]).split('df1m_')[1] in tmp_stock_list]

#
# for d in tqdm(date_list):
#     tmp = Parallel(n_jobs=-1, verbose=1)(delayed(single_date_df)(path,date) for path,date in zip(data_list,[d]*len(data_list)))
#     tmp_df = pd.concat(tmp, ignore_index=True)
#     tmp_close = tmp_df.pivot(index='date', columns='stock_id', values='close')
#     close_1m = close_1m.append(tmp_close)


# names = ['close','open', 'high', 'low','volume', 'money']

close_1m = pd.DataFrame()
for path in tqdm(data_list):
    tmp_code = (path.split('.parquet')[0]).split('df1m_')[1]
    tmp = pd.read_parquet(path)
    # tmp['money'] = tmp['money'] * 10 ** (-4)
    close_1m[tmp_code] = tmp['money']

close_1m.to_parquet('./data/1m_data_agg/money_1m.parquet')


tmp = pd.read_parquet(data_list[0])
tmp['money'] = tmp['money'] * 10**(-4)
close_1m = pd.read_parquet('./data/1m_data_agg/low_1m.parquet')









