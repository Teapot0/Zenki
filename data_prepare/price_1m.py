import glob
import os
import joblib
from joblib import Parallel,delayed
import pandas as pd
import time
from tqdm import tqdm
from jqdatasdk import get_price,auth,get_query_count

auth('13382017213', 'Aasd120120')
get_query_count()

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']
stock_ids = sorted(all_name['code'])

fields = ['open', 'close', 'high', 'low', 'volume','money']

start_date = '2018-01-01'
end_date = '2021-06-30'

save_path = './data/1m_data/'


for stock_id in tqdm(stock_ids[4400:]):
    # st_tm = time.time()
    tmp_df = get_price(security=stock_id, start_date=start_date, end_date=end_date, frequency='1m',
                          fields=fields, fill_paused=False)
    tmp_df.to_parquet(save_path + 'df1m_{}.parquet'.format(stock_id))
    # print(time.time() - st_tm)


