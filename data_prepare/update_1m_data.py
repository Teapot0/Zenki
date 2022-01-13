import glob
import os
import joblib
from joblib import Parallel,delayed
import pandas as pd
import time
from tqdm import tqdm
from jqdatasdk import get_price,auth,get_query_count

# update 1m data
old_end_date = '2021-06-30'
New_end_date = '2021-10-29'


save_path = './data/1m_data/'


for stock_id in tqdm(stock_ids[975:1500]):
    # st_tm = time.time()
    tmp_df = get_price(security=stock_id, start_date=start_date, end_date=end_date, frequency='1m',
                          fields=fields, fill_paused=False)
    tmp_df.to_parquet(save_path + 'df1m_{}.parquet'.format(stock_id))
    # print(time.time() - st_tm)

