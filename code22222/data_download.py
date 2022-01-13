import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info, get_mtss,get_money_flow, get_all_securities
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

# auth('15951961478', '961478')
auth('13382017213', 'Aasd120120')
get_query_count()


# 所有公司名称
all_stocks = get_all_securities(types=['stock'], date=None)
all_stock = finance.run_query(query(finance.STK_COMPANY_INFO.code,
                                    finance.STK_COMPANY_INFO.short_name).filter(
    finance.STK_COMPANY_INFO.code.in_(list(all_stocks.index))))

all_stock.to_excel('./all_stock_names.xlsx')

# 510300.XSHG 等大盘指数沪深300行情数据
New_end_date = '2021-10-08'

index_code = ['510300.XSHG','510050.XSHG', '510500.XSHG','159948.XSHE', '512100.XSHG']
for code in index_code:
    p = get_price(code, start_date='2014-01-01', end_date=New_end_date,
                             fields=['open', 'close', 'high', 'low', 'volume', 'high_limit', 'low_limit'])
    p.to_excel('./{}.xlsx'.format(code))

# ST数据
close = pd.read_csv('./close.csv', index_col='Unnamed: 0')
st_df = get_extras('is_st', list(close.columns), start_date='2014-01-01', end_date=New_end_date)
st_df.to_csv('./is_st.csv')
