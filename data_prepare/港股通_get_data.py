import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, valuation
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os

auth('15951961478', '961478')
get_query_count()

# 目标公司
roe_15_df = pd.read_excel('/Users/caichaohong/Desktop/Zenki/roe_15.xlsx', index_col='Unnamed: 0')

# 市场通标的
cc_df = finance.run_query(
    query(finance.STK_EL_CONST_CHANGE).filter(
        finance.STK_EL_CONST_CHANGE.change_date >= '2010-01-01')
)
cc_df.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/company_list_inout.xlsx')

cc_df = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/company_list_inout.xlsx', index_col='Unnamed: 0')
cc_code_list_north = sorted(
    (cc_df['code'][(cc_df['link_id'] == 310001) | (cc_df['link_id'] == 310002)]).unique())  # 北向资金股票池
cc_code_list_south = sorted(
    (cc_df['code'][(cc_df['link_id'] == 310003) | (cc_df['link_id'] == 310004)]).unique())  # 南向资金股票池

# 成交&额度
id_list = [310001, 310002, 310003, 310004]
# 310001	沪股通
# 310002	深股通
# 310003	港股通（沪）
# 310004	港股通（深）
for link_id in id_list:
    q = query(finance.STK_ML_QUOTA).filter(
        finance.STK_ML_QUOTA.day >= '2015-01-01', finance.STK_ML_QUOTA.link_id == link_id)
    df = finance.run_query(q)
    df = df.drop(columns='id')
    df.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/{}.xlsx'.format(df['link_name'][0]))

# 汇率

rate_df = finance.run_query(
    query(finance.STK_EXCHANGE_LINK_RATE).filter(finance.STK_EXCHANGE_LINK_RATE.day >= '2015-01-01',
                                                 finance.STK_EXCHANGE_LINK_RATE.link_id == '310003')
)

rate_df = rate_df.drop(columns='id')
rate_df.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/exchange_rate.xlsx')

# 持股数据

# hugutong = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/沪股通.xlsx', index_col='Unnamed: 0')
# # 个股持仓的日期list 比 沪股通list日期要多？


hold_df1 = finance.run_query(
    query(finance.STK_HK_HOLD_INFO.day,
          finance.STK_HK_HOLD_INFO.link_id,
          finance.STK_HK_HOLD_INFO.link_name,
          finance.STK_HK_HOLD_INFO.code,
          finance.STK_HK_HOLD_INFO.name,
          finance.STK_HK_HOLD_INFO.share_number,
          finance.STK_HK_HOLD_INFO.share_ratio).filter(finance.STK_HK_HOLD_INFO.code == cc_code_list_north[0])
)
hold_df1.index = hold_df1['day']

hold_df2 = finance.run_query(
    query(finance.STK_HK_HOLD_INFO.day,
          finance.STK_HK_HOLD_INFO.link_id,
          finance.STK_HK_HOLD_INFO.link_name,
          finance.STK_HK_HOLD_INFO.code,
          finance.STK_HK_HOLD_INFO.name,
          finance.STK_HK_HOLD_INFO.share_number,
          finance.STK_HK_HOLD_INFO.share_ratio).filter(finance.STK_HK_HOLD_INFO.code == cc_code_list_north[1])
)
hold_df2.index = hold_df2['day']

out = pd.concat([hold_df1['share_number'], hold_df2['share_number']], axis=1, join='outer')

# share_out = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/hold_share_north.xlsx',index_col='Unnamed: 0')
# ratio_out = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/hold_ratio_north.xlsx',index_col='Unnamed: 0')


# =================================================

start = '2017-03-17'
end = '2021-04-18'
dateList = get_trade_days(start_date=start, end_date=end)

share = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/share.xlsx', index_col='Unnamed: 0')
ratio = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/ratio.xlsx', index_col='Unnamed: 0')
price = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/price.xlsx', index_col='Unnamed: 0')
value = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/value.xlsx', index_col='Unnamed: 0')

for i in tqdm(range(len(dateList))):
    date = dateList[i]

    temp = finance.run_query(query(finance.STK_HK_HOLD_INFO.code,
                                   finance.STK_HK_HOLD_INFO.share_number,
                                   finance.STK_HK_HOLD_INFO.share_ratio).filter(
        finance.STK_HK_HOLD_INFO.code.in_(cc_code_list_north),
        finance.STK_HK_HOLD_INFO.day == date))

    share.loc[date][temp['code']] = temp['share_number'].values
    ratio.loc[date][temp['code']] = temp['share_ratio'].values

share.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/share.xlsx')
ratio.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/ratio.xlsx')

# valuation 市值数据：======================
market_cap = pd.DataFrame(index=dateList, columns=cc_code_list_north)
circulating_market_cap = pd.DataFrame(index=dateList, columns=cc_code_list_north)
pe_ratio = pd.DataFrame(index=dateList, columns=cc_code_list_north)
ps_ratio = pd.DataFrame(index=dateList, columns=cc_code_list_north)

for i in tqdm(range(len(dateList))):
    date = dateList[i]

    temp = get_fundamentals(query(valuation.code, valuation.day, valuation.market_cap, valuation.circulating_market_cap,
                                  valuation.pe_ratio, valuation.ps_ratio).filter(
        valuation.code.in_(cc_code_list_north)), date=dateList[i])

    market_cap.loc[date][temp['code']] = temp['market_cap'].values
    circulating_market_cap.loc[date][temp['code']] = temp['circulating_market_cap'].values
    pe_ratio.loc[date][temp['code']] = temp['pe_ratio'].values
    ps_ratio.loc[date][temp['code']] = temp['ps_ratio'].values

market_cap.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/market_cap.xlsx')
circulating_market_cap.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/circulating_market_cap.xlsx')
pe_ratio.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/pe_ratio.xlsx')
ps_ratio.to_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/ps_ratio.xlsx')


# 日级别改为周级别：
# price_week = price.resample('W',   how={'open': 'first',
#                                         'high': 'max',
#                                         'low': 'min',
#                                         'close': 'last',
#                                         'Volume': 'sum'})


def resample_data_weekly(df, df_type):
    # period W 是周
    # df_type, close,open,high,low

    week_datelist = []
    out_week = []
    temp_start = 0  # 新的一周从i开始, 主要算第一周的open

    for i in tqdm(range(1, df.shape[0])):
        date_delta = df.index[i] - df.index[i-1]  # 每两个交易日时间差

        if date_delta.days != 1:  #时间差不为1，则换周

            temp_friday = df.index[i-1]  # 不一定是周五
            week_datelist.append(temp_friday)

            if df_type == 'close':
                out_week.append(df.iloc[i-1, :].values)
            if df_type == 'open':
                out_week.append(df.iloc[temp_start, :].values)
                temp_start = i  # 相当于每周一的i
            if df_type == 'high':
                out_week.append(df.iloc[temp_start:i, :].max(axis=0))
            if df_type == 'low':
                out_week.append(df.iloc[temp_start:i, :].min(axis=0))

        if i == df.shape[0]:  # 最后一行如果是周五，则加上
            if df.index[-1].date().weekday() == 4:
                if df_type == 'close':
                    out_week.append(df.iloc[i, :].values)
                if df_type == 'open':
                    out_week.append(df.iloc[temp_start, :].values)
                if df_type == 'high':
                    out_week.append(df.iloc[temp_start:i, :].max(axis=0))
                if df_type == 'low':
                    out_week.append(df.iloc[temp_start:i, :].min(axis=0))

    out = pd.DataFrame(columns=df.columns, index=week_datelist)
    for j in range(len(out_week)):
        out.iloc[j, :] = out_week[j]
    return out


price_week = resample_data_weekly(price, df_type='close')






