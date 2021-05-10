import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_all_securities,get_query_count, get_price, opt, query, get_fundamentals, finance, income, balance, cash_flow, indicator, valuation
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os


auth('15951961478', '961478')
get_query_count()


stock_list = get_all_securities()

years = ['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']

rev= pd.DataFrame(index=stock_list, columns=years)
rev_growth = pd.DataFrame(index=stock_list, columns=years)
np = pd.DataFrame(index=stock_list, columns=years)
np_growth = pd.DataFrame(index=stock_list, columns=years)
pe = pd.DataFrame(index=stock_list, columns=years)


for i in tqdm(range(len(stock_list))):
    ret = [get_fundamentals(query(indicator.statDate,
                                  income.np_parent_company_owners,
                                  income.total_operating_revenue,
                                  indicator.inc_total_revenue_year_on_year,
                                  indicator.inc_net_profit_to_shareholders_year_on_year,
                                  valuation.pe_ratio).filter(income.code == stock_list[i]),
                           statDate=y) for y in years]
    temp = pd.DataFrame()
    for rr in ret:
        temp = pd.concat([temp, rr], join='outer')
    temp.index = [x.split('-')[0] for x in temp['statDate']]
    rev.loc[stock_list[i]][temp.index] = temp['total_operating_revenue'].values * 10**(-8)
    rev_growth.loc[stock_list[i]][temp.index] = temp['inc_total_revenue_year_on_year'].values
    np.loc[stock_list[i]][temp.index] = temp['np_parent_company_owners'].values*10**(-8)
    np_growth.loc[stock_list[i]][temp.index] = temp['inc_net_profit_to_shareholders_year_on_year'].values
    pe.loc[stock_list[i]][temp.index] = temp['pe_ratio'].values

with pd.ExcelWriter('/Users/caichaohong/Desktop/Zenki/4.20_financials_raw.xlsx') as writer:
    rev.to_excel(writer, sheet_name='营收')
    rev_growth.to_excel(writer, sheet_name='营收增速')
    np.to_excel(writer, sheet_name='净利润')
    np_growth.to_excel(writer, sheet_name='净利润增速')
    pe.to_excel(writer, sheet_name='pe')
    writer.save()
    writer.close()




