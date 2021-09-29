import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from basic_funcs.basic_funcs_open import *
from jqdatasdk import bond,query,attribute_history_engine

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def convert_premium_rate(context): #计算昨日收盘转债溢价率，从2018-09-13开始
    '''查债券收盘价'''
    df=bond.run_query(query(bond.CONBOND_DAILY_PRICE).filter(bond.CONBOND_DAILY_PRICE.code==context))
    df.sort_values(by='date',ascending=False,inplace=True)  #按照日期降序排序
    dfa=df.dropna() #删除有空值的行，剔除停牌债券
    dfa.reset_index(inplace=True,drop=True) #重置索引
    close=dfa.loc[0,'close'] #提取元素
    '''查最新转股价'''
    df2=bond.run_query(query(bond.CONBOND_CONVERT_PRICE_ADJUST).filter(bond.CONBOND_CONVERT_PRICE_ADJUST.code==context))
    if df2.empty: #判断是否为空表
        convert_price_now=1000000 #这个表存在BUG，权宜之计
    else:
        df2.sort_values(by='adjust_date',ascending=False,inplace=True) #按照日期降序排序
        df2.reset_index(inplace=True,drop=True) #重置索引
        convert_price_now=df2.loc[0,'new_convert_price']
    '''查正股代码'''
    df3=bond.run_query(query(bond.CONBOND_BASIC_INFO).filter(bond.CONBOND_BASIC_INFO.code==context))
    stock_code=df3.loc[0,'company_code'] #提取元素
    '''查正股收盘价'''
    df4=attribute_history_engine(stock_code, 1, unit='1d',
            fields=['close'],
            skip_paused=True, df=True, fq='pre')
    df4.reset_index(inplace=True,drop=True)
    stock_close=df4.loc[0,'close'] #提取元素 loc只能通过index和columns来取,若不重设索引则不能这么提取
    '''计算溢价率'''
    convert_value=(100/convert_price_now)*stock_close #转股价值=（面值/转股价）*正股价
    convert_premium_rate=(close-convert_value)/convert_value #转股溢价率=(转债价-转股价值)/转股价值
    return convert_premium_rate

print(convert_premium_rate('123103'))







