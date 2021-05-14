import pandas as pd
import numpy as np
import jqdatasdk as jq
from jqdatasdk import auth, get_query_count, get_price, opt, query, get_fundamentals, finance, get_trade_days, \
    valuation, get_security_info
from tqdm import tqdm
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import os
from basic_funcs.basic_function import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler

auth('15951961478', '961478')
get_query_count()

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

def transform_rts_to_daily_intervals(rts):
    out = (rts >= 0.09) * 3 + ((rts > 0.06) & (rts < 0.09)) * 2 + ((rts >= 0.03) & (rts <= 0.06)) * 1 + (
            (rts < 0.03) & (rts > -0.03)) * 0 + ((rts >= -0.06) & (rts <= -0.03)) * -1 + (
                      (rts > -0.09) & (rts < -0.06)) * -2 + (rts <= -0.09) * -3
    return out

# def divide_quantiles(df):
#     # 7层
#     x1 = np.nanquantile(df,1/7)
#     x2 = np.nanquantile(df,2/7)
#     x3 = np.nanquantile(df,3/7)
#     x4 = np.nanquantile(df,4/7)
#     x5 = np.nanquantile(df,5/7)
#     x6 = np.nanquantile(df,6/7)
#
#     out = (df >= x0) * 3 + ((df > ) & (rts < 0.09)) * 2 + ((rts >= 0.03) & (rts <= 0.06)) * 1 + (
#             (rts < 0.03) & (rts > -0.03)) * 0 + ((rts >= -0.06) & (rts <= -0.03)) * -1 + (
#                       (rts > -0.09) & (rts < -0.06)) * -2 + (rts <= -0.09) * -3
#
#     return out



hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts_1'] = hs300['close'].pct_change(1)
hs300['rts_5'] = hs300['close'].pct_change(5)
hs300['rts_10'] = hs300['close'].pct_change(10)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

share = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/share.csv', index_col='Unnamed: 0', date_parser=dateparse)
ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/ratio.csv', index_col='Unnamed: 0', date_parser=dateparse)
value = pd.read_csv('/Users/caichaohong/Desktop/Zenki/南北向资金/value.csv', index_col='Unnamed: 0', date_parser=dateparse)

top_10_net_buy = pd.read_excel('/Users/caichaohong/Desktop/Zenki/南北向资金/TOP_10_net_buy.xlsx', index_col='Unnamed: 0')

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
high = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv', index_col='Unnamed: 0',
                   date_parser=dateparse)
low = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv', index_col='Unnamed: 0', date_parser=dateparse)
high_limit = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv', index_col='Unnamed: 0',
                         date_parser=dateparse)
close = clean_close(close, low, high_limit)  # 新股一字板
close = clean_st_exit(close)  # 退市和ST
close_rts_1 = transform_rts_to_daily_intervals(close.pct_change(1))
close_rts_5 = close_rts_1.rolling(5).sum()
close_rts_10 = close_rts_1.rolling(10).sum()

out_0 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor0.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_1 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor1.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_2 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor2.csv', index_col='Unnamed: 0',
                    date_parser=dateparse)
out_012 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/沪深300秒懂舆情因子/sentimentfactor012.csv', index_col='Unnamed: 0',
                      date_parser=dateparse)

margin_buy_value = pd.read_csv('/Users/caichaohong/Desktop/Zenki/融资融券/margin_buy_value.csv', index_col='Unnamed: 0',
                               date_parser=dateparse)
margin_rts_1 = margin_buy_value.pct_change(1)
margin_rts_5 = margin_buy_value.pct_change(5)
margin_rts_10 = margin_buy_value.pct_change(10)

# 北向资金+融资融券
stock_list = sorted(list(set(close_rts_1.columns).intersection(set(share.columns), set(margin_buy_value.columns))))

share = share[stock_list]
ratio = ratio[stock_list]
value = value[stock_list]
share_rts_1 = share.pct_change(1)
share_rts_5 = share.pct_change(5)
share_rts_10 = share.pct_change(10)

close_rts_north_1 = close_rts_1[stock_list]
close_rts_north_5 = close_rts_5[stock_list]
close_rts_north_10 = close_rts_10[stock_list]

# 不加舆情因子----------------
buy_list = pd.Series(index=close_rts_north_1.index)
daily_rts = pd.Series(index=close_rts_north_1.index)
acc_score = pd.Series(index=close_rts_north_1.index)

for i in tqdm(range(11, share.shape[0] - 1)):
    date = share.index[i]
    date_1 = share.index[i + 1]
    date_yes = share.index[i - 1]

    if i % 5 == 0:  # 持股5天

        def get_factor_north(date, date_tom):
            f1 = pd.DataFrame(index=stock_list)
            f1['close_rts_1'] = close_rts_north_1.loc[date][stock_list].values
            f1['close_rts_5'] = close_rts_north_5.loc[date][stock_list].values
            f1['close_rts_10'] = close_rts_north_10.loc[date][stock_list].values
            f1['excess_rts_1'] = f1['close_rts_1'].values - hs300['rts_1'].loc[date]
            f1['excess_rts_5'] = f1['close_rts_5'].values - hs300['rts_5'].loc[date]
            f1['excess_rts_10'] = f1['close_rts_10'].values - hs300['rts_10'].loc[date]
            f1['share_rts_1'] = share_rts_1.loc[date][stock_list].values
            f1['share_rts_5'] = share_rts_5.loc[date][stock_list].values
            f1['share_rts_10'] = share_rts_10.loc[date][stock_list].values
            f1['value_rts_res_1'] = value.pct_change(1).loc[date][stock_list].values - f1['close_rts_1'] - f1[
                'share_rts_1']
            f1['value_rts_res_5'] = value.pct_change(5).loc[date][stock_list].values - f1['close_rts_5'] - f1[
                'share_rts_5']
            f1['value_rts_res_10'] = value.pct_change(10).loc[date][stock_list].values - f1['close_rts_10'] - f1[
                'share_rts_10']
            f1['margin_buy_rts_1'] = margin_rts_1.loc[date][stock_list].values
            f1['margin_buy_rts_5'] = margin_rts_5.loc[date][stock_list].values
            f1['margin_buy_rts_10'] = margin_rts_10.loc[date][stock_list].values

            f1['is_buy'] = ((close_rts_north_5.loc[date_tom][stock_list].rank() <= int(f1.shape[0] * 0.2)) * 1).values
            return f1


        f1 = get_factor_north(date_yes, date)

        clf = XGBClassifier(base_score=0.5, booster='gbtree', learning_rate=0.05, max_depth=8, n_estimators=50)
        clf.fit(f1.drop(columns=['is_buy']), f1['is_buy'])

        f2 = get_factor_north(date, date_1)
        pred_y = clf.predict(f2.drop(columns=['is_buy']))

        pred_df = pd.DataFrame()
        pred_df['real_y'] = f2['is_buy']
        pred_df['pred_y'] = pred_y
        buy_list.loc[date_1] = str(list(pred_df[pred_df['pred_y'] == 1].index))

    daily_rts.loc[date_1] = close_rts_north_1.loc[date_1][list(pred_df[pred_df['pred_y'] == 1].index)].mean()
    acc_score.loc[date_1] = (pred_df.sum(axis=1) == 2).sum() / pred_y.sum()

plt.plot((1 + daily_rts).cumprod())

'''
# 第一天
date = share.index[21]  # 用昨日数据，预测明天的数据
date_tom = share.index[22]

f1 = pd.DataFrame(index=stock_list)
f1['close_rts_1'] = close_rts_north_1.loc[date][stock_list].values
f1['close_rts_5'] = close_rts_north_5.loc[date][stock_list].values
f1['close_rts_10'] = close_rts_north_10.loc[date][stock_list].values
f1['excess_rts_1'] = f1['close_rts_1'].values - hs300['rts_1'].loc[date]
f1['excess_rts_5'] = f1['close_rts_5'].values - hs300['rts_5'].loc[date]
f1['excess_rts_10'] = f1['close_rts_10'].values - hs300['rts_10'].loc[date]
f1['share_rts_1'] = share_rts_1.loc[date][stock_list].values
f1['share_rts_5'] = share_rts_5.loc[date][stock_list].values
f1['share_rts_10'] = share_rts_10.loc[date][stock_list].values
f1['value_rts_res_1'] = value.pct_change(1).loc[date][stock_list].values - f1['close_rts_1'] - f1['share_rts_1']
f1['value_rts_res_5'] = value.pct_change(5).loc[date][stock_list].values - f1['close_rts_5'] - f1['share_rts_5']
f1['value_rts_res_10'] = value.pct_change(10).loc[date][stock_list].values - f1['close_rts_10'] - f1['share_rts_10']
f1['margin_buy_rts_1'] = margin_rts_1.loc[date][stock_list].values
f1['margin_buy_rts_5'] = margin_rts_5.loc[date][stock_list].values
f1['margin_buy_rts_10'] = margin_rts_10.loc[date][stock_list].values



# top 0.1 rts
f1['is_buy'] = ((close_rts_north_5.loc[date_tom][stock_list].rank() <= int(f1.shape[0]*0.1)) * 1).values

train_x, test_x, train_y, test_y = train_test_split(f1.drop(columns=['is_buy']), f1['is_buy'], test_size=0.3, random_state=42)

clf = XGBClassifier(base_score=0.5, booster='gbtree', learning_rate=0.05, max_depth=8, n_estimators=50)
clf.fit(train_x, train_y)

pred_y = clf.predict(test_x)

pred_df = pd.DataFrame()
pred_df['test_y'] = test_y
pred_df['pred_y'] =pred_y
acc1 = (pred_df.sum(axis=1) ==2).sum()/test_y.sum() # 原买预测正确百分比
acc2 = (pred_df.sum(axis=1) ==2).sum()/pred_y.sum() # 预测正确百分比
'''

fig, ax = plt.subplots(figsize=(8, 8))
plot_importance(clf, height=0.5, ax=ax, max_num_features=64)
plt.show()

# 加入舆情因子，股票数量300--------------------------------------------
sent_date_list = list(set(out_0.index).intersection(set(share.index)))  # 990
sent_stock_list = sorted(list(set(out_0.columns).intersection(stock_list)))  # 201个

hold_n = 10
if hold_n == 5:
    rts_df = close_rts_5
if hold_n == 10:
    rts_df = close_rts_10

hold_percent = 0.2

buy_list = pd.Series(index=close_rts_north_1.index)
daily_rts = pd.Series(index=close_rts_north_1.index)
acc_score = pd.Series(index=close_rts_north_1.index)

for i in tqdm(range(9, len(sent_date_list) - 1)):
    date = sent_date_list[i]
    date_1 = sent_date_list[i + 1]
    date_yes = sent_date_list[i - 1]

    if (i-9) % hold_n == 0:  # 持股N天

        def get_factor_north(date, date_tom):
            f1 = pd.DataFrame(index=sent_stock_list)
            f1['close_rts_1'] = close_rts_1.loc[date][sent_stock_list].values
            f1['close_rts_5'] = close_rts_5.loc[date][sent_stock_list].values
            f1['close_rts_10'] = close_rts_10.loc[date][sent_stock_list].values

            # f1['excess_rts_1'] = f1['close_rts_1'].values - hs300['rts_1'].loc[date]
            # f1['excess_rts_5'] = f1['close_rts_5'].values - hs300['rts_5'].loc[date]
            # f1['excess_rts_10'] = f1['close_rts_10'].values - hs300['rts_10'].loc[date]

            f1['share_rts_1'] = share_rts_1.loc[date][sent_stock_list].values
            f1['share_rts_5'] = share_rts_5.loc[date][sent_stock_list].values
            f1['share_rts_10'] = share_rts_10.loc[date][sent_stock_list].values
            f1['value_rts_res_1'] = value.pct_change(1).loc[date][sent_stock_list].values - f1['close_rts_1'] - f1[
                'share_rts_1']
            f1['value_rts_res_5'] = value.pct_change(5).loc[date][sent_stock_list].values - f1['close_rts_5'] - f1[
                'share_rts_5']
            f1['value_rts_res_10'] = value.pct_change(10).loc[date][sent_stock_list].values - f1['close_rts_10'] - f1[
                'share_rts_10']
            f1['margin_buy_rts_1'] = margin_rts_1.loc[date][sent_stock_list].values
            f1['margin_buy_rts_5'] = margin_rts_5.loc[date][sent_stock_list].values
            f1['margin_buy_rts_10'] = margin_rts_10.loc[date][sent_stock_list].values
            f1['neu_news'] = out_0.loc[date][sent_stock_list].values
            f1['neu_news_rts_1'] = out_0.pct_change(1).loc[date][sent_stock_list].values
            f1['neu_news_rts_5'] = out_0.pct_change(5).loc[date][sent_stock_list].values
            f1['neg_news'] = out_1.loc[date][sent_stock_list].values
            f1['neg_news_rts1'] = out_1.pct_change(1).loc[date][sent_stock_list].values
            f1['neg_news_rts5'] = out_1.pct_change(5).loc[date][sent_stock_list].values
            f1['pos_news'] = out_2.loc[date][sent_stock_list].values
            f1['pos_news_rts1'] = out_2.pct_change(1).loc[date][sent_stock_list].values
            f1['pos_news_rts5'] = out_2.pct_change(5).loc[date][sent_stock_list].values
            f1['is_buy'] = (
                    (rts_df.loc[date_tom][sent_stock_list].rank() <= int(f1.shape[0] * hold_percent)) * 1).values

            return f1

        f1 = get_factor_north(date_yes, date)

        clf = XGBClassifier(base_score=0.5, booster='gbtree', learning_rate=0.05, max_depth=8, n_estimators=50)
        clf.fit(f1.drop(columns=['is_buy']), f1['is_buy'])

        f2 = get_factor_north(date, date_1)
        pred_y = clf.predict(f2.drop(columns=['is_buy']))

        pred_df = pd.DataFrame()
        pred_df['real_y'] = f2['is_buy']
        pred_df['pred_y'] = pred_y
        buy_list.loc[date_1] = str(list(pred_df[pred_df['pred_y'] == 1].index))

    daily_rts.loc[date_1] = close_rts_north_1.loc[date_1][list(pred_df[pred_df['pred_y'] == 1].index)].mean()
    acc_score.loc[date_1] = (pred_df.sum(axis=1) == 2).sum() / pred_y.sum()

plt.plot((1 + daily_rts).cumprod())

plot_rts(daily_rts, comm_fee=0.003, hold_time=5)

code = '600519.XSHG'
ax1 = plt.subplot()
ax11 = ax1.twinx()
ax1.plot(value[code].values, 'red')
ax11.plot(close[code].values)

top_10_net_buy.isna().sum().sort_values(ascending=False)















