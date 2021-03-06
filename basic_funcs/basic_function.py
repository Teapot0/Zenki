import numpy as np
from numpy import sqrt, pi, e
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqdatasdk import get_industries, get_industry_stocks,get_extras, finance, query, bond
from sklearn.linear_model import LinearRegression
import talib

color_list = ['grey', 'rosybrown', 'saddlebrown', 'orange', 'goldenrod',
              'olive', 'yellow', 'darkolivegreen', 'lime', 'lightseagreen',
              'cyan', 'cadetblue', 'deepskyblue', 'steelblue', 'lightslategrey',
              'navy', 'slateblue', 'darkviolet', 'thistle', 'orchid',
              'deeppink', 'lightpink']


def read_csv_select(path, start_time=False, end_time=False, stock_list=False):
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv(path,index_col='Unnamed: 0')
    df = df[(df.index>=start_time) & (df.index<=end_time)]
    if stock_list == False:
        return df
    else:
        return df[stock_list]


def read_excel_select(path, start_date, end_date, stocks=False):
    temp_df = pd.read_excel(path, index_col='Unnamed: 0')
    temp_df = temp_df[(temp_df.index >= start_date) & (temp_df.index<= end_date)]
    if stocks == False:
        return temp_df
    else:
        return temp_df[stocks]


def normal_pdf(x):
    return (1 / sqrt(2 * pi)) * e^((-x ^ 2) / 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def vwap(price, volume):
    return price.div(volume,axis=0)


def mad(factor, n=3 * 1.4826):  # n=几倍的mad
    """3倍中位数去极值"""

    # 求出因子值的中位数
    median = np.median(factor)

    # 求出因子值与中位数的差值, 进行绝对值
    mad = np.median(abs(factor - median))

    # 定义几倍的中位数上下限
    high = median + (n * mad)
    low = median - (n * mad)

    # 替换上下限
    factor = np.where(factor > high, high, factor)
    factor = np.where(factor < low, low, factor)
    return factor


def quantile_drop(series, min, max):
    q = series.quantile([min, max])
    return np.clip(series, q.iloc[0], q.iloc[1])


def MaxDrawdown(return_list):  # 必须是净值的list，返回array
    return (np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)


def plot_hold_position(data, risk_free_rate=0.04):
    # data_prepare must have net_value, rts , and benchmark的net_value
    df = data.copy(deep=True)
    df['year'] = [x.year for x in df.index]
    N = len(df['year'][df['year'] == 2020])  # 每年多少天，估算一共多少年
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ax1.plot(df['net_value'], 'black', label='port_net_value')
    ax1.plot(df['benchmark_net_value'], 'blue', label='benchmark_net_value')
    ax1.plot((1 + df['rts'] - df['benchmark_rts']).cumprod(), 'gold', label='cumulative alpha')  # 画超额收益 Alpha
    ax2.plot(df['nv_max_draw'], 'red', linestyle='-.',linewidth=1, label='port_max_draw')
    ax1.legend()
    ax2.legend()
    annual_rts = df['net_value'].values[-1] ** (1 / (round(df.shape[0] / N, 2))) - 1
    plt.title('years_={} Max_Drawdown={} \n total_rts={} annualized rts ={}\n Sharpe={}'.format(
        round(df.shape[0] / N, 2),
        np.round(MaxDrawdown(list(df['net_value'].dropna())).max(),4),
        np.round(df['net_value'].values[-1],2),
        np.round(annual_rts,4),
        ((df['rts'] - risk_free_rate/N).mean() / df['rts'].std()) * np.sqrt(N) ))
    plt.show()


def plot_rts(value_rts, benchmark_df,comm_fee=0.003, hold_time=1):
    # bencnmark rts is series or column of df for the rts of close，必须第一个是NA
    # valur rts = daily rts series
    out_df = pd.DataFrame()
    NA_num = value_rts.isna().sum()  # 只留一个NA 用于归net-value to 1
    out_df['rts'] = value_rts.iloc[NA_num - 1:, ]
    out_df['rts'][::hold_time] = out_df['rts'][::hold_time] - comm_fee
    out_df['net_value'] = (1 + out_df['rts'].fillna(0)).cumprod()  # 应该只有第一个NA

    out_df['nv_max_draw'] = list(MaxDrawdown(list(out_df['net_value'])).reshape(-1))
    out_df['benchmark_rts'] = benchmark_df.loc[out_df.index]['close'].pct_change(1)
    out_df['benchmark_net_value'] = (1 + out_df['benchmark_rts'].fillna(0)).cumprod()
    plot_hold_position(data=out_df, risk_free_rate=0.04)


def get_top_value_factor_rts(factor, rts, top_number=10, hold_time=3, weight="avg", return_holdings_list=False):
    # factor的值是numbers 不是signal；
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # 日收益率
    holdings = pd.DataFrame(np.nan, index=factor.index, columns=['holdings'])

    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    if NA_rows != 1:
        print('na_rows Not one')
    out.iloc[:NA_rows, ] = np.nan
    holdings.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓
        date = factor.index[i]
        date_1 = factor.index[i + 1]

        temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
        if temp_ii == 0:
            temp_factor = factor.loc[date].dropna().sort_values(ascending=False) # 每天从大到小
            temp_stock_list = list(temp_factor.index[:top_number])  # 未来hold_time的股票池

        temp_rts_daily = rts.loc[date_1][temp_stock_list]
        holdings.loc[date_1] = str(temp_stock_list)

        if weight == 'avg':  # 每天收益率均值
            out.loc[date_1] = temp_rts_daily.mean()

    if return_holdings_list == False:
        return out
    else:
        return out, holdings


def get_top_value_factor_rts_open(factor, close,open, top_number=10, hold_time=3, weight="avg", return_holdings_list=False):
    # 用开盘价作为买入价
    # factor的值是numbers 不是signal；
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    rts = close.pct_change(1)
    # 去掉每天开盘价影响
    rts_open = close/open - 1
    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # 日收益率
    holdings = pd.DataFrame(np.nan, index=factor.index, columns=['holdings'])

    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    if NA_rows != 1:
        print('na_rows Not one')
    out.iloc[:NA_rows, ] = np.nan
    holdings.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓
        date = factor.index[i]
        date_1 = factor.index[i + 1]

        temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
        if temp_ii == 0:
            temp_factor = factor.loc[date].dropna().sort_values(ascending=False)  # 每天从大到小
            temp_stock_list = list(temp_factor.index[:top_number])  # 未来hold_time的股票池

        temp_rts_daily = rts_open.loc[date_1][temp_stock_list]
        holdings.loc[date_1] = str(temp_stock_list)

        if weight == 'avg':  # 每天收益率均值
            out.loc[date_1] = temp_rts_daily.mean()

    if return_holdings_list == False:
        return out
    else:
        return out, holdings


def get_signal_factor_rts(factor, rts, hold_time=3, weight="avg", return_holdings_list=False):
    # factor的值是1，0;当期值代表当期收盘
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # 日收益率
    holdings = pd.DataFrame(np.nan, index=factor.index, columns=['holdings'])

    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    if NA_rows != 1:
        print('na_rows Not one')
    out.iloc[:NA_rows, ] = np.nan
    holdings.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓
        date = factor.index[i]
        date_1 = factor.index[i + 1]

        temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
        if temp_ii == 0:
            temp_factor = factor.loc[date].dropna()  # 每天从大到小
            temp_stock_list = list(temp_factor[temp_factor == 1].index)  # 未来hold_time的股票池

        temp_rts_daily = rts.loc[date_1][temp_stock_list]
        holdings.loc[date_1] = str(temp_stock_list)

        if weight == 'avg':  # 每天收益率均值
            out.loc[date_1] = temp_rts_daily.mean()

    if return_holdings_list == False:
        return out
    else:
        return out, holdings


def get_top_factor_rts_lowhighmean(factor, price, high, low, top_number=10, hold_time=3, weight="avg",
                                   return_holdings_list=False):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel
    # 用每天最高最低均价作为成本价

    hold_cost = (high + low) * 0.5
    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # 收益率的list
    holdings = pd.DataFrame(np.nan, index=factor.index, columns=['holdings'])

    temp_stock_list = []

    for i in tqdm(range(1, len(factor.index) - 1)):  # 第一天没有收益，每hold_time换仓
        date = factor.index[i]
        date_1 = factor.index[i + 1]

        temp_ii = (i - 1) % hold_time  # 判断是否换仓
        if temp_ii == 0:
            temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
            temp_stock_list = list(temp_factor.index[:top_number])  # 未来hold_time的股票池
            temp_hold_cost = hold_cost.loc[date][temp_stock_list]

        temp_rts_daily = (price.loc[date_1][temp_stock_list] / temp_hold_cost) - 1
        holdings.loc[date_1] = str(temp_stock_list)

        if weight == 'avg':  # 每天收益率均值
            out.loc[date_1] = temp_rts_daily.mean()

    if return_holdings_list == False:
        return out
    else:
        return out, holdings


def get_params_out(top_number_list, hold_time_list, factor_df, rts_df, comm_fee=0.003):
    # factor_df 和 rts_df（price rts） 必须是same时间index, stocks为columns
    out = pd.DataFrame()
    annual_rts_list = []
    sharpe_list = []
    index_list = []

    # 不同持股数，不同持仓周期的表现
    for n in tqdm(top_number_list):
        for t in hold_time_list:
            value_rts = get_top_value_factor_rts(factor=factor_df, rts=rts_df, top_number=n, hold_time=t, weight='avg')
            temp = pd.DataFrame(index=factor_df.index)
            temp['rts'] = value_rts
            temp['rts'][::t] = temp['rts'][::t] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
            temp['rts'].fillna(0)  # 未交易日，收益率为0，也不扣手续费
            temp['net_value'] = (1 + temp['rts']).cumprod()

            temp['year'] = [x.year for x in temp.index]
            n_one_year = len(temp['year'][temp['year'] == 2019])  # 每年多少天，估算一共多少年
            annual_rts = temp['net_value'].values[-1] ** (1 / (round(temp.shape[0] / (n_one_year), 2))) - 1
            sharpe = (annual_rts - 0.03) / (np.std(temp['rts']) * np.sqrt(n_one_year))
            annual_rts_list.append(annual_rts)
            sharpe_list.append(sharpe)
            index_list.append("n={}_t={}".format(n, t))
    out['annual_rts'] = annual_rts_list
    out['sharpe'] = sharpe_list
    out.index = index_list
    return out


def quantile_factor_test_plot(factor, rts, benchmark_rts, quantiles, hold_time, plot_title=False, weight="avg",
                              comm_fee=0.003):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA

    plt.figure(figsize=(9, 9))
    plt.plot((1 + benchmark_rts[factor.index]).cumprod(), color='black', label='benchmark_net_value')

    quantile_df = pd.DataFrame(np.nan,index = factor.index, columns=range(quantiles))
    hold_stock = {}
    for q in range(quantiles):
        hold_stock[q] = []

    for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓

        date = factor.index[i]
        date_1 = factor.index[i + 1]

        temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
        if temp_ii == 0:  # 换仓日
            temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
            stock_number = len(temp_factor)  # 一直都na去掉
            if len(temp_factor) > 0:  # 若无股票，则持仓不变
                for q in range(quantiles):
                    start_i = q * int(stock_number / quantiles)
                    if q == quantiles - 1:  # 最后一层
                        end_i = stock_number
                    else:
                        end_i = (q + 1) * int(stock_number / quantiles)
                    temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池
                    hold_stock[q] = temp_stock_list

        for q in range(quantiles):
            temp_rts_daily = rts.loc[date_1][hold_stock[q]]
            if weight == 'avg':  # 每天收益率均值
                quantile_df[q].loc[date_1] = temp_rts_daily.mean()

    quantile_df.iloc[::hold_time] = quantile_df.iloc[::hold_time] - comm_fee
    quantile_df = quantile_df.fillna(0)

    for q in range(quantiles):
        net_value = (1 + quantile_df[q]).cumprod()
        plt.plot(net_value, color=color_list[q], label='Quantile{}'.format(q))

    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    if plot_title == False:
        plt.title('quantiles={}\n持股时间={}'.format(quantiles, hold_time))
    else :
        plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title, quantiles, hold_time))

    return quantile_df


def weekly_quantile_factor_test_plot(factor, rts, benchmark_rts, quantiles, buy_date_list,plot_title=False, weight="avg",
                              comm_fee=0.003):
    # factor是time index, stocks columns的df
    # top number is the number of top biggest factor values each day
    # Factor and rts must have same time index, default type is timestamp from read_excel

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # daily 收益率的df
    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    out.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    stock_number = factor.dropna(axis=1, how='all').shape[1] # 一直都na去掉

    plt.figure(figsize=(9, 9))
    plt.plot((1 + benchmark_rts[factor.index]).cumprod().values, color='black', label='benchmark_net_value')

    for q in tqdm(range(quantiles)):
        # 默认q取0-9，共10层
        start_i = q * int(stock_number / quantiles)
        if q == quantiles - 1:  # 最后一层
            end_i = stock_number
        else:
            end_i = (q + 1) * int(stock_number / quantiles)

        for i in range(NA_rows, len(factor.index) - 1):  # 每hold_time换仓

            date = factor.index[i]
            date_1 = factor.index[i + 1]

            if date in buy_date_list:  # 换仓日
                temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
                if len(temp_factor) > 0:  # 若无股票，则持仓不变
                    temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池
            temp_rts_daily = rts.loc[date_1][temp_stock_list]

            if weight == 'avg':  # 每天收益率均值
                out['daily_rts'].loc[date_1] = temp_rts_daily.mean()

        out['daily_rts'].loc[buy_date_list] = out['daily_rts'].loc[buy_date_list] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
        out['daily_rts'] = out['daily_rts'].fillna(0)
        out['net_value'] = (1 + out['daily_rts']).cumprod()
        plt.plot(out['net_value'].values, color=color_list[q], label='Quantile{}'.format(q))
    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    if plot_title == False:
        plt.title('quantiles={}'.format(quantiles,))
    else :
        plt.title('{}\nquantiles={}'.format(plot_title, quantiles))

    return out



def resample_data_weekly(df, df_type):
    # period W 是周
    # df_type, close,open,high,low

    week_datelist = []
    out_week = []
    temp_start = 0  # 新的一周从i开始, 主要算第一周的open

    for i in tqdm(range(1, df.shape[0])):
        date_delta = df.index[i] - df.index[i - 1]  # 每两个交易日时间差

        if date_delta.days != 1:  # 时间差不为1，则换周

            temp_friday = df.index[i - 1]  # 不一定是周五
            week_datelist.append(temp_friday)

            if df_type == 'close':
                out_week.append(df.iloc[i - 1, :].values)
            if df_type == 'open':
                out_week.append(df.iloc[temp_start, :].values)
                temp_start = i  # 相当于每周一的i
            if df_type == 'high':
                out_week.append(df.iloc[temp_start:i, :].max(axis=0))
            if df_type == 'low':
                out_week.append(df.iloc[temp_start:i, :].min(axis=0))

        if i == df.shape[0] - 1:  # 最后一行如果是周五，则加上
            if df.index[-1].date().weekday() == 4:
                week_datelist.append(df.index[-1])
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



def clean_close(close_df, low_df, high_limit_df):
    # 去掉新股上市未开板的一字无量涨停板
    all_new_stock = []

    # 每天上市新股
    new_stocks = {}
    for i in tqdm(range(1, close_df.shape[0])):
        date = close_df.index[i]
        yesterday_date = close_df.index[i - 1]
        new_stocks[date] = list(
            set(close_df.loc[date].dropna().index).difference(set(close_df.loc[yesterday_date].dropna().index)))

    # 上市第一天涨停，44% 最低价不等于涨停价
    for i in tqdm(range(1, close_df.shape[0])):
        tmp_date = close_df.index[i]
        tmp_close = close_df.iloc[i,]
        tmp_high_limit = high_limit_df.iloc[i,]
        tmp_new_stock = new_stocks[tmp_date]  # 每天新股名单
        first_day_zt = list(tmp_close[tmp_new_stock][tmp_close[tmp_new_stock] == tmp_high_limit[tmp_new_stock]].index)
        close_df.loc[tmp_date][first_day_zt] = np.nan

    for i in tqdm(range(1, close_df.shape[0])):
        tmp_date = close_df.index[i]
        tmp_new_stock = new_stocks[tmp_date]  # 每天新股名单

        all_new_stock = list(set(all_new_stock).union(set(tmp_new_stock)))

        # 去掉开板的
        kaiban = list(low_df.iloc[i,][all_new_stock][
                          low_df.iloc[i,][all_new_stock] != high_limit_df.iloc[i,][all_new_stock]].index)
        all_new_stock = list(set(all_new_stock).difference(kaiban))

        # 未开板新股去掉
        close_df.iloc[i,][all_new_stock] = np.nan

    return close_df


def clean_st_exit(close):
    # 去掉 ST的
    is_st = get_extras('is_st', list(close.columns), end_date='2021-04-23', count=1)
    not_st = (is_st.loc['2021-04-23'] == False)
    not_st_list = list(not_st[not_st == True].index)
    df_not_st = close[not_st_list]
    
    is_exit = close.iloc[-1].isna()
    exit_list = list(is_exit[is_exit == True].index)  # 退市名单
    df_not_st_not_exit = df_not_st[sorted(list(set(df_not_st.columns).difference(set(exit_list))))]
    return df_not_st_not_exit


def zt_yesterday(close, high_limit):
    close_rts = close.pct_change(1)
    out = pd.DataFrame(index=close.index, columns=['zt_yesterday'])
    for i in range(close.shape[0]-1):
        date = close.index[i]
        temp = close.loc[date]
        temp_zt = temp[temp == high_limit.loc[date]].index
        out.loc[close.index[i+1]] = round(close_rts.loc[close.index[i+1]][temp_zt].mean(),3)
    return out


def transform_rts_to_daily_intervals(rts):
    out = (rts >= 0.09) * 7 + ((rts > 0.06) & (rts < 0.09)) * 6 + ((rts >= 0.03) & (rts <= 0.06)) * 5 + (
            (rts < 0.03) & (rts > -0.03)) * 4 + ((rts >= -0.06) & (rts <= -0.03)) * 3 + (
                      (rts > -0.09) & (rts < -0.06)) * 2 + (rts <= -0.09) * 1
    return out


def transform_300_rts_to_daily_intervals(rts):
    out = (rts >= 0.02) * 7 + ((rts > 0.01) & (rts < 0.02)) * 6 + ((rts >= 0.005) & (rts <= 0.01)) * 5 + (
            (rts < 0.005) & (rts > -0.005)) * 4 + ((rts >= -0.01) & (rts <= -0.005)) * 3 + (
                      (rts > -0.02) & (rts < -0.01)) * 2 + (rts <= -0.02) * 1
    return out



def get_rsrs(high,low,N,n):
    rsrs=pd.DataFrame(index=high.index, columns=['rsrs', 'rsrs_mu','rsrs_std', 'signal'])
    for i in range(n,len(high)):
        tmp_high = high[i-n:i].dropna()
        tmp_low = low[i-n:i].dropna()

        if len(tmp_high)==0:
            rsrs['rsrs'].loc[high.index[i]] = np.nan
        else :
            m = LinearRegression()
            m.fit(X=tmp_high.values.reshape(-1,1), y=tmp_low.values.reshape(-1,1))
            rsrs['rsrs'].loc[high.index[i]] = m.coef_[0,0]
    rsrs['rsrs_mu'] = rsrs['rsrs'].rolling(N).mean()
    rsrs['rsrs_std'] = rsrs['rsrs'].rolling(N).std()
    # rsrs_high = rsrs['rsrs_mu'] + rsrs['rsrs_std']
    # rsrs_low = rsrs['rsrs_mu'] - rsrs['rsrs_std']
    # rsrs['signal'] = (rsrs['rsrs'] > rsrs_high)*1 + (rsrs['rsrs'] < rsrs_low)*-1
    rsrs['signal'] = (rsrs['rsrs'] - rsrs['rsrs_mu']) / rsrs['rsrs_std']
    return rsrs


def get_rsi_df(close,n):
    #close is dataframe of all stock, not series
    out = pd.DataFrame(columns=close.columns, index = close.index)
    for s in tqdm(close.columns):
        out[s] = talib.RSI(close[s], timeperiod=n)
    return out


def get_rsrs(high,low,N,n):
    rsrs=pd.DataFrame(index=high.index, columns=['rsrs', 'rsrs_mu','rsrs_std', 'signal'])
    for i in range(n,len(high)):
        tmp_high = high[i-n:i].dropna()
        tmp_low = low[i-n:i].dropna()

        if len(tmp_high)==0:
            rsrs['rsrs'].loc[high.index[i]] = np.nan
        else :
            m = LinearRegression()
            m.fit(X=tmp_high.values.reshape(-1,1), y=tmp_low.values.reshape(-1,1))
            rsrs['rsrs'].loc[high.index[i]] = m.coef_[0,0]
    rsrs['rsrs_mu'] = rsrs['rsrs'].rolling(N).mean()
    rsrs['rsrs_std'] = rsrs['rsrs'].rolling(N).std()
    # rsrs_high = rsrs['rsrs_mu'] + rsrs['rsrs_std']
    # rsrs_low = rsrs['rsrs_mu'] - rsrs['rsrs_std']
    # rsrs['signal'] = (rsrs['rsrs'] > rsrs_high)*1 + (rsrs['rsrs'] < rsrs_low)*-1
    rsrs['signal'] = (rsrs['rsrs'] - rsrs['rsrs_mu']) / rsrs['rsrs_std']
    return rsrs


def get_standard_df(df,n=False):
    if n ==False:
        return (df - df.mean())/df.std()
    else:
        df_n = df.rolling(n).mean()
        df_std = df.rolling(n).std()
        return (df-df_n)/df_std


def get_rsi_df(close,n):
    #close is dataframe of all stock, not series
    out = pd.DataFrame(columns=close.columns, index = close.index)
    for s in tqdm(close.columns):
        out[s] = talib.RSI(close[s], timeperiod=n)
    return out


def get_rsrs(high,low,N,n):
    rsrs=pd.DataFrame(index=high.index, columns=['rsrs', 'rsrs_mu','rsrs_std', 'signal'])
    for i in range(n,len(high)):
        tmp_high = high[i-n:i].dropna()
        tmp_low = low[i-n:i].dropna()

        if len(tmp_high)==0:
            rsrs['rsrs'].loc[high.index[i]] = np.nan
        else :
            m = LinearRegression()
            m.fit(X=tmp_high.values.reshape(-1,1), y=tmp_low.values.reshape(-1,1))
            rsrs['rsrs'].loc[high.index[i]] = m.coef_[0,0]
    rsrs['rsrs_mu'] = rsrs['rsrs'].rolling(N).mean()
    rsrs['rsrs_std'] = rsrs['rsrs'].rolling(N).std()
    # rsrs_high = rsrs['rsrs_mu'] + rsrs['rsrs_std']
    # rsrs_low = rsrs['rsrs_mu'] - rsrs['rsrs_std']
    # rsrs['signal'] = (rsrs['rsrs'] > rsrs_high)*1 + (rsrs['rsrs'] < rsrs_low)*-1
    rsrs['signal'] = (rsrs['rsrs'] - rsrs['rsrs_mu']) / rsrs['rsrs_std']
    return rsrs



def get_short_ma_order(close, n1, n2, n3):
    ma1 = close.rolling(n1).mean()
    ma2 = close.rolling(n2).mean()
    ma3 = close.rolling(n3).mean()
    return (ma1 < ma2) & (ma2 < ma3) & (ma1 < ma3)


def get_close_ma_stock(close,n1,n2,ma_n):
    ma_stock = {}
    close_ma1 = close.rolling(n1).mean()
    close_ma2 = close.rolling(n2).mean()
    close_ma_order = ((close_ma1 > close_ma2) * 1).rolling(ma_n).sum()
    for i in range(close.shape[0]):
        date = close.index[i]
        # 均线连续n日多
        ma_stock[date] = list(close_ma_order.loc[date][close_ma_order.loc[date] == ma_n].index)
    return ma_stock


def get_financial_stock_list(market_cap,roe_5,pe,money,roe_mean,mc_min, pe_min,money_min):
    stock_list_panel = {}

    for i in range(market_cap.shape[0]):
        date = market_cap.index[i]
        # 5年平均roe大于12%
        tmp_year = '{}-12-31'.format(market_cap.index[i].year - 1)
        roe_list = list((roe_5.loc[tmp_year][roe_5.loc[tmp_year] > roe_mean]).index)
        # 市值大于100 pe大于25
        mc_100 = list(market_cap.iloc[i, :][market_cap.iloc[i, :] > mc_min].index)
        pe_25 = list(pe.iloc[i, :][pe.iloc[i, :] > pe_min].index)
        # 成交额大于1000万
        money_list = list(money.iloc[i, :][money.iloc[i, :] > money_min].index)
        tmp_list = list(set(roe_list).intersection(set(mc_100), set(pe_25), set(money_list)))
        stock_list_panel[date] = tmp_list
    return stock_list_panel


def get_std_list(close_rts_1,std_n_list,std_list):
    std_stock = {}
    df_list = [close_rts_1.rolling(std_n).std() * sqrt(std_n) for std_n in std_n_list]
    for date in close_rts_1.index:
        tmp = [list(df_list[i].loc[date][df_list[i].loc[date] < std_list[i]].index) for i in range(len(std_list))]
        std_stock[date] = list(set.intersection(*map(set,tmp))) # 每一天股票池,map(set,a) simply converts it to a list of sets
    return std_stock


def get_alpha_list(close,hs300,rts_n_list,rts_list):
    rts_stock = {}
    df_list = [close.pct_change(rts_n).sub(hs300['close'].pct_change(rts_n), axis=0) for rts_n in rts_n_list]

    for date in close.index:  # 去掉NA
        tmp =[list(df_list[i].loc[date][df_list[i].loc[date] > rts_list[i]].index) for i in range(len(rts_n_list))]
        rts_stock[date] = list(set.intersection(*map(set,tmp)))
    return rts_stock


def get_rps(close,rps_n):
    min = close.rolling(rps_n).min()
    max = close.rolling(rps_n).max()
    rps = (close-min)/(max-min) * 100
    return rps


def get_down_list(close,hs300,bench_rts=0.005,down_rts=-0.02):
    down_list = {}
    tmp_down_list = []
    close_rts_1 = close.pct_change(1)
    for i in tqdm(range(close.shape[0])):
        if i == close.shape[0]-1:
            down_list[close.index[-1]] = []
        else:
            date = close.index[i]
            date1 = close.index[i+1]
            tmp_week = date.week
            week1 = date1.week
            if hs300['rts_1'].loc[date] > bench_rts:
                cp = list(close_rts_1.loc[date][close_rts_1.loc[date] <= down_rts].index)
                tmp_down_list = tmp_down_list + cp
                down_list[date] = cp
            else:
                down_list[date] = []

            if tmp_week != week1:
                down_list[date] = tmp_down_list
                tmp_down_list = []
    return down_list


def get_licha():
    df_bank = finance.run_query(query(finance.SW1_DAILY_VALUATION).filter(finance.SW1_DAILY_VALUATION.code == '801780'))
    # 回购
    df_bond = bond.run_query(query(bond.REPO_DAILY_PRICE).filter(bond.REPO_DAILY_PRICE.name == 'GC182'))
    df_t1 = pd.merge(df_bond, df_bank, on='date')
    df_t1 = df_t1[['date', 'close', 'dividend_ratio']]
    df_t1.index = pd.to_datetime(df_t1['date'])
    return df_t1


def get_ic_table(factor, rts, buy_date_list):
    # buy list 是换仓日
    out = pd.DataFrame(np.nan,index=buy_date_list,columns=['ic', 'rank_ic'])
    for i in tqdm(range(1,len(buy_date_list))):
        date = buy_date_list[i]
        date1 = buy_date_list[i-1]
        tmp = pd.concat([factor.loc[date1],rts.loc[date]],axis=1)
        tmp.columns=['date1', 'date']
        out['ic'].loc[date] = tmp.dropna().corr().iloc[0,1]
        out['rank_ic'].loc[date] = tmp.rank().corr().iloc[0,1]
    return out


def get_single_ic_table(factor, rts):
    out = pd.DataFrame(np.nan,index=factor.index,columns=['ic'])
    for i in tqdm(range(1,factor.shape[0])):
        date = factor.index[i]
        date1 = factor.index[i-1]
        tmp = pd.concat([factor.loc[date1],rts.loc[date]],axis=1)
        tmp.columns=['date1', 'date']
        out['ic'].loc[date] = tmp.dropna().corr().iloc[0,1]
    return out






