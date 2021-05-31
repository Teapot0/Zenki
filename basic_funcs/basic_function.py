import numpy as np
from numpy import sqrt, pi, e
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jqdatasdk import get_industries, get_industry_stocks,get_extras

color_list = ['grey', 'rosybrown', 'saddlebrown', 'orange', 'goldenrod',
              'olive', 'yellow', 'darkolivegreen', 'lime', 'lightseagreen',
              'cyan', 'cadetblue', 'deepskyblue', 'steelblue', 'lightslategrey',
              'navy', 'slateblue', 'darkviolet', 'thistle', 'orchid',
              'deeppink', 'lightpink']


def read_csv_select(path, start_time=False, end_time=False, stock_list=False):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv(path,index_col='Unnamed: 0', date_parser=dateparse)
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


def plot_hold_position(data, risk_free_rate=0.03):
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
    annual_rts = df['net_value'].values[-1] ** (1 / (round(df.shape[0] / (N), 2))) - 1
    plt.title('years_={} Max_Drawdown={} \n total_rts={} annualized rts ={}\n Sharpe={}'.format(
        round(df.shape[0] / (N), 2),
        np.round(MaxDrawdown(list(df['net_value'].dropna())).max(),4),
        np.round(df['net_value'].values[-1],2),
        np.round(annual_rts,4),
        (annual_rts - risk_free_rate) / (np.std(df['rts']) * np.sqrt(N))))
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
    plot_hold_position(data=out_df, risk_free_rate=0.03)


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

    out = pd.DataFrame(np.nan, index=factor.index, columns=['daily_rts'])  # daily 收益率的df
    NA_rows = rts.isna().all(axis=1).sum()  # 判断是否只有第一行全NA
    out.iloc[:NA_rows, ] = np.nan
    temp_stock_list = []

    stock_number = factor.dropna(axis=1, how='all').shape[1] # 一直都na去掉

    plt.figure(figsize=(9, 9))
    plt.plot((1 + benchmark_rts).cumprod().values, color='black', label='benchmark_net_value')

    for q in range(quantiles):
        # 默认q取0-9，共10层
        start_i = q * int(stock_number / quantiles)
        if q == quantiles - 1:  # 最后一层
            end_i = stock_number
        else:
            end_i = (q + 1) * int(stock_number / quantiles)

        for i in tqdm(range(NA_rows, len(factor.index) - 1)):  # 每hold_time换仓

            date = factor.index[i]
            date_1 = factor.index[i + 1]

            temp_ii = (i - NA_rows) % hold_time  # 判断是否换仓
            if temp_ii == 0:  # 换仓日
                temp_factor = factor.loc[date].sort_values(ascending=False).dropna()  # 每天从大到小
                if len(temp_factor) > 0:  # 若无股票，则持仓不变
                    temp_stock_list = list(temp_factor.index[start_i:end_i])  # 未来hold_time的股票池
            temp_rts_daily = rts.loc[date_1][temp_stock_list]

            if weight == 'avg':  # 每天收益率均值
                out.loc[date_1] = temp_rts_daily.mean()

        out['daily_rts'][::hold_time] = out['daily_rts'][::hold_time] - comm_fee  # 每隔 hold_time 减去手续费和滑点，其中有一些未交易日，NA值自动不动
        out['daily_rts'] = out['daily_rts'].fillna(0)
        out['net_value'] = (1 + out['daily_rts']).cumprod()
        plt.plot(out['net_value'].values, color=color_list[q], label='Quantile{}'.format(q))
    plt.legend(bbox_to_anchor=(1.015, 0), loc=3, borderaxespad=0, fontsize=7.5)
    if plot_title == False:
        plt.title('quantiles={}\n持股时间={}'.format(quantiles, hold_time))
    else :
        plt.title('{}\nquantiles={}\n持股时间={}'.format(plot_title, quantiles, hold_time))

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
    #去掉新股上市未开板的一字无量涨停板
    all_new_stock = []
    for i in tqdm(range(1, close_df.shape[0])):
        tmp_close = close_df.iloc[i,]
        yesterday_close = close_df.iloc[i - 1,]
        new_stock = list(set(tmp_close.dropna().index).difference(set(yesterday_close.dropna().index)))  # 每天新股名单
        for ss in new_stock:
            if close_df.iloc[i,][ss] == high_limit_df.iloc[i,][ss]:
                close_df.iloc[i,][ss] = np.nan  # 第一天上市等于涨停价则去掉
        # 加上第一天上市涨停的
        all_new_stock = list(set(all_new_stock).union(
            set(close_df.iloc[i,][new_stock][close_df.iloc[i,][new_stock] == high_limit_df.iloc[i,][new_stock]].index)))
        # 去掉开板的
        new_stock_kai = list(low_df.iloc[i,][all_new_stock][low_df.iloc[i,][all_new_stock] != high_limit_df.iloc[i,][all_new_stock]].index)
        # 所有未开板新股
        new_stock_not_kai = list(set(all_new_stock).difference(set(new_stock_kai)))
        # 未开板新股去掉
        close_df.iloc[i,][new_stock_not_kai] = np.nan

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





