# 克隆自聚宽文章：https://www.joinquant.com/post/28737
# 标题：研报复现 | 遗传规划
# 作者：龙174

# 导入函数库
import jqlib.technical_analysis as tech
import datetime as dt
import pandas as pd
import statsmodels.api as sm
from jqdata import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import warnings


# 获取收益率函数
def get_ret(codes, start, end, unit):
    res = []
    global unit_days
    count = int(len(get_trade_days(start_date=start, end_date=end)) / unit_days[unit])
    for code in codes:
        tmp = get_bars(code, count, unit=unit, include_now=True, df=True).set_index('date')
        tmp = tmp.close.pct_change().rename(code)
        res.append(tmp)
    ret = pd.concat(res, axis=1)
    return ret


# ATR指标
def get_atr(codes, dates, timeperiod=14, unit='1d'):
    import jqlib.technical_analysis as tech
    res = []
    for date in dates:
        MTR, ATR = tech.ATR(codes, date, timeperiod=timeperiod, unit=unit, include_now=True)
        tmp = pd.Series(ATR, name=date)
        res.append(tmp)
    atr = pd.concat(res, axis=1).T
    return atr


# macd指标
def get_macd(codes, dates, SHORT=12, LONG=26, MID=9, unit='1d'):
    res = []
    for date in dates:
        dif, dea, macd = tech.MACD(codes, date, SHORT=SHORT, LONG=LONG, MID=MID, unit=unit, include_now=True)
        tmp = pd.Series(macd, name=date)
        res.append(tmp)
    macd = pd.concat(res, axis=1).T
    return macd


# 移动均线指标
def get_emaDiff(codes, dates, fast=12, slow=50, unit='1d'):
    res = []
    for date in dates:
        EXPMA_fast = pd.Series(tech.EXPMA(codes, check_date=date, timeperiod=fast, unit=unit, include_now=True),
                               name=date)
        EXPMA_slow = pd.Series(tech.EXPMA(codes, check_date=date, timeperiod=slow, unit=unit, include_now=True),
                               name=date)
        diff = EXPMA_fast - EXPMA_slow
        res.append(diff)
    emaDiff = pd.concat(res, axis=1).T
    return emaDiff


# 获取gain因子
def get_gain(codes, period, start, end, unit='1d'):
    res = []
    global unit_days
    count = int(len(get_trade_days(start_date=start, end_date=end)) / unit_days[unit])
    for code in codes:
        tmp = get_price(code, start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False,
                        fq='pre')
        close = tmp.close.rename(code)
        volume = tmp.volume.rename(code)
        perGain = pd.Series(index=close.index, name=code)

        for i in range(period, len(close)):
            perClose = close.iloc[i - period:i].copy()
            perClose = (close.iloc[i] - perClose) / close.iloc[i]

            perVolume = volume.iloc[i - period:i].copy()
            perVolume = perVolume / sum(perVolume)
            weight = pd.Series(1, index=perVolume.index)
            for j in range(len(perVolume) - 1):
                tmp = 1 - perVolume.iloc[j + 1:len(perVolume)].copy()
                weight.iloc[j] = tmp.prod()
            weight = weight * perVolume
            weight = weight / sum(weight)

            perGain.iloc[i] = sum(perClose * weight)

        res.append(perGain)

    gain = pd.concat(res, axis=1)
    return gain


# 获取RSRS因子
def get_rsrs(codes, period, window, start, end, unit='1d'):
    res = []
    global unit_days
    count = int(len(get_trade_days(start_date=start, end_date=end)) / unit_days[unit])
    for code in codes:
        tmp = get_price(code, start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False,
                        fq='pre')
        beta = pd.Series(index=tmp.index, name=code)
        rsquare = pd.Series(index=tmp.index, name=code)
        high = tmp.high.copy()
        low = tmp.low.copy()

        for i in range(period, len(high)):
            tmpHigh = high.iloc[i - period + 1:i + 1].copy()
            tmpLow = low.iloc[i - period + 1:i + 1].copy()
            if (sum(pd.isnull(tmpHigh)) + sum(pd.isnull(tmpLow))) > 0:
                continue
            x = sm.add_constant(tmpLow)
            model = sm.OLS(tmpHigh, x)
            results = model.fit()
            beta.iloc[i] = results.params.low
            rsquare.iloc[i] = results.rsquared

        mean = beta.rolling(window=window).mean()
        std = beta.rolling(window=window).std()
        beta_std = (beta - mean) / std
        right = beta_std * beta * rsquare

        res.append(right)
    rsrs = pd.concat(res, axis=1)
    return rsrs


# 获取交易日
def get_tradeDates(start, end, unit='1d'):
    global unit_days
    tmp = get_price('000300.XSHG', start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False,
                    fq='pre')['close']
    return list(tmp.index)


# 基础函数
def __rolling_rank(data):
    value = rankdata(data)[-1]
    return value


def __rolling_prod(data):
    return np.prod(data)


d = 10
a = 1


# 自定义函数，make_function 函数群
def rank(data):
    value = pd.Series(data).rank()
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def correlation(data1, data2):
    global d
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            x1 = pd.Series(data1)
            x2 = pd.Series(data2)
            df = pd.concat([x1, x2], axis=1)
            temp = pd.Series(dtype='float64')
            for i in range(len(df)):
                if i <= d - 2:
                    temp[str(i)] = np.nan
                else:
                    df2 = df.iloc[i - d + 1:i + 1, :]
                    temp[str(i)] = df2.corr('spearman').iloc[1, 0]
            return np.nan_to_num(temp)
        except:
            return np.zeros(data1.shape[0])


def covariance(data1, data2):
    global d
    x1 = pd.Series(data1).reset_index(drop=True)
    x2 = pd.Series(data2).reset_index(drop=True)
    df = pd.concat([x1, x2], axis=1)
    temp = pd.Series(dtype='float64')
    for i in range(len(df)):
        if i <= d - 2:
            temp[str(i)] = np.nan
        else:
            df2 = df.iloc[i - d + 1:i + 1, :]
            temp[str(i)] = df2.cov().iloc[1, 0]
    return np.nan_to_num(temp)


def scale(data):
    global a
    value = data.mul(a).div(np.abs(data).sum())
    return np.nan_to_num(value)


def delta(data):
    global d
    value = pd.Series(data)
    delay = pd.Series(data).shift(d)
    return np.nan_to_num(value - delay)


def signedpower(data):
    global a
    value = pd.Series(data)
    value = np.sign(value) * abs(value) ** a
    return np.nan_to_num(value)


def ts_min(data):
    global d
    value = np.array(data.rolling(d).min().tolist())
    return np.nan_to_num(value)


def ts_max(data):
    global d
    value = np.array(pd.Series(data).rolling(d).max().tolist())
    return np.nan_to_num(value)


def ts_rank(data):
    global d
    value = pd.Series(data).rolling(d).apply(__rolling_rank)
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def ts_argmin(data):
    global d
    value = pd.Series(data).rolling(d).apply(np.argmin) + 1
    return np.nan_to_num(value)


def ts_argmax(data):
    global d
    value = pd.Series(data).rolling(d).apply(np.argmax) + 1
    return np.nan_to_num(value)


def ts_sum(data):
    global d
    value = pd.Series(data).rolling(d).sum()
    value = np.array(value.tolist())
    return np.nan_to_num(value)


def alpha1(index_data, codes, tradeDates):
    atr = index_data['atr']
    close = index_data['close']
    score_table = pd.DataFrame(index=tradeDates, columns=codes)
    for code in codes:
        pclose = close[code]
        patr = atr[code]
        score = correlation(pclose, patr)
        score_table[code] = score
    return score_table


def alpha2(index_data, codes, tradeDates):
    rsrs = index_data['rsrs']
    close = index_data['close']
    score_table = pd.DataFrame(index=tradeDates, columns=codes)
    for code in codes:
        pclose = close[code]
        prsrs = rsrs[code]
        score = ts_argmin(prsrs - pclose)
        score_table[code] = score
    return score_table


def alpha3(index_data, codes, tradeDates):
    volume = index_data['volume']
    score_table = pd.DataFrame(index=tradeDates, columns=codes)
    for code in codes:
        pvolume = volume[code]
        score = abs(ts_argmin(ts_sum(pvolume)))
        score_table[code] = score
    return score_table


def alpha4(index_data, codes, tradeDates):
    money = index_data['money']
    score_table = pd.DataFrame(index=tradeDates, columns=codes)
    for code in codes:
        pmoney = money[code]
        score = 1 / (delta(ts_argmin(pmoney)) + 1)
        score_table[code] = score
        score_table.replace(np.inf, 0)
    return score_table


# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')
    warnings.filterwarnings('ignore')
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
    # 开盘时运行
    run_daily(market_open, time='open', reference_security='000300.XSHG')


## 开盘前运行函数
def before_market_open(context):
    # 获取数据
    days = 250  # 获取数据的天数
    end = context.current_dt - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days)

    global unit_days
    unit_days = {'1m': 1 / 240, '5m': 1 / 48, '15m': 1 / 16, '30m': 1 / 8, '60m': 1 / 4, '120m': 1 / 2, '1d': 1,
                 '1w': 5, '1M': 20}
    # 如果数据频率不是一天，需要对获取数据的个数进行变换，这个字典就是用来执行这个变换的，比较难表达可以忽略这个字典
    unit = '1d'  # 数据频率

    codes = ['510300.XSHG']

    ohlc = get_price(codes, start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False, fq='pre')
    _open = ohlc['open'].pct_change()
    high = ohlc['high'].pct_change()
    low = ohlc['low'].pct_change()
    close = ohlc['close'].pct_change()
    volume = ohlc['volume'].pct_change()
    money = ohlc['money'].pct_change()
    ret = _open
    atr = get_atr(codes=codes, dates=ret.index, unit=unit)
    macd = get_macd(codes=codes, dates=ret.index, unit=unit)
    emaDiff = get_emaDiff(codes=codes, dates=ret.index, unit=unit)
    gain = get_gain(codes, 60, start, end, unit=unit)
    rsrs = get_rsrs(codes, 18, 60, start, end, unit=unit)

    # 取近期数据
    days = 100
    end = context.current_dt - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days)
    tradeDates = get_tradeDates(start, end)

    index_data = {}
    tech_indexs = ['ret', 'atr', 'macd', 'emaDiff', 'gain', 'rsrs', '_open', 'high', 'low', 'close', 'volume', 'money']

    scaler = StandardScaler()
    for name in tech_indexs:
        table = eval(name).loc[tradeDates]
        table = pd.DataFrame(scaler.fit_transform(table), index=table.index, columns=table.columns)
        index_data[name] = table

    alpha = pd.DataFrame(columns=['alpha' + str(i) for i in range(1, 5)], index=tradeDates)
    for col in alpha.columns:
        alpha[col] = eval(col)(index_data, codes, tradeDates)
        mean = alpha[col].mean()
        std = alpha[col].std()
        alpha[col].apply(lambda x: x if abs(x - mean) < 3 * std else np.sign(x) * std + mean)
    alpha = sm.add_constant(alpha)

    x_train = alpha.iloc[31:-1, :].reset_index(drop=True)
    x_train = x_train.replace(np.inf, 0)
    ret = ret.loc[tradeDates]
    y_train = ret.iloc[32:, :].reset_index(drop=True)

    LR = LogisticRegression()
    LR.fit(x_train, y_train > 0)
    x_test = np.array(alpha.iloc[-1, :].replace(inf, 0)).reshape(1, -1)
    g.signal = LR.predict_proba(x_test)[0][1]
    # 输出运行时间
    log.info('数据取到(before_market_open)：' + str(close.index[-1]))
    # 给微信发送消息（添加模拟交易，并绑定微信生效）
    # send_message('美好的一天~')
    log.info('预测值：' + str(LR.predict_proba(x_test)))
    # 要操作的股票：平安银行（g.为全局变量）
    g.security = '159915.XSHE'


## 开盘时运行函数
def market_open(context):
    log.info('函数运行时间(market_open):' + str(context.current_dt.time()))
    security = g.security

    cash = context.portfolio.available_cash

    if g.signal > 0.4 and cash > 0:
        order_value(security, cash)
        log.info("Buying %s" % (security))
    elif g.signal < 0.4 and context.portfolio.positions[security].closeable_amount > 0:
        order_target(security, 0)
        # 记录这次卖出
        log.info("Selling %s" % (security))


