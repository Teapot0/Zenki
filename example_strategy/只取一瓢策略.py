# 克隆自聚宽文章：https://www.joinquant.com/post/34394
# 标题：回踩均线搏反弹，21年来70%，无未来（测试第一弹）
# 作者：Mr.翟

# 标题：只取一瓢
# 作者：mrzhai
'''
1.阿波罗11计算股票评分
2.股票筛选
3.止盈止损
'''
# 导入聚宽函数库
import jqdata
import pandas as pd
import talib
import numpy as np
from functools import reduce
from jqlib.technical_analysis import *
from jqlib.alpha101 import *


# 获取近期n天大于value_list的个数
def get_bigger_than_val_counter(close, n, value_list):
    np_close = np.array(close[-n:])
    np_value = np.array(value_list[-n:])
    np_value = np_value * 0.98  # 运行向下浮动2%

    diff = np_close - np_value
    return sum(diff > 0)


# 去掉浮动 获取近期n天大于value_list的个数
def get_bigger_than_val_counter_noflow(close, n, value_list):
    np_close = np.array(close[-n:])
    np_value = np.array(value_list[-n:])

    diff = np_close - np_value
    return sum(diff > 0)


# 均线
def get_ma(close, timeperiod=5):
    return talib.SMA(close, timeperiod)


# 获取均值
def get_avg_price(close, day: int):
    return get_ma(close, day)[-1]


# 获取macd技术线
def get_macd(close):
    diff, dea, macd = talib.MACDEXT(close,
                                    fastperiod=12,
                                    fastmatype=1,
                                    slowperiod=26,
                                    slowmatype=1,
                                    signalperiod=9,
                                    signalmatype=1)
    macd = macd * 2
    return diff, dea, macd


# 获取day最大价格
def get_max_price(high_values, day):
    return high_values[-day:].max()


# 获取最小价格
def get_min_price(low_values, day):
    return low_values[-day:].min()


# 单日成交量 大于该股的前5日所有成交量的2倍
def is_multiple_stocks(volume, days=5):
    vol = volume[-1]

    for i in range(2, days + 2):
        if volume[-i] * 2 > vol:
            return False

    return True


# 获取波动的百分比的标准差
def get_std_percentage(var_list):
    v_array = np.array(var_list)

    # 获取每个单元的涨跌幅百分比
    ratio = (v_array - np.median(v_array)) / np.median(v_array)

    # 每个单元的平方，消除负号
    ratio_p2 = ratio * ratio

    # 得到平均涨跌幅的平方
    ratio = ratio_p2.sum() / len(v_array)

    # 开方 得到平均涨跌幅
    return np.sqrt(ratio)


# 均线多头排列
def get_avg_array(avg_list):
    ratio = 0
    count = len(avg_list)

    for i in range(1, count):
        if avg_list[-i - 1] < avg_list[-i]:
            return 0
        ratio += (avg_list[-i - 1] - avg_list[-i]) / avg_list[-i]

    return ratio


# 给股票评分
def get_stocks_score(tick_list, data_frame, pass_time):
    score = {}
    avg_score = {}

    for code in tick_list:
        high = data_frame[data_frame['code'] == code]['high'].values
        low = data_frame[data_frame['code'] == code]['low'].values
        close = data_frame[data_frame['code'] == code]['close'].values

        # 获取均线矩阵， 不要5日线
        avg_list = np.array(
            [get_ma(close, 13),
             get_ma(close, 21),
             get_ma(close, 55)])

        box_ratio = 0
        for i in range(2, 22):  # 不考虑当日的影响
            box_ratio += get_std_percentage(avg_list.T[-i])

        # print (str(code)+" box " + str(box_ratio))

        array_ratio = 0
        for i in range(1, 6):
            array_ratio += get_avg_array(avg_list.T[-i])
        # print (str(code)+" array " + str(array_ratio))

        avg_score[code] = array_ratio
        score[code] = box_ratio - array_ratio

    tick_list.clear()
    # 降序排列
    score = sorted(score.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    for key in score:
        sec_info = get_security_info(key[0])
        if key[1] > 0:
            print("\t" + str(sec_info.display_name) + ": " + str(key[0]) +
                  " 得分是 " + str(key[1]))
            tick_list.append(key[0])

    return tick_list


# 获取过去180天
def select_ticks(check_date):
    tick_list = get_fundamentals(
        query(valuation.code).filter(
            valuation.turnover_ratio > 3,
            valuation.turnover_ratio <= 10,
        ).order_by(
            # 按换手率降序排列
            valuation.turnover_ratio.desc()),
        date=check_date)['code'].values

    tick_list = list(tick_list)
    print(str(check_date) + " 筛选到股票数量：" + str(len(tick_list)))

    # 获取股票基本信息
    stock_base = get_all_securities(['stock'])

    # 科创、 次新、ST股票不考虑
    filter_list = []
    for code in tick_list:
        # 查看是否有该股票信息
        try:
            if str(code).find('688', 0, 3) != -1 or len(
                    jqdata.get_trade_days(stock_base.loc[code]['start_date'],
                                          check_date)
            ) < 365 or stock_base.loc[code]['display_name'].find(
                'ST') != -1 or stock_base.loc[code]['display_name'].find(
                '退') != -1:
                filter_list.append(code)
        except KeyError:
            filter_list.append(code)
            continue

    for code in filter_list:
        tick_list.remove(code)

    print(str(check_date) + " 过滤掉ST、次新、停牌、涨停、科创后，股票数量：" + str(len(tick_list)))

    if len(tick_list) == 0:
        return tick_list

    # 均线多头
    filter_list = []
    print("获取：" + str(check_date) + "的数据")
    df = get_price(tick_list, count=260, end_date=check_date, panel=False)
    for code in tick_list:
        tdf = df[df['code'] == code]

        # 股票每天基本价格信息
        low = tdf['low'].values
        open = tdf['open'].values
        high = tdf['high'].values
        close = tdf['close'].values
        volume = tdf['volume'].values

        # 均线
        avg21_list = get_ma(close, 21)
        avg55_list = get_ma(close, 55)
        avg120_list = get_ma(close, 120)
        # avg250_list = get_ma(close, 250)

        # 至少连续5天，多头，21>55>120, 股价<250均线，前天回踩55，昨日拉升55
        count = get_bigger_than_val_counter(avg21_list, 6, avg55_list)
        if count < 5:
            filter_list.append(code)
            continue

        count = get_bigger_than_val_counter(avg55_list, 6, avg120_list)
        if count < 5:
            filter_list.append(code)
            continue

        # 至少连续5天，股价高于55日均线
        count = get_bigger_than_val_counter(close, 6, avg55_list)
        if count < 5:
            filter_list.append(code)
            continue

        # 前日是跌回踩55
        if close[-2] > open[-2]:
            filter_list.append(code)
            continue

        # 前日回踩55
        if close[-2] > avg55_list[-2]:
            filter_list.append(code)
            continue

        # 昨日拉升大于55
        if close[-1] < avg55_list[-1]:
            filter_list.append(code)
            continue

        # 55日线呈多头趋势
        if avg55_list[-1] < avg55_list[-2]:
            filter_list.append(code)
            continue

    # 剔除前面不满足条件的股票
    for code in filter_list:
        tick_list.remove(code)

    # 通过阿尔法11进行排序，优质股放在最前面
    tick_list = get_stocks_score(tick_list, df, check_date)
    print(str(check_date) + " 符合条件的股票数量：" + str(len(tick_list)))

    # 只获取前三个
    if len(tick_list) > 3:
        tick_list = tick_list[:3]

    return tick_list


# 初始化代码
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000905.XSHG')
    # True为开启动态复权模式，使用真实价格交易
    set_option('use_real_price', True)
    # 去除未来函数
    set_option("avoid_future_data", True)
    # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, \
                             open_commission=0.0003, close_commission=0.0003, \
                             close_today_commission=0, min_commission=5), type='stock')
    # 设定成交占比，避免价格冲击
    set_option('order_volume_ratio', 0.25)
    # 开盘价成交，理论上无滑点
    set_slippage(FixedSlippage(0))
    # 最大建仓数量
    g.max_hold_stocknum = 3
    # 运行函数
    run_daily(before_trading_start, time='09:00')
    run_daily(trade, time='9:30')


## 获取 购买标的
def before_trading_start(context):
    # 调仓时间
    t_date = context.current_dt.strftime('%Y-%m-%d')
    # 上一个调仓日期
    p_date = context.previous_date.strftime('%Y-%m-%d')
    # 获取目标标的
    g.security = select_ticks(p_date)


## 交易函数
def trade(context):
    log.info("当前持股数： %s" % len(context.portfolio.positions))
    log.info("当前资金量： %s" % (context.portfolio.available_cash))
    # 止盈止损
    stop_loss(context)
    # 是否存在符合条件的股票
    if len(g.security) > 0:
        # 判断当前有持仓股票只有1只
        if len(context.portfolio.positions) == 1:
            # 其余资金买入
            value = context.portfolio.available_cash / 2  # 资金分成两份
            for security in g.security[:2]:
                order_value(security, value, side='long')
                log.info("买入 %s" % (g.security))
        # 判断当前有持仓股票只有2只
        elif len(context.portfolio.positions) == 2:
            # 其余资金买入
            value = context.portfolio.available_cash  # 资金量
            for security in g.security[:1]:
                order_value(security, value, side='long')
                log.info("买入 %s" % (g.security))
        # 判断当前有持仓股票大于2只
        elif len(context.portfolio.positions) > 2:
            log.info("当前已满仓 %s" % (context.portfolio.positions.keys()))
        else:
            value = context.portfolio.available_cash / 3  # 资金分成三份
            for security in g.security:
                order_value(security, value, side='long')
                log.info("买入 %s" % (g.security))


## 止盈止损
# 1.收益小于等于-7%直接平仓
# 2.收益大于15%后，回撤大于5%时平仓；收益大于20%后，回撤大于3%时平仓
# 3.持仓3天后，收益仍然小于5%，直接平仓
# 4.持仓股数小于3时，从股票池买入，直到持股数为3
def stop_loss(context, profit=0.2, lose=0.07):
    for stock in context.portfolio.positions.keys():
        df = get_price(stock, start_date=context.portfolio.positions[stock].init_time, \
                       end_date=context.previous_date, frequency='minute', fields=['high'], skip_paused=True)
        df_max_high = df["high"].max()  # 从买入至今的最高价

        avg_cost = context.portfolio.positions[stock].avg_cost  # 持仓股票的平均成本
        current_price = context.portfolio.positions[stock].price  # 持仓股票的当前价
        hold_day = hold_days(context, stock)  # 计算持股天数

        # 1.收益小于等于-7%直接平仓
        if current_price / avg_cost < (1 - lose):
            log.info(str(stock) + '  达个股止损线，平仓止损！')
            order_target(stock, 0)
            continue

        # 2.收益大于20%后，回撤大于3%时平仓
        if df_max_high / avg_cost > (1 + profit) and (
                df_max_high - current_price) / current_price > 0.03:
            log.info(str(stock) + '  回撤达个股止盈线，平仓止盈！')
            order_target(stock, 0)
            continue

        # 3.收益大于15%后，回撤大于5%时平仓
        if df_max_high / avg_cost > 1.15 and df_max_high / avg_cost <= (
                1 + profit) and (df_max_high -
                                 current_price) / current_price > 0.05:
            log.info(str(stock) + '  回撤达个股止盈线，平仓止盈！')
            order_target(stock, 0)
            continue

        # 4.持仓3天后，收益仍然小于5%，直接平仓
        if current_price / avg_cost < 1.05 and hold_day >= 3:
            log.info(str(stock) + '  未达个股盈利标准，平仓！')
            order_target(stock, 0)
            continue

        # 5.持仓1天后，收益仍然小于0%，直接平仓
        if current_price / avg_cost < 1 and hold_day == 1:
            log.info(str(stock) + '  未达个股盈利标准，平仓！')
            order_target(stock, 0)
            continue


## 计算持股天数
def hold_days(context, stocks):
    start_date = context.portfolio.positions[stocks].init_time
    today = context.current_dt
    trade_days = jqdata.get_trade_days(start_date, today)
    return len(trade_days)
