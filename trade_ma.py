# 导入函数库
from jqdata import *
# 导入需要的程序包
import pandas as pd
import lightgbm as lgb
import time
from datetime import timedelta
import datetime
import pickle


def get_today_buyer_stocks(end_dt=datetime.datetime(2021, 8, 13), stocks_top_n=10):
    buyerlist = pickle.loads(read_file('buyerlist_nost.pkl'))
    return buyerlist[end_dt]  # .date()


# 初始化函数，设定基准等等
def initialize(context):
    import os

    print('Start', context.current_dt.date())

    set_slippage(FixedSlippage(0.00))
    # 设定沪深300作为基准
    # set_benchmark('000300.XSHG')
    set_benchmark('000905.XSHG')
    # set_benchmark('000099.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)

    g.hold_N = 4  # 持有数量

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    # set_order_cost(OrderCost(close_tax=0.0024, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    run_daily(before_market_open, time='before_open')

    # 开盘时运行
    # run_daily(market_open,  time='every_bar')
    run_daily(market_open, time='09:30')
    # run_daily(market_open, time='14:30')

    # 收盘后运行
    run_daily(after_market_close, time='after_close')


## 开盘前运行函数
def before_market_open(context):
    g.tmp_buylist = get_today_buyer_stocks(end_dt=context.current_dt.date())
    log.info('today_buy_list is--' + str(g.tmp_buylist))

    tmp_hold = context.portfolio.positions

    g.sell_list = []

    if len(tmp_hold) > 0:
        for s in tmp_hold:
            price = get_bars(s, count=5, unit='1d', fields='close')
            ma5 = price['close'].mean()
            if price['close'][-1] < ma5:
                g.sell_list.append(s)


## 开盘时运行函数
def market_open(context):
    tmp_hold = context.portfolio.positions

    if len(tmp_hold) == 0:
        amt = context.portfolio.available_cash
        for s in g.tmp_buylist[:g.hold_N]:
            order_target_value(s, amt / g.hold_N)
    else:
        for s in g.sell_list:
            order_target_value(s, 0)

        tmp_hold = context.portfolio.positions
        if len(tmp_hold) < 4:
            per_cash = context.portfolio.available_cash / (g.hold_N - len(tmp_hold))
            buylist = [x for x in g.tmp_buylist if x not in tmp_hold]
            buylist = [x for x in buylist if x not in g.sell_list]
            buylist = buylist[:(g.hold_N - len(tmp_hold))]
            # log.info('新买入:--' + str(buylist))
            for s in buylist:
                order_target_value(s, per_cash)


## 收盘后运行函数
def after_market_close(context):
    log.info('一天结束')
    log.info('##############################################################')
