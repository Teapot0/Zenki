import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cha_cha(x1, x2, direction=False):
    diff = (x1 - x2) > 0
    jincha = ((diff - diff.shift(1)) > 0) * 1
    sicha = ((diff - diff.shift(1)) < 0) * -1
    if direction == 'jincha':
        return jincha
    if direction == 'sicha':
        return sicha
    else:
        return jincha + sicha  # 金叉是1，死叉-1，其他情况是0


def MaxDrawdown(return_list):  # 必须是净值的list，返回array
    return (np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)


def transform_to_3_categorical(data):
    df = data.dropna()
    low = np.percentile(df,5)
    high = np.percentile(df, 95)
    return (data >= high )*1 + (data <= low)*-1 + ((data>low) & (data<high))*0



def plot_rts(df):
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ax1.plot(df['net_value'].values, 'black', label='port_net_value')
    # ax1.plot(df['benchmark_net_value'], 'blue', label='benchmark_net_value')
    # ax1.plot((1 + df['rts'] - df['benchmark_rts']).cumprod(), 'gold', label='cumulative alpha')  # 画超额收益 Alpha
    ax2.plot(df['nv_max_draw'].values, 'red', linestyle='-.',linewidth=1, label='port_max_draw')
    ax1.legend()
    ax2.legend()
    annual_rts = df['net_value'].values[-1] ** (1 / (round(df.shape[0] / (244), 2))) - 1
    plt.title('years_={} Max_Drawdown={} \n total_rts={} annualized rts ={}\n Sharpe={}'.format(
        round(df.shape[0] / (244), 2),
        np.round(MaxDrawdown(list(df['net_value'].dropna())).max(),4),
        np.round(df['net_value'].values[-1],2),
        np.round(annual_rts,4),
        (annual_rts - 0.03) / (np.std(df['rts_1']) * np.sqrt(244))))
    plt.show()


