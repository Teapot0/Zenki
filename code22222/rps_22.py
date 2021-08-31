import pandas as pd
import warnings
warnings.filterwarnings('ignore')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']

close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1) # 某列全NA
close_rts_1 = close.pct_change(1)

# 上市大于N天的,一共1816天
ipo_days = close.shape[0] - close.isna().sum()
stock_list_days = list(close.isna().sum()[close.isna().sum() <= 244].index) # 大于一年的
close = close[stock_list_days]

# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)
st_list = list((st_df.iloc[-1,:][st_df.iloc[-1,:] == True]).index)
close = close.drop(columns=st_list)

#RPS
# 一二三个月
rps_n1 = 120
rps_n2 = 250


def get_rps_df(rps_n):
    close_n_min = close.rolling(rps_n).min()
    close_n_max = close.rolling(rps_n).max()
    rps = ((close - close_n_min)/(close_n_max - close_n_min)) * 100
    return rps

rps1 = get_rps_df(rps_n1)
rps2 = get_rps_df(rps_n2)

tmp = pd.DataFrame(columns=['120rps_5', '250rps_5','120rps_10', '250rps_10'])
tmp['120rps_5'] = rps1.rolling(5).mean().iloc[-1,]
tmp['250rps_5'] = rps2.rolling(5).mean().iloc[-1,]
tmp['120rps_10'] = rps1.rolling(10).mean().iloc[-1,]
tmp['250rps_10'] = rps2.rolling(10).mean().iloc[-1,]
tmp.to_excel('/Users/caichaohong/Desktop/rps123.xlsx')








