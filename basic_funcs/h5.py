import h5py
import numpy as np
import pandas as pd

data = h5py.File('/Users/caichaohong/Desktop/Data.h5','a')

price_daily = data.create_group("/price_data/stocks/daily")
price_30m = data.create_group("/price_data/stocks/30m")
price_5m = data.create_group("/price_data/stocks/5m")

price_index_etf_daily = data.create_group("/price_data/index_etf/daily")
price_index_etf_5m = data.create_group("/price_data/index_etf/5m")

factor_data = data.create_group("factor_data")

close_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv',index_col='Unnamed: 0')
open_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/open.csv',index_col='Unnamed: 0')
high_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high.csv',index_col='Unnamed: 0')
low_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low.csv',index_col='Unnamed: 0')
highlimit_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/high_limit.csv',index_col='Unnamed: 0')
lowlimit_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/low_limit.csv',index_col='Unnamed: 0')
vol_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/volume.csv',index_col='Unnamed: 0')
mon_daily = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/money.csv',index_col='Unnamed: 0')

data['/price_data/stocks/daily/close'] = close_daily
data['/price_data/stocks/daily/open'] = open_daily
data['/price_data/stocks/daily/high'] = high_daily
data['/price_data/stocks/daily/low'] = low_daily
data['/price_data/stocks/daily/high_limit'] = highlimit_daily
data['/price_data/stocks/daily/low_limit'] = lowlimit_daily
data['/price_data/stocks/daily/volume'] = vol_daily
data['/price_data/stocks/daily/money'] = mon_daily
data['/price_data/stocks/daily/index'] = [x.encode() for x in close_daily.index]
data['/price_data/stocks/daily/columns'] = [x.encode() for x in close_daily.columns]

# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0')
data['/price_data/stocks/daily/st_df'] = st_df

#
hs300 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510300.XSHG.xlsx', index_col='Unnamed: 0')
sz50 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510050.XSHG.xlsx', index_col='Unnamed: 0')
zz500 = pd.read_excel('/Users/caichaohong/Desktop/Zenki/price/510500.XSHG.xlsx', index_col='Unnamed: 0')
hs300.index = [x.strftime('%Y-%m-%d') for x in hs300.index]

data['/price_data/index_etf/daily/hs300'] = hs300
data['/price_data/index_etf/daily/sz50'] = sz50
data['/price_data/index_etf/daily/zz500'] = zz500
data['/price_data/index_etf/daily/index'] = [x.encode() for x in hs300.index]
data['/price_data/index_etf/daily/columns'] = [x.encode() for x in hs300.columns]

hs300_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/510300_5m.csv',index_col='Unnamed: 0')
hs300_5m.index = [x.split(' ')[0] for x in hs300_5m.index]
data['/price_data/index_etf/5m/hs300'] = hs300_5m
data['/price_data/index_etf/5m/index'] = [x.encode() for x in hs300_5m.index]
data['/price_data/index_etf/5m/columns'] = [x.encode() for x in hs300_5m.columns]

data.close()
# del data['/price_data/daily'].attrs['columns']
# data['/price_data/daily'].attrs.modify('columns', list(close_daily.columns))

close_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/close_5m.csv',index_col='Unnamed: 0')
open_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/open_5m.csv',index_col='Unnamed: 0')
high_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/high_5m.csv',index_col='Unnamed: 0')
low_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/low_5m.csv',index_col='Unnamed: 0')
vol_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/volume_5m.csv',index_col='Unnamed: 0')
mon_5m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/5m/money_5m.csv',index_col='Unnamed: 0')

data['/price_data/stocks/5m/close'] = close_5m
data['/price_data/stocks/5m/open'] = open_5m
data['/price_data/stocks/5m/high'] = high_5m
data['/price_data/stocks/5m/low'] = low_5m
data['/price_data/stocks/5m/volume'] = vol_5m
data['/price_data/stocks/5m/money'] = mon_5m
data['/price_data/stocks/5m/index'] = [x.encode() for x in close_5m.index]
data['/price_data/stocks/5m/columns'] = [x.encode() for x in close_5m.columns]

data.close()
#
# 大单
net_amount_main = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv', index_col='Unnamed: 0')
net_pct_main = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_main.csv', index_col='Unnamed: 0')
net_amount_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_xl.csv', index_col='Unnamed: 0')
net_pct_xl = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_xl.csv', index_col='Unnamed: 0')
net_amount_l = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_l.csv', index_col='Unnamed: 0')
net_pct_l = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_l.csv', index_col='Unnamed: 0')
net_amount_m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_m.csv', index_col='Unnamed: 0')
net_pct_m = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_m.csv', index_col='Unnamed: 0')
net_amount_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_s.csv', index_col='Unnamed: 0')
net_pct_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_pct_s.csv', index_col='Unnamed: 0')


data['/price_data/stocks/daily/net_amount_main'] = net_amount_main
data['/price_data/stocks/daily/net_pct_main'] = net_pct_main
data['/price_data/stocks/daily/net_amount_xl'] = net_amount_xl
data['/price_data/stocks/daily/net_pct_xl'] = net_pct_xl
data['/price_data/stocks/daily/net_amount_l'] = net_amount_l
data['/price_data/stocks/daily/net_pct_l'] = net_pct_l
data['/price_data/stocks/daily/net_amount_m'] = net_amount_m
data['/price_data/stocks/daily/net_pct_m'] = net_pct_m
data['/price_data/stocks/daily/net_amount_s'] = net_amount_s
data['/price_data/stocks/daily/net_pct_s'] = net_pct_s

data.close()

# Financial Data
market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/market_cap.csv', index_col='Unnamed: 0')
circulating_market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/circulating_market_cap.csv', index_col='Unnamed: 0')
pe_ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/pe_ratio.csv', index_col='Unnamed: 0')
ps_ratio = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/ps_ratio.csv', index_col='Unnamed: 0')

data['/financial_data/daily/market_cap'] = market_cap
data['/financial_data/daily/circulating_market_cap'] = circulating_market_cap
data['/financial_data/daily/circulating_pe_ratio'] = pe_ratio
data['/financial_data/daily/circulating_ps_ratio'] = ps_ratio
data['/financial_data/daily/index'] = [x.encode() for x in market_cap.index]
data['/financial_data/daily/columns'] = [x.encode() for x in market_cap.columns]

data.close()

# Factor Data
big_player = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/big_player.csv',index_col='Unnamed: 0')
amtEntropy = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/amtRatioEntropy.csv',index_col='Unnamed: 0')
rankinglist = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/rankinglist.csv',index_col='Unnamed: 0')
Turnrankinglist = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/Turnrankinglist.csv',index_col='Unnamed: 0')
extremevol_std = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/extremevol_std.csv',index_col='Unnamed: 0')
alpha_083 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_083.csv',index_col='Unnamed: 0')
alpha_062 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_062.csv',index_col='Unnamed: 0')
alpha_064 = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/191/alpha_064.csv',index_col='Unnamed: 0')
industry_reverse = pd.read_csv('/Users/caichaohong/Desktop/Zenki/factors/industry_reverse.csv',index_col='Unnamed: 0')


def save_factor_toh5(path, factor_df):
    value_path = path + 'value'
    index_path = path + 'index'
    column_path = path + 'columns'
    data[value_path] = factor_df
    data[index_path] = [x.encode() for x in factor_df.index]
    data[column_path] = [x.encode() for x in factor_df.columns]


save_factor_toh5('/financial_data/factor/big_player/', factor_df=big_player)
save_factor_toh5('/financial_data/factor/amtEntropy/', factor_df=amtEntropy)
save_factor_toh5('/financial_data/factor/rankinglist/', factor_df=rankinglist)
save_factor_toh5('/financial_data/factor/Turnrankinglist/', factor_df=Turnrankinglist)
save_factor_toh5('/financial_data/factor/extremevol_std/', factor_df=extremevol_std)
save_factor_toh5('/financial_data/factor/alpha191_083/', factor_df=alpha_083)
save_factor_toh5('/financial_data/factor/alpha191_062/', factor_df=alpha_062)
save_factor_toh5('/financial_data/factor/alpha191_064/', factor_df=alpha_064)
save_factor_toh5('/financial_data/factor/industry_reverse/', factor_df=industry_reverse)

data.close()
# read
index = [x.decode() for x in data['price_data/5m/index']]
z = pd.DataFrame(data['price_data/5m/close'],index=index)





