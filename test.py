import trade




# 大单
net_amount_s = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/money_flow/net_amount_main.csv', index_col='Unnamed: 0')
circulating_market_cap = pd.read_csv('/Users/caichaohong/Desktop/Zenki/financials/circulating_market_cap.csv', index_col='Unnamed: 0', date_parser=dateparse)

hs300 = pd.read_excel('~/510300.XSHG.xlsx', index_col='Unnamed: 0')
hs300['rts'] = hs300['close'].pct_change(1)
hs300['net_value'] = hs300['close'] / hs300['close'][0]

factor = net_amount_s / circulating_market_cap

