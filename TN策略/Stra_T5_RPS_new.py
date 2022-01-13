import pandas as pd
import warnings
from jqdatasdk import get_all_securities, auth, finance, query, get_fundamentals,valuation,get_query_count
from datetime import datetime, timedelta
from tqdm import tqdm
warnings.filterwarnings('ignore')

auth('13382017213', 'Aasd120120')
get_query_count()

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

all_name = pd.read_excel('/Users/caichaohong/Desktop/Zenki/all_stock_names.xlsx',index_col='Unnamed: 0')
all_name.index = all_name['code']

# 是否基金持仓数据，若不更新改为False即可
is_update_fund = False


close = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/daily/close.csv', index_col='Unnamed: 0', date_parser=dateparse)
close = close.dropna(how='all', axis=1) # 某列全NA
close_rts_1 = close.pct_change(1)


# 上市大于N天的,一共1816天
ipo_days = close.shape[0] - close.isna().sum()
stock_list_days = list(ipo_days[ipo_days >= 250].index) # 大于一年的
close = close[stock_list_days]


#RPS
# 一二三个月
rps_N1 = 20
rps_N2 = 40
rps_N3 = 60
rps_N4 = 120
rps_N5 = 250

ret_n1 = close.pct_change(rps_N1)
ret_n2 = close.pct_change(rps_N2)
ret_n3 = close.pct_change(rps_N3)
ret_n4 = close.pct_change(rps_N4)
ret_n5 = close.pct_change(rps_N5)

rps_n1_raw = ret_n1.iloc[-1,].sort_values(ascending=False).rank(pct=True)
rps_n2_raw = ret_n2.iloc[-1,].sort_values(ascending=False).rank(pct=True)
rps_n3_raw = ret_n3.iloc[-1,].sort_values(ascending=False).rank(pct=True)
rps_n4_raw = ret_n4.iloc[-1,].sort_values(ascending=False).rank(pct=True)
rps_n5_raw = ret_n5.iloc[-1,].sort_values(ascending=False).rank(pct=True)

rps_n1_85 = rps_n1_raw[rps_n1_raw>=0.85]
rps_n2_85 = rps_n2_raw[rps_n2_raw>=0.85]
rps_n3_85 = rps_n3_raw[rps_n3_raw>=0.85]
rps_n4_85 = rps_n4_raw[rps_n4_raw>=0.85]
rps_n5_85 = rps_n5_raw[rps_n5_raw>=0.85]


# ST
st_df = pd.read_csv('/Users/caichaohong/Desktop/Zenki/price/is_st.csv', index_col='Unnamed: 0',date_parser=dateparse)

stock_n1 = list(set(rps_n1_85.index).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))
stock_n2 = list(set(rps_n2_85.index).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))
stock_n3 = list(set(rps_n3_85.index).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))
stock_n4 = list(set(rps_n4_85.index).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))
stock_n5 = list(set(rps_n5_85.index).difference(st_df.iloc[-1,][st_df.iloc[-1,]==True].index))

rps_n1 = rps_n1_85.loc[stock_n1].sort_values(ascending=False)
rps_n2 = rps_n2_85.loc[stock_n2].sort_values(ascending=False)
rps_n3 = rps_n3_85.loc[stock_n3].sort_values(ascending=False)
rps_n4 = rps_n4_85.loc[stock_n4].sort_values(ascending=False)
rps_n5 = rps_n5_85.loc[stock_n5].sort_values(ascending=False)

inter_list = list(set.intersection(*map(set,[stock_n1, stock_n2,stock_n3, stock_n4, stock_n5])))
union_list = list(set.union(*map(set,[stock_n1, stock_n2,stock_n3, stock_n4, stock_n5])))


today = datetime.today().strftime('%Y-%m-%d')
yesterday = datetime.now() - timedelta(1) # mc的数据用
today_date = pd.to_datetime(today)
yesterday = datetime.strftime(yesterday, '%Y-%m-%d')

# 更新基金持仓数据
if is_update_fund == True:

    def get_end_dt(today_date):
        if (today_date.month >=1) & (today_date.month<=3):
            end_dt_pre = str(today_date.year ) + '-09-30'
            end_dt = str(today_date.year - 1) + '-12-31'
        elif (today_date.month >=4) & (today_date.month<=6):
            end_dt_pre = str(today_date.year - 1) + '-12-31'
            end_dt = str(today_date.year ) + '-03-31'
        elif (today_date.month >=7) & (today_date.month<=9):
            end_dt_pre = str(today_date.year) + '-03-31'
            end_dt = str(today_date.year ) + '-06-30'
        elif (today_date.month >=10) & (today_date.month<=12):
            end_dt_pre = str(today_date.year) + '-06-30'
            end_dt = str(today_date.year ) + '-09-30'
        return datetime.strptime(end_dt_pre, '%Y-%m-%d'),datetime.strptime(end_dt, '%Y-%m-%d')

    end_dt_pre, end_dt = get_end_dt(today_date)

    fund_list = list(get_all_securities(types=['stock_fund','mixture_fund'],date=today).index)
    for i in range(0,len(fund_list)):
        fund_list[i] = fund_list[i][0:6]    #取前六位,不带后缀


    def get_fund_hold(fund_code):
        q = query(finance.FUND_PORTFOLIO_STOCK).filter(
            finance.FUND_PORTFOLIO_STOCK.code == fund_code,
            finance.FUND_PORTFOLIO_STOCK.period_end == end_dt)
        df = finance.run_query(q)

        if df.empty:
            q = query(finance.FUND_PORTFOLIO_STOCK).filter(
                finance.FUND_PORTFOLIO_STOCK.code == fund_code,
                finance.FUND_PORTFOLIO_STOCK.period_end == end_dt_pre)
            df = finance.run_query(q)

        df = df.loc[df['symbol'].str.isdigit()] # 去掉美股
        df = df.loc[df['symbol'].str.len() == 6] # 去掉港股
        return df

    fund_df = pd.DataFrame()

    for fund in tqdm(fund_list):
        tmp = get_fund_hold(fund)
        fund_df = fund_df.append(tmp)

    fund_df.to_parquet('./data/all_fund_holdings_{}.parquet'.format(end_dt.strftime('%Y-%m-%d')))

    fund_sum = fund_df[['symbol', 'market_cap']].groupby('symbol').sum()
    fund_sum.columns = ['fund_sum']
    all_name_copy = all_name.copy(deep=True)
    all_name_copy.index = [x.split('.')[0] for x in all_name_copy.index]
    fund_sum.index = all_name_copy['code'].loc[fund_sum.index]
    fund_sum['fund_sum'] = fund_sum['fund_sum'] * 10**(-8)
    fund_sum.to_csv('./data/基金持仓比例_{}.csv'.format(end_dt.strftime('%Y-%m-%d')))


# 市值数据 market cap
mc = get_fundamentals(query(valuation.code,valuation.market_cap), date=yesterday)
mc.index = mc['code']

mc_0630 = get_fundamentals(query(valuation.code,valuation.market_cap), date='2021-06-30')
mc_0630.index = mc_0630['code']

fund_sum = pd.read_csv('./data/基金持仓比例_2021-09-30.csv')
fund_sum_last = pd.read_csv('./data/基金持仓比例_2021-06-30.csv')
fund_sum.index = fund_sum['code']
fund_sum_last.index = fund_sum_last['code']

fund_sum = fund_sum[fund_sum.index.notnull()]
fund_sum_last = fund_sum_last[fund_sum_last.index.notnull()]
fund_sum['mc'] = mc['market_cap'].loc[fund_sum.index]
fund_sum['ratio'] = fund_sum['fund_sum'] / fund_sum['mc']
fund_sum_last['mc'] = mc_0630['market_cap'].loc[fund_sum_last.index]
fund_sum_last['ratio'] = fund_sum_last['fund_sum'] / fund_sum_last['mc']
# 每天根据市值更新
fund_sum = fund_sum.loc[fund_sum['ratio'] >= 0.03]
fund_sum.to_csv('./data/基金持仓大于3.csv')

fund_sum = pd.read_csv('./data/基金持仓大于3.csv')
fund_sum.index = fund_sum['code']
big_fund_hold_list = list(set.intersection(*map(set,[list(fund_sum.index),union_list])))
fund_sum['name'] = all_name['short_name'].loc[fund_sum.index]

fund_sum_df = fund_sum.loc[big_fund_hold_list]
fund_sum_df = fund_sum_df.sort_values(by='ratio', ascending=False)
fund_sum_df['ratio_last'] = fund_sum_last['ratio'].loc[fund_sum_df.index]

rpsn1_2 = rps_n1_raw.loc[big_fund_hold_list].sort_values(ascending=False)
rpsn2_2 = rps_n2_raw.loc[big_fund_hold_list].sort_values(ascending=False)
rpsn3_2 = rps_n3_raw.loc[big_fund_hold_list].sort_values(ascending=False)
rpsn4_2 = rps_n4_raw.loc[big_fund_hold_list].sort_values(ascending=False)
rpsn5_2 = rps_n5_raw.loc[big_fund_hold_list].sort_values(ascending=False)

rps_dict = {'rps20_stock': all_name['short_name'].loc[rps_n1.index].values,
            'rps20_value': rps_n1.values,
            'rps40_stock': all_name['short_name'].loc[rps_n2.index].values,
            'rps40_value': rps_n2.values,
            'rps60_stock': all_name['short_name'].loc[rps_n3.index].values,
            'rps60_value': rps_n3.values,
            'rps120_stock': all_name['short_name'].loc[rps_n4.index].values,
            'rps120_value': rps_n4.values,
            'rps250_stock': all_name['short_name'].loc[rps_n5.index].values,
            'rps250_value': rps_n5.values,
            '交集': all_name['short_name'].loc[inter_list].values,
            '基金持仓大于3%': fund_sum_df['name'].values,
            '基金持仓比例_当季': fund_sum_df['ratio'].values,
            '基金持仓比例_上季': fund_sum_df['ratio_last'].values,

            '基金股rps20': all_name['short_name'].loc[rpsn1_2.index].values,
            'rps20': rpsn1_2.values,
            '基金持仓比例当季_20': fund_sum_df['ratio'].loc[rpsn1_2.index].values,
            '基金持仓比例上季_20': fund_sum_df['ratio_last'].loc[rpsn1_2.index].values,

            '基金股rps40': all_name['short_name'].loc[rpsn2_2.index].values,
            'rps40': rpsn2_2.values,
            '基金持仓比例当季_40': fund_sum_df['ratio'].loc[rpsn2_2.index].values,
            '基金持仓比例上季_40': fund_sum_df['ratio_last'].loc[rpsn2_2.index].values,

            '基金股rps60': all_name['short_name'].loc[rpsn3_2.index].values,
            'rps60': rpsn3_2.values,
            '基金持仓比例当季_60': fund_sum_df['ratio'].loc[rpsn3_2.index].values,
            '基金持仓比例上季_60': fund_sum_df['ratio_last'].loc[rpsn3_2.index].values,

            '基金股rps120': all_name['short_name'].loc[rpsn4_2.index].values,
            'rps120': rpsn4_2.values,
            '基金持仓比例当季_120': fund_sum_df['ratio'].loc[rpsn4_2.index].values,
            '基金持仓比例上季_120': fund_sum_df['ratio_last'].loc[rpsn4_2.index].values,

            '基金股rps250': all_name['short_name'].loc[rpsn5_2.index].values,
            'rps250': rpsn5_2.values,
            '基金持仓比例当季_250': fund_sum_df['ratio'].loc[rpsn5_2.index].values,
            '基金持仓比例上季_250': fund_sum_df['ratio_last'].loc[rpsn5_2.index].values,
            }
rps_df = pd.DataFrame({key: pd.Series(value) for key, value in rps_dict.items() })

# rps_df.to_excel('./rps_df.xlsx')

# 单元格颜色
writer = pd.ExcelWriter('./data/rps_df_{}.xlsx'.format(today), engine='xlsxwriter')
rps_df.to_excel(writer, 'Sheet1')
workbook = writer.book
worksheet = writer.sheets['Sheet1']
format = workbook.add_format({'bg_color': '#FF4500'})
# format2 = workbook.add_format({'bg_color': '#9D9D9D'})

big_fund_name = list(all_name['short_name'].loc[big_fund_hold_list])
for s in big_fund_name:
    worksheet.conditional_format('A1:L400', {'type':     'text',
                                             'criteria': 'containing',
                                             'value':    s,
                                             'format':   format})
writer.save()








