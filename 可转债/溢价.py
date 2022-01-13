from jqdatasdk import bond, auth, query
import pandas as pd

import tushare as ts

auth('13382017213', 'Aasd120120')

stock = '603377.XSHG'

limit_day = 100


kzz = '113575'
df1 = bond.run_query(query(bond.CONBOND_CONVERT_PRICE_ADJUST.code,bond.CONBOND_CONVERT_PRICE_ADJUST.adjust_date,bond.CONBOND_CONVERT_PRICE_ADJUST.new_convert_price)
        .filter(bond.CONBOND_CONVERT_PRICE_ADJUST.code ==kzz ).limit(9))
print(df1)

price_kts0403=df1.new_convert_price[0]
price_kts0616= df1.new_convert_price[1]
print(type(price_kts0403),price_kts0616)
    #获取KZZ的日数据
df2 = bond.run_query(query(bond.CONBOND_DAILY_PRICE.code,bond.CONBOND_DAILY_PRICE.date,
            bond.CONBOND_DAILY_PRICE.open,bond.CONBOND_DAILY_PRICE.close,bond.CONBOND_DAILY_PRICE.high,bond.CONBOND_DAILY_PRICE.low)
            .filter(bond.CONBOND_DAILY_PRICE.code ==kzz  ).order_by(bond.CONBOND_DAILY_PRICE.date.desc()).limit(limit_day))

#获取每张转债对应的股价
df2['price_kts'] = price_kts0616

df2.update(df2.iloc[:, 2:6].mul(df2.price_kts/100, 0))

data = df2[['date','open','close','high','low']]

x= len(data)

data =data.sort_values(by='date', ascending=True)

kzz_data=data.reset_index(drop=True)
#print(kzz_data)

# 获取股票的信息

df4 = attribute_history(stock, x, unit='1d',
            fields=['open', 'close','high','low'],
            skip_paused=True, df=True, fq='pre')


#df3 = df4.sort_index(ascending=False)

stock_data = df4.reset_index()
#print(stock_data)







#获取转股溢价率=（转债价*转股价/正股价/100）-1
kp=kzz_data[['date','close']]
sp=stock_data

#print(kp,sp)


radio = kp/sp-1
kts_radio = radio.close

#转股收益率

radio1 = sp/kp -1
stk_radio = radio1.close
#print(stk_radio)

#print(kzz_price_open,kzz_price_close,s_open,s_close)
#print(kts_rate_open,kts_rate_close)
#s_open .plot()
plt.figure(figsize=[18,7])


#kts_radio.plot()
stk_radio.plot( linewidth = '5', label = "radio", linestyle=':', marker='|')
kp['close'].plot(secondary_y=True, style='g',linewidth = '3')#设置第二个y轴（右y轴）
sp.close.plot(secondary_y=True, style='g')#设置第二个y轴（右y轴）