
def get_rsrs(high,low,N,n):
    rsrs=pd.DataFrame(index=high.index, columns=['rsrs', 'rsrs_mu','rsrs_std', 'signal'])
    for i in range(n,len(high)):
        tmp_high = high[i-n:i]
        tmp_low = low[i-n:i]
        m = LinearRegression()
        m.fit(X=tmp_high.values.reshape(-1,1), y=tmp_low.values.reshape(-1,1))
        rsrs['rsrs'].loc[high.index[i]] = m.intercept_
    rsrs['rsrs_mu'] = rsrs.rolling(N).mean()
    rsrs['rsrs_std'] = rsrs.rolling(N).std()
    rsrs_high = rsrs['rsrs_mu'] + rsrs['rsrs_std']
    rsrs_low = rsrs['rsrs_mu'] - rsrs['rsrs_std']
    rsrs['signal'] = (rsrs['rsrs'] > rsrs_high)*1 + (rsrs['rsrs'] < rsrs_low)*-1
    return rsrs




z=get_rsrs(high=high['600519.XSHG'], low=low['600519.XSHG'], N=20,n=5)

ax1=plt.subplot()
ax2=ax1.twinx()
ax1.plot(close['600519.XSHG'],'black')
ax2.plot(z['signal'],'red')