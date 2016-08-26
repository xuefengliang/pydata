from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

now=datetime.now()
now.year
delta=datetime(2011,1,7)-datetime(2008,6,24,8,15)
start=datetime(2011,1,7)
start+timedelta(12)

stamp=datetime(2011,1,3)
str(stamp)

datestrs=['7/6/2011','8/6/2011']
[datetime.strptime(x,'%m/%d/%Y') for x in datestrs]
parse('2011-01-03')
parse('Jan 31,1997 10:45 PM')
parse('6/12/2001',dayfirst=True)

pd.to_datetime(datestrs)
idx=pd.to_datetime(datestrs+[None])

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
            datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts=pd.Series(np.random.randn(6),index=dates)
type(ts)
ts.index
ts.index.dtype

stamp=ts.index[2]
longer_ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))
ts[datetime(2011,1,7):]

dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4),
                    index=dates,
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.ix['5-2001']

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                              '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)
grouped=dup_ts.groupby(level=0)
grouped.mean()

ts
ts.resample('D')

from pandas.tseries.offsets import Hour,Minute
hour=Hour()
four_hour=Hour(4)

import pytz
pytz.common_timezones[-5:]
tz=pytz.timezone('US/Eastern')

rng=pd.date_range('3/9/2012 9:30',periods=6,freq='D')
ts=pd.Series(np.random.randn(len(rng)),index=rng)
print(ts.index.tz)
pd.date_range('3/9/2012 9:30',periods=10,freq='D',tz='UTC')
ts_utc=ts.tz_localize('UTC')

stamp=pd.Timestamp('2-11-03-12 04:00')
stamp_utc=stamp.tz_localize('utc')
stamp_utc.tz_convert('US/Eastern')

p=pd.Period(2007,freq='A-DEC')
rng=pd.date_range('1/1/2000',periods=3,freq='M')
ts=pd.Series(np.random.randn(3),index=rng)
pts=ts.to_period()

data=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch08/macrodata.csv')
data.year
data.quarter
index=pd.PeriodIndex(year=data.year,quarter=data.quarter,freq='Q-DEC')
data.index=index
data.head()

rng=pd.date_range('1/1/2000',periods=100,freq='D')
ts=pd.Series(np.random.randn(len(rng)),index=rng)
ts.resample('M',how='mean')

rng=pd.date_range('1/1/2000',periods=12,freq='T')
ts=pd.Series(np.arange(12),index=rng)
ts.resample('5min',how='sum')
ts.resample('5min',how='sum',loffset='-1s')

frame=pd.DataFrame(np.random.randn(2,4),index=pd.date_range('1/1/2000',periods=2,freq='W-WED'),columns=['Colorado','Texas','New York','Ohio'])
df_daily=frame.resample('D')
frame=pd.DataFrame(np.random.randn(24,4),index=pd.date_range('1-2000','1-2002',freq='M'),columns=['Colorado','Texas','New York','Ohio'])
annual_frame=frame.resample('A-DEC',how='mean')
annual_frame.resample('Q-DEC',fill_method='ffill')

close_px_all=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch09/stock_px.csv',parse_dates=True,index_col=0)
close_px=close_px_all[['AAPL','MSFT','XOM']]
close_px=close_px.resample('B',fill_method='ffill')
close_px['AAPL'].plot()
close_px.AAPL.plot()
close_px['2009'].plot()
close_px['AAPL'].ix['01-2011':'03-2011'].plot()
appl_q=close_px['AAPL'].resample('Q-DEC',fill_method='ffill')
appl_q.ix['2009':].plot()
close_px.AAPL.plot()
pd.rolling_mean(close_px.AAPL,250).plot()

appl_std250=pd.rolling_std(close_px.AAPL,250,min_periods=10)
appl_std250.plot()
expanding_mean=lambda x:rolling_mean(x,len(x),min_periods=1)
pd.rolling_mean(close_px,60).plot(logy=True)

fig,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(12,7))
aapl_px=close_px.AAPL['2005':'2009']
ma60=pd.rolling_mean(aapl_px,50,min_periods=50)
ewma6=pd.ewma(aapl_px,span=60)
aapl_px.plot(style='k-',ax=axes[0])
ma60.plot(style='k--',ax=axes[0])
aapl_px.plot(style='k--',ax=axes[1])
ewma6.plot(style='k--',ax=axes[1])
axes[0].set_title('Simple MA')
axes[0].set_title('Exponentially-weighted MA')

spx_px=close_px_all['SPX']
spx_rets=spx_px/spx_px.shift(1)-1
returns=close_px.pct_change()
corr=pd.rolling_corr(returns.AAPL,spx_rets,125,min_periods=100)
corr.plot()
corr=pd.rolling_corr(returns,spx_rets,125,min_periods=100)
corr.plot()

from scipy.stats import percentileofscore
score_at_2percent=lambda x:percentileofscore(x,0.02)
result=pd.rolling_apply(returns.AAPL,250,score_at_2percent)
result.plot()

rng=pd.date_range('1/1/2000',periods=10000000,freq='10ms')
ts=pd.Series(np.random.randn(len(rng)),index=rng)
ts.resample('15min',how='ohlc')
%timeit ts.resample('15min',how='0hlc')
