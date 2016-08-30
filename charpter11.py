import numpy as np
import pandas as pd

s1=pd.Series(range(3),index=['a','b','c'])
s2=pd.Series(range(4),index=['d','b','c','e'])
s3=pd.Series(range(3),index=['f','a','c'])

pd.DataFrame({'one':s1,'two':s2,'three':s3})
pd.DataFrame({'one':s1,'two':s2,'three':s3},index=list('face'))

ts1=pd.Series(np.random.randn(3),index=pd.date_range('2006-6-13',periods=3,freq='W-WED'))
ts1.resample('B')
ts1.resample('B',fill_method='ffill')

dates=pd.DatetimeIndex(['2012-6-12','2012-6-17','2012-6-18','2012-6-21','2012-6-22','2012-6-29'])
ts2=pd.Series(np.random.randn(6),index=dates)
ts1.reindex(ts2.index,method='ffill')
ts2+ts1.reindex(ts2.index,method='ffill')

gdp=pd.Series([1.78,1.94,2.08,2.01,2.15,2.31,2.46],index=pd.period_range('1984Q2',periods=7,freq='Q-SEP'))
infl=pd.Series([0.025,0.045,0.037,0.04],index=pd.period_range('1982',periods=4,freq='A-DEC'))
infl_q=infl.asfreq('Q-SEP',how='end')
infl_q.reindex(gdp.index,method='ffill')

rng=pd.date_range('2012-06-01 06:30','2012-06-01 15:59',freq='T')
rng=rng.append([rng+pd.offsets.BDay(i) for i in range(1,4)])
ts=pd.Series(np.arange(len(rng),dtype=float),index=rng)

from datetime import time
ts[time(10,0)]
ts.at_time(time(10,0))
ts.between_time(time(10,0),time(10,1))

indexer=np.sort(np.random.permutation(len(ts))[700:])
irr_ts=ts.copy()
irr_ts[indexer]=np.nan
irr_ts['2012-06-01 09:50':'2012-06-01 10:00']
selection=pd.date_range('2012-06-01 10:00',periods=4,freq='B')
irr_ts.asof(selection)

data1=pd.DataFrame(np.ones((6,3),dtype=float),columns=['a','b','c'],index=pd.date_range('6/12/2012',periods=6))
data2=pd.DataFrame(np.ones((6,4),dtype=float)*2,columns=['a','b','c','d'],index=pd.date_range('6/13/2012',periods=6))
spliced=pd.concat([data1.ix[:'2012-06-14'],data2.ix['2012-06-15':]])
spliced_filled=spliced.combine_first(data2)

spliced.update(data2,overwrite=False)
cp_spliced=spliced.copy()
cp_spliced[['a','c']]=data1[['a','c']]

import random
import string
np.random.seed(0)
N=1000
def rands(n):
    choices=string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in xrange(n)])

tickers=np.array([rands(5) for _ in xrange(N)])
M=500
df=pd.DataFrame({'Momentum':np.random.randn(M)/200+0.03,'Value':np.random.randn(M)/200+0.08,'ShortInterest':np.random.randn(M)/200-0.02},index=tickers[:M])

ind_names=np.array(['FINANCIAL','TECH'])
sampler=np.random.randint(0,len(ind_names),N)
industries=pd.Series(ind_names[sampler],index=tickers,name='industry')
by_industry=df.groupby(industries)
by_industry.mean()

def zscore(group):
    return (group-group.mean())/group.std()
df_stand=by_industry.apply(zscore)
df_stand.groupby(industries).agg(['mean','std'])

ind_rank=by_industry.rank(ascending=False)
ind_rank.groupby(industries).agg(['min','max'])
by_industry.apply(lambda x:zscore(x.rank()))

from numpy.random import rand
fac1,fac2,fac3=np.random.rand(3,1000)
tickers_subset=tickers.take(np.random.permutation(N)[:1000])
port=pd.Series(0.7*fac1-1.2*fac2+0.3*fac3+np.random.rand(1000),index=tickers_subset)
factors=pd.DataFrame({'f1':fac1,'f2':fac2,'f3':fac3},index=tickers_subset)
factors.corrwith(port)
pd.ols(y=port,x=factors).beta
def beta_exposure(chuck,factors=None):
    return pd.ols(y=chuck,x=factors).beta
by_ind=port.groupby(industries)
exposures=by_ind.apply(beta_exposure,factors=factors)
exposures.unstack()

names=['AAPL','GOOG','MSFT','DELL','GS','MS','BAC','C']
