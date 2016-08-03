import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

obj = Series([4, 7, -5, 3])
obj.values
obj.index
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2.index
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
'b' in obj2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()
obj3 + obj4
obj4.name = 'population'
obj4.index.name = 'state'
obj.index = ['index', 'Steve', 'Jeff', 'Ryan']

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2.columns
frame2['state']
frame2.year

frame2.ix['three']
frame2['debt'] = 16.5
frame2['debt'] = np.arange(5.)
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
frame2['eastern'] = frame2.state == 'Ohio'
del frame2['eastern']
frame2.columns
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
DataFrame(pop, index=[2001, 2002, 2003])
pdata = {'Ohio': frame3['Ohio'][:-1], 'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)
frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3.values

obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index[1:]
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=states)
frame.ix[['a', 'b', 'c', 'd'], states]
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio'])

obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
data = DataFrame(np.arange(16).reshape(4, 4),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2

frame = DataFrame(np.arange(12.).reshape(4, 3), columns=list('dbe'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
frame - series
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2
series3 = frame['b']
frame.sub(series3, axis=0)

frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])

np.abs(frame)
f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis=1)


def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])


frame.apply(f)

format = lambda x: '%.2f' % x
frame.applymap(format)
frame['e'].map(format)

obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2, 4)),
                  index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
frame.sort_index()
frame.sort_index(axis=1)

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_index(by='b')
obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
obj.rank(method='first')

obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique
df=DataFrame(np.random.randn(4,3),index=['a','a','b','b'])
df.ix['b']

df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df.sum()
df.sum(axis=0)
df.idxmax()
df.describe()

obj=Series(['a','a','b','c']*4)
obj.describe()

import pandas.io.data as web
all_data={}
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker]=web.get_data_yahoo(ticker,'1/1/2000','1/1/2010')

price=DataFrame({tic:data['Adj Close'] for tic,data in all_data.iteritems()})
volume=DataFrame({tic:data['Volume'] for tic,data in all_data.iteritems()})
returns=price.pct_change()
returns.tail()

string_data=Series(['aardvark','artichoke',np.nan,'avocado'])
string_data.isnull()
string_data[0]=None
string_data.isnull()

from numpy import nan as NA
data=Series([1,NA,3.5,NA,7])
data.dropna()

data=DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
cleaned=data.dropna()
data.dropna(how='all')
data.dropna(axis=1,how='all')

df=DataFrame(np.random.randn(7,3))
df.ix[:4,1]=NA
df.ix[:2,2]=NA
df.dropna(thresh=3)

df=DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
df.fillna(0)
df.fillna({1:4,1:2,-1:4})

df=DataFrame(np.random.randn(6,3))
df.ix[2:,1]=NA
df.ix[4:,2]=NA
df.fillna(method='ffill')
df.fillna(method='ffill',limit=2)

ser=Series(np.arange(3.))
ser2=Series(np.arange(3.),index=['a','b','c'])

ser3=Series(range(3),index=[-5,1,3])
ser3.iget_value(2)

frame=DataFrame(np.arange(6).reshape(3,2),index=[2,0,1])
frame.irow(0)