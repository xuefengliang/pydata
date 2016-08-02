import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

obj=Series([4,7,-5,3])
obj.values
obj.index
obj2=Series([4,7,-5,3],index=['d','b','a','c'])
obj2.index
obj2[obj2>0]
obj2*2
np.exp(obj2)
'b' in obj2

sdata={'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3=Series(sdata)

states=['California','Ohio','Oregon','Texas']
obj4=Series(sdata,index=states)
pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()
obj3+obj4
obj4.name='population'
obj4.index.name='state'
obj.index=['index','Steve','Jeff','Ryan']

data={'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
      'year':[2000,2001,2002,2001,2002],
      'pop':[1.5,1.7,3.6,2.4,2.9]}
frame=DataFrame(data)
frame2=DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five'])
frame2.columns
frame2['state']
frame2.year

frame2.ix['three']
frame2['debt']=16.5
frame2['debt']=np.arange(5.)
val=Series([-1.2,-1.5,-1.7],index=['two','four','five'])
frame2['debt']=val
frame2
frame2['eastern']=frame2.state=='Ohio'
del frame2['eastern']
frame2.columns
pop={'Nevada':{2001:2.4,2002:2.9},
     'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3=DataFrame(pop)
DataFrame(pop,index=[2001,2002,2003])
pdata={'Ohio':frame3['Ohio'][:-1],'Nevada':frame3['Nevada'][:2]}
DataFrame(pdata)
frame3.index.name='year'
frame3.columns.name='state'
frame3.values

obj=Series(range(3),index=['a','b','c'])
index=obj.index
index[1:]
index=pd.Index(np.arange(3))
obj2=Series([1.5,-2.5,0],index=index)
obj2.index is index

obj=Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
obj
obj2=obj.reindex(['a','b','c','d','e'])
obj.reindex(['a','b','c','d','e'],fill_value=0)
obj3=Series(['blue','purple','yellow'],index=[0,2,4])
obj3.reindex(range(6),method='ffill')
frame=DataFrame(np.arange(9).reshape((3,3)),index=['a','c','d'],columns=['Ohio','Texas','California'])
states=['Texas','Utah','California']
frame.reindex(columns=states)
frame.reindex(index=['a','b','c','d'],method='ffill',columns=states)
frame.ix[['a','b','c','d'],states]
obj=Series(np.arange(5.),index=['a','b','c','d','e'])
new_obj=obj.drop('c')
data=DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],
               columns=['one','two','three','four'])
data.drop(['Colorado','Ohio'])

obj=Series(np.arange(4.),index=['a','b','c','d'])
data=DataFrame(np.arange(16).reshape(4,4),
               index=['Ohio','Colorado','Utah','New York'],
               columns=['one','two','three','four'])

s1=Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
s2=Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
s1+s2

frame=DataFrame(np.arange(12.).reshape(4,3),columns=list('dbe'),index=['Utah','Ohio','Texas','Oregon'])
series=frame.ix[0]
frame-series
series2=Series(range(3),index=['b','e','f'])
frame+series2
series3=frame['b']
frame.sub(series3,axis=0)






































































