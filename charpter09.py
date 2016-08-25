import numpy as np
import pandas as pd

df=pd.DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'],'data1':np.random.randn(5),'data2':np.random.randn(5)})
states=np.array(['Ohio','California','California','Ohio','Ohio'])
years=np.array([2005,2005,2006,2005,2006])

df['data1'].groupby([states,years]).mean()
df.groupby('key1').mean()
df.groupby(['key1','key2']).mean()
df.groupby(['key1','key2']).size()

for name,group in df.groupby('key1'):
    print name
    print group

for (k1,k2), group in df.groupby(['key1','key2']):
    print k1,k2
    print group

pieces=dict(list(df.groupby('key1')))
pieces['b']
df.dtypes
grouped=df.groupby(df.dtypes,axis=1)
dict(list(grouped))
df.groupby('key1')['data1']
df['data1'].groupby(df['key1'])
df.groupby(['key1','key2'])[['data2']].mean()

s_grouped=df.groupby(['key1','key2'])['data2']
s_grouped.mean()

people=pd.DataFrame(np.random.randn(5,5),columns=['a','b','c','d','e'],index=['Joe','Steve','Wes','Jim','Travis'])
people.ix[2:3,['b','c']]=np.nan
people
mapping={'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
by_column=people.groupby(mapping,axis=1)
by_column.sum()
map_series=pd.Series(mapping)
people.groupby(map_series,axis=1).count()
people.groupby(len).sum()
key_list=['one','one','one','two','two']
people.groupby([len,key_list]).min()

columns=pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],[1,3,5,1,3]],names=['cty','tenor'])
hier_df=pd.DataFrame(np.random.randn(4,5),columns=columns)
hier_df.groupby(level='cty',axis=1).count()

grouped=df.groupby('key1')
grouped['data1'].quantile(0.9)

def peak_to_peak(arr):
    return arr.max()-arr.min()

grouped.agg(peak_to_peak)
grouped.describe()

tips=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch08/tips.csv')
tips['tip_pct']=tips['tip']/tips['total_bill']
tips[:6]
grouped=tips.groupby(['sex','smoker'])
group_pct=grouped['tip_pct']
group_pct.agg('mean')
group_pct.agg(['mean','std',peak_to_peak])
group_pct.agg([('foo','mean'),('bar',np.std)])

functions=['count','mean','max']
result=grouped['tip_pct','total_bill'].agg(functions)
ftuples=[('Durchschnitt','mean'),('Abweichung',np.var)]
grouped['tip_pct','total_bill'].agg(ftuples)
group.agg({'tip':np.max,'size':'sum'})
grouped.agg({'tip_pct':['min','max','mean','std'],'size':'sum'})
tips.groupby(['sex','smoker'],as_index=False).mean()

k1_means=df.groupby('key1').mean().add_prefix('mean_')
pd.merge(df,k1_means,left_on='key1',right_index=True)

key=['one','two','one','two','one']
people.groupby(key).mean()
people.groupby(key).transform(np.mean)

def demean(arr):
    return arr-arr.mean()
demeaned=people.groupby(key).transform(demean)
demeaned.groupby(key).mean()

def top(df,n=5,column='tip_pct'):
    return df.sort_index(by=column)[-n:]

top(tips,n=6)
tips.groupby('smoker').apply(top)
tips.groupby(['smoker','day']).apply(top,n=1,column='total_bill')
tips.groupby('smoker',group_keys=False).apply(top)

frame=pd.DataFrame({'data1':np.random.randn(1000),'data2':np.random.randn(1000)})
factor=pd.cut(frame.data1,4)

def get_stats(group):
    return {'min':group.min(),'max':group.max(),'count':group.count(),'mean':group.mean()}
grouped=frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()

s=pd.Series(np.random.randn(6))
s[::2]=np.nan
s.fillna(s.mean())

states=['Ohio','New York','Vermont','Florida','Oregon','Nevada','California','Idaho']
group_key=['East']*4+['West']*4
data=pd.Series(np.random.randn(8),index=states)
data[['Vermont','Nevada','Idaho']]=np.nan
data.groupby(group_key).mean()
fill_mean=lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

fill_values={'East':0.5,'West':-1}
fill_func=lambda g:g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)

suits=['H','S','C','D']
card_val=(range(1,11)+[10]*3)*4
base_names=['A']+range(2,11)+['J','K','Q']
cards=[]
for suit in ['H','S','C','D']:
    cards.extend(str(num)+suit for num in base_names)
    
deck=pd.Series(card_val,index=cards)
deck[:13]
def draw(deck,n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
draw(deck)

get_suit=lambda card:card[-1]
deck.groupby(get_suit).apply(draw,n=2)
deck.groupby(get_suit,group_keys=False).apply(draw,n=2)

df=pd.DataFrame({'category':['a','a','a','a','b','b','b','b'],'data':np.random.randn(8),'weights':np.random.rand(8)})
grouped=df.groupby('category')
get_wavg=lambda g: np.average(g['data'],weights=g['weights'])
grouped.apply(get_wavg)

close_px=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch09/stock_px.csv',parse_dates=True,index_col=0)
rets=close_px.pct_change().dropna()
spx_corr=lambda x:x.corrwith(x['SPX'])
by_year=rets.groupby(lambda x:x.year)
by_year.apply(spx_corr)
by_year.apply(lambda g : g['AAPL'].corr(g['MSFT']))

import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params
by_year.apply(regress,'AAPL',['SPX'])

tips.pivot_table(index=['sex', 'smoker'])
tips.pivot_table(['tip_pct','size'],index=['sex','day'],columns='smoker')
tips.pivot_table(['tip_pct','size'],index=['sex','day'],columns='smoker',margins=True)
tips.pivot_table('tip_pct',index=['sex','smoker'],columns='day',aggfunc=len,margins=True)
tips.pivot_table('size',index=['time','sex','smoker'],columns='day',aggfunc='sum',fill_value=0)

pd.crosstab([tips.time,tips.day],tips.smoker,margins=True)

fec=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch09/P00000001-ALL.csv')
fec.ix[123456]
unique_cands=fec.cand_nm.unique()
parties = {'Bachmann, Michelle': 'Republican',
        'Cain, Herman': 'Republican',
        'Gingrich, Newt': 'Republican',
        'Huntsman, Jon': 'Republican',
        'Johnson, Gary Earl': 'Republican',
       'McCotter, Thaddeus G': 'Republican',
       'Obama, Barack': 'Democrat',
       'Paul, Ron': 'Republican',
       'Pawlenty, Timothy': 'Republican',
       'Perry, Rick': 'Republican',
       "Roemer, Charles E. 'Buddy' III": 'Republican',
        'Romney, Mitt': 'Republican',
        'Santorum, Rick': 'Republican'}
        
fec.cand_nm[123456:123461].map(parties)
fec['party']=fec.cand_nm.map(parties)
fec['party'].value_counts()
fec.contbr_occupation.value_counts()[:10]

occ_mapping = {
       'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
       'INFORMATION REQUESTED' : 'NOT PROVIDED',
       'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
       'C.E.O.': 'CEO'
    }
    
f=lambda x:occ_mapping.get(x,x)
fec.contbr_occupation=fec.contbr_occupation.map(f)

emp_mapping = {
       'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
       'INFORMATION REQUESTED' : 'NOT PROVIDED',
       'SELF' : 'SELF-EMPLOYED',
       'SELF EMPLOYED' : 'SELF-EMPLOYED',
    }
f=lambda x:emp_mapping.get(x,x)
fec.contbr_employer=fec.contbr_employer.map(f)
by_occupation = fec.pivot_table('contb_receipt_amt',
                                    index='contbr_occupation',
                                   columns='party', aggfunc='sum')
over_2mm=by_occupation[by_occupation.sum(1)>2000000]
over_2mm.plot(kind='barh')

def get_top_amounts(group,key,n=5):
    totals=group.groupby(key)['contb_receipt_amt'].sum()
    return totals.order(ascending=False)[n:]

fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
grouped=fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)

bins=np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels=pd.cut(fec_mrbo.contb_receipt_amt,bins)
grouped=fec_mrbo.groupby(['cand_nm',labels])
grouped.size().unstack(0)
bucket_sums=grouped.contb_receipt_amt.sum().unstack(0)
normed_sums=bucket_sums.div(bucket_sums.sum(axis=1),axis=0)
normed_sums[:-2].plot(kind='barh',stacked=True)

grouped=fec_mrbo.groupby(['cand_nm','contbr_st'])
totals=grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
percent=totals.div(totals.sum(1),axis=1)

from mpl_toolkits.basemap import Basemap,cm
import numpy as np
from matplotlib import  rcParams
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt