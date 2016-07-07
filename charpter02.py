# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:05:15 2016

@author: xuefliang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

path='/home/xuefliang/Downloads/pydata-book-master/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records=[json.loads(line) for line in open(path)]
records[0]

time_zones=[rec['tz'] for rec in records if 'tz' in rec]
time_zones[:10]
def get_counts(sequence):
    counts={}
    for x in sequence:
        if x in counts:
            counts[x]+=1
        else:
            counts[x]=1
    return counts
    
from collections import  defaultdict
def get_counts(sequence):
    counts=defaultdict(int)
    for x in sequence:
        counts[x]+=1
    return counts

counts=get_counts(time_zones)
counts['America/New_York']
len(time_zones)

def top_counts(count_dict,n=10):
    value_key_pairs=[(count,tz) for tz,count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

top_counts(counts)

from collections import Counter
counts=Counter(time_zones)
counts.most_common(10)

frame=pd.DataFrame(records)
frame['tz'][:10]
tz_counts=frame['tz'].value_counts()
tz_counts[:10]

clean_tz=frame['tz'].fillna('Missing')
clean_tz[clean_tz=='']='Unknown'
tz_counts=clean_tz.value_counts()

tz_counts[:10].plot(kind='barh',rot=0)
results=pd.Series([x.split()[0] for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]
cframe=frame[frame.a.notnull()]
operating_system=np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
operating_system[:5]
by_tz_os=cframe.groupby(['tz',operating_system])
agg_counts=by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

indexer=agg_counts.sum(1).argsort()
indexer[:10]

count_subset=agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh',stacked=True)
normed_subset=count_subset.div(count_subset.sum(1),axis=0)
normed_subset.plot(kind='barh',stacked=True)


unames=['user_id','gender','age','occupation','zip']
users=pd.read_table('/home/xuefliang/Downloads/pydata-book-master/ch02/movielens/users.dat',sep='::',header=None,names=unames)
rnames=['user_id','movie_id','rating','timestamp']
ratings=pd.read_table('/home/xuefliang/Downloads/pydata-book-master/ch02/movielens/ratings.dat',sep='::',header=None,names=rnames)
mnames=['movie_id','title','genres']
movies=pd.read_table('/home/xuefliang/Downloads/pydata-book-master/ch02/movielens/movies.dat',sep='::',header=None,names=mnames)

users[:5]
ratings[:5]
movies[:5]

data=pd.merge(pd.merge(ratings,users),movies)
data.ix[0]
mean_ratings = data.pivot_table('rating', index='title',columns='gender', aggfunc='mean')
ratings_by_title=data.groupby('title').size()
ratings_by_title[:10]
active_titles=ratings_by_title.index[ratings_by_title>=250]
mean_ratings=mean_ratings.ix[active_titles]
top_female_ratings=mean_ratings.sort_index(by='F',ascending=False)

mean_ratings['diff']=mean_ratings['M']-mean_ratings['F']
sorted_by_diff=mean_ratings.sort_index(by='diff')
sorted_by_diff[:15]
sorted_by_diff[::-1][:15]
rating_std_by_title=data.groupby('title')['rating'].std()
rating_std_by_title=rating_std_by_title.ix[active_titles]
rating_std_by_title.order(ascending=False)[:10]

names1880=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch02/names/yob1880.txt',names=['name', 'sex', 'births'])
names1880.groupby('sex').births.sum()
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = '/home/xuefliang/Downloads/pydata-book-master/ch02/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)

# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)

total_births = names.pivot_table('births', index='year',columns='sex', aggfunc=sum)
total_births.tail()
total_births.plot(title="Total birth by sex and year")

def add_prop(group):
    births=group.births.astype(float)
    group['prop']=births/births.sum()
    return group
names=names.groupby(['year','sex']).apply(add_prop)
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)

def get_top1000(group):
    return group.sort_index(by='births',ascending=False)[:1000]
grouped=names.groupby(['year','sex'])
top1000=grouped.apply(get_top1000)

pieces=[]
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000]),
top1000 = pd.concat(pieces, ignore_index=True)

boys=top1000[top1000.sex=='M']
girls=top1000[top1000.sex=='F']
total_births = top1000.pivot_table('births', index='year', columns='name',aggfunc=sum)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True,figsize=(12,10),grid=False,title="Number of births per year")

table=top1000.pivot_table('prop',index='year',columns='sex',aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))
df=boys[boys.year==2010]

prop_cumsum=df.sort_index(by='prop',ascending=False).prop.cumsum()
prop_cumsum[:10]

df=boys[boys.year==1900]
in1900=df.sort_index(by='prop',ascending=False).prop.cumsum()
in1900.searchsorted(0.5)+1

    
def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1

diversity=top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity=diversity.unstack('sex')
diversity.head()
diversity.plot(title="Number of popular names in to 50%")

get_last_letter=lambda x:x[-1]
last_letters=names.name.map(get_last_letter)
last_letters.name='last_letter'

table = names.pivot_table('births', index=last_letters,columns=['sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()
subtable.sum()

letter_prop = subtable / subtable.sum().astype(float)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar',rot=0, ax=axes[1], title='Female',legend=False)

letter_prop=table/table.sum().astype(float)
dny_ts=letter_prop.ix[['d','n','y'],'M']
dny_ts.head()
dny_ts.plot()

all_names=top1000.name.unique()
mask=np.array(['lesl' in x.lower() for x in all_names])
lesley_like=all_names[mask]

filtered=top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

table=table.div(table.sum(1),axis=0)
table.tail()

table.plot(style={'M':'k-','F':'k--'})








































































































