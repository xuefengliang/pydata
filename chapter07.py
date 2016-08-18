import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json

s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3], axis=0)
s4 = pd.concat([s1 * 5, s3])

data = pd.Series([1., -999., 2., -999, -1000., 3.])
data.replace(-999, np.nan)
data.replace([-999, -1000], np.nan)

data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data.index.map(str.upper)
data.index=data.index.map(str.upper)
data.rename(index=str.title,columns=str.upper)
data.rename(index={'Ohio':'INDIANA'},columns={'three':'peekaboo'})
_=data.rename(index={'Ohio':'INDIANA'},inplace=True)

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)

data=np.random.rand(20)
pd.cut(data,4,precision=2)

data=np.random.randn(1000)
cats=pd.qcut(data,4)

np.random.seed(12345)
data=pd.DataFrame(np.random.randn(1000,4))
data.describe()
col=data[3]
col[np.abs(col)>3]
data[(np.abs(data)>3).any(1)]
data[np.abs(data)>3]=np.sign(data)*3

df=pd.DataFrame(np.arange(5*4).reshape(5,4))
sampler=np.random.permutation(5)
sampler
df
df.take(sampler)
df.take(np.random.permutation(len(df))[:3])

bag=np.array([5,7,-1,6,4])
sampler=np.random.randint(0,len(bag),size=10)
draws=bag.take(sampler)
draws

df=pd.DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})
pd.get_dummies(df['key'])
dummies=pd.get_dummies(df['key'],prefix='key')
df_with_dummy=df[['data1']].join(dummies)

mnames=['movie_id','title','genres']
movies=pd.read_table('/home/xuefliang/Downloads/pydata-book-master/ch02/movielens/movies.dat',sep='::',header=None,names=mnames)
movies[:10]
genre_iter=(set(x.split('|')) for x in movies.genres)
genres=sorted(set.union(*genre_iter))
dummies=pd.DataFrame(np.zeros((len(movies),len(genres))),columns=genres)
for i,gen in enumerate(movies.genres):
    dummies.ix[i,gen.split('|')]==1
movies_windic=movies.join(dummies.add_prefix('Genre_'))
movies_windic.ix[0]
values=np.random.rand(10)
bins=[0,0.2,0.4,0.6,0.8,1]
pd.get_dummies(pd.cut(values,bins))

val='a,b,  guido'
val.split(',')
pieces=[x.strip() for x in val.split(',')]
first,second,third=pieces
first+'::'+second+'::'+third
'::'.join(pieces)

text="foo bar\t baz \tqux"
re.split('\s+',text)
regex=re.compile('\s+')
regex.split(text)
regex.findall(text)

text="Dave dave@google.com Steve steve@gmail.com Rob rob@gmail.com Ryan ryan@yahoo.com"
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex=re.compile(pattern,flags=re.IGNORECASE)
regex.findall(text)
m=regex.search(text)
text[m.start():m.end()]
print regex.match(text)
print regex.sub('REDACTED',text)
pattern=r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex=re.compile(pattern,flags=re.IGNORECASE)
m=regex.match('wesm@bright.net')
m.groups()
regex.findall(text)
print regex.sub(r'Username:\1,Domain:\2,Suffix:\3',text)

regex = re.compile(r"""
(?P<username>[A-Z0-9._%+-]+)
@
(?P<domain>[A-Z0-9.-]+)
\.
(?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)
m=regex.match('wesm@bright.com')
m.groupdict()

data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com','Rob': 'rob@gmail.com', 'Wes': np.nan}
data=pd.Series(data)
data.str.contains('gmail')
data.str.findall(pattern,flags=re.IGNORECASE)
matches=data.str.match(pattern,flags=re.IGNORECASE)
matches.str.get(1)
matches.str[0]
data.str[:5]
db=json.load(open('/home/xuefliang/Downloads/pydata-book-master/ch07/foods-2011-10-03.json'))
len(db)
db[0].keys()
nutrients=pd.DataFrame(db[0]['nutrients'])
info_keys = ['description', 'group', 'id', 'manufacturer']
info = pd.DataFrame(db, columns=info_keys)
pd.value_counts(info.group)[:10]
nutrients=[]
for rec in db:
    fnuts=pd.DataFrame(rec['nutrients'])
    fnuts['id']=rec['id']
    nutrients.append(fnuts)
nutrients=pd.concat(nutrients,ignore_index=True)
nutrients.duplicated().sum()
nutrients=nutrients.drop_duplicates()
col_mapping={'description':'food','group':'fgroup'}
info=info.rename(columns=col_mapping,copy=False)
col_mapping={'description':'nutrient','group':'nutgroup'}
nutrients=nutrients.rename(columns=col_mapping,copy=False)
ndata=pd.merge(nutrients,info,on='id',how='outer')
result=ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].order().plot(kind='barh')
by_nutrient=ndata.groupby(['nutgroup','nutrient'])
get_maxinum=lambda x:x.xs(x.value.idxmax())
get_minmum=lambda x:x.xs(x.value.idxmin())
max_food=by_nutrient.apply(get_maxinum)[['value','food']]
max_food.food=max_food.food.str[:50]
max_food.ix['Amino Acids']['food']
