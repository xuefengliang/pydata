import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn

plt.plot(np.arange(10))
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
plt.plot([1.5,3.5,-2,1.6])
plt.plot(randn(50).cumsum(),'k--')
_=ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))
fig,axes=plt.subplots(2,3)
axes
fig,axes=plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(randn(500),bins=50,color='k',alpha=0.5)
        
plt.subplots_adjust(wspace=0,hspace=0)
plt.plot(randn(30).cumsum(),'ko--')
plt.plot(randn(30).cumsum(),color='k',linestyle='dashed',marker='o')

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum())
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(['oen','two','three','four','five'],rotation=30,fontsize='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(),'k',label='one')
ax.plot(randn(1000).cumsum(),'k--',label='two')
ax.plot(randn(1000).cumsum(),'k.',label='three')

ax.text(x,y,'Hello world!',family='monospace',fontsize=10)

from datetime import datetime
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
data=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch08/spx.csv',index_col=0,parse_dates=True)
spx=data['SPX']
spx.plot(ax=ax, style='k-')
crisis_data= [(datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 50),
                xytext=(date, spx.asof(date) + 200),
                arrowprops=dict(facecolor='black'),
                horizontalalignment='left', verticalalignment='top')
ax.set_xlim(['1/1/2007','1/1/2007'])
ax.set_ylim([600,800])
ax.set_title('Import dates in 2008-2--9 financial crisisi')

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],color='g', alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
plt.savefig('/home/xuefliang/pydata/figpath.svg')

from io import StringIO
buffer=StringIO()
plt.savefig(buffer)
plot_data=buffer.getvalue()
s=pd.Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
s.plot()


df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),columns=['A', 'B', 'C', 'D'],index=np.arange(0, 100, 10))
df.plot()

fig, axes = plt.subplots(2, 1)
data=pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='bar',ax=axes[0],color='k',alpha=0.7)
data.plot(kind='barh',ax=axes[1],color='k',alpha=0.7)

df = pd.DataFrame(np.random.rand(6, 4),
                index=['one', 'two', 'three', 'four', 'five', 'six'],
                columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df.plot(kind='bar')
df.plot(kind='barh',stacked=True,alpha=0.5)
tips=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch08/tips.csv')
party_counts=pd.crosstab(tips.day,tips.size)
party_counts
party_counts=party_counts.ix[:,2:5]
party_pcts=party_counts.div(party_counts.sum(1).astype(float),axis=0)

tips['tip_pct']=tips['tip']/tips['total_bill']
tips['tip_pct'].hist(bins=50)
#tips['tip_pct'].plot(kind='kde')

com1=np.random.normal(0,1,size=200)
com2=np.random.normal(10,2,size=200)
values=pd.Series(np.concatenate([com1,com2]))
values.hist(bins=100,alpha=0.3,color='k',normed=True)
#values.plot(kind='kde',style='k--')

macro=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch08/macrodata.csv')
data=macro[['cpi','m1','tbilrate','unemp']]
trans_data=np.log(data).diff().dropna()
trans_data[-5:]
plt.scatter(trans_data['m1'],trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' %('m1','unemp'))
pd.scatter_matrix(trans_data,diagonal='kde',color='k',alpha=0.3)

data=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch08/Haiti.csv')
data[['INCIDENT DATE','LATITUDE','LONGITUDE']][:10]
data.describe()
data=data[(data.LATITUDE>18) & (data.LATITUDE<20) & (data.LONGITUDE>-75) &(data.LONGITUDE<-70) &(data.CATEGORY.notnull())]

def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]
    
def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))
    
def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split(' | ')[1]
    return code, names.strip()
    
get_english('2.Urgences logistiques | Vital Lines')

all_cats=get_all_categories(data.CATEGORY)
english_mapping = dict(get_english(x) for x in all_cats)
english_mapping['2a']
english_mapping['6c']

def get_code(seq):
     return [x.split('.')[0] for x in seq if x]
    
all_codes=get_code(all_cats)
code_index=pd.Index(np.unicode(all_codes))
dummy_frame=pd.DataFrame(np.zeros((len(data),len(all_codes))),index=data.index,columns=all_codes)

for row,cat in zip(data.index,data.CATEGORY):
    codes=get_code(to_cat_list(cat))
    dummy_frame.ix[row,codes]=1
data=data.join(dummy_frame.add_prefix('category_'))

from mpl_toolkits.basemap import Basemap
def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25,lllon=-75, urlon=-71):
    m = Basemap(ax=ax, projection='stere',
        lon_0=(urlon + lllon) / 2,
        lat_0=(urlat + lllat) / 2,
        llcrnrlat=lllat, urcrnrlat=urlat,
        llcrnrlon=lllon, urcrnrlon=urlon,
         resolution='f')
        # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
to_plot = ['2a', '1', '3c', '7a']
lllat=17.25; urlat=20.25; lllon=-75; urlon=-71;
for code, ax in zip(to_plot, axes.flat):
    m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,lllon=lllon, urlon=urlon)
    
cat_data = data[data['category_%s' % code] == 1]
x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)
m.plot(x, y, 'k.', alpha=0.5)
ax.set_title('%s: %s' % (code, english_mapping[code]))
