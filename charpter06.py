import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex1.csv')
pd.read_table('/home/xuefliang/Downloads/pydata-book-master/ch06/ex1.csv',sep=',')

pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex2.csv',header=None)
pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex2.csv',names=['a','b','c','d','message'])

names=['a','b','c','d','message']
pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex2.csv',names=names,index_col='message')

parsed=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/csv_mindex.csv',index_col=['key1','key2'])
result=pd.read_table('/home/xuefliang/Downloads/pydata-book-master/ch06/ex3.csv',sep='\s+')
pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex4.csv',skiprows=[0,2,3])

pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex5.csv',na_values=['NULL'])
sentinels={'message':['foo','NA'],'something':['two']}
pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex5.csv',na_values=sentinels)

chunker=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex6.csv',chunksize=1000)
tot=pd.Series([])
for piece in chunker:
    tot=tot.add(piece['key'].value_counts(),fill_value=0)
tot=tot.order(ascending=False)

data=pd.read_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/ex5.csv')
data.to_csv('out.csv')

dates=pd.date_range('1/1/2000',periods=7)
ts=pd.Series(np.arange(7),index=dates)
ts.to_csv('tseries.csv')

pd.Series.from_csv('/home/xuefliang/Downloads/pydata-book-master/ch06/tseries.csv',parse_dates=True)
import csv
f=open('/home/xuefliang/Downloads/pydata-book-master/ch06/ex7.csv')
reader=csv.reader(f)
for line in reader:
    print line
lines=list(csv.reader(open('/home/xuefliang/Downloads/pydata-book-master/ch06/ex7.csv')))
header,values=lines[0],lines[1:]
data_dict={h:v for h,v in zip(header,zip(*values))}

obj = """
{"name":"Wes",
 "places_lived":["United States","Spain","Germany"],
 "pet":null,
 "siblings":[{"name":"Scott","age"：25，"pet":"Zuko"},
                {"name":"Katie","age":33,"pet":"Cisco"}]
}
"""

import json
result=json.loads(obj)
