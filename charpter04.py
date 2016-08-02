# -*- coding: utf-8 -*-
"""
@author: xuefliang
@file: charpter04.py
@time: 8/1/16 8:16 PM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data1=[6,7.5,8,0,1]
arr1=np.array(data1)

data2=[[1,2,3,4],[5,6,7,8]]
arr2=np.array(data2)

arr2.ndim
arr2.shape

arr1.dtype
arr2.dtype

np.zeros(10)
np.zeros((3,6))
np.empty((2,3,2))
np.arange(15)

arr1=np.array([1,2,3],dtype=np.float64)
arr2=np.array([1,2,3],dtype=np.int32)

float_arr=arr1.astype(np.float64)
float_arr.dtype

numeric_strings=np.array(['1.25','-9.6','42'],dtype=np.string_)
numeric_strings.astype(float)

int_array=np.arange(10)
calibers=np.array([.22,.270,.357,.380,.44,.50],dtype=np.float64)
int_array.astype(calibers.dtype)


empty_unint32=np.empty(8,dtype='u4')
empty_unint32

arr=np.array([[1.,2.,3.],[4.,5.,6.]])
arr*arr
arr-arr
1/arr
arr*0.5

arr=np.arange(10)
arr
arr[5]
arr[5:8]=12

arr_slice=arr[5:8]
arr_slice[1]=12345
arr

arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2]
arr2d[0][2]
arr2d[0,2]

names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data=np.random.randn(7,4)
names=='Bob'
data[names=='Bob']

arr=np.empty((8,4))
for i in range(8):
    arr[i]=i

arr=np.arange(32).reshape((8,4))
arr[[1,5,7,2],[0,3,1,3]]
arr[[1,5,7,2]][:,[0,3,1,2]]
arr[np.ix_([1,5,7,2],[0,3,1,2])]

arr=np.arange(15).reshape((3,5))
arr.T
arr=np.random.randn(6,3)
np.dot(arr.T,arr)

arr=np.arange(16).reshape(2,2,4)
arr.transpose((1,0,2))
arr.swapaxes(1,2)

arr=np.arange(10)
np.sqrt(arr)
np.exp(arr)
x=np.random.randn(8)
y=np.random.randn(8)
np.maximum(x,y)

arr=np.random.randn(7)*5
np.modf(arr)

points=np.arange(-5,5,0.01)
xs,ys=np.meshgrid(points,points)
z=np.sqrt(xs**2+ys**2)
plt.imshow(z,cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x*2+y*2}$ for a grid of values")

xarr=np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])

result=[(x if c else y)
        for x,y,c in zip(xarr,yarr,cond)]
result=np.where(cond,xarr,yarr)

arr=np.random.randn(4,4)
np.where(arr>0,2,-2)
np.where(arr>0,2,arr)

cond1=np.array([True,False])
cond2=np.array([False,True])
np.where(cond1 & cond2,0,np.where(cond1,1,np.where(cond2,2,3)))
result=1*(cond1-cond2)+2*(cond2&-cond1)+3*-(cond1|cond2)

arr=np.random.randn(5,4)
arr.mean()
np.mean(arr)
arr.sum()
arr.mean(axis=1)
arr.sum(0)

arr=np.array([[0,1,2],[3,4,5],[6,7,8]])
arr.cumsum(0)
arr.cumprod(1)

arr=np.random.randn(100)
(arr>0).sum()
bools=np.array([False,True,False,False])
bools.any()
bools.all()

arr=np.random.randn(8)
arr.sort()
arr=np.random.randn(5,3)
arr.sort(1)

large_arr=np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05*len(large_arr))]

np.sort(arr)

names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
np.unique(names)
ints=np.array([3,3,3,2,2,2,11,4,4])
np.unique(ints)

values=np.array([6,0,0,3,2,5,6])
np.in1d(values,[2,3,6])

arr=np.arange(10)
np.save('some_array',arr)
np.load('some_array.npy')
np.savez('array_archive.npz',a=arr,b=arr)
arch=np.load('array_archive.npz')
arch['b']

x=np.array([[1.,2.,3.,],[4.,5.,6.]])
y=np.array([[6.,23.],[-1,7],[8,9]])
x.dot(y)
np.dot(x,np.ones(3))

from numpy.linalg import inv,qr
x=np.random.randn(5,5)
mat=x.T.dot(x)
inv(mat)
mat.dot(inv(mat))
q,r=qr(mat)

samples=np.random.normal(size=(4,4))

position=0
walk=[position]
steps=1000
for i in xrange(steps):
    step=1 if random.randint(0,1) else -1
    position+=step
    walk.append(position)
nsteps=1000
draws=np.random.randint(0,2,size=nsteps)
steps=np.where(draws>0,1,-1)
walk=steps.cumsum()

walk.min()
np.abs(walk)>=10
(np.abs(walk)>=10).argmax()

nwalks=5000
nsteps=1000
draws=np.random.randint(0,2,size=(nwalks,nsteps))
steps=np.where(draws>0,1,-1)
walks=steps.cumsum(1)
walks.max()
hits30=(np.abs(walks)>=30).any(1)
hits30.sum()
crossing_times=(np.abs(walks[hits30])>=30).argmax(1)




