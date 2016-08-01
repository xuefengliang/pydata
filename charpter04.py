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










