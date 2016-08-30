import numpy as np
ints=np.ones(10,dtype=np.uint16)
floats=np.ones(10,dtype=np.float32)
np.issubdtype(ints.dtype,np.integer)
np.issubdtype(floats.dtype,np.floating)
np.float64.mro()

arr=np.arange(8)
arr.reshape((4,2))

arr=np.arange(15)
arr.reshape((5,-1))
other_arr=np.ones((3,5))
other_arr.shape
arr.reshape(other_arr.shape)

arr=np.arange(15).reshape((5,3))
arr.ravel()

arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[10,11,12]])
np.concatenate([arr1,arr2],axis=0)
np.concatenate([arr1,arr2],axis=1)

from numpy.random import randn
arr=randn(5,2)
first,second,third=np.split(arr,[1,3])

arr=np.arange(6)
arr1=arr.reshape((3,2))
arr2=randn(3,2)
np.r_[arr1,arr2]
np.c_[np.r_[arr1,arr2],arr]
np.c_[1:6,-10:-5]

arr=np.arange(10)*100
inds=[7,1,2,6]
arr[inds]
arr.take(inds)
arr.put(inds,42)

arr=np.arange(5)
arr*4
arr=randn(4,3)
arr.mean(0)
demeaned=arr-arr.mean(0)
row_means=arr.mean(1)
row_means.reshape((4,1))
demeaned=arr-row_means.reshape((4,1))

arr=np.zeros((4,4))
arr_3d=arr[:,np.newaxis,:]
arr_3d.shape
arr_1d=np.random.normal(size=3)
arr_1d[:,np.newaxis]
arr_1d[np.newaxis,:]

arr=randn(3,4,5)
depth_means=arr.mean(2)
demeaned=arr-depth_means[:,:,np.newaxis]
demeaned.mean(2)
def demean_axis(arr,axis=0):
    means=arr.mean(axis)
    indexer=[slice(NOne)]*arr.ndim
    indexer[axis]=np.newaxis
    return arr-means[indexer]
arr=np.zeros((4,3))
arr[:]=5
col=np.array([1.28,-0.42,0.44,1.6])
arr[:]=col[:,np.newaxis]

arr=np.arange(10)
np.add.reduce(arr)
arr.sum()
arr=randn(5,5)
arr[::2].sort(1)
arr[:,:-1]<arr[:,1:]
np.logical_and.reduce(arr[:,:-1]<arr[:,1:],axis=1)

arr=np.arange(15).reshape((3,5))
np.add.accumulate(arr,axis=1)
arr=np.arange(3).repeat([1,2,2])
np.multiply.outer(arr,np.arange(5))
result=np.subtract.outer(randn(3,4),randn(5))
result.shape

arr=np.arange(10)
np.add.reduceat(arr,[0,5,8])
arr=np.multiply.outer(np.arange(4),np.arange(5))

def add_elements(x,y):
    return x+y
add_them=np.frompyfunc(add_elements,2,1)
add_them(np.arange(8),np.arange(8))
add_them=np.vectorize(add_elements,otypes=[np.float64])
add_them(np.arange(8),np.arange(8))

arr=randn(10000)
%timeit add_them(arr,arr)
%timeit np.add(arr,arr)

dtype=[('x',np.float64),('y',np.int32)]
sarr=np.array([(1.5,6),(np.pi,-2)],dtype=dtype)

dtype=[('x',np.int64,3),('y',np.int32)]
arr=np.zeros(4,dtype=dtype)
dtype=[('x',[('a','f8'),('b','f4')]),('y',np.int32)]
data=np.array([((1,2),5),((3,4),6)],dtype=dtype)
data['x']

values=np.array([5,0,1,3,2])
indexer=values.argsort()
arr=randn(3,6)
arr[0]=values

values=np.array(['2:first','2:second','1:first','1:second','1:third'])
key=np.array([2,2,1,1,1])
indexer=key.argsort(kind="mergesort")
values.take(indexer)

arr=np.array([0,1,7,12,15])
arr.searchsorted(9)
arr.searchsorted([0,8,11,16])
arr=np.array([0,0,0,1,1,1,1])
arr.searchsorted([0,1])
arr.searchsorted([0,1],side='right')

data=np.floor(np.random.uniform(0,10000,size=50))
bins=np.array([0,100,1000,5000,10000])
labels=bins=bins.searchsorted(data)
pd.Series(data).groupby(labels).mean()
np.digitize(data,bins)

X =  np.array([[ 8.82768214,  3.82222409, -1.14276475,  2.04411587],
                   [ 3.82222409,  6.75272284,  0.83909108,  2.08293758],
                   [-1.14276475,  0.83909108,  5.01690521,  0.79573241],
                  [ 2.04411587,  2.08293758,  0.79573241,  6.24095859]])
X[:,0]
y=X[:,:1]
np.dot(y.T,np.dot(X,y))

Xm=np.matrix(X)
ym=Xm[:,0]
ym.T*Xm*ym
Xm.I*X
