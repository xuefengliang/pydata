# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.random import randn
from numpy.linalg import eigvals

data ={i:randn() for i in range(7)}
an_apple=27
an_example=42
b=[1,2,3]

def add_number(a,b):
  """
  Add two numbers together
  Returns
  -------
  the_sum:type of argument
  """
  return a+b
  
%run "E:\pydata\ipython_script_test.py"

a=np.random.randn(100,100)
%timeit np.dot(a,a)

2**27
foo='bar'

def run_experiment(niter=100):
  K=100
  results=[]
  for _ in xrange(niter):
    mat=np.random.randn(K,K)
    max_eigenvalue=np.abs(eigvals(mat)).max()
    results.append(max_eigenvalue)
  return results

some_result=run_experiment()
print 'Largest one we saw: %s' %np.max(some_result)


def add_and_sum(x,y):
	added=x+y
	summed=added.sum(axis=1)
	return summed

def call_function():
	x=randn(1000,1000)
	y=randn(1000,1000)
	return add_and_sum(x+y)

