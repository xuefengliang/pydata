# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.random import randn
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
