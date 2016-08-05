import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s1=pd.Series([0,1],index=['a','b'])
s2=pd.Series([2,3,4],index=['c','d','e'])
s3=pd.Series([5,6],index=['f','g'])
pd.concat([s1,s2,s3],axis=0)
s4=pd.concat([s1*5,s3])

