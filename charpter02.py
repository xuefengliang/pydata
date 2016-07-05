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