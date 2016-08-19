import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *

ggplot(mtcars, aes('mpg', 'qsec')) + \
geom_point(colour='steelblue') + \
scale_x_continuous(breaks=[10,20,30], labels=["horrible", "ok", "awesome"])

ggplot(mtcars, aes(x='mpg',color='clarity'))+geom_density()

ggplot(diamonds, aes(x='price', color='clarity')) + geom_density()


