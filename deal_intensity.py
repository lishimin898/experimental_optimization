import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

path='E:/python/code/intensity/2020-07-31/'
choice='gainopt'
path+=choice
file_name=os.listdir(path)
n=len(file_name)
for i in range(n):
    filepath=path+'/'+str(i)+'/'+choice+'.csv'
    df=pd.read_csv(filepath,header=None)