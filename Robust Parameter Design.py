# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:13:47 2018

@author: A-Bibeka
"""
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("%reset -f")

import os
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd 
from sklearn import linear_model
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
import statistics
import math
sys.path.append("C:/Users/a-bibeka/Documents/GitHub/Python-Code-Compilation")
from HalfNrm_LenthTest import HalfPlt_V1
from HalfNrm_LenthTest import LenthsTest
print('Current working directory ',os.getcwd())


data1=StringIO('''A B C D E F G H B_M-1 B_M-2 B_M-3 B_M-4 T_M-1 T_M-2 T_M-3 T_M-4
- - - + - - - - 14.2908 14.1924 14.2714 14.1876 15.3182 15.4279 15.2657 15.4056
- - - + + + + + 14.8030 14.7193 14.6960 14.7635 14.9306 14.8954 14.9210 15.1349
- - + - - - + + 13.8793 13.9213 13.8532 14.0849 14.0121 13.9386 14.2118 14.0789
- - + - + + - - 13.4054 13.4788 13.5878 13.5167 14.2444 14.2573 14.3951 14.3724
- + - - - + - + 14.1736 14.0306 14.1398 14.0796 14.1492 14.1654 14.1487 14.2765
- + - - + - + - 13.2539 13.3338 13.1920 13.4430 14.2204 14.3028 14.2689 14.4104
- + + + - + + - 14.0623 14.0888 14.1766 14.0528 15.2969 15.5209 15.4200 15.2077
- + + + + - - + 14.3068 14.4055 14.6780 14.5811 15.0100 15.0618 15.5724 15.4668
+ - - - - + + - 13.7259 13.2934 12.6502 13.2666 14.9039 14.7952 14.1886 14.6254
+ - - - + - - + 13.8953 14.5597 14.4492 13.7064 13.7546 14.3229 14.2224 13.8209
+ - + + - + - + 14.2201 14.3974 15.2757 15.0363 14.1936 14.4295 15.5537 15.2200
+ - + + + - + - 13.5228 13.5828 14.2822 13.8449 14.5640 14.4670 15.2293 15.1099
+ + - + - - + + 14.5335 14.2492 14.6701 15.2799 14.7437 14.1827 14.9695 15.5484
+ + - + + + - - 14.5676 14.0310 13.7099 14.6375 15.8717 15.2239 14.9700 16.0001
+ + + - - - - - 12.9012 12.7071 13.1484 13.8940 14.2537 13.8368 14.1332 15.1681
+ + + - + + + + 13.9532 14.0830 14.1119 13.5963 13.8136 14.0745 14.4313 13.6862''')

# Read the data above 
dat=pd.read_csv(data1,delimiter=r"\s+",header=0)

dat.loc[:,'A']=np.where(dat['A']=='-',-1,1)
dat.loc[:,'B']=np.where(dat['B']=='-',-1,1)
dat.loc[:,'C']=np.where(dat['C']=='-',-1,1)
dat.loc[:,'D']=np.where(dat['D']=='-',-1,1)
dat.loc[:,'E']=np.where(dat['E']=='-',-1,1)
dat.loc[:,'F']=np.where(dat['F']=='-',-1,1)
dat.loc[:,'G']=np.where(dat['G']=='-',-1,1)
dat.loc[:,'H']=np.where(dat['H']=='-',-1,1)

dat['id']=dat.index
#Transform wide data to long - First on M
dat1=pd.wide_to_long(dat,stubnames=['B_M-','T_M-'],i="id",j='M')
dat1=dat1.reset_index()
dat1.loc[:,"id"]=dat1.index
dat1=dat1.rename(index=str,columns={"B_M-":"L1","T_M-":"L2"})

#Transform wide data to long - Second on L
dat1=pd.wide_to_long(dat1,stubnames=['L'],i="id",j='NosL')
dat1=dat1.reset_index()
dat1.loc[:,"id"]=dat1.index
dat1=dat1.rename(index=str,columns={"L":"Y"})
dat1=dat1.rename(index=str,columns={"NosL":"L"})

def contrast_l(ef):
    contr=int()
    if(ef==1):contr=1# M1 +ve
    elif(ef==2):contr=1# M2 +ve
    elif(ef==3):contr=-1# M3 -ve
    else:contr=-1# M4 -ve
    return contr
def contrast_q(ef):
    contr=int()
    if(ef==1):contr=1# M1 +ve
    elif(ef==2):contr=-1# M2 -ve
    elif(ef==3):contr=-1# M3 -ve
    else:contr=1# M4 +ve
    return contr
def contrast_c(ef):
    contr=int()
    if(ef==1):contr=1# M1 +ve
    elif(ef==2):contr=-1# M2 -ve
    elif(ef==3):contr=1# M3 +ve
    else:contr=-1# M4 -ve
    return contr

dat1["Ml"]=dat1.M.apply(contrast_l)
dat1["Mq"]=dat1.M.apply(contrast_q)
dat1["Mc"]=dat1.M.apply(contrast_c)
dat1.loc[:,'L']=np.where(dat1['L']==2,1,-1)
