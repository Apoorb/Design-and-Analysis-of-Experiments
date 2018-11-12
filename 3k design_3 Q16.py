#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:58:34 2018

@author: Apoorb
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:59:57 2018

@author: Apoorb
Q16a
"""

import os
import pandas as pd
from io import StringIO # Needed to read the data
import math #To use math.log
import numpy as np 
from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn import linear_model
import statistics




print('Current working directory ',os.getcwd())
os.chdir('/Users/Apoorb/Documents/GitHub/Python-Code-Compilation')
print('Current working directory ',os.getcwd())


# Function to plot the Half-Normal Plot
def HalfPlt_V1(DatTemp,Theta,Var_,PltName):
    '''
    DatTemp : Dataset with the effecs {"FactEff":[#,#,....],"Var1":["A","B"....]}
    Theta : column name for effects; "FactEff"
    Var_ : column name for list of variables; "Var1"
    PltName : Name of the Half plot
    '''
    #Get the # of effects
    len1 =len(DatTemp[Var_])
    DatTemp['absTheta']=DatTemp[Theta].apply(abs)
    DatTemp=DatTemp.sort_values(by=['absTheta'])
    #Need to reset index after sort orderwise ploting will have error
    DatTemp = DatTemp.reset_index(drop=True)
    #Get the index of each sorted effect
    DatTemp['i']= np.linspace(1,len1,len1).tolist()
    DatTemp['NorQuant']=DatTemp['i'].apply(lambda x:norm.ppf(0.5+0.5*(x-0.5)/len1))
    fig1, ax1 =plt.subplots()
    ax1.scatter(DatTemp['NorQuant'], DatTemp['absTheta'], marker='x', color='red')
    #Name all the points using Var1, enumerate gives index and value
    for j,type in enumerate(DatTemp[Var_]):
        x = DatTemp['NorQuant'][j]
        y = DatTemp['absTheta'][j]
        ax1.text(x+0.05, y+0.05, type, fontsize=9)
    ax1.set_title("Half-Normal Plot")
    ax1.set_xlabel("Normal Quantile")
    ax1.set_ylabel("effects")
    fig1.savefig(PltName)
    
# Function to perform Lenth test
#Lenth's Method for  testing signficance for experiments without
# variance estimate
def LenthsTest(dat,fef,fileNm,IER_Alpha=2.30):
    '''
    dat: Dataset with the effecs {"FactEff":[#,#,....],"Var1":["A","B"....]}
    fef = column name for effects; "FactEff"
    IER_Alpha = IER for n effects and alpha
    '''
    #Get the # of effects
    len1=len(dat[fef])
    dat['absEff']=dat[fef].apply(abs)
    s0=1.5*statistics.median(map(float,dat['absEff']))
    #Filter the effects
    tpLst=[i for i in dat['absEff'] if i<2.5*s0]
    #Get PSE 
    PSE =1.5 * statistics.median(tpLst)
    #Lenth's t stat
    dat['t_PSE'] = (round(dat[fef]/PSE,2))
    dat['IER_Alpha']=[IER_Alpha]*len1
    dat['Significant'] = dat.apply(lambda x : 'Significant' if abs(x['t_PSE']) > x['IER_Alpha'] else "Not Significant", axis=1)
    dat=dat[["Var1","FactEff","t_PSE","IER_Alpha","Significant"]]
    dat.to_csv(fileNm)
    return(dat)
    
    
TempIO= StringIO('''Run A B C D M1 M2 P1 P2
1 0 0 0 0 34.0 31.8 49 54
2 0 1 1 1 102.8 98.4 96 67
3 0 2 2 2 57.6 57.2 50 44
4 1 0 1 2 31.8 24.8 30 48
5 1 1 2 0 162.2 150.8 74 111
6 1 2 0 1 32.4 27.5 37 51
7 2 0 2 1 31.9 37.7 88 96
8 2 1 0 2 53.1 57.2 101 50
9 2 2 1 0 53.8 36.9 45 54
''')

Df=pd.read_csv(TempIO,delimiter=r"\s+",header=0)
Df.head()
Df.A=Df.A.apply(int)

DesMat=Df[["A","B","C","D"]]


def contrast_l(ef):
    contr=int()
    if(ef==0):contr=-1/math.sqrt(2)
    elif(ef==1):contr=0/math.sqrt(2)
    else:contr=1/math.sqrt(2)
    return contr
def contrast_q(ef):
    contr=int()
    if(ef==0):contr=1/math.sqrt(6)
    elif(ef==1):contr=-2/math.sqrt(6)
    else:contr=1/math.sqrt(6)
    return contr

# 6 Main effects
X=pd.DataFrame()
X["Al"]=DesMat.A.apply(contrast_l)
X["Bl"]=DesMat.B.apply(contrast_l)
X["Cl"]=DesMat.C.apply(contrast_l)
X["Dl"]=DesMat.D.apply(contrast_l)
X["Aq"]=DesMat.A.apply(contrast_q)
X["Bq"]=DesMat.B.apply(contrast_q)
X["Cq"]=DesMat.C.apply(contrast_q)
X["Dq"]=DesMat.D.apply(contrast_q)


# Get sum y^2 for Microfinish
#Microfinish Y
Mic_y2= Df[["M1","M2"]].apply(lambda x: x[0]**2+x[1]**2,axis=1)
regM=linear_model.LinearRegression()
ModSum=regM.fit(X,Mic_y2)
ef2=ModSum.coef_
dat=pd.DataFrame({"FactEff":ef2,"Var1":X.columns})
#dat

HalfPlt_V1(dat,'FactEff','Var1','Q16_MF.png')

#IER at alpha =5% and 
LenthsTest(dat,'FactEff',"Q16_MF.csv",IER_Alpha=2.20)


# Get sum y^2 for polish time
#Microfinish Y
Pol_y2= Df[["P1","P2"]].apply(lambda x: x[0]**2+x[1]**2,axis=1)
regM=linear_model.LinearRegression()
ModSum=regM.fit(X,Mic_y2)
ef2=ModSum.coef_
dat=pd.DataFrame({"FactEff":ef2,"Var1":X.columns})
#dat

HalfPlt_V1(dat,'FactEff','Var1','Q16_PT.png')

#IER at alpha =5% and 
LenthsTest(dat,'FactEff',"Q16_PT.csv",IER_Alpha=2.20)