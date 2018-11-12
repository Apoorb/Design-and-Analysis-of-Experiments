#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:08:07 2018
@author: Apoorb
#HW6 3**k Design of Experiment
"""
#Q3
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
    
TempIO= StringIO('''Run A B C D S1 S2 S3 F1 F2 F3
1 0 0 0 0 5164 6615 5959 12.89 12.70 12.74
2 0 0 1 1 5356 6117 5224 12.83 12.73 13.07
3 0 0 2 2 3070 3773 4257 12.37 12.47 12.44
4 0 1 0 1 5547 6566 6320 13.29 12.86 12.70
5 0 1 1 2 4754 4401 5436 12.64 12.50 12.61
6 0 1 2 0 5524 4050 4526 12.76 12.72 12.94
7 0 2 0 2 5684 6251 6214 13.17 13.33 13.98
8 0 2 1 0 5735 6271 5843 13.02 13.11 12.67
9 0 2 2 1 5744 4797 5416 12.37 12.67 12.54
10 1 0 0 1 6843 6895 6957 13.28 13.65 13.58
11 1 0 1 2 6538 6328 4784 12.62 14.07 13.38
12 1 0 2 0 6152 5819 5963 13.19 12.94 13.15
13 1 1 0 2 6854 6804 6907 14.65 14.98 14.40
14 1 1 1 0 6799 6703 6792 13.00 13.35 12.87
15 1 1 2 1 6513 6503 6568 13.13 13.40 13.80
16 1 2 0 0 6473 6974 6712 13.55 14.10 14.41
17 1 2 1 1 6832 7034 5057 14.86 13.27 13.64
18 1 2 2 2 4968 5684 5761 13.00 13.58 13.45
19 2 0 0 2 7148 6920 6220 16.70 15.85 14.90
20 2 0 1 0 6905 7068 7156 14.70 13.97 13.66
21 2 0 2 1 6933 7194 6667 13.51 13.64 13.92
22 2 1 0 0 7227 7170 7015 15.54 16.16 16.14
23 2 1 1 1 7014 7040 7200 13.97 14.09 14.52
24 2 1 2 2 6215 6260 6488 14.35 13.56 13.00
25 2 2 0 1 7145 6868 6964 15.70 16.45 15.85
26 2 2 1 2 7161 7263 6937 15.21 13.77 14.34
27 2 2 2 0 7060 7050 6950 13.51 13.42 13.07
''')
Df=pd.read_csv(TempIO,delimiter=r"\s+",header=0)
Df.head()

DesMat=Df[["A","B","C"]]
DesMat.loc[:,"AB"]=(DesMat.A+DesMat.B)%3
DesMat.loc[:,"AB2"]=(DesMat.A+2*DesMat.B)%3
DesMat.loc[:,"AC"]=(DesMat.A+DesMat.C)%3
DesMat.loc[:,"AC2"]=(DesMat.A+2*DesMat.C)%3
DesMat.loc[:,"BC"]=(DesMat.B+DesMat.C)%3
DesMat.loc[:,"BC2"]=(DesMat.B+2*DesMat.C)%3

DesMat.loc[:,"ABC"]=(DesMat.A+DesMat.B+DesMat.C)%3
DesMat.loc[:,"ABC2"]=(DesMat.A+DesMat.B+2*DesMat.C)%3
DesMat.loc[:,"AB2C"]=(DesMat.A+2*DesMat.B+DesMat.C)%3
DesMat.loc[:,"AB2C2"]=(DesMat.A+2*DesMat.B+2*DesMat.C)%3

Df['S_lnsBar']= Df[['S1','S2','S3']].apply(statistics.variance,axis=1).apply(math.log)

Df['F_lnsBar']=((Df[['F1','F2','F3']].std(axis=1))**2).apply(math.log)
Df=Df.drop(["D","S1","S2","S3","F1","F2","F3"],axis=1)
Df.head()

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
X["Aq"]=DesMat.A.apply(contrast_q)
X["Bq"]=DesMat.B.apply(contrast_q)
X["Cq"]=DesMat.C.apply(contrast_q)

# 12 2 way effects
X["ABl"]=DesMat.AB.apply(contrast_l)
X["ABq"]=DesMat.AB.apply(contrast_q)
X["AB2l"]=DesMat.AB2.apply(contrast_l)
X["AB2q"]=DesMat.AB2.apply(contrast_q)
X["ACl"]=DesMat.AC.apply(contrast_l)
X["ACq"]=DesMat.AC.apply(contrast_q)
X["AC2l"]=DesMat.AC2.apply(contrast_l)
X["AC2q"]=DesMat.AC2.apply(contrast_q)
X["BCl"]=DesMat.BC.apply(contrast_l)
X["BCq"]=DesMat.BC.apply(contrast_q)
X["BC2l"]=DesMat.BC2.apply(contrast_l)
X["BC2q"]=DesMat.BC2.apply(contrast_q)
# 8 3 way effects
X["ABCl"]=DesMat.ABC.apply(contrast_l)
X["ABCq"]=DesMat.ABC.apply(contrast_q)
X["ABC2l"]=DesMat.ABC2.apply(contrast_l)
X["ABC2q"]=DesMat.ABC2.apply(contrast_q)
X["AB2Cl"]=DesMat.AB2C.apply(contrast_l)
X["AB2Cq"]=DesMat.AB2C.apply(contrast_q)
X["AB2C2l"]=DesMat.AB2C2.apply(contrast_l)
X["AB2C2q"]=DesMat.AB2C2.apply(contrast_q)

regr=linear_model.LinearRegression()
# Train the model using the training sets
Zs1=Df.S_lnsBar
R1=regr.fit(X,Zs1)
R1.intercept_

ef2=R1.coef_
ef2
dat=pd.DataFrame({"FactEff":ef2,"Var1":X.columns})
#dat

HalfPlt_V1(dat,'FactEff','Var1','HalfPlotStrn.png')

#IER at alpha =5% and 
LenthsTest(dat,'FactEff',"LenthTestDisper_Str.csv",IER_Alpha=2.08)

##############################################################################
#Flash Exp
regr=linear_model.LinearRegression()
# Train the model using the training sets
Zs1=Df.F_lnsBar
R1=regr.fit(X,Zs1)
R1.intercept_
ef2=R1.coef_
ef2
dat=pd.DataFrame({"FactEff":ef2,"Var1":X.columns})
#dat

HalfPlt_V1(dat,'FactEff','Var1','HalfPlotFlash.png')

#IER at alpha =5% and 
LenthsTest(dat,'FactEff',"LenthTestDisper_Flash.csv",IER_Alpha=2.08)
