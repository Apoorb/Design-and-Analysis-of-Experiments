#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:59:57 2018

@author: Apoorb
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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats
import seaborn as sns


print('Current working directory ',os.getcwd())
#os.chdir('/Users/Apoorb/Documents/GitHub/Python-Code-Compilation')
os.chdir("C:\\Users\\a-bibeka\\Documents\\GitHub\\Python-Code-Compilation")
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
    
    
TempIO= StringIO('''
Run A C E B D F S1 S2
1 0 0 0 0 0 0 18.25 17.25
2 0 0 1 1 1 1 4.75 7.5
3 0 0 2 2 2 2 11.75 11.25
4 0 1 0 1 1 1 13.0 8.75
5 0 1 1 2 2 2 12.5 11.0
6 0 1 2 0 0 0 9.25 13.0
7 0 2 0 2 2 2 21.0 15.0
8 0 2 1 0 0 0 3.5 5.25
9 0 2 2 1 1 1 4.0 8.5
10 1 0 0 0 1 2 6.75 15.75
11 1 0 1 1 2 0 5.0 13.75
12 1 0 2 2 0 1 17.25 13.5
13 1 1 0 1 2 0 13.5 21.25
14 1 1 1 2 0 1 9.0 10.25
15 1 1 2 0 1 2 15.0 9.75
16 1 2 0 2 0 1 10.5 8.25
17 1 2 1 0 1 2 11.0 11.5
18 1 2 2 1 2 0 19.75 14.25
19 2 0 0 0 2 1 17.0 20.0
20 2 0 1 1 0 2 17.75 17.5
21 2 0 2 2 1 0 13.0 12.0
22 2 1 0 1 0 2 8.75 12.25
23 2 1 1 2 1 0 12.25 9.0
24 2 1 2 0 2 1 13.0 11.25
25 2 2 0 2 1 0 10.0 10.0
26 2 2 1 0 2 1 14.5 17.75
27 2 2 2 1 0 2 8.0 11.0
''')

Df=pd.read_csv(TempIO,delimiter=r"\s+",header=0)
Df.head()
Df.A=Df.A.apply(int)

DesMat=Df[["A","B","C","D","E","F"]]
DesMat.loc[:,"AC"]=(DesMat.A+DesMat.C)%3
DesMat.loc[:,"AC2"]=(DesMat.A+2*DesMat.C)%3
DesMat.loc[:,"AE"]=(DesMat.A+DesMat.E)%3
DesMat.loc[:,"AE2"]=(DesMat.A+2*DesMat.E)%3

Df.loc[:,"Sbar"]=Df[["S1","S2"]].apply(statistics.mean,axis=1)
Df.loc[:,"S_lns2"]=Df[["S1","S2"]].apply(statistics.variance,axis=1).apply(lambda x : math.log(x) if x!=0 else math.log(0.1**20))


f,axes=plt.subplots(2,3,sharex=True,sharey=True)
g=sns.factorplot(x="A",y="Sbar",data=Df,ci=None,ax=axes[0,0])
g=sns.factorplot(x="B",y="Sbar",data=Df,ci=None,ax=axes[0,1])
g=sns.factorplot(x="C",y="Sbar",data=Df,ci=None,ax=axes[0,2])
g=sns.factorplot(x="D",y="Sbar",data=Df,ci=None,ax=axes[1,0])
g=sns.factorplot(x="E",y="Sbar",data=Df,ci=None,ax=axes[1,1])
g=sns.factorplot(x="F",y="Sbar",data=Df,ci=None,ax=axes[1,2])
plt.tight_layout()
f.savefig("MainEffPlt.png")


fig1=interaction_plot(Df.A,Df.C,Df.Sbar)
fig2=interaction_plot(Df.A,Df.E,Df.Sbar)

#frames=[DesMat,DesMat]
#Df1=pd.concat(frames)
#Df1.loc[:,"Y"]=Df.S1.tolist()+Df.S2.tolist()
#
#Df1.to_csv("Q9Dat.csv")

f2,axes1=plt.subplots(2,3,sharex=True,sharey=True)
g=sns.factorplot(x="A",y="S_lns2",data=Df,ci=None,ax=axes1[0,0])
g=sns.factorplot(x="B",y="S_lns2",data=Df,ci=None,ax=axes1[0,1])
g=sns.factorplot(x="C",y="S_lns2",data=Df,ci=None,ax=axes1[0,2])
g=sns.factorplot(x="D",y="S_lns2",data=Df,ci=None,ax=axes1[1,0])
g=sns.factorplot(x="E",y="S_lns2",data=Df,ci=None,ax=axes1[1,1])
g=sns.factorplot(x="F",y="S_lns2",data=Df,ci=None,ax=axes1[1,2])
plt.tight_layout()
fig1=interaction_plot(Df.A,Df.C,Df.S_lns2)
fig2=interaction_plot(Df.A,Df.E,Df.S_lns2)

regr=linear_model.LinearRegression()
# Train the model using the training sets
Zs1=Df.S_lns2
R1=regr.fit(DesMat,Zs1)
R1.intercept_

ef2=R1.coef_
ef2
dat=pd.DataFrame({"FactEff":ef2,"Var1":DesMat.columns})
#dat

HalfPlt_V1(dat,'FactEff','Var1','HalfPlotStrn_Q9.png')

#IER at alpha =5% and 
LenthsTest(dat,'FactEff',"LenthTestDisper_Str_Q9.csv",IER_Alpha=2.21)


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
datXy=pd.DataFrame()
datXy["Al"]=DesMat.A.apply(contrast_l)
datXy["Bl"]=DesMat.B.apply(contrast_l)
datXy["Cl"]=DesMat.C.apply(contrast_l)
datXy["Dl"]=DesMat.D.apply(contrast_l)
datXy["El"]=DesMat.E.apply(contrast_l)
datXy["Fl"]=DesMat.F.apply(contrast_l)

datXy["Aq"]=DesMat.A.apply(contrast_q)
datXy["Bq"]=DesMat.B.apply(contrast_q)
datXy["Cq"]=DesMat.C.apply(contrast_q)
datXy["Dq"]=DesMat.D.apply(contrast_q)
datXy["Eq"]=DesMat.E.apply(contrast_q)
datXy["Fq"]=DesMat.F.apply(contrast_q)

datXy["ACll"]=datXy.Al*datXy.Cl
datXy["AClq"]=datXy.Al*datXy.Cq
datXy["ACql"]=datXy.Aq*datXy.Cl
datXy["ACqq"]=datXy.Aq*datXy.Cq

datXy["AEll"]=datXy.Al*datXy.El
datXy["AElq"]=datXy.Al*datXy.Eq
datXy["AEql"]=datXy.Aq*datXy.El
datXy["AEqq"]=datXy.Aq*datXy.Eq

datXy.loc[:,"Y"]=Df.Sbar

datXy.to_csv("DesDatQ9.csv")