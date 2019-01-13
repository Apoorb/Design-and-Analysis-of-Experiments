# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:15:03 2018

@author: A-Bibeka
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import statistics

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
    
