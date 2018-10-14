#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:08:07 2018
@author: Apoorb
"""
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


print('Current working directory ',os.getcwd())
#os.chdir('C:/Users/a-bibeka/Documents/GitHub/Python-Code-Compilation')
os.chdir('/Users/Apoorb/Documents/GitHub/Python-Code-Compilation')
 
print('Current working directory ',os.getcwd())

# Function to plot the Half-Normal Plot
def HalfPlt_V1(DatTemp,Theta,Var_,PltName):
    len1 =len(DatTemp[Var_])
    DatTemp['absTheta']=DatTemp[Theta].apply(abs)
    DatTemp=DatTemp.sort_values(by=['absTheta'])
    DatTemp = DatTemp.reset_index(drop=True)
    DatTemp['i']= np.linspace(1,len1,len1).tolist()
    DatTemp['NorQuant']=DatTemp['i'].apply(lambda x:norm.ppf(0.5+0.5*(x-0.5)/len1))
    fig1, ax1 =plt.subplots()
    ax1.scatter(DatTemp['NorQuant'], DatTemp['absTheta'], marker='x', color='red')
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
def LenthsTest(dat,fef,fileNm,IER_5per=2.30):
    len1=len(dat[fef])
    dat['absEff']=dat[fef].apply(abs)
    s0=1.5*statistics.median(map(float,dat['absEff']))
    tpLst=[i for i in dat['absEff'] if i<2.5*s0]
    PSE =1.5 * statistics.median(tpLst)
    #Lenth's t stat
    dat['t_PSE'] = (round(dat[fef]/PSE,2))
    dat['IER_0.05']=[IER_5per]*len1
    dat['Significant'] = dat.apply(lambda x : 'Significant' if abs(x['t_PSE']) > x['IER_0.05'] else "Not Significant", axis=1)
    dat.to_csv(fileNm)
    return(dat)

data1=StringIO('''OilTemp CarbonCont TempStBfQuench PerCrackSpring
70 0.50 1450 67
70 0.50 1600 79
70 0.70 1450 61
70 0.70 1600 75
120 0.50 1450 59
120 0.50 1600 90
120 0.70 1450 52
120 0.70 1600 87''')

# Read the data above 
CrSp_data=pd.read_csv(data1,delimiter=r"\s+",header=0)

# Y value
y=CrSp_data['PerCrackSpring']
y=np.matrix(y)
yTrans=np.transpose(y)
# Get the dimensions of about data
dim=CrSp_data.shape
nrow=dim[0]
#Create a numpy matrix with same number of rows as the data and
# 7 columns (3 main effects, 3 two way interactions and 1 three way 
#interaction)
DesignMat=np.empty([nrow,7])
DesignMat[:,0] =[-1 if(i==70) else +1 for i in CrSp_data['OilTemp']]
DesignMat[:,1] =[-1 if(i<0.51) else +1 for i in CrSp_data['CarbonCont']]
DesignMat[:,2] =[-1 if(i==1450) else +1 for i in CrSp_data['TempStBfQuench']]
DesignMat[:,3]=DesignMat[:,0]*DesignMat[:,1]
DesignMat[:,4]=DesignMat[:,0]*DesignMat[:,2]
DesignMat[:,5]=DesignMat[:,1]*DesignMat[:,2]
DesignMat[:,6]=DesignMat[:,0]*DesignMat[:,1]*DesignMat[:,2]

print(np.around(DesignMat,decimals=3))

#Convert the desgn matrix in pandas
DesignMat_Panda= pd.DataFrame(DesignMat)
DesignMat_Panda.rename(columns={0: 'A', 1: 'B', 2: 'C',3:'AB',4:'AC',5:'BC',6:'ABC'}, inplace=True)
DesignMat_Panda
#nrow/2 will give the average for +Zi's and -ve Zi's
YBar= (y*DesignMat)/(nrow/2)

#############################################33
#Verify if we can get the factorial effects from the regression
#Coef
X=DesignMat_Panda
#X['Const']=[1]*8
#cols=X.columns.tolist()
#cols=cols[-1:]+cols[:-1]
#X=X[cols]
Y=pd.DataFrame(yTrans)
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
R1=regr.fit(X,Y)
R1.intercept_
Coefs=R1.coef_
#Factorial effect is twice the coeff
Coefs=2*Coefs
Coefs=np.round(Coefs,2)
#Factorial effect from manual calculation
":)"if(np.allclose(YBar,Coefs)) else ":("

#Effect for Full Factorial Design
YBar.tolist()[0]

#Optimal Settings
# A -ve, B +ve, C-ve,AB na, AC -ve ,BC -ve, ABC -ve
#AC is an important effect. Need to ensure AC is -ve

# So A +ve, B is +ve , C -ve, AB +ve,  AC -ve, BC is -ve, ABC -ve
#######################################################

# One Factor at a time design
CrSp_data.groupby(['OilTemp'])['PerCrackSpring'].mean()
CrSp_data.groupby(['CarbonCont'])['PerCrackSpring'].mean()
CrSp_data.groupby(['TempStBfQuench'])['PerCrackSpring'].mean()

# If we start with A. -ve A will give smaller value so we would
# set the value of A to -ve and then change settings of B and C thus
# we don't account for the interaction of A and C. 


#********************************************************************
#********************************************************************
ExpDa=StringIO('''No X1 X2 X3 Y RandOrd
1  A  low   Y  11  1 
2  B  low   Y  12  4 
3  A  high  Y  10  5 
4  B  high  Y  11  3 
5  A  low   Z  16  2 
6  B  low   Z  14  6 
7  A  high  Z  15  7 
8  B  high  Z  19  8  
               ''')
ExpDa1=pd.read_csv(ExpDa,delimiter=r"\s+",header=0)
Y=np.matrix(ExpDa1['Y'])
X3=[-1 if(i=='Y') else +1 for i in ExpDa1['X3']]
nrows1=len(X3)
#Main effect of X3
ME_x3=X3*np.transpose(Y)/(nrows1/2)
ME_x3

DatX=ExpDa1[[i for i in ExpDa1.columns.tolist() if i.find('X')==0]]
DatX.loc[:,'x1']=np.where(DatX['X1']=='A',-1,1)
DatX.loc[:,'x2']=np.where(DatX['X2']=='low',-1,1)
DatX.loc[:,'x3']=np.where(DatX['X3']=='Y',-1,1)
DatX.loc[:,'x1x2']=DatX['x1']*DatX['x2']
DatX.loc[:,'x1x3']=DatX['x1']*DatX['x3']
DatX.loc[:,'x2x3']=DatX['x2']*DatX['x3']
DatX.loc[:,'x1x2x3']=DatX['x1']*DatX['x2']*DatX['x3']

X_DzMat=np.matrix(DatX.loc[:,['x1','x2','x3','x1x2','x1x3','x2x3','x1x2x3']])

#Get the factorial effect
FactorialEff=Y*X_DzMat*(1/4)

DatIndex= pd.DataFrame({'Var':['x1','x2','x3','x1x2','x1x3','x2x3','x1x2x3'],
                        'Effect':FactorialEff.tolist()[0]})
DatIndex=DatIndex.sort_values(by=['Effect'])
#DatIndex.reset_index(drop=True)
Var=DatIndex['Var']
d={'FaEff':DatIndex['Effect'],'Var':Var}
dat2=pd.DataFrame(d)
HalfPlt_V1(dat2,'FaEff','Var','HalfPlot.png')


LenthsTest(dat2,'FaEff',"LenthTab.csv",2.30)
################################################################################
################################################################################
# Finding nominal the best solution
RoughDat=StringIO('''No A B C Yi1 Yi2 Yi3 Yi4 Yi5
1 -1 -1 -1 54.6 73.0 139.2 55.4 52.6 
2 -1 -1 +1 86.2 66.2 79.2 86.0 82.6 
3 -1 +1 -1 41.4 51.2 42.6 58.6 58.4 
4 -1 +1 +1 62.8 64.8 74.6 74.6 64.6 
5 +1 -1 -1 59.6 52.8 55.2 61.0 61.0 
6 +1 -1 +1 82.0 72.8 76.6 73.4 75.0 
7 +1 +1 -1 43.4 49.0 48.6 49.6 55.2 
8 +1 +1 +1 65.6 65.0 64.2 60.8 77.4 
               ''')
RoughDat1=pd.read_csv(RoughDat,delimiter=r"\s+",header=0)


RoughDat1['YBar']=RoughDat1[['Yi1','Yi2','Yi3','Yi4','Yi5']].mean(axis=1)
RoughDat1['lnsBar']=((RoughDat1[['Yi1','Yi2','Yi3','Yi4','Yi5']].std(axis=1))**2).apply(math.log)
#RoughDat1['Const']=[1]*(RoughDat1.shape[0])
RoughDat1.loc[:,'AB']=RoughDat1['A']*RoughDat1['B']
RoughDat1.loc[:,'AC']=RoughDat1['A']*RoughDat1['C']
RoughDat1.loc[:,'BC']=RoughDat1['B']*RoughDat1['C']
RoughDat1.loc[:,'ABC']=RoughDat1['A']*RoughDat1['B']*RoughDat1['C']
RoughDat1.to_csv("RoughnessDat.csv")
X2= RoughDat1.loc[:,['A','B','C','AB','AC','BC','ABC']]
Ybar=pd.DataFrame(RoughDat1['YBar'])
LnSBar=pd.DataFrame(RoughDat1['lnsBar'])


# Get the factorial effects
yBarMod=linear_model.LinearRegression()
ModSum=yBarMod.fit(X2,Ybar)
#Factorial effect is twice the coeff
FactEff=np.round(2*ModSum.coef_[0],2).tolist()
Var1=['A','B','C','AB','AC','BC','ABC']
Dat2=pd.DataFrame({'FactEff':FactEff,'Var1':Var1})
Dat2.to_excel("RoughlocEff.xls")
    
HalfPlt_V1(Dat2,'FactEff','Var1','HalfPlot1.png')
'''
DatTemp=Dat2
Theta='FactEff'
i='i_'
Var_='Var1'
'''
LenthsTest(Dat2,'FactEff',"LenthTestLoc.csv",IER_5per=2.30)
#Only consider B and C for regresion based on Lenth Test Results
X2= RoughDat1.loc[:,['B','C']]
# Get the factorial effects
M1=linear_model.LinearRegression()
M1Sum=yBarMod.fit(X2,Ybar)
M1Sum.intercept_
M1Sum.coef_


#####################
#Dispersion Effect
X2= RoughDat1.loc[:,['A','B','C','AB','AC','BC','ABC']]
lnsBarMod=linear_model.LinearRegression()
ModSum2=lnsBarMod.fit(X2,LnSBar)
ModSum2.intercept_
ModSum2.coef_
FactEff2=np.round(2*ModSum2.coef_[0],2).tolist()
Var1=['A','B','C','AB','AC','BC','ABC']


################
#Dispersion Effect
Dat3=pd.DataFrame({'FactEff':FactEff2,'Var1':Var1})
Dat3.to_excel("RoughDisperEff.xls")
HalfPlt_V1(Dat3,'FactEff','Var1','HalfPlot2.png')
LenthsTest(Dat3,'FactEff',"LenthTestDisper.csv",IER_5per=2.30)
#Only consider B and C for regresion based on Lenth Test Results
X2= RoughDat1.loc[:,['A']]
# Get the factorial effects
M2=linear_model.LinearRegression()
M2Sum=yBarMod.fit(X2,LnSBar)
M2Sum.intercept_
M2Sum.coef_




