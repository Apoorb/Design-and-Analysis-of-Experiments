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
import statsmodels.api as sm
import pandas as pd 
from sklearn import linear_model
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
import statistics
import math
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
Var=DatIndex['Var']

SrtEff=np.sort(FactorialEff).tolist()[0]

i=np.linspace(1,7,7).tolist()
d={'i':i,'SrtEff':SrtEff}
dat2=pd.DataFrame(d)

dat2['absEff']=dat2['SrtEff'].apply(abs)

def HalfPlt(i_):
    return (norm.ppf(0.5+0.5*(i_-0.5)/7))
dat2['NorQuant']=dat2['i'].apply(HalfPlt)

print('Current working directory ',os.getcwd())
os.chdir('C:/Users/a-bibeka/Documents/GitHub/Python-Code-Compilation')
print('Current working directory ',os.getcwd())
for i,type in enumerate(Var):
    x = dat2['NorQuant'][i]
    y = dat2['absEff'][i]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+0.05, y+0.05, type, fontsize=9)
    
plt.title("Half-Normal Plot")
plt.xlabel("Normal Quantile")
plt.savefig('HalfPlot.png')


# Lenth's Method for  testing signficance for experiments without
# variance estimate
s0=1.5*statistics.median(map(float,dat2['absEff']))
tpLst=[i for i in dat2['absEff'] if i<2.5*s0]
PSE =1.5 * statistics.median(tpLst)
#Lenth's t stat
DatIndex['t_PSE'] = round(DatIndex['Effect']/PSE,2)
DatIndex['IER_0.05']=[2.30]*7
DatIndex['Significant'] = DatIndex.apply(lambda x : 'Significant' if x['t_PSE'] > x['IER_0.05'] else "Not Significant", axis=1)
DatIndex.to_csv("LenthTab.csv")


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
RoughDat1['lnsBar']=(RoughDat1[['Yi1','Yi2','Yi3','Yi4','Yi5']].std(axis=1)).apply(math.log)
#RoughDat1['Const']=[1]*(RoughDat1.shape[0])
RoughDat1.loc[:,'AB']=RoughDat1['A']*RoughDat1['B']
RoughDat1.loc[:,'AC']=RoughDat1['A']*RoughDat1['C']
RoughDat1.loc[:,'BC']=RoughDat1['B']*RoughDat1['C']
RoughDat1.loc[:,'ABC']=RoughDat1['A']*RoughDat1['B']*RoughDat1['C']
X2= RoughDat1.loc[:,['A','B','C','AB','AC','BC','ABC']]
Ybar=pd.DataFrame(RoughDat1['YBar'])
LnSBar=pd.DataFrame(RoughDat1['lnsBar'])


def HalfPlt(DatTheta):
    


yBarMod=linear_model.LinearRegression()
ModSum=yBarMod.fit(X2,Ybar)
ModSum.intercept_
ModSum.coef_

lnsBarMod=linear_model.LinearRegression()
ModSum1=lnsBarMod.fit(X2,LnSBar)
ModSum1.intercept_
ModSum1.coef_