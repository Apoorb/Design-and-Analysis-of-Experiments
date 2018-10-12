#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:08:07 2018
@author: Apoorb
"""
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

import pandas as pd 
from sklearn import linear_model
import numpy as np

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