#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:30:49 2018

@author: Apoorb
"""

'''
# 12 2 way effects
Df["ABll"]=Df.loc[:,["Al","Bl"]].apply(np.product,axis=1)
Df["ABlq"]=Df.loc[:,["Al","Bq"]].apply(np.product,axis=1)
Df["ABql"]=Df.loc[:,["Aq","Bl"]].apply(np.product,axis=1)
Df["ABqq"]=Df.loc[:,["Aq","Bq"]].apply(np.product,axis=1)

Df["ACll"]=Df.loc[:,["Al","Cl"]].apply(np.product,axis=1)
Df["AClq"]=Df.loc[:,["Al","Cq"]].apply(np.product,axis=1)
Df["ACql"]=Df.loc[:,["Aq","Cl"]].apply(np.product,axis=1)
Df["ACqq"]=Df.loc[:,["Aq","Cq"]].apply(np.product,axis=1)

Df["BCll"]=Df.loc[:,["Bl","Cl"]].apply(np.product,axis=1)
Df["BClq"]=Df.loc[:,["Bl","Cq"]].apply(np.product,axis=1)
Df["BCql"]=Df.loc[:,["Bq","Cl"]].apply(np.product,axis=1)
Df["BCqq"]=Df.loc[:,["Bq","Cq"]].apply(np.product,axis=1)

# 8 3 way effects
Df["ABClll"]=Df.loc[:,["Al","Bl","Cl"]].apply(np.product,axis=1)
Df["ABCllq"]=Df.loc[:,["Al","Bl","Cq"]].apply(np.product,axis=1)
Df["ABClql"]=Df.loc[:,["Al","Bq","Cl"]].apply(np.product,axis=1)
Df["ABCqll"]=Df.loc[:,["Aq","Bl","Cl"]].apply(np.product,axis=1)
Df["ABClqq"]=Df.loc[:,["Al","Bq","Cq"]].apply(np.product,axis=1)
Df["ABCqql"]=Df.loc[:,["Aq","Bq","Cl"]].apply(np.product,axis=1)
Df["ABCqlq"]=Df.loc[:,["Aq","Bl","Cq"]].apply(np.product,axis=1)
Df["ABCqqq"]=Df.loc[:,["Aq","Bq","Cq"]].apply(np.product,axis=1)
'''