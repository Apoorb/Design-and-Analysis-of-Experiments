# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:09:49 2019

@author: A-Bibeka
Read data from travel time files
"""

import os
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
#os.chdir("D:/Dropbox/TTI_Projects/Road User Cost/VISSIM AM Peak V14/NB/NB 2045")
os.getcwd()
ListFiles=glob.glob('D:/Dropbox/TTI_Projects/Road User Cost/VISSIM AM Peak V14/NB/NB 2045/*Vehicle Travel Time Results.att')
Lfs2=glob.glob("D:/Dropbox/TTI_Projects/Road User Cost/VISSIM PM Peak V14/NB/NB 2045/*Vehicle Travel Time Results.att")
ListFiles=ListFiles+Lfs2
#file = ListFiles[1]
#file = "AM 2045 NB Base_Vehicle Travel Time Results.att"
def TTSegName(x):
    if(x==1):Nm="69_SB"
    elif(x==2):Nm="69_NB"
    elif(x==3):Nm="Spur_SB"
    else:Nm="Spur_NB"
    return Nm
Findat= pd.DataFrame()
for file_Fu in ListFiles:
    file=file_Fu.split("\\")[1]
    desc1= file[0:10]
    Time,Year, SenDir=desc1.split(" ")
    Scenario=file[11:].split("_Vehicle Travel Time Results.att")[0]
    Scenario=Scenario.split(" Cor2")[0]
    dat=pd.read_csv(file_Fu,sep =';',skiprows=17)
    mask=dat["$VEHICLETRAVELTIMEMEASUREMENTEVALUATION:SIMRUN"]=="AVG"
    dat["TTSegNm"]=dat['VEHICLETRAVELTIMEMEASUREMENT'].apply(TTSegName)
    mask2=(dat["TTSegNm"]=="69_NB")|(dat["TTSegNm"]=="Spur_NB")
    dat = dat[mask & mask2]
    dat
    dat["Scenario"] = Scenario
    dat["Time"]= Time
    dat["Year"] = np.int(Year)
    dat["SenDir"]= SenDir
    Findat=Findat.append(dat)
    Findat["Delay"]= Findat["VEHS(ALL)"]*(Findat["TRAVTM(ALL)"]-(Findat["DISTTRAV(ALL)"]/(1.47*65)))/3600
    Findat["Delay"]=np.round( Findat["Delay"])
Findat["SMS"] =np.round(Findat["DISTTRAV(ALL)"]/Findat["TRAVTM(ALL)"]/1.47,1)



RdFl1= os.path.join("D:/Dropbox/TTI_Projects/Road User Cost","InpVolKey.csv")
InputVolDat=pd.read_csv(RdFl1)
InputVolDat=InputVolDat.drop("Scenario2",axis=1)
Findat=Findat.merge(InputVolDat,how='inner',on=["Scenario","Year","Time","TTSegNm"])
Findat["LatentDelay"]=(Findat["InpVol"]-Findat["VEHS(ALL)"])*(Findat["TRAVTM(ALL)"]-(Findat["DISTTRAV(ALL)"]/(1.47*65)))/3600
Findat["LatentDelay"]=np.round(Findat["LatentDelay"],1)
Findat["TotalDelay"] = np.round(Findat["Delay"]+Findat["LatentDelay"],1)
Findat_NB_69= Findat[Findat["TTSegNm"]=="69_NB"]
Findat_NB_69["Sen"]=Findat_NB_69["Scenario"].apply(lambda x: str.split(x,"_")[0])
Findat_NB_69["LaneDrLoc"]=Findat_NB_69["Scenario"].apply(lambda x: str.split(x," ")[-1])
Findat_NB_69["LaneDrLoc"]=Findat_NB_69["LaneDrLoc"].apply(lambda x: "L" if x=="Ln" else x)

g=sns.catplot(x="Sen",y="SMS",hue="LaneDrLoc",col="Time",data=Findat_NB_69,kind="bar",order=["Base","4+2","5+1","5+2"])
Findat_Wd=pd.pivot_table(Findat_NB_69,index="Scenario",columns=["Time","Year"],values =["SMS","VEHS(ALL)","Delay","LatentDelay","TotalDelay"])
new_index=['Base','4+2_NoShared Ln R','4+2_NoShared Ln','5+1_Flared Ln R','5+1_Flared Ln','5+2_Shared Ln R','5+2_Shared Ln']
Findat_Wd=Findat_Wd.reindex(new_index)

Findat_Wd

#################################
# SB 
ListFiles=glob.glob('D:/Dropbox/TTI_Projects/Road User Cost/VISSIM AM Peak V14/SB/SB 2045/*Vehicle Travel Time Results.att')
Lfs2=glob.glob("D:/Dropbox/TTI_Projects/Road User Cost/VISSIM PM Peak V14/SB/SB 2045/*Vehicle Travel Time Results.att")
ListFiles=ListFiles+Lfs2
#file = ListFiles[1]
#file = "AM 2045 NB Base_Vehicle Travel Time Results.att"
def TTSegName(x):
    if(x==1):Nm="69_SB"
    elif(x==2):Nm="69_NB"
    elif(x==3):Nm="Spur_SB"
    else:Nm="Spur_NB"
    return Nm
Findat= pd.DataFrame()
for file_Fu in ListFiles:
    file=file_Fu.split("\\")[1]
    desc1= file[0:10]
    Time,Year, SenDir=desc1.split(" ")
    Scenario2=file[11:].split("_Vehicle Travel Time Results.att")[0]
    Scenario2=Scenario2.split(" Cor2")[0]
    dat=pd.read_csv(file_Fu,sep =';',skiprows=17)
    mask=dat["$VEHICLETRAVELTIMEMEASUREMENTEVALUATION:SIMRUN"]=="AVG"
    dat["TTSegNm"]=dat['VEHICLETRAVELTIMEMEASUREMENT'].apply(TTSegName)
    mask2=(dat["TTSegNm"]=="69_SB")|(dat["TTSegNm"]=="Spur_SB")
    dat = dat[mask & mask2]
    dat
    dat["Scenario2"] = Scenario2
    dat["Time"]= Time
    dat["Year"] = np.int(Year)
    dat["SenDir"]= SenDir
    Findat=Findat.append(dat)
    Findat["Delay"]= Findat["VEHS(ALL)"]*(Findat["TRAVTM(ALL)"]-(Findat["DISTTRAV(ALL)"]/(1.47*65)))/3600
    Findat["Delay"]=np.round( Findat["Delay"])
Findat["SMS"] =np.round(Findat["DISTTRAV(ALL)"]/Findat["TRAVTM(ALL)"]/1.47,1)



RdFl1= os.path.join("D:/Dropbox/TTI_Projects/Road User Cost","InpVolKey.csv")
InputVolDat=pd.read_csv(RdFl1)
InputVolDat=InputVolDat.drop("Scenario",axis=1)
Findat=Findat.merge(InputVolDat,how='inner',on=["Scenario2","Year","Time","TTSegNm"])
Findat["LatentDelay"]=(Findat["InpVol"]-Findat["VEHS(ALL)"])*(Findat["TRAVTM(ALL)"]-(Findat["DISTTRAV(ALL)"]/(1.47*65)))/3600
Findat["LatentDelay"]=np.round(Findat["LatentDelay"],1)
Findat["TotalDelay"] = np.round(Findat["Delay"]+Findat["LatentDelay"],1)
Findat_SB_69= Findat[Findat["TTSegNm"]=="69_SB"]
Findat_Wd_SB=pd.pivot_table(Findat_SB_69,index="Scenario2",columns=["Time","Year"],values =["SMS","VEHS(ALL)","Delay","LatentDelay","TotalDelay"])
new_index=['Base','4+2_1500ft','4+2_2500ft','4+2_No LnDrop','5+1_1500ft','5+1_2500ft','5+1_No LnDrop']
Findat_Wd_SB=Findat_Wd_SB.reindex(new_index)



