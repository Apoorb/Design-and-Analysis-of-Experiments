# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:57:21 2018

@author: A-Bibeka
"""

import sys
import googlemaps
import pandas as pd 

sys.path.insert(0,'C:/Users/a-bibeka/Dropbox/TTI_Projects/Google map API')
#Client key - my key
f=open("Ap-test API key.txt")
keyAp = f.readline()
gmaps_ap = googlemaps.Client(key=keyAp)

from GmapsPythonAPI_fun import distance_matrix
from GmapsPythonAPI_fun import elevation

path = {'Matlock EB':[[32.677600, -97.111051],[32.677799, -97.108480]],
'S Collins EB':[[32.677162, -97.088917],[32.677400, -97.085181]],
'S Great SW EB':[[32.675228, -97.042483],[32.675847, -97.037290]],
'LakeRidge EB':[[32.675190, -97.022844],[32.675801, -97.017893]],
'Robinson EB':[[32.675598, -97.020201],[32.675762, -97.014198]],
'S Carrier EB':[[32.674547, -97.006681],[32.674408, -97.002187]],
'Beltline EB':[[32.671589, -96.984894],[32.671461, -96.980340]]}
df=pd.DataFrame()
for k,value in path.items():
        a=elevation(gmaps_ap,value)
        v1=','.join(str(e) for e in value[0])
        v2=','.join(str(e) for e in value[1])
        b=distance_matrix(gmaps_ap,v1,v2,units='metric')
        b=b['rows'][0]['elements'][0]['distance']['value']
        res1=pd.DataFrame(a)
        #Convert the location column into separate lat lng column 
        temp=res1.iloc[:,1].apply(pd.Series)
        #Add the new columns to res1
        res1['lat']=temp['lat']
        res1['lng']=temp['lng']
        res1['distance']=b
        del res1['location']
        res1['path']=k
        res1['location']='NaN'
        res1.loc[res1.index[0], 'location']="Ramp_st"
        res1.loc[res1.index[1], 'location']="Ramp_end"
        df=df.append(res1, ignore_index=True)

#Create a seperate data frame with distance 
#(Only elevation can be used to transform df to wide data)
tp=df[['path','distance']]
#Distance for both rows is the same
tp=tp.drop_duplicates(subset='path',keep='first')
#Long to Wide
df_wide=df.pivot(index='path',columns='location',values='elevation')
df_wide.reset_index(inplace=True)
#Combine data 
df1=pd.merge(df_wide,tp,on='path')
#Buffer
df1['elevation']=(df1['Ramp_end']-df1['Ramp_st'])
df1['elevation']=df1['elevation']*(1/df1['distance'])
#Save as csv file 
df1.to_csv('C:/Users/a-bibeka/Dropbox/TTI_Projects/Google map API/test_data1.csv', sep=',', encoding='utf-8',index=False)
print(df1)

org = [32.677600, -97.111051]
des= [32.677799, -97.108480]
l=[org,des]
elevation(gmaps_ap,l)
org= ','.join(str(e) for e in org)
des=','.join(str(e) for e in des)
res=distance_matrix(gmaps_ap,org,des,units='metric')
a=res['rows'][0]['elements'][0]['distance']['value']

