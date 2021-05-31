# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:57:58 2020

@author: Eric Russell
eric.s.russell@wsu.edu
LAR-CEE
Washington State University


Created for use to investigate climate data for the Pullman-Moscow Airport historical data (KPUW)
First/Last Freeze plot

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime

data = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\NCDC\USC00456789_2020.csv',header=0, index_col='DATE')
data.index = pd.to_datetime(data.index)
dg = data.groupby(pd.Grouper(freq='Y'))
#%%
bf,cf = [],[]
for k in dg.groups.keys():
    df = dg['TMIN'].get_group(k)
    qn = df[df.index[182:]]<=0
    x = np.cumsum(qn)==1
    if sum(qn) >=1:
        bf.append([str(qn.index[x][0])[0:10],str(qn.index[x][0])[0:4]])
    else:
        bf.append([0,str(qn.index[0])[0:4]])
     
    df = dg['TMAX'].get_group(k)
    qn = df[df.index[182:]]<=0
    x = np.cumsum(qn)==1
    if sum(qn) >=1:
        cf.append([str(qn.index[x][0])[0:10],str(qn.index[x][0])[0:4]])
    else:
        cf.append([0,str(qn.index[0])[0:4]])

cf = pd.DataFrame(cf, columns=['doy','year'])            
cf.index = cf['year']
FirstMin = pd.DataFrame(pd.to_datetime(cf['doy']).dt.dayofyear)
yr = pd.DataFrame(FirstMin.index.astype(int))
test = []
for k in range (0,len(yr)):
    test.append(datetime.datetime(yr['year'][k], 1, 1) + datetime.timedelta(FirstMin['doy'][k].astype(float)- 1))
a = pd.DataFrame(pd.to_datetime(test)).astype(str)
# FirstMin['Date']  =a


bf = pd.DataFrame(bf, columns=['doy','year'])            
bf.index = bf['year']
FirstMax = pd.DataFrame(pd.to_datetime(bf['doy']).dt.dayofyear)
yr = pd.DataFrame(FirstMax.index.astype(int))
test = []
for k in range (0,len(yr)):
    test.append(datetime.datetime(yr['year'][k], 1, 1) + datetime.timedelta(FirstMax['doy'][k].astype(float)- 1))
FirstMaxDate = pd.DataFrame(pd.to_datetime(test)).astype(str)

#%%
qn = FirstMin['doy']>2
qn1 = FirstMax['doy']>2
tF = FirstMax['doy'][qn1].rolling(5).mean()
tM = FirstMin['doy'][qn].rolling(5).mean()

plt.figure(figsize=(8,6))
plt.plot(FirstMax[qn1].index.values.astype(int),FirstMax['doy'][qn1],'-s')
plt.plot(FirstMin[qn].index.values.astype(int),FirstMin['doy'][qn],'-s')
plt.yticks(np.linspace(200, 365,12),fontsize = 14)
# plt.ylim([205, 365])
plt.ylabel('Day of Year', fontsize = 14)
plt.xticks(np.linspace(1940, 2020,17),rotation = 90,fontsize = 14)
plt.xlabel('Year', fontsize = 14)
plt.legend(['TMIN','TMAX'], fontsize = 16)
plt.tight_layout()
# plt.savefig(r'C:\Users\russe\Desktop\LTAR\Website Blogs\FirstFreeze.png',dpi= 350)

