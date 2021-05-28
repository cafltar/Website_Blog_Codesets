# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:58:37 2021

@author: russe
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 08:59:17 2019

@author: Eric
"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:\\Users\\russe\\Desktop\\PyScripts\\Flux_Processing_Code')       
import LTAR_QAQC_Online as LLT

def doy(df):
    day_of_year = []
    for k in range (0,len(df)):
        day_of_year.append(time.strptime(str(df.index[k])[0:10], "%Y-%m-%d").tm_yday)
    return day_of_year
        
def Monthly(CE):
    CEM = CE.groupby(pd.Grouper(freq = 'M')).mean()
    CEMSS = CE.groupby(pd.Grouper(freq = 'M')).sum()
    z = []
    for k in range (0,len(CEM)):
        z.append(str(CEM.index[k])[5:7])
    CEM['M'] = z
    CEMSS['M'] = z
    CEM2 = CEM.groupby('M').mean()
    CEMS = CEM.groupby('M').std()
    CEMSS = CEMSS.groupby('M').mean()

    return CEM2, CEMS, CEMSS

Climo = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\NCDC\USC00456789.csv',header = 0, index_col = 'DATE')
Climo.index = pd.to_datetime(Climo.index)
Climo['PRCP'] = Climo['PRCP']/25.4

Climo['TMIN'] = Climo['TMIN']*1.8+32
Climo['TMAX'] = Climo['TMAX']*1.8+32

Climo = LLT.indx_fill(Climo, 'D')
P_Climo = Climo.groupby(pd.Grouper(freq = 'Y'))
pld = []
pld_yr = []
tmin = [];tmin = pd.DataFrame(tmin)
tmax = [];tmax = pd.DataFrame(tmax)
prc = [];prc = pd.DataFrame(prc)
q=[]
Climo = LLT.indx_fill(Climo, 'D')
P_Climo = Climo.groupby(pd.Grouper(freq = 'Y'))
pld = []
for k in P_Climo.groups.keys():
    q = P_Climo.get_group(k)
    pn = q['PRCP']==0 
    tmn = q['TMIN']>=45
    tmn40 = q['TMIN']>=40
    tmx = q['TMAX']<=85
    tm = ((q['TMAX']+q['TMIN'])/2 >=55) & ((q['TMAX']+q['TMIN'])/2 <= 75)
    pld.append([np.sum(pn&tmn&tmx&tm),np.sum(~tm),np.sum(~tmx),np.sum(~tmn),np.sum(~pn),np.sum(~tmn40),np.sum(pn&tmn40&tmx&tm), str(k)[0:4]])
    dy = doy(q)
    q.index =dy
    tmin = pd.concat([tmin,q['TMIN']],axis = 1)
    tmax = pd.concat([tmax,q['TMAX']],axis = 1)
    # prc = pd.concat([prc,q['PRCP']],axis = 1)
    pld_yr.append([pn&tmn&tmx&tm,tm,tmx,tmn,pn,pn&tmn40&tmx&tm])
pld= pd.DataFrame(pld)

#%%
plt.figure(111, figsize=(8,4))
plt.plot(pld[7].astype(float),pld[0], linewidth =2)
plt.xticks(np.arange(1970,2021,5), fontsize = 14)
plt.xlim([1968,2022])
plt.yticks(np.arange(20,81,10), fontsize = 14)
plt.ylabel('Number of pleasant days',fontsize = 18)
plt.xlabel('Year',fontsize = 18)
plt.tight_layout()

# %% Percent based on temperature 40 vs 45
Full45 = []; Full45 = pd.DataFrame(Full45)
Full40 = []; Full40 = pd.DataFrame(Full40)
for k in range (0,len(pld_yr)):
    Full45 = pd.concat([Full45,pd.DataFrame(pld_yr[k][0].values)],axis = 1)
    Full40 = pd.concat([Full40,pd.DataFrame(pld_yr[k][5].values)],axis = 1)
Full45 = np.nanmean(Full45, axis = 1); 
Full45 = pd.DataFrame(Full45)
Full40 = np.nanmean(Full40, axis = 1); 
Full40 = pd.DataFrame(Full40)
#%
plt.figure(4,figsize=(8,4))
plt.plot(Full45*100,linewidth = 2,label='MinTemp-45F')
plt.plot(Full40*100,linewidth = 2,label='MinTemp-40F')
plt.xticks(np.arange(1,366,30), fontsize = 14)
plt.yticks(np.arange(0,70.1,10), fontsize = 14)
plt.ylabel('Percent Days',fontsize = 18)
plt.xlabel('DoY',fontsize = 18)
plt.legend(fontsize = 14)
plt.legend()
plt.tight_layout()
#%% Daily max/min temperature across years


plt.figure(12,figsize=(8,6))
plt.plot(np.nanmean(tmin,axis=1),'b',linewidth = 3,label = 'Min. Temp.')
plt.plot(np.nanmean(tmax,axis=1),'r',linewidth = 3,label = 'Max. Temp.')
plt.xticks(np.arange(1,366,30), fontsize = 15)
# plt.xlim([1968,2022])
plt.yticks(np.arange(10,100.1,10), fontsize = 16)
plt.ylabel('Temperature [Fahrenheit]',fontsize = 18)
plt.xlabel('DoY',fontsize = 18)
plt.legend(fontsize = 18)
plt.tight_layout()

#%% Breakdown of each component

plt.figure(5,figsize=(11,8))
ax= plt.subplot(211)
plt.plot(pld[7].astype(float),pld[0], label = 'Tmin 45F')
plt.plot(pld[7].astype(float),pld[6], label = 'Tmin 40F')
plt.xticks(np.arange(1970,2021,5), fontsize = 14)
plt.xlim([1968,2022])
plt.yticks(np.arange(20,81,10), fontsize = 14)
plt.ylabel('Number of pleasant days',fontsize = 18)
plt.xlabel('Year',fontsize = 18)
plt.legend()
plt.tight_layout()

ax= plt.subplot(212)
plt.plot(pld[7].astype(float),pld[1], label = 'MeanTemp')
plt.plot(pld[7].astype(float),pld[2], label = 'MaxTemp')
plt.plot(pld[7].astype(float),pld[3], label = 'MinTemp-45')
plt.plot(pld[7].astype(float),pld[5], label = 'MinTemp-40')
plt.plot(pld[7].astype(float),pld[4], label = 'Precip')
plt.xticks(np.arange(1970,2021,5), fontsize = 14)
plt.xlim([1968,2022])
plt.yticks(np.arange(0,350,25), fontsize = 14)
plt.ylabel('Number of days removed',fontsize = 18)
plt.xlabel('Year',fontsize = 18)
plt.legend()
plt.tight_layout()