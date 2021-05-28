# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:32:04 2019

@author: Eric

Note: This is not the prettiest code there is but it is functional and does as advertised.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
pd.options.mode.chained_assignment = None  # default='warn'

def indx_fill(df, time):   
    df.index = pd.to_datetime(df.index)
        # Sort index in case it came in out of order, a possibility depending on filenames and naming scheme
    df = df.sort_index()
        # Remove any duplicate times, can occur if files from mixed sources and have overlapping endpoints
    df = df[~df.index.duplicated(keep='first')]
        # Fill in missing times due to tower being down and pad dataframe to midnight of the first and last day
    idx = pd.date_range(df.index[0].floor('D'),df.index[len(df.index)-1].ceil('D'),freq = time)
    df = df.reindex(idx, fill_value=np.NaN)
    return df

def Yearly(data, col):
    data['doy'] = np.NaN
    for z in range(0,len(data)):
        data['doy'][z] = time.strptime(str(data.index[z])[0:10], "%Y-%m-%d").tm_yday
    pld=[]
    gg = data.groupby('doy')
    for kk in gg.groups.keys():
        ab = gg.get_group(kk)[col]
        pld.append([np.sum(ab),np.nanmean(ab),np.std(ab),kk])
    pld = pd.DataFrame(pld, columns =['SUM','MEAN','STD','doy'])
    return pld

#%%
dt = []
data = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\NCDC\PUW_Hourly_Data.csv', header = 0)
#data = pd.read_csv(r'C:\Users\Eric\Desktop\LTAR\Climatology\NCDC\Omak_Ephrata_hourlyData_2.csv', header = 0)
for k in range (0,len(data)):
    Y =  str(int(data['YRMODAHRMN'][k]))[0:4]
    M =  str(int(data['YRMODAHRMN'][k]))[4:6]  
    D =  str(int(data['YRMODAHRMN'][k]))[6:8]
    hh = str(int(data['YRMODAHRMN'][k]))[8:10] 
    mm = str(int(data['YRMODAHRMN'][k]))[10:12]
    dt.append(Y+'-'+M+'-'+D+' '+hh+':'+mm)
dt = pd.DataFrame(dt);data.index = dt[0]
data.index=pd.to_datetime(data.index) # Time-based index
data = data.sort_index()
data = data.replace('^\*+$', np.nan, regex=True)
data = data[~data.index.duplicated(keep='first')]
hld = data['TEMP'].astype(float).resample('H').mean()
hld = hld.between_time('7:00','20:00')
check = hld.groupby(pd.Grouper(freq='D'))
#%%
no = []
for k in check.groups.keys():
    if len(data[str(k)[0:10]]) >0:
        hld = check.get_group(k)
        no.append([str(k)[0:10],len(hld)/24, sum(~np.isnan(hld))/14])
tt = pd.DataFrame(no)
tt.index = pd.to_datetime(tt[0])

plt.figure(1, figsize = (8,4))
plt.plot(tt[2],'k',alpha = 0.7)
plt.fill_betweenx(tt[2],pd.to_datetime('1974'),pd.to_datetime('1998'), color =  'red',alpha = 0.25)
plt.fill_betweenx(tt[2],pd.to_datetime('1998'),pd.to_datetime('1999'), color =  'yellow',alpha = 0.25)
plt.fill_betweenx(tt[2],pd.to_datetime('1999'),pd.to_datetime('2006'), color =  'green',alpha = 0.25)
plt.ylabel('Fractional Data Coverage',fontsize = 14)
plt.xlabel('Year',fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylim([0, 1.1])
plt.tight_layout()
#plt.savefig(r'C:\Users\Eric\Desktop\LTAR\Climatology\Pleasant\PDFractionalCoverage_Temperature.png',dpi = 322)




#%%
data = data['01-01-1999':'12-31-2018']
Day = data.between_time('7:00','20:00')
Day['TEMP']  = Day['TEMP'].astype(float)
Day['PCP01']  = Day['PCP01'].astype(float)
# Rest is basically the same as the current usage but in F, not C and need to change the way the precip is dealt with.
Day['TEMP'] = (Day['TEMP']-32)*(5/9)

prcp = Day['PCP01'].resample('D').sum()
prcp = pd.DataFrame(prcp)

py = Day['PCP01'].resample('Y').sum() # Precip before 2000 is mostly crap so not sure how useful it is for this purpose but need to show all the data

prcp = pd.DataFrame(prcp)
Precip = prcp == 0

mea = Day.groupby(pd.Grouper(freq = 'D')).mean()
mi = Day.groupby(pd.Grouper(freq = 'D')).min()
ma = Day.groupby(pd.Grouper(freq = 'D')).max()
ME = (ma['TEMP']+mi['TEMP'])/2

MEM = (ME>12.7) & (ME < 23.9)
MY = (mea>12.7) & (mea < 23.9)
MaxY = (ma['TEMP'] < 29.4); MaxY = pd.DataFrame(MaxY, columns=['TEMP'])
MinY = (mi['TEMP']>7.2); MinY = pd.DataFrame(MinY, columns=['TEMP'])
#%
PD = MY['TEMP']&MinY['TEMP']&MaxY['TEMP']&Precip['PCP01']
PD_M = MEM&MinY['TEMP']&MaxY['TEMP']&Precip['PCP01']
y = PD.groupby(pd.Grouper(freq='Y')).sum()
y_M = PD_M.groupby(pd.Grouper(freq='Y')).sum()

Max_Miss = MaxY['TEMP'].resample('Y', loffset ='-364D').sum()
Min_Miss = MinY['TEMP'].resample('Y', loffset ='-364D').sum()
Mean_Miss =MY['TEMP'].resample('Y', loffset ='-364D').sum()
Precip_Miss =Precip['PCP01'].resample('Y', loffset ='-364D').sum()

y.index = (y.index.strftime('%Y'))
y.index = pd.to_datetime(y.index)

y_M.index = (y_M.index.strftime('%Y'))  
y_M.index = pd.to_datetime(y_M.index)

#%%
plt.figure(2, figsize=(8,4))
plt.plot(y.index,y,'.-',linewidth = 3, ms = 15)
plt.ylabel('Pleasant Days',fontsize = 14)
plt.xlabel('Year',fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylim([50, 100])
#plt.xlim(['1998','2019'])
plt.tight_layout()
#plt.savefig(r'C:\Users\Eric\Desktop\LTAR\Climatology\Pleasant\DaytimeDaysWithPrecip.png',dpi = 322)

#%%

MY_Y = Yearly(MY,'TEMP')
Min_Y = Yearly(MinY,'TEMP')
Max_Y = Yearly(MaxY,'TEMP')
P_Y = Yearly(Precip,'PCP01')
mi['TEMP'] = mi['TEMP']*(9/5)+32
mintemp = Yearly(mi,'TEMP')
mea['TEMP'] = mea['TEMP']*(9/5)+32
meantemp = Yearly(mea,'TEMP')
ma['TEMP'] = ma['TEMP']*(9/5)+32
maxtemp = Yearly(ma,'TEMP')

#%%
plt.figure(4, figsize=(9,6))
ax = plt.subplot(221)
plt.plot(mintemp['MEAN'],'b', label = 'T$_{Min}$')
plt.plot(meantemp['MEAN'],'green', label = 'T$_{Mean}$')
plt.plot(maxtemp['MEAN'],'k', label = 'T$_{Max}$')
plt.hlines(45,0,366, colors = 'red')
plt.xticks(np.arange(0,390,60),fontsize =12)
plt.xlim([-5,370])
plt.ylim([20,90])
plt.legend(fontsize = 12)
plt.ylabel('Temperature [F]',fontsize = 12)
plt.xlabel('Day of Year',fontsize = 12)
plt.yticks(fontsize =12)
ax.fill_between(mintemp['doy'],55,75,color = 'red',alpha = 0.15)
plt.hlines(85,0,366,colors = 'r')

ax = plt.subplot(222)
plt.plot(meantemp['MEAN'],'green',linewidth = 1.75)
ax.fill_between(mintemp['doy'],meantemp['MEAN']-meantemp['STD'],meantemp['MEAN']+meantemp['STD'],color = 'red',alpha = 0.15)
plt.xticks(np.arange(0,390,60),fontsize =12)
plt.xlim([-5,370])
plt.ylim([10,80])
plt.ylabel('Temperature [F]',fontsize = 12)
plt.xlabel('Day of Year',fontsize = 12)
plt.yticks(fontsize =12)
ax.fill_between(mintemp['doy'],55,75,color = 'gray',alpha = 0.15)
plt.text(10,60,'T$_{mean}$',fontsize = 18)

ax = plt.subplot(223)
plt.plot(maxtemp['MEAN'],'k',linewidth = 1.75)
ax.fill_between(maxtemp['doy'],maxtemp['MEAN']-maxtemp['STD'],maxtemp['MEAN']+maxtemp['STD'],color = 'red',alpha = 0.15)
plt.xticks(np.arange(0,390,60),fontsize =12)
plt.xlim([-5,370])
plt.ylim([20,90])
plt.ylabel('Temperature [F]',fontsize = 12)
plt.xlabel('Day of Year',fontsize = 12)
plt.yticks(fontsize =12)
plt.text(10,70,'T$_{max}$',fontsize = 18)
plt.hlines(85,0,366, colors = 'gray')

ax = plt.subplot(224)
plt.plot(mintemp['MEAN'],'blue',linewidth = 1.75)
ax.fill_between(mintemp['doy'],mintemp['MEAN']-mintemp['STD'],mintemp['MEAN']+mintemp['STD'],color = 'red',alpha = 0.15)
plt.xticks(np.arange(0,390,60),fontsize =12)
plt.xlim([-5,370])
plt.ylim([5,70])
plt.ylabel('Temperature [F]',fontsize = 12)
plt.xlabel('Day of Year',fontsize = 12)
plt.yticks(fontsize =12)
plt.text(10,50,'T$_{min}$',fontsize = 18)
plt.hlines(45,0,366, colors = 'grey')

plt.tight_layout()

# plt.savefig(r'C:\Users\Eric\Desktop\LTAR\Climatology\Pleasant\Temp_MMM_hourly_STD.png',dpi = 322)



#%%

plt.figure(3, figsize=(9,6))
ax = plt.subplot(211)
plt.plot(Max_Miss,'.-',linewidth = 3, ms = 15, label = 'Max. Temp')
plt.plot(Precip_Miss,'.-',linewidth = 3, ms = 15, label = 'Precipitation')
plt.xlabel('Year',fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, ncol = 2)
plt.ylabel('Pleasant Days',fontsize = 14)
# plt.xlim(['1998','2020'])

plt.ylim([240,360])
ax = plt.subplot(212)
plt.plot(Min_Miss,'.-',linewidth = 3, ms = 15, label = 'Min. Temp')
plt.plot(Mean_Miss,'.-',linewidth = 3, ms = 15, label = 'Mean Temp')
plt.ylabel('Pleasant Days',fontsize = 14)
plt.xlabel('Year',fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 14, ncol = 2)
# plt.xlim(['1998','2020'])
plt.ylim([80,180])
plt.tight_layout()
# plt.savefig(r'C:\Users\russe\Desktop\LTAR\Climatology\Pleasant\DaytimeDaysWithPrecip.png',dpi = 322)

