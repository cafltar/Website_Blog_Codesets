# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:02:53 2021

@author: russe
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
os.chdir('C:\\Users\\russe\\Desktop\\PyScripts\\Flux_Processing_Code')       
import LTAR_QAQC_Online as LLT

NWS = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\ot\NWS_Precip_April_2019.csv',header = 0, index_col = 'DY')
NWS.index = pd.to_datetime(NWS.index)

Climo = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\NCDC\USW00094846.csv',header = 0, index_col = 'DATE')
Climo.index = pd.to_datetime(Climo.index)
Climo['PRCP'] = Climo['PRCP']
Climo = LLT.indx_fill(Climo, 'D')
P_Climo = Climo['PRCP'].groupby(pd.Grouper(freq = 'M'))
xt = []
fig = plt.figure(1,figsize=(8,6))
for k in P_Climo.groups.keys():
    if str(k)[5:7] == '05':
        plt.plot(np.cumsum(P_Climo.get_group(k).values),'k',alpha = 0.5)
        qn0 = P_Climo.get_group(k).values>0
        qn25 = P_Climo.get_group(k).values>=0.25
        qn50 = P_Climo.get_group(k).values>=0.5
        qn100 = P_Climo.get_group(k).values>=1
        xt.append([sum(qn0),sum(qn25),sum(qn50),sum(qn100),str(k)[0:4], sum(P_Climo.get_group(k).values)])
        print(str(k)[0:4]+' had this much precip:'+str(np.sum(P_Climo.get_group(k).values)))
plt.xlabel('Day in April',fontsize = 14)
plt.ylabel('Cumuative. Precip [Inches]',fontsize = 14)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
#plt.savefig(r'C:\Users\russe\Desktop\LTAR\Precip\April_Cumulative_NWS_1970_2018_with2019.png',dpi = 350)
qn0 = NWS['WTR'].values>0
qn25 = NWS['WTR'].values>=0.25
qn50 = NWS['WTR'].values>=0.5
qn100 = NWS['WTR'].values>=1

xt.append([sum(qn0),sum(qn25),sum(qn50),sum(qn100),2019, sum(NWS['WTR'].values)])

#%%
#% Plot the sum of precip over the number of days with rain
xt = pd.DataFrame(xt)
xt.index = pd.to_datetime(xt[4])
plt.figure(2, figsize=(8,6))
plt.bar(xt[4].astype(int),xt[0])
plt.bar(xt[4].astype(int),xt[1])
plt.bar(xt[4].astype(int),xt[2])
plt.bar(xt[4].astype(int),xt[3])
plt.xticks(np.arange(1970,2021,5), fontsize = 14)
plt.xlim([1968,2022])
plt.legend(['>0"','>=0.25"','>=0.5"','>=1"'],loc = 2, fontsize = 14,ncol=2)
plt.yticks(np.arange(0,21,2), fontsize = 14)
plt.ylabel('Number of days',fontsize = 18)
plt.xlabel('Year',fontsize = 18)
plt.tight_layout()
#plt.savefig(r'C:\Users\russe\Desktop\LTAR\Precip\Rainfall_Events_Severity_1970_2018.png',dpi = 350)

#%%

xl = pd.ExcelFile("C:\\Users\\russe\\Desktop\\Projects\\Active\\LTAR\\Climatology\\PCFS_Climo_Data.xlsx")
q = xl.sheet_names
PCFS = xl.parse('MonthlySumPrecip')
PCFS = pd.DataFrame(PCFS)
PCFS.index = PCFS['Year']
mnt = 'April'
plt.figure(4, figsize=(8,6))
ax = plt.subplot(111)
plt.bar(PCFS['Year'],PCFS[mnt].astype(float))
plt.bar(PCFS['Year'][2019],PCFS[mnt][2019].astype(float))
plt.text(PCFS['Year'][2009],PCFS[mnt][2019].astype(float)+0.33,'2019:\n 3.65"',fontsize = 14)
plt.bar(PCFS['Year'][1996],PCFS[mnt][1996].astype(float))
plt.text(PCFS['Year'][1986],PCFS[mnt][1996].astype(float)+0.33,'1996:\n 4.66"',fontsize = 14)
plt.bar(PCFS['Year'][1993],PCFS[mnt][1993].astype(float))
plt.text(PCFS['Year'][1983],PCFS[mnt][1993].astype(float)+0.33,'1993:\n 3.75"',fontsize = 14)
plt.bar(PCFS['Year'][1958],PCFS[mnt][1958].astype(float))
plt.text(PCFS['Year'][1948],PCFS[mnt][1958].astype(float)+0.33,'1958:\n 5.04"',fontsize = 14)
plt.bar(PCFS['Year'][1917],PCFS[mnt][1917].astype(float))
plt.text(PCFS['Year'][1907],PCFS[mnt][1917].astype(float)+0.33,'1917:\n 4.21"',fontsize = 14)
plt.yticks(np.arange(0,7,1), fontsize = 18)
plt.ylabel('Total Precipitation [inches]',fontsize = 18)
plt.xlabel('Year',fontsize = 18)
plt.xticks(np.arange(1890,2026,15), fontsize = 18)
plt.tight_layout()
#plt.savefig(r'C:\Users\russe\Desktop\LTAR\Precip\Monthly_Rain_April_1893_2019.png',dpi = 350)
#
