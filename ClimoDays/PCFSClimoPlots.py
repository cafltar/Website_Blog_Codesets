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
from matplotlib.ticker import AutoMinorLocator

def format_plot(ax,yf,xf,xminor,yminor,yl,yu,xl,xu):
    #subplot has to have ax as the axis handle
    plt.yticks(fontsize = yf);plt.xticks(fontsize = xf)
    minor_locator = AutoMinorLocator(xminor)
    ax.xaxis.set_minor_locator(minor_locator)
    minor_locator = AutoMinorLocator(yminor)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(axis='both',direction='in',length=12.5,width=2)
    ax.tick_params(axis='both',which = 'minor',direction='in',length=5)
    plt.ylim([yl,yu])
    plt.xlim([xl,xu])  
    return

#%%

data = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\NCDC\USC00456789_2020.csv',header=0)
data.index = pd.to_datetime(data['DATE'])
dg = data.groupby(pd.Grouper(freq='Y'))
msum = data.groupby(pd.Grouper(freq='M')).sum()
#%% Initial data processing/checks
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

#%% Cold as ice first freeze plot
# Add in formatplot
qn = FirstMin['doy']>2
qn1 = FirstMax['doy']>2
tF = FirstMax['doy'][qn1].rolling(5).mean()
tM = FirstMin['doy'][qn].rolling(5).mean()

plt.figure(1,figsize=(8,6))
ax = plt.subplot(111)
plt.plot(FirstMax[qn1].index.values.astype(int),FirstMax['doy'][qn1],'-s')
plt.plot(FirstMin[qn].index.values.astype(int),FirstMin['doy'][qn],'-s')
plt.yticks(np.linspace(200, 365,12),fontsize = 14)
plt.ylabel('Day of Year', fontsize = 14)
plt.xticks(np.linspace(1940, 2020,17),rotation = 90,fontsize = 14)
plt.xlabel('Year', fontsize = 14)
format_plot(ax, 14, 14, 5, 5, 200, 365, 1940, 2020)
plt.legend(['TMIN','TMAX'], fontsize = 16)
plt.tight_layout()
# plt.savefig(r'C:\Users\russe\Desktop\LTAR\Website Blogs\FirstFreeze.png',dpi= 350)

#%% Fourth of July climo
# Add in formatplot
Dates = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\JulyFourth_Pullman_PCFS.csv', header = 0,index_col='Date')
Dates.index = pd.to_datetime(Dates.index)
Dates = Dates.astype(float)

IndDay = data[['4/7/' in x for x in data['DATE']]]

msum['date'] = msum.index
msum_july = msum[['-04-30' in x for x in msum['date'].astype(str)]]


plt.figure(2,figsize=(6,4))
ax = plt.subplot(111)
plt.plot(Dates['TMIN'], label ='Tmax')
plt.plot(Dates['TMAX'], label ='Tmin')
plt.plot((Dates['TMIN']+Dates['TMAX'])/2, label ='Tmean')
plt.ylim([30,100]); plt.yticks(fontsize = 12)
plt.xticks(fontsize = 12)
format_plot(ax, 12, 12, 5, 5, 30, 100, pd.to_datetime('1900'), pd.to_datetime('2020'))
plt.legend(fontsize=14, loc = 3, ncol = 3)
plt.ylabel('Temperature on 4-July \n [Fahrenheit]', fontsize = 12)
plt.xlabel('Year', fontsize = 12)
plt.tight_layout()

plt.figure(3, figsize=(8,4))
ax = plt.subplot(212)
x = np.linspace(1900,2018,119)
plt.bar(x,Dates['PRCP'])
plt.xlim([1970,2020])
plt.ylim([0,0.6])
format_plot(ax, 12, 12, 5, 5, 0, 0.6, 1970, 2020)
plt.xticks(rotation = 15)
plt.xlabel('Year',fontsize = 12)
plt.ylabel('Total July Precipitation \n [Inches]', fontsize = 10)


ax = plt.subplot(211)
x = np.linspace(1941,2020,80)
plt.bar(x,msum_july['PRCP']/25.4)
plt.xlim([1970,2020])
format_plot(ax, 12, 12, 5, 5, 0, 6, 1970, 2020)
plt.xticks(rotation = 15)
plt.xlabel('Year',fontsize = 12)
plt.ylabel('Precipitation on 4-July \n [Inches]', fontsize = 10)
plt.show()
plt.tight_layout()
#%% Memorial Day 
# Add in formatplot
Dates = pd.read_csv(r'C:\Users\russe\Desktop\Projects\Active\LTAR\Climatology\MemorialDayDates.csv', header = 0)
Dates['Date'] = pd.to_datetime(Dates['Date'])
MemDay = data.loc[Dates['Date']]
MemDay.index =MemDay.index.year

plt.figure(4, figsize=(7,4))
ax = plt.subplot(211)
plt.plot(MemDay['TMIN']*(9/5)+32,'b', label='TMin')
plt.plot(MemDay['TMAX']*(9/5)+32,'r', label='TMax')
plt.ylabel('Temperature\n [degF]', fontsize = 11)
plt.legend(loc = 4, ncol = 2)
plt.xticks(rotation = 15)
format_plot(ax, 12, 12, 5, 5, 20, 100, 1970, 2020)

ax = plt.subplot(212)
plt.bar(MemDay.index, MemDay['PRCP']/25.4, label = 'Total Precip')
plt.ylabel('Daily Total Rainfall \n [inch]', fontsize = 11)
plt.legend(loc = 1)
format_plot(ax, 12, 12, 5, 5, 0, 1.5, 1970, 2020)
plt.xticks(rotation = 15)
plt.tight_layout()


