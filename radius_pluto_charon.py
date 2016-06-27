# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 00:04:58 2016

@author: throop
"""

# Just a quick routine to make a pair of plots. These plot the table in Nimmo's Pluto Shape / Size paper,
# e-mail 21-Jun-2016 (for Icarus special issue).

# HBT 21-Jun-2016

import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1986, 1150, 50, 750, 50],
        [1987, 1100, 80, 580, 50],
        [1988, 1142, 9, 596, 17],
        [1989, 1180, 23, 0, 0],
        [1990, 1151, 6, 593, 13],
        [1991, 0, 0, 601.5, 0],
        [1992, 1206, 11, 0, 0],
        [1992.5, 1150, 7, 593, 10],
        [1993, 1195, 5, 0, 0],
        [1994, 1152, 7, 595, 5],
        [1994.3, 1179.5, 23.5, 629, 21],
        [1994.7, 1160, 12, 635, 13],
        [1995, 1155, 20, 612, 30],
        [2006, 0, 0, 606, 1.5],
        [2006.5, 0, 0, 603.6, 1.4],
        [2009, 1181, 12, 0, 0],
        [2015, 1187, 4, 606, 3],
        [2016, 1188.3, 1.6, 606.0, 1.0]])
        
year = data[:,0]
rp   = data[:,1]
drp  = data[:,2]
rc    = data[:,3]
drc  = data[:,4]

plt.subplots(1,2, figsize=(12,12))

fs=15

plt.subplot(2,1,1)
#plt.plot(year, rp, marker='+', linestyle='none')
plt.errorbar(year, rp, xerr=0, yerr=drp, marker='d', linestyle='none', mew=1, color='blue', ms=3, linewidth=1)

plt.ylim((1080,1250))
plt.title('Pluto Radius Measurement vs Time', fontsize=fs)
plt.xlim((1985, 2017))
plt.ylabel('Radius [km]', fontsize=fs)
plt.subplot(2,1,2)

#plt.plot(year, rc, marker='o', linestyle='none')
plt.errorbar(year, rc, xerr=0, yerr=drc, marker='d', linestyle='none', color='blue', ms=3, linewidth=1)

plt.errorbar(1991, 601.5, yerr=20, lolims=True, color='blue', ms=3, marker='d') # Add one single point as lower-limit

plt.ylim((550,800))     
plt.title('Charon Radius Measurement vs Time', fontsize=fs)
plt.ylabel('Radius [km]', fontsize=fs)
plt.xlim((1985, 2017))
plt.xlabel('Year', fontsize=fs)
   

plt.show()
