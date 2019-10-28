# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:02:35 2018

@author: jhodges
"""

import glob
import matplotlib.pyplot as plt
import numpy as np

def getTime(file):
    with open(file,'r') as f:
        lines = f.readlines()
    simTime = -1
    for line in lines:
        if 'Total Farsite Run Time' in line:
            simTime = float(line.split()[4])
    return simTime

if __name__ == "__main__":
    
    inDir = "E://projects//wildfire-research//farsite//data//"
    
    files = glob.glob(inDir+"*_Timings.txt")
    
    simTimes = []
    for i in range(0,len(files)):
        simTime = getTime(files[i])
        simTimes.append(simTime)
    simTimes = np.array(simTimes)
    
    simTimes2 = simTimes.copy()
    simTimes2 = simTimes2[simTimes2>60]
    simTimes2 = simTimes2[simTimes2<3600*10]
    
    simTimeMean = np.mean(simTimes2)
    simTimeStd = np.std(simTimes2)
    
    plt.figure(figsize=(12,12))
    fs = 32
    lw = 3
    plt.hist(simTimes/3600,bins=10000,cumulative=True,normed=True,histtype='step',linewidth=lw)
    plt.xlim(0,6)
    plt.xlabel('FARSITE computational time (hours)',fontsize=fs)
    plt.ylabel('Cumulative Probality',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.tight_layout()
    plt.savefig('wfsm_farsiteCompTime.pdf',dpi=300)