# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:34:23 2018

@author: JHodges
"""

import subprocess
import os
import behavePlus as bp
import numpy as np
import datetime
import queryLandFire as qlf
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
import rasterio.plot as rsplot
import parse_asos_file as paf

def findBestThreshold(predictionImgs,truthImgs):
    thresh = -0.01
    threshes = []
    fMeasures = []
    confusionMatrixes = []
    while thresh < 1.0:
        thresh = thresh + 0.01
        confusionMatrix = []
        for i in range(0,len(truthImgs)):
            pImg = predictionImgs[i].copy()
            tImg = truthImgs[i].copy()
            confusionMatrix.append(findConfusionMatrix(pImg,tImg,thresh))
        confusionMatrix = np.array(confusionMatrix)
        threshes.append(thresh)
        fMeasures.append(np.mean(confusionMatrix[:,-1]))
        confusionMatrixes.append(confusionMatrix)
    bestThresh = threshes[np.argmax(fMeasures)]
    bestConfusionMatrix = np.mean(confusionMatrixes[np.argmax(fMeasures)],axis=0)
    return bestThresh, bestConfusionMatrix, threshes, fMeasures
    
def postProcessFirePerimiter(pImg,thresh):
    pImg[pImg < thresh] = 0.0
    corners = [pImg[-1,-1].copy(),pImg[0,0].copy(),pImg[0,-1].copy(),pImg[-1,0].copy()]
    centers = pImg[24:26,24:26].copy()
    pImg = scsi.medfilt2d(pImg)
    (pImg[-1,-1],pImg[0,0],pImg[0,-1],pImg[-1,0]) = corners
    pImg[24:26,24:26] = centers
    pImg[pImg>thresh] = 1.0
    return pImg
    
def findConfusionMatrix(pImg,tImg,thresh):
    pImg = postProcessFirePerimiter(pImg,thresh)

    TN = float(len(np.where(np.array(pImg.flatten() == 0) & np.array(tImg.flatten() == 0))[0]))
    FN = float(len(np.where(np.array(pImg.flatten() == 0) & np.array(tImg.flatten() == 1))[0]))
    FP = float(len(np.where(np.array(pImg.flatten() == 1) & np.array(tImg.flatten() == 0))[0]))
    TP = float(len(np.where(np.array(pImg.flatten() == 1) & np.array(tImg.flatten() == 1))[0]))
    
    try:
        accuracy = round((TP+TN)/(TP+TN+FP+FN),2)
    except ZeroDivisionError:
        accuracy = round((TP+TN)/(TP+TN+FP+FN+1),2)
    try:
        recall = round((TP)/(TP+FN),2)
    except ZeroDivisionError:
        recall = round((TP)/(TP+FN+1),2)
    try:
        precision = round((TP)/(TP+FP),2)
    except ZeroDivisionError:
        precision = round((TP)/(TP+FP+1),2)
    try:
        fMeasure = round((2*recall*precision)/(recall+precision),2)
    except ZeroDivisionError:
        fMeasure = round((2*recall*precision)/(recall+precision+1),2)
    
    confusionMatrix = [TN,FN,FP,TP,accuracy,recall,precision,fMeasure]
    return confusionMatrix

def plotThresholdFMeasure(threshes,fMeasures):
    plt.figure(figsize=(8,8))
    plt.plot(threshes,fMeasures,'-k',linewidth=3)
    fs = 32
    plt.xlabel('Threshold',fontsize=fs)
    plt.ylabel('F-Measure',fontsize=fs)
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,1.0)
    plt.tick_params(labelsize=fs)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.grid()
    plt.savefig('optimalThreshold.png',dpi=300)


if __name__ == "__main__":
    
    bestThresh, bestConfusionMatrix, threshes, fMeasures = findBestThreshold(predictionImgs,truthImgs)
    plotThresholdFMeasure(threshes,fMeasures)
    
    print("Best Threshold: %.2f"%(bestThresh))
    print("Best F-Measure: %.2f"%(bestConfusionMatrix[-1]))
    
    '''
    
    thresh = -0.01
    threshes = []
    fMeasures = []
    confusionMatrixes = []
    while thresh < 1.0:
        thresh = thresh + 0.01
        confusionMatrix = []
        errors = []
        if generatePlots:
            for toPlot in range(0,99):
                pImg = predictionImgs[toPlot].copy()
                tImg = truthImgs[toPlot].copy()
                pImg[pImg < thresh] = 0.0
                corners = [pImg[-1,-1].copy(),pImg[0,0].copy(),pImg[0,-1].copy(),pImg[-1,0].copy()]
                centers = pImg[24:26,24:26].copy()
                pImg = scsi.medfilt2d(pImg)
                (pImg[-1,-1],pImg[0,0],pImg[0,-1],pImg[-1,0]) = corners
                pImg[24:26,24:26] = centers
                
                pImg[pImg>thresh] = 1.0

                TN = float(len(np.where(np.array(pImg.flatten() == 0) & np.array(tImg.flatten() == 0))[0]))
                FN = float(len(np.where(np.array(pImg.flatten() == 0) & np.array(tImg.flatten() == 1))[0]))
                FP = float(len(np.where(np.array(pImg.flatten() == 1) & np.array(tImg.flatten() == 0))[0]))
                TP = float(len(np.where(np.array(pImg.flatten() == 1) & np.array(tImg.flatten() == 1))[0]))
                
                try:
                    accuracy = round((TP+TN)/(TP+TN+FP+FN),2)
                except ZeroDivisionError:
                    accuracy = round((TP+TN)/(TP+TN+FP+FN+1),2)
                try:
                    recall = round((TP)/(TP+FN),2)
                except ZeroDivisionError:
                    recall = round((TP)/(TP+FN+1),2)
                try:
                    precision = round((TP)/(TP+FP),2)
                except ZeroDivisionError:
                    precision = round((TP)/(TP+FP+1),2)
                try:
                    fMeasure = round((2*recall*precision)/(recall+precision),2)
                except ZeroDivisionError:
                    fMeasure = round((2*recall*precision)/(recall+precision+1),2)
                
                confusionMatrix.append([TN,FN,FP,TP,accuracy,recall,precision,fMeasure])

            confusionMatrix = np.array(confusionMatrix)
            print(thresh,np.mean(confusionMatrix[:,-1]))
            threshes.append(thresh)
            fMeasures.append(np.mean(confusionMatrix[:,-1]))
            confusionMatrixes.append(confusionMatrix)
    plt.figure(figsize=(8,8))
    plt.plot(threshes,fMeasures,'-k',linewidth=3)
    fs = 32
    plt.xlabel('Threshold',fontsize=fs)
    plt.ylabel('F-Measure',fontsize=fs)
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,1.0)
    plt.tick_params(labelsize=fs)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.grid()
    
    print(threshes[np.argmax(fMeasures)],np.mean(confusionMatrixes[np.argmax(fMeasures)],axis=0))
    '''