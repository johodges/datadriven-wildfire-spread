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

def parseFarsiteInput(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    moistures = []
    weathers = []
    winds = []
    for line in lines:
        if 'FUEL_MOISTURES_DATA' in line:
            switch = 'moisture'
        elif 'WEATHER_DATA' in line:
            switch = 'weather'
        elif 'WIND_DATA' in line:
            switch = 'wind'
        lineSplit = line.split(' ')
        if lineSplit[0].isdigit():
            lineArray = np.array([float(x) for x in lineSplit])
            if switch == 'moisture':
                moistures.append(lineArray)
            elif switch == 'weather':
                weathers.append(lineArray)
            elif switch == 'wind':
                winds.append(lineArray)
    moistures = np.array(moistures)
    weathers = np.array(weathers)
    winds = np.array(winds)
    return moistures, weathers, winds

def lineStringToPolygon(data):
    for i in range(0,data.shape[0]):
        data['geometry'][i] = Polygon(list(data['geometry'][i].coords))
    return data

def loadFarsiteOutput(inDir,namespace,commandFarsite=True):
    imgs, names, headers = qlf.readLcpFile(inDir+namespace+'.LCP')
    header = qlf.parseLcpHeader(headers)
    dataOut = gpd.GeoDataFrame.from_file(inDir+namespace+'_out_Perimeters.shp')
    if commandFarsite:
        dataOut = lineStringToPolygon(dataOut)
    testDate = [datetime.datetime(year=2016, month=int(x), day=int(y), hour=int(z)).timestamp() for x,y,z in zip(dataOut['Month'].values,dataOut['Day'].values,dataOut['Hour'].values/100)]
    testDate = np.array(testDate,dtype=np.float32)
    testDate = np.array((testDate-testDate[0])/3600,dtype=np.int16) #/3600
    dataOut['time'] = testDate
    
    lcpData = rasterio.open(inDir+namespace+'.LCP')
    return dataOut, lcpData

def downsampleImage(img,interval):
    newImg = img[::interval,::interval].copy()
    return newImg

def plotTimeContour(img,imgBand,contours):
    contours = contours.reindex(index=contours.index[::-1])
    fig, ax = plt.subplots(figsize=(12,8))
    rsplot.show((img,imgBand),with_bounds=True,ax=ax,cmap='gray')
    
    vmin = np.min(contours.time)
    vmax = np.max(contours.time)
    
    contours.plot(ax=ax, cmap='jet', scheme='time')
    
    sm = plt.cm.ScalarMappable(cmap='jet_r', norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    cbar.ax.set_ylabel('Time since Ignition (Hours)',rotation=270,fontsize=fs)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.invert_yaxis()
    
    plt.tick_params(labelsize=fs)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    plt.tight_layout()
    
    plt.xlabel('NAD83 EW %s'%(ax.xaxis.offsetText.get_text()),fontsize=fs)
    plt.ylabel('NAD83 NS %s'%(ax.yaxis.offsetText.get_text()),fontsize=fs)
    ax.yaxis.offsetText.set_visible(False)
    ax.xaxis.offsetText.set_visible(False)
    return fig

def getRasterFromPolygon(data,tF,ind,value,sz):
    outArr = np.zeros(sz)
    shapes = ((geom,value) for geom, value in zip(data.iloc[:ind,:].geometry, np.zeros((data.iloc[:ind,:].shape[0],),dtype=np.int16)+value)) #dataOut.iloc[:i+1,:].time))
    raster = np.array(features.rasterize(shapes=shapes, fill=1, out=outArr, transform=tF)).copy()
    return raster

def parseMoistures(moistures):
    fm, m1h, m10h, m100h, lhm, lwm = np.median(moistures,axis=0)
    return m1h, m10h, m100h, lhm, lwm

def makeConstImage(sz,value):
    img = np.zeros(sz)+value
    return img

if __name__ == "__main__":
    
    commandFile = 'commonDir/farsite/example/Panther/runPanther.txt'
    inDir = 'E:/projects/wildfire-research/farsite/data/'
    #inDir = 'E:/projects/wildfire-research/farsiteData/'
    #namespace = 'n117-9343_36-5782_3000'
    namespace = 'n114-0177_38-3883_25000'
    #lcpFile = inDir+namespace+'.LCP'
    #elevation = getLcpElevation(lcpFile)
    #ignitionShape = makeCenterIgnition(lcpFile)    
    
    moistures, weathers, winds = parseFarsiteInput(inDir+namespace+'.input')
    
    windSpeed = np.median(winds[:,3])
    windDir = np.median(winds[:,4])
    windX,windY = paf.convertVector(windSpeed,windDir)
    m1h, m10h, m100h, lhm, lwm = parseMoistures(moistures)
    
    
    
    #fileName = inDir+namespace
    #params = generateFarsiteInput(fileName,elevation)
    #runFarsite(commandFile)
    
    
    
    #p = subprocess.Popen([dockerStart,dockerCmd],stdout=subprocess.PIPE)
    #print(p.communicate())
    
    fs = 16
    interval = int(np.ceil(1000/30))
    dataOut, lcpData = loadFarsiteOutput(inDir,namespace)
    
    fig = plotTimeContour(lcpData,1,dataOut)
    
    elevImg = lcpData.read(1)
    fuelImg = lcpData.read(4)
    canopyImg = lcpData.read(5)
    canopyHeightImg = lcpData.read(6)
    canopyDensityImg = lcpData.read(7)
    sz = elevImg.shape
    
    elevImg = downsampleImage(elevImg,interval)
    fuelImg = downsampleImage(fuelImg,interval)
    canopyImg = downsampleImage(canopyImg,interval)
    smallSz = elevImg.shape
    
    windXImg = makeConstImage(smallSz,windX)
    windYImg = makeConstImage(smallSz,windY)
    
    t = dataOut['time']
    tOff = 6
    
    tF = lcpData.transform
    
    for i in range(1,dataOut.shape[0]-1,601):
        endInd = np.argwhere(t-t[i]>=tOff)[0][0]
        """ This is how command line farsite outputs
        """
        currentFire = getRasterFromPolygon(dataOut,tF,i,1,sz)
        nextFire = getRasterFromPolygon(dataOut,tF,endInd,1,sz)
        
        currentFire = downsampleImage(currentFire,interval)
        nextFire = downsampleImage(nextFire,interval)
        
        fig = plt.figure(figsize=(12,12))
        plt.subplot(3,4,1)
        plt.imshow(currentFire,cmap='jet')
        plt.subplot(3,4,2)
        plt.imshow(elevImg,cmap='jet')
        plt.subplot(3,4,3)
        plt.imshow(windXImg,cmap='jet')
        plt.subplot(3,4,4)
        plt.imshow(windYImg,cmap='jet')
        plt.subplot(3,4,6)
        plt.imshow(canopyImg,cmap='jet')
        plt.subplot(3,4,7)
        plt.imshow(fuelImg,cmap='jet')

        plt.subplot(3,4,9)
        plt.imshow(nextFire,cmap='jet')
        