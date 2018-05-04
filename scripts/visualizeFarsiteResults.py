# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:11:26 2018

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
import math
import glob

def convertVector(speed,direction):
    ''' This function will convert wind speed measurements from polar
    coordinates to Cartesian coordinates.
    '''
    if direction == -1:
        speedX = speed/(2**0.5)
        speedY = speed/(2**0.5)
    elif direction is None:
        pass
        print("Wind direction was not set.")
    elif speed is None:
        pass
        print("Wind speed was not set.")
    else:
        try:
            speedX = speed*np.sin(direction/180*math.pi)
            speedY = speed*np.cos(direction/180*math.pi)
        except:
            assert False, "Unknown wind vector: %s Mps %s Deg" % (str(speed),str(direction))
    return speedX, speedY

def plotWildfireData(datas,names,
                     clims=None,closeFig=None,
                     saveFig=False,saveName=''):
    totalPlots = np.ceil(float(len(datas))**0.5)
    colPlots = totalPlots
    rowPlots = np.ceil((float(len(datas)))/colPlots)
    currentPlot = 0
    
    if saveFig:
        fntsize = 20
        lnwidth = 5
        fig = plt.figure(figsize=(colPlots*12,rowPlots*10))#,tight_layout=True)      
        if closeFig is None:
            closeFig = True
    else:
        fig = plt.figure(figsize=(colPlots*6,rowPlots*5))#,tight_layout=True)
        fntsize = 20
        lnwidth = 2
        if closeFig is None:
            closeFig = False
        
    xmin = 0
    xmax = datas[0].shape[1]
    xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/10)+1))
    ymin = 0
    ymax = datas[0].shape[0]
    yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/10)+1))

    for i in range(0,len(names)):
        key = names[i]
        currentPlot = currentPlot+1

        ax = fig.add_subplot(rowPlots,colPlots,currentPlot)
        ax.tick_params(axis='both',labelsize=fntsize)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.title(key,fontsize=fntsize)

        if clims is None:
            clim = np.linspace(0,1,10)
            label = ''
        else:
            clim = clims[i]
            label = ''
        img = ax.imshow(datas[i],cmap='jet',vmin=clim[0],vmax=clim[-1])#,vmin=0,vmax=1)
        img_cb = plt.colorbar(img,ax=ax,label=label)

        img_cb.set_label(label=label,fontsize=fntsize)
        img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        ax.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        for ln in ax.lines:
            ln.set_linewidth(lnwidth)
    if saveFig:
        fig.savefig(saveName)
        
    if closeFig:
        plt.clf()
        plt.close(fig)

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

def loadFarsiteOutput(namespace,commandFarsite=True):
    #imgs, names, headers = qlf.readLcpFile(inDir+namespace+'.LCP')
    #header = qlf.parseLcpHeader(headers)
    dataOut = gpd.GeoDataFrame.from_file(namespace+'_out_Perimeters.shp')
    if commandFarsite:
        dataOut = lineStringToPolygon(dataOut)
    testDate = [datetime.datetime(year=2016, month=int(x), day=int(y), hour=int(z)).timestamp() for x,y,z in zip(dataOut['Month'].values,dataOut['Day'].values,dataOut['Hour'].values/100)]
    testDate = np.array(testDate,dtype=np.float32)
    testDate = np.array((testDate-testDate[0])/3600,dtype=np.int16) #/3600
    dataOut['time'] = testDate
    
    #lcpData = rasterio.open(inDir+namespace+'.LCP')
    return dataOut

def loadFarsiteLcp(lcpNamespace):
    lcpData = rasterio.open(lcpNamespace)
    return lcpData

def downsampleImage(img,interval):
    newImg = img[::interval,::interval].copy()
    return newImg

def plotTimeContour(img,imgBand,contours,namespace):
    fs = 16
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
    plt.savefig(namespace)
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

def remapFuelImg(img):
    modelIndexDict = bp.determineFastestModel()
    fuelModelsIdx = bp.buildFuelModelsIdx()
    sz = img.shape
    imgRs = img.reshape((sz[0]*sz[1],))
    imgStr = fuelModelsIdx[imgRs]
    imgRsNew = [modelIndexDict[x] for x in imgStr]
    imgNew = np.reshape(imgRsNew,(sz[0],sz[1]))
    return imgNew

def visualizeFarsiteResult(namespace,lcpNamespace,perimeterOnly=True):

    moistures, weathers, winds = parseFarsiteInput(namespace+'.input')
    windSpeed = np.median(winds[:,3])
    windDir = np.median(winds[:,4])
    windX,windY = convertVector(windSpeed,windDir)
    m1h, m10h, m100h, lhm, lwm = parseMoistures(moistures)
    
    fs = 16
    interval = int(np.ceil(1000/30))
    dataOut = loadFarsiteOutput(namespace)
    lcpData = loadFarsiteLcp(inDir+lcpNamespace)
    
    fig = plotTimeContour(lcpData,1,dataOut,namespace+'_p.png')
    plt.close()
    if not perimeterOnly:
        elevImg = np.array(lcpData.read(1),dtype=np.float32)
        elevImg = elevImg-np.median(elevImg)
        fuelImg = lcpData.read(4)
        canopyImg = np.array(lcpData.read(5),dtype=np.float32)/100
        canopyHeightImg = np.array(lcpData.read(6),dtype=np.float32)/10
        canopyBaseHeightImg = np.array(lcpData.read(7),dtype=np.float32)/10
        canopyDensityImg = np.array(lcpData.read(8),dtype=np.float32)/100
        sz = elevImg.shape
        
        elevImg = downsampleImage(elevImg,interval)
        fuelImg = downsampleImage(fuelImg,interval)
        canopyImg = downsampleImage(canopyImg,interval)
        canopyHeightImg = downsampleImage(canopyHeightImg,interval)
        canopyBaseHeightImg = downsampleImage(canopyBaseHeightImg,interval)
        canopyDensityImg = downsampleImage(canopyDensityImg,interval)
        fuelImg = remapFuelImg(fuelImg)
        smallSz = elevImg.shape
        
        windXImg = makeConstImage(smallSz,windX)
        windYImg = makeConstImage(smallSz,windY)
        lhmImg = makeConstImage(smallSz,lhm)
        lwmImg = makeConstImage(smallSz,lwm)
        m1hImg = makeConstImage(smallSz,m1h)
        m10hImg = makeConstImage(smallSz,m10h)
        m100hImg = makeConstImage(smallSz,m100h)
        
        t = dataOut['time']
        tOff = 6
        
        tF = lcpData.transform
        
        clims = [[0,1],[-250,250],
                 [-20,20],[-20,20],
                 [30,150],[30,150],
                 [0,40],[0,40],
                 [0,40],[0,1],
                 [0,20],[0,20],
                 [0,0.4],
                 [0,52],
                 [0,1],[0,1]]
        names = ['Input Fire','Input Elev',
                 'Input WindX','Input WindY',
                 'Live Herb M','Live Wood M',
                 'Moisture 1-h','Moisture 10-h',
                 'Moisture 100-h','Canopy Cover',
                 'Canopy Height','Canopy Base Height',
                 'Canopy Density',
                 'model',
                 'Network','Truth']
        
        
        (t.max()-t.min())/tOff
        for i in range(0,t.max(),tOff):
            """ This is how command line farsite outputs
            """
            try:
                startInd = np.argwhere(t-i>=0)[0][0]
                endInd = np.argwhere(t-i>=tOff)[0][0]
                currentFire = getRasterFromPolygon(dataOut,tF,startInd,1,sz)
                nextFire = getRasterFromPolygon(dataOut,tF,endInd,1,sz)
                
                currentFire = downsampleImage(currentFire,interval)
                nextFire = downsampleImage(nextFire,interval)
                data = [currentFire,elevImg,windXImg,windYImg,lhmImg,lwmImg,m1hImg,m10hImg,m100hImg,canopyImg,canopyHeightImg,canopyBaseHeightImg,canopyDensityImg,fuelImg,nextFire,nextFire]
                plotWildfireData(data,names,clims=clims,saveFig=True,saveName=namespace+'_'+str(i)+'_'+str(i+tOff)+'_summary.png')
            except:
                pass

if __name__ == "__main__":
    
    commandFile = 'commonDir/farsite/example/Panther/runPanther.txt'
    inDir = 'E:/projects/wildfire-research/farsite/data/'
    
    files = glob.glob(inDir+"*run_*_Perimeters.shp")
    print("Total files: %.0f"%(len(files)))
    for i in range(0,len(files)):#file in files:
        file = files[i]
        #file = inDir[:-1]+'\\'+'run_0_8_n116-9499_38-9950_25000_out_Perimeters.shp'
        namespace = file.split('\\')[1].split('_out_Perimeters.shp')[0]
        namespace = inDir+namespace
        try:
            lcpNamespace = namespace.split('_')[3]+'_'+namespace.split('_')[4]+'_'+namespace.split('_')[5]+'.LCP'
            
            if len(glob.glob(namespace+'_p.png')) == 0:
                visualizeFarsiteResult(namespace,lcpNamespace)
        except:
            print(namespace)
    #lcpNamespace = namespace+'.LCP'
    
    #inDir = 'E:/projects/wildfire-research/farsiteData/'
    #namespace = 'n117-9343_36-5782_3000'
    #namespace = 'n114-0177_38-3883_25000'
    #lcpNamespace = 'n114-0177_38-3883_25000.LCP'
    #namespace = 'run_3_n115-5637_40-2028_25000'
    #lcpNamespace = 'n115-5637_40-2028_25000.LCP'
    #namespace = 'run_0_5_n116-6648_37-3162_25000'
    #lcpNamespace = 'n116-6648_37-3162_25000.LCP'
    #namespace = 'run_0_n118-4236_43-0131_25000'
    #lcpNamespace = 'n118-4236_43-0131_25000.LCP'
    #namespace = 'run_1_n120-8917_34-9586_25000'
    #lcpNamespace = 'n120-8917_34-9586_25000.LCP'
    #namespace = 'run_2_n115-6195_39-1428_25000'
    #lcpNamespace = 'n115-6195_39-1428_25000.LCP'
    #lcpFile = inDir+namespace+'.LCP'
    #elevation = getLcpElevation(lcpFile)
    #ignitionShape = makeCenterIgnition(lcpFile)    
    
    
    """
    moistures, weathers, winds = parseFarsiteInput(inDir+namespace+'.input')
    
    windSpeed = np.median(winds[:,3])
    windDir = np.median(winds[:,4])
    windX,windY = convertVector(windSpeed,windDir)
    m1h, m10h, m100h, lhm, lwm = parseMoistures(moistures)
    
    
    
    #fileName = inDir+namespace
    #params = generateFarsiteInput(fileName,elevation)
    #runFarsite(commandFile)
    
    
    
    #p = subprocess.Popen([dockerStart,dockerCmd],stdout=subprocess.PIPE)
    #print(p.communicate())
    
    fs = 16
    interval = int(np.ceil(1000/30))
    dataOut = loadFarsiteOutput(inDir,namespace)
    lcpData = loadFarsiteLcp(inDir+lcpNamespace)
    
    fig = plotTimeContour(lcpData,1,dataOut,inDir+namespace+'_p.png')
    
    elevImg = np.array(lcpData.read(1),dtype=np.float32)
    elevImg = elevImg-np.median(elevImg)
    fuelImg = lcpData.read(4)
    canopyImg = np.array(lcpData.read(5),dtype=np.float32)/100
    canopyHeightImg = np.array(lcpData.read(6),dtype=np.float32)/10
    canopyBaseHeightImg = np.array(lcpData.read(7),dtype=np.float32)/10
    canopyDensityImg = np.array(lcpData.read(8),dtype=np.float32)/100
    sz = elevImg.shape
    
    elevImg = downsampleImage(elevImg,interval)
    fuelImg = downsampleImage(fuelImg,interval)
    canopyImg = downsampleImage(canopyImg,interval)
    canopyHeightImg = downsampleImage(canopyHeightImg,interval)
    canopyBaseHeightImg = downsampleImage(canopyBaseHeightImg,interval)
    canopyDensityImg = downsampleImage(canopyDensityImg,interval)
    fuelImg = remapFuelImg(fuelImg)
    smallSz = elevImg.shape
    
    windXImg = makeConstImage(smallSz,windX)
    windYImg = makeConstImage(smallSz,windY)
    lhmImg = makeConstImage(smallSz,lhm)
    lwmImg = makeConstImage(smallSz,lwm)
    m1hImg = makeConstImage(smallSz,m1h)
    m10hImg = makeConstImage(smallSz,m10h)
    m100hImg = makeConstImage(smallSz,m100h)
    
    t = dataOut['time']
    tOff = 6
    
    tF = lcpData.transform
    
    clims = [[0,1],[-250,250],
             [-20,20],[-20,20],
             [30,150],[30,150],
             [0,40],[0,40],
             [0,40],[0,1],
             [0,20],[0,20],
             [0,0.4],
             [0,52],
             [0,1],[0,1]]
    names = ['Input Fire','Input Elev',
             'Input WindX','Input WindY',
             'Live Herb M','Live Wood M',
             'Moisture 1-h','Moisture 10-h',
             'Moisture 100-h','Canopy Cover',
             'Canopy Height','Canopy Base Height',
             'Canopy Density',
             'model',
             'Network','Truth']
    
    """
    '''
    (t.max()-t.min())/tOff
    for i in range(0,t.max(),tOff):
        """ This is how command line farsite outputs
        """
        try:
            startInd = np.argwhere(t-i>=0)[0][0]
            endInd = np.argwhere(t-i>=tOff)[0][0]
            currentFire = getRasterFromPolygon(dataOut,tF,startInd,1,sz)
            nextFire = getRasterFromPolygon(dataOut,tF,endInd,1,sz)
            
            currentFire = downsampleImage(currentFire,interval)
            nextFire = downsampleImage(nextFire,interval)
            data = [currentFire,elevImg,windXImg,windYImg,lhmImg,lwmImg,m1hImg,m10hImg,m100hImg,canopyImg,canopyHeightImg,canopyBaseHeightImg,canopyDensityImg,fuelImg,nextFire,nextFire]
            plotWildfireData(data,names,clims=clims,saveFig=True,saveName=inDir+namespace+'_'+str(i)+'_'+str(i+tOff)+'_summary.png')
        except:
            pass
        
        """
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
        """
    '''