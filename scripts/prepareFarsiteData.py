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
import tensorflow as tf
import sys
import h5py

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
    if data.shape[0] <= 200:
        #assert False, "Stopped"
        for i in range(0,data.shape[0]):
            data['geometry'][i] = Polygon(list(data['geometry'][i].coords))
    else:
        #assert False, "Stopped"
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

def generateBurnMap(inDir,lcpNamespace,namespace):
    rst = rasterio.open(inDir+lcpNamespace)
    meta = rst.meta.copy()
    meta.update(driver='GTiff')
    meta.update(count=1)
    meta.update(nodata=255)
    meta.update(crs="EPSG:3759")
    meta.update(dtype='uint8')
    dataOut = loadFarsiteOutput(namespace)
    dataOut = dataOut.reindex(index=dataOut.index[::-1])
    with rasterio.open(outDir+namespace.split(inDir)[1]+'.tif','w', **meta) as out:
        out_arr = out.read(1)
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(dataOut.geometry, dataOut.time))
        burned = features.rasterize(shapes=shapes, fill=100, out=out_arr, transform=out.transform)
        out.write_band(1, burned)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

#def _int64_feature(value):
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def writeConstH5(name,elevImg,canopyImg,canopyHeightImg,canopyBaseHeightImg,fuelImg):
    hf = h5py.File(outDir+constsName,'w')
    hf.create_dataset('elevation',data=elevImg,compression="gzip",compression_opts=9)
    hf.create_dataset('canopyCover',data=canopyImg,compression="gzip",compression_opts=9)
    hf.create_dataset('canopyHeight',data=canopyHeightImg,compression="gzip",compression_opts=9)
    hf.create_dataset('canopyBaseHeight',data=canopyBaseHeightImg,compression="gzip",compression_opts=9) # or /canopyHeightImg
    hf.create_dataset('fuelModel',data=fuelImg,compression="gzip",compression_opts=9)
    hf.close()
    
def writeSpecH5(specName,pointData,inputBurnmap,outputBurnmap,constsName):
    hf = h5py.File(specName,'w')
    hf.create_dataset('pointData',data=pointData,compression="gzip",compression_opts=9)
    hf.create_dataset('inputBurnmap',data=inputBurnmap,compression="gzip",compression_opts=9)
    hf.create_dataset('outputBurnmap',data=outputBurnmap,compression="gzip",compression_opts=9)
    hf.create_dataset('constsName',data=bytes(constsName,'utf-8'),dtype=h5py.special_dtype(vlen=bytes))
    hf.close()
    
def readSpecH5(specName):
    hf = h5py.File(specName,'r')
    pointData = hf.get('pointData')
    inputBurnmap = np.array(hf.get('inputBurnmap'),dtype=np.float)
    outputBurnmap = np.array(hf.get('outputBurnmap'),dtype=np.float)
    constsName = hf.get('constsName').value.decode('utf-8')
    
    [windX,windY,lhm,lwm,m1h,m10h,m100h] = pointData
    
    hf.close()
    hf = h5py.File(specName.split('run_')[0]+constsName,'r')
    elev = np.array(hf.get('elevation'),dtype=np.float)
    canopyCover = np.array(hf.get('canopyCover'),dtype=np.float)
    canopyHeight = np.array(hf.get('canopyHeight'),dtype=np.float)
    canopyBaseHeight = np.array(hf.get('canopyBaseHeight'),dtype=np.float)
    fuelModel = np.array(hf.get('fuelModel'),dtype=np.float)
    
    data = np.zeros((elev.shape[0],elev.shape[1],13))
    data[:,:,0] = inputBurnmap
    data[:,:,1] = elev
    data[:,:,2] = windX
    data[:,:,3] = windY
    data[:,:,4] = lhm
    data[:,:,5] = lwm
    data[:,:,6] = m1h
    data[:,:,7] = m10h
    data[:,:,8] = m100h
    data[:,:,9] = canopyCover
    data[:,:,10] = canopyHeight
    data[:,:,11] = canopyBaseHeight
    data[:,:,12] = fuelModel
    
    return data, outputBurnmap

if __name__ == "__main__":
    
    commandFile = 'commonDir/farsite/example/Panther/runPanther.txt'
    inDir = 'E:/projects/wildfire-research/farsite/data/'
    outDir = 'E:/projects/wildfire-research/farsite/results/'
    
    inputNames = ['Current Fire','Elevation','Current WindX','Current WindY','Live Herb M','Live Wood M','Moisture 1-h','Moisture 10-h','Moisture 100-h','Canopy Cover','Canopy Height','Crown Ratio','Fuel Model']
    
    files = glob.glob(inDir+"run_*_*_*_*_*_*_Perimeters.shp")
    print("Total files: %.0f"%(len(files)))
    #assert False, "Stopped"
    for i in range(1060,len(files)):#file in files:
        try:
            #i = -1
            file = files[i]
            #file = inDir[:-1]+'\\'+'run_0_8_n116-9499_38-9950_25000_out_Perimeters.shp'
            namespace = file.split('\\')[1].split('_out_Perimeters.shp')[0]
            outNamespace = outDir+namespace+'_p.png'
            namespace = inDir+namespace
            lcpNamespace = namespace.split('_')[3]+'_'+namespace.split('_')[4]+'_'+namespace.split('_')[5]+'.LCP'
            
            if len(glob.glob(outDir+namespace.split(inDir)[1]+'.tif'))==0:
                #generateBurnMap(inDir,lcpNamespace,namespace)
                #assert False, "Stopped"
                pass
            else:
                interval = 6
                burnmapData = rasterio.open(outDir+namespace.split(inDir)[1]+'.tif')
                burnmap = np.array(burnmapData.read_band(1),dtype=np.float)
                rInd = int(burnmap.shape[0]/2)
                cInd = int(burnmap.shape[1]/2)
                burnmap[rInd-2:rInd+2,cInd-2:cInd+2] = -1
                maxBurnTime = np.max(burnmap[burnmap <= 200].max())
                maxBurnSteps = int(maxBurnTime/interval)
                burnmaps = np.zeros((burnmap.shape[0],burnmap.shape[1],maxBurnSteps),dtype=np.float)
                
                for i in range(0,maxBurnSteps):
                    burnTime = float(i*interval)
                    burnTemp = burnmap.copy()
                    burnTemp[burnTemp < burnTime] = -1
                    burnTemp[burnTemp >= burnTime] = 0
                    burnTemp[burnTemp == -1] = 1
                    burnmaps[:,:,i] = burnTemp
                #print("Found %s"%(outDir+namespace.split(inDir)[1]+'.tif'))
                lcpData = loadFarsiteLcp(inDir+lcpNamespace)
                elevImg = np.array(lcpData.read(1),dtype=np.float32)
                elevImg = elevImg-np.median(elevImg)
                fuelImg = lcpData.read(4)
                canopyImg = np.array(lcpData.read(5),dtype=np.float32)/100
                canopyHeightImg = np.array(lcpData.read(6),dtype=np.float32)/10
                canopyBaseHeightImg = np.array(lcpData.read(7),dtype=np.float32)/10
                canopyDensityImg = np.array(lcpData.read(8),dtype=np.float32)/100
                
                sz = elevImg.shape
                
                plt.imshow(burnmaps[:,:,-1])
                
                moistures, weathers, winds = parseFarsiteInput(namespace+'.input')
                windSpeed = np.median(winds[:,3])
                windDir = np.median(winds[:,4])
                windX,windY = convertVector(windSpeed,windDir)
                m1h, m10h, m100h, lhm, lwm = parseMoistures(moistures)
                
                sz = elevImg.shape
                fuelImg = remapFuelImg(fuelImg)
                smallSz = elevImg.shape
                
                #writer = tf.python_io.TFRecordWriter('train.tfrecords')
                constsName = lcpNamespace.split('.LCP')[0]+'.h5'
                if len(glob.glob('%s%s'%(outDir,constsName))) == 0:
                    writeConstH5(outDir+constsName,elevImg,canopyImg,canopyHeightImg,canopyBaseHeightImg,fuelImg)
                
                for i in range(0,maxBurnSteps-1):
                    pointData = np.array([windX,windY,lhm,lwm,m1h,m10h,m100h])
                    inputBurnmap = np.array(burnmaps[:,:,i],dtype=np.float)
                    outputBurnmap = np.array(burnmaps[:,:,i+1],dtype=np.float)
                    specName = outDir+namespace.split(inDir)[1]+'_%0.0f.h5'%(i)
                    writeSpecH5(specName,pointData,inputBurnmap,outputBurnmap,constsName)
                    
                    #feature = {'train/label': _int64_feature(np.array(data[:,:,-1],dtype=np.int64).flatten()),
                    #           'train/image': _float_feature(np.array(data[:,:,:-1]).flatten())}
                    #example = tf.train.Example(features=tf.train.Features(feature=feature))
                    #writer.write(example.SerializeToString())
                #writer.close()
                sys.stdout.flush()
        except:
            print("Failed: %s"%(file))