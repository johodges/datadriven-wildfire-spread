# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 07:46:43 2018

@author: jhodges
"""

import h5py
import numpy as np
import glob
import skimage.measure as skme
import matplotlib.pyplot as plt

def writeConstH5(name,elevImg,canopyImg,canopyHeightImg,canopyBaseHeightImg,fuelImg):
    hf = h5py.File(name,'w')
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
    #print(specName.split('run_')[0]+"..//"+constsName)
    hf = h5py.File(specName.split('run_')[0]+"..//"+constsName,'r')
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
    
    return data, outputBurnmap, constsName

def downsampleData(data,outputBurnmap):
    #data = data[::33,::33,:]
    #data = data[:-1,:-1,:]
    data2 = np.zeros((50,50,13))
    for j in range(0,data.shape[2]):
        data2[:,:,j] = skme.block_reduce(data[:,:,j],(33,33),np.median)[:50,:50]
    outputBurnmap = skme.block_reduce(outputBurnmap,(33,33),np.median)[:50,:50]
    
    #outputBurnmap = outputBurnmap[::33,::33]
    #outputBurnmap = outputBurnmap[:-1,:-1]
    
    return data2, outputBurnmap

def extractConsts(data):
    elev = data[:,:,1]
    canopyImg = data[:,:,9]
    canopyHeightImg = data[:,:,10]
    canopyBaseHeightImg = data[:,:,11]
    fuelImg = data[:,:,12]
    return elev, canopyImg, canopyHeightImg, canopyBaseHeightImg, fuelImg

def extractPointData(data):
    (windX,windY) = (data[0,0,2],data[0,0,3])
    (lhm,lwm) = (data[0,0,4],data[0,0,5])
    (m1h,m10h,m100h) = (data[0,0,6],data[0,0,7],data[0,0,8])
    pointData = [windX,windY,lhm,lwm,m1h,m10h,m100h]
    return pointData

if __name__ == "__main__":
    inDir = "E:\\projects\\wildfire-research\\farsite\\results\\train\\"
    outDir = inDir+"lowres_2\\"
    files = glob.glob(inDir+"run*.h5")
    for file in files:
        #file = files[10]
        data, outputBurnmap, constsName = readSpecH5(file)
        data, outputBurnmap = downsampleData(data,outputBurnmap)
        
        elev, canopy, canopyHeight, canopyBaseHeight, fuel = extractConsts(data)
        pointData = extractPointData(data)
        inputBurnmap = data[:,:,0]
        
        specName = outDir+file.split('\\')[-1]
        writeConstH5(outDir+constsName,elev,canopy,canopyHeight,canopyBaseHeight,fuel)
        writeSpecH5(specName,pointData,inputBurnmap,outputBurnmap,constsName)
    
    pass