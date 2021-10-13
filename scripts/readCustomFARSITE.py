# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 08:34:06 2018

@author: jhodges
"""

import glob
import numpy as np
import h5py

def readAllData(files,interval=1):
    datas = []
    truths = []
    for file in files[::interval]:
        data, truth = readSpecH5(bytes(file,'utf-8'))
        datas.append(np.reshape(data,(50*50*13)).copy())
        truths.append(np.reshape(truth,(50*50*2)).copy())
    datas = np.array(datas)
    truths = np.array(truths)
    return datas, truths

def readSpecH5(specName):
    specName = specName.decode('utf-8')
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
    
    output = np.zeros((outputBurnmap.shape[0],outputBurnmap.shape[1],2))
    output[:,:,0] = outputBurnmap.copy()
    output[:,:,1] = outputBurnmap.copy()
    output[:,:,0] = 1 - output[:,:,0]
    
    return data, output

if __name__ == "__main__":
    trainDir = 'I:\\wildfire-research\\train\\lowres_2\\'
    testDir = 'I:\\wildfire-research\\test\\lowres_2\\'
    
    trainFiles = glob.glob(trainDir+'run*.h5')
    testFiles = glob.glob(testDir+'run*.h5')
    
    train_datas, train_truths = readAllData(trainFiles)
    test_datas, test_truths = readAllData(testFiles)