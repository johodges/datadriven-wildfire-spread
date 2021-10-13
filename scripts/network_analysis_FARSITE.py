# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 08:34:06 2018

@author: jhodges
"""

import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from networkDesign import cnnModel3, cnnModel4
import network_analysis as na
import util_common as uc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

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
    #print(specName,constsName)
    
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
    inDir = 'E:/projects/wildfire-research/farsite/results/train/lowres_2/'
    testDir = 'E:/projects/wildfire-research/farsite/results/test/lowres_2/'
    
    inDir = 'I:\\wildfire-research\\train\\lowres_2\\'
    testDir = 'I:\\wildfire-research\\test\\lowres_2\\'
    
    trainFiles = glob.glob(inDir+'run*.h5')[:2000]
    trainFiles2 = glob.glob(inDir+'run*.h5')[2000:]
    testFiles = glob.glob(testDir+'run*.h5')[:2000]
    
    allFiles = trainFiles+testFiles
    np.random.seed(0)
    np.random.shuffle(allFiles)
    
    trainFiles = allFiles[:2000]+trainFiles2
    testFiles = allFiles[2000:]
    
    modelDir = "../models/farsiteTest_lowres_4"
    #modelDir = "../models/rothermelFull_cnnModel_3"
    modelFnc = cnnModel3
    batchSize = 100
    epochs = 200001
    bestThresh = 0.41 # Training Data
    index = 0
    ns = '../outputs/farsite_results_redo'
    generatePlots = True
    train = False
    
    #assert False, "stopped"

    
    #datas, truths = na.readPickledRawData('../rothermelData/dataRemakeTest3000')
    #datas = np.array(datas,dtype=np.float32)
    #truths = np.array(truths/255,dtype=np.int64)
    #labels = na.datas2labels(truths)
    #datas = (datas,truths)
    #testing_data, training_data = na.splitdata_tf(datas,test_number=1936,fakeRandom=True)
    
    #train_data = np.array(training_data[0],dtype=np.float32)
    #train_labels = np.array(training_data[1]/255,dtype=np.int64)
    #train_labels_exp = na.datas2labels(train_labels)
    
    if train:
        datas, truths = readAllData(trainFiles)
        datas = np.array(datas,dtype=np.float32)
        labels = np.array(truths,dtype=np.int64)
        na.convolve_wildfire_train(datas,labels,modelFnc,epochs=epochs,model_dir=modelDir)
    
    datas, truths = readAllData(testFiles)
    datas = np.array(datas,dtype=np.float32)
    labels = np.array(truths,dtype=np.int64)
        
    t1 = uc.tic()
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    evalSummary, prediction_exp, truth_exp = na.convolve_wildfire_test(datas,labels,modelFnc,model_dir=modelDir)
    pImgs = np.reshape(prediction_exp,(prediction_exp.shape[0],50,50,2))
    tImgs = np.reshape(truth_exp,(truth_exp.shape[0],50,50,2))
    iImgs = np.reshape(datas,(datas.shape[0],50,50,13))
    
    predictionImgs = pImgs[:,:,:,1]/(pImgs[:,:,:,0]+pImgs[:,:,:,1])
    truthImgs = tImgs[:,:,:,1]/(tImgs[:,:,:,0]+tImgs[:,:,:,1])
    print(uc.toc(t1))
    #inputs = na.inputs2labels(datas,labels)
    #inputsImgs = na.arrayToImage(inputs)
    #prediction_exp[prediction_exp < 0] = 0
    #prediction = na.labels2probs(prediction_exp,fireThresh=1.0)
    #truth = na.labels2datas(truth_exp,fireThresh=0.75)
    
    #predictionImgs = na.arrayToImage(prediction,outStyle=True)
    #truthImgs = na.arrayToImage(truth,outStyle=True)
    
    confusionMatrix = []
    for i in range(0,len(truthImgs)):
        pImg = predictionImgs[i].copy()
        tImg = truthImgs[i].copy()
        iImg = iImgs[i,:,:,0].copy()
        confusionMatrix.append(na.findConfusionMatrix(pImg,tImg,bestThresh,iImg))
    confusionMatrix = np.array(confusionMatrix)
    averageConfusionMatrix = np.mean(confusionMatrix,axis=0)
    stdConfusionMatrix = np.std(confusionMatrix,axis=0)
    print("True Negative: %.2f +- %.2f"%(averageConfusionMatrix[0],stdConfusionMatrix[0]))
    print("True Positive: %.2f +- %.2f"%(averageConfusionMatrix[3],stdConfusionMatrix[3]))
    print("False Negative: %.2f +- %.2f"%(averageConfusionMatrix[1],stdConfusionMatrix[1]))
    print("False Positive: %.2f +- %.2f"%(averageConfusionMatrix[2],stdConfusionMatrix[2]))
    print("Accuracy: %.2f +- %.2f"%(averageConfusionMatrix[4],stdConfusionMatrix[4]))
    print("Recall: %.2f +- %.2f"%(averageConfusionMatrix[5],stdConfusionMatrix[5]))
    print("Precision: %.2f +- %.2f"%(averageConfusionMatrix[6],stdConfusionMatrix[6]))
    print("fMeasure: %.2f +- %.2f"%(averageConfusionMatrix[7],stdConfusionMatrix[7]))
    
    print("IGNORING SMALL FIRE RESULTS:")
    confusionMatrix2 = confusionMatrix[confusionMatrix[:,8] > 9,:]
    averageConfusionMatrix = np.mean(confusionMatrix2,axis=0)
    stdConfusionMatrix = np.std(confusionMatrix2,axis=0)
    print("True Negative: %.2f +- %.2f"%(averageConfusionMatrix[0],stdConfusionMatrix[0]))
    print("True Positive: %.2f +- %.2f"%(averageConfusionMatrix[3],stdConfusionMatrix[3]))
    print("False Negative: %.2f +- %.2f"%(averageConfusionMatrix[1],stdConfusionMatrix[1]))
    print("False Positive: %.2f +- %.2f"%(averageConfusionMatrix[2],stdConfusionMatrix[2]))
    print("Accuracy: %.2f +- %.2f"%(averageConfusionMatrix[4],stdConfusionMatrix[4]))
    print("Recall: %.2f +- %.2f"%(averageConfusionMatrix[5],stdConfusionMatrix[5]))
    print("Precision: %.2f +- %.2f"%(averageConfusionMatrix[6],stdConfusionMatrix[6]))
    print("fMeasure: %.2f +- %.2f"%(averageConfusionMatrix[7],stdConfusionMatrix[7]))
    #print(averageConfusionMatrix)
    
    fs = 32
    plt.figure(figsize=(12,12))
    plt.hist(confusionMatrix[:,7],bins=20,range=(0,1))
    plt.xlabel('F-Measure',fontsize=fs)
    plt.ylabel('Number of Occurrences',fontsize=fs)
    plt.xlim(-0.01,1.01)
    plt.ylim(0,1000)
    plt.tick_params(labelsize=fs)
    plt.tight_layout()
    plt.savefig(ns+'_Fa_pdf.eps')
    nBins = 1000
    
    recallP80, precisionP80, fMeasureP80 = na.getPercentile(confusionMatrix,nBins,600)
    recallP90, precisionP90, fMeasureP90 = na.getPercentile(confusionMatrix,nBins,300)
    recallP95, precisionP95, fMeasureP95 = na.getPercentile(confusionMatrix,nBins,150)
    
    recallM = averageConfusionMatrix[5]
    precisionM = averageConfusionMatrix[6]
    fMeasureM = averageConfusionMatrix[7]
    
    recallP = confusionMatrix2[index,5]
    precisionP = confusionMatrix2[index,6]
    fMeasureP = confusionMatrix2[index,7]
    '''
    print("Metric\t\tM\t80\t90\t95")
    print("%s\t\t%.2f\t%.2f\t%.2f\t%.2f"%("Recall",recallM,recallP80,recallP90,recallP95))
    print("%s\t%.2f\t%.2f\t%.2f\t%.2f"%("Precision",precisionM,precisionP80,precisionP90,precisionP95))
    print("%s\t%.2f\t%.2f\t%.2f\t%.2f"%("FMeasure",fMeasureM,fMeasureP80,fMeasureP90,fMeasureP95))
    '''
    print("Metric\t\tP\tM\t80\t90\t95")
    print("%s\t\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%("Recall",recallP,recallM,recallP80,recallP90,recallP95))
    print("%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%("Precision",precisionP,precisionM,precisionP80,precisionP90,precisionP95))
    print("%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%("FMeasure",fMeasureP,fMeasureM,fMeasureP80,fMeasureP90,fMeasureP95))
    
    
    inds = np.where(confusionMatrix[:,7] < 0.8)[0]
    plt.figure(figsize=(12,12))
    plt.hist(confusionMatrix[inds,8],bins=25,range=(0,100))#,normed=True)
    plt.xlabel('Input Fire Size (px)',fontsize=fs)
    plt.ylabel('Number of Occurrences',fontsize=fs)
    plt.xlim(0,100)
    plt.ylim(0,len(inds))
    plt.tick_params(labelsize=fs)
    plt.tight_layout()
    plt.savefig(ns+'_fireSize_when_F_lt_0.8.eps')
    
if generatePlots:
    fs = 48
    lnwidth = 3
    xmin = 0
    xmax = iImgs.shape[2]
    xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/10)+1))
    ymin = 0
    ymax = iImgs.shape[1]
    yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/10)+1))
    #toPlots = [25,46,159,349,404,405,406,407,408,409,410,544,545,546,547,548,549,550]
    toPlots = [int(x) for x in list(np.linspace(0,1000,251))]
    toPlots = [0,1,2,3,4,5]
    ns = '../results/farsite_redo'
    for i in range(0,len(toPlots),1):
        toPlot = toPlots[i]
        fusedFire = truthImgs[toPlot].copy()
        fusedFire[fusedFire == 0] = -6.0
        fusedFire[iImgs[toPlot,:,:,0] > 0] = 0.0
        fusedFire[fusedFire > 0] = 6.0
        #saveName = ns+'independentTest_'+str(toPlot)+'%s.png'%('fused')
        saveName = ns+'farsite_exampleFusedFire%.0f.pdf'%(i)
        clim = [-6,6]
        
        fusedFire[1,2] = 0
        fusedFire[5,2] = 6
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='both',labelsize=fs)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('km',fontsize=fs)
        plt.ylabel('km',fontsize=fs)
        img = ax.imshow(fusedFire,cmap='hot_r',vmin=clim[0],vmax=clim[-1])
        plt.tight_layout()
        ax.annotate('Initial Burn Map\nBurn Map After 6 hours',xy=(5,6.0),xycoords='data',textcoords='data',xytext=(5,6.0),fontsize=fs)
        fig.savefig(saveName)
        plt.clf()
        plt.close(fig)


        pImg = predictionImgs[toPlot].copy()
        #saveName = ns+'independentTest_'+str(toPlot)+'%s.png'%('networkRaw')
        saveName = ns+'farsite_exampleNetworkRaw%.0f.pdf'%(i)
        clim = [0,1]
        np.savetxt(ns+'farsite_exampleNetworkRaw%.0f.csv'%(i),pImg,delimiter=',')
        
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='both',labelsize=fs)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('km',fontsize=fs)
        plt.ylabel('km',fontsize=fs)
        img = ax.imshow(pImg,cmap='hot_r',vmin=clim[0],vmax=clim[-1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%", pad=0.05)
        c = plt.colorbar(img,ticks=[1.0,0.8,0.6,0.4,0.2,0.0],cax=cax)
        plt.tick_params(labelsize=fs)
        plt.ylabel('Probability of Fire',fontsize=fs)
        plt.tight_layout()
        #ax.annotate('Initial Burn Map\nBurn Map After 6 hours',xy=(5,4.75),xycoords='data',textcoords='data',xytext=(5,4.75),fontsize=fs)
        fig.savefig(saveName)
        plt.clf()
        plt.close(fig)





        pImg = na.postProcessFirePerimiter(predictionImgs[toPlot].copy(),bestThresh)
        #saveName = ns+'independentTest_'+str(toPlot)+'%s.png'%('networkProcessed')
        saveName = ns+'farsite_exampleNetworkProcessed%.0f.pdf'%(i)
        clim = [0,1]
        np.savetxt(ns+'exampleNetworkProcessed%.0f.csv'%(i),pImg,delimiter=',')
        
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='both',labelsize=fs)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('km',fontsize=fs)
        plt.ylabel('km',fontsize=fs)
        img = ax.imshow(pImg,cmap='hot_r',vmin=clim[0],vmax=clim[-1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%", pad=0.05)
        c = plt.colorbar(img,ticks=[1.0,0.8,0.6,0.4,0.2,0.0],cax=cax)
        plt.tick_params(labelsize=fs)
        plt.ylabel('Probability of Fire',fontsize=fs)
        plt.tight_layout()
        #ax.annotate('Initial Burn Map\nBurn Map After 6 hours',xy=(5,4.75),xycoords='data',textcoords='data',xytext=(5,4.75),fontsize=fs)
        fig.savefig(saveName)
        plt.clf()
        plt.close(fig)
        
        pImg[pImg>bestThresh] = 1.0
        errorImg = pImg-truthImgs[toPlot]
        #saveName = ns+'independentTest_'+str(toPlot)+'%s.png'%('error')
        saveName = ns+'farsite_exampleNetworkError%.0f.pdf'%(i)
        errorImg[errorImg == 1] = 2
        errorImg[errorImg == 0] = -2
        errorImg[errorImg == -1] = 1
        errorImg[errorImg == -2] = 0
        
        errorImg[1,2] = 1
        errorImg[5,2] = 2
        
        clim = [0,2]

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='both',labelsize=fs)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('km',fontsize=fs)
        plt.ylabel('km',fontsize=fs)
        img = ax.imshow(errorImg,cmap='hot_r',vmin=clim[0],vmax=clim[-1])
        ax.annotate('Omission\nCommission',xy=(5,6.0),xycoords='data',textcoords='data',xytext=(5,6.0),fontsize=fs)
        plt.tight_layout()
        #ax.annotate('Initial Burn Map\nBurn Map After 6 hours',xy=(5,4.75),xycoords='data',textcoords='data',xytext=(5,4.75),fontsize=fs)
        fig.savefig(saveName)
        plt.clf()
        plt.close(fig)
    
    
    
    
    
    
    
    #na.convolve_wildfire_train(datas,truths[:,2500:],modelFnc,epochs=epochs,model_dir=modelDir)
    #evalSummary, prediction_exp, truth_exp = na.convolve_wildfire_test(datas,truths,modelFnc,model_dir=modelDir)
    
    #convolve_wildfire_train_preload(datas,truths,modelFnc,epochs=100001,batchSize=batchSize,model_dir=modelDir)
    #convolve_wildfire_train(files,modelFnc,epochs=100001,batchSize=batchSize,model_dir=modelDir)
    #prediction = convolve_wildfire_test(files[99:110],modelFnc,batchSize=batchSize,model_dir=modelDir)
    
    '''
    p2 = np.array([np.reshape(prediction[x,:],(50,50,2)) for x in range(prediction.shape[0])])
    p3 = p2[:,:,:,1]/np.sum(p2,axis=3)
    
    truths = []
    datas = []
    for i in range(0,prediction.shape[0]):
        data, truth = readSpecH5(bytes(files[i+99],'utf-8'))
        truth2 = np.reshape(truth,(50,50,2))
        datas.append(data)
        truths.append(truth2[:,:,1])
        
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(p3[i,:,:],vmin=0.5,vmax=1)
        plt.subplot(1,2,2)
        plt.imshow(truth2[:,:,1],vmin=0,vmax=1)
    '''
