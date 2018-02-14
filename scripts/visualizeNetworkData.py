# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import generate_dataset as gd
from generate_dataset import GriddedMeasurementPair
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

import util_common as uc
import numpy as np
import glob
import sklearn.decomposition as skd
import pickle

def loadCandidates(indirs):
    if type(indirs) is not list:
        indirs = [indirs]
    files = []
    for indir in indirs:
        if indir[-1] != '/':
            indir = indir+'/'
        files.extend(glob.glob(indir+'*.pkl'))
    
    datas = []
    for i in range(0,len(files)):
        file = files[i]
        data = uc.readPickle(file)
        if data is not None:
            for i in range(0,len(data)):
                for key in data[i].__dict__.keys():
                    d = getattr(data[i],key)
                    toMod = True
                    if 'FireMask' in key:
                        d = d-6
                        dmin = 0
                        dmax = 3
                    elif 'Elevation' in key:
                        dmin = -600
                        dmax = 4200
                    elif 'VegetationIndex' in key:
                        dmin = 0
                        dmax = 10000
                    elif 'Wind' in key:
                        dmin = -12
                        dmax = 12
                    else:
                        toMod = False
                    if toMod:
                        d[d < dmin] = dmin
                        d[d >= dmax] = dmax
                        d = (d-dmin)/(dmax-dmin)
                        setattr(data[i],key,d)
            datas.extend(data)
    return datas

def im2vector(img,k=3):
    data = []
    try:
        u,s,v = np.linalg.svd(img)
        u = np.reshape(u[:,:k],(u.shape[0]*k,))
        v = np.reshape(v[:k,:],(v.shape[0]*k,))
        s = s[:k]
        data.extend(u)
        data.extend(v)
        data.extend(s)
        return np.array(data)
    except np.linalg.LinAlgError:
        return None


def rearrangeDataset(datas,debugPrint=False):
    
    
    allInputs = []
    allOutputs = []
    for i in range(0,len(datas)):
        data = datas[i]
        inputs = []
        outputs = []
        nanError = False
        nanKey = ''
        for key in data.__dict__.keys():
            if "In_" in key:
                d = getattr(data,key)
                sz = np.shape(d)
                d = np.reshape(d,(sz[0]*sz[1],))
                if len(np.where(np.isnan(d))[0]) > 0:
                    #print(key)
                    #nanError = True
                    if np.isnan(np.nanmin(d)):
                        nanError = True
                        nanKey = nanKey+key+", "
                    d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                d = np.reshape(d,(sz[0],sz[1]))
                d = im2vector(d)
                if d is not None:
                    inputs.extend(d)
                else:
                    nanError = True
            if "Out_" in key:
                d = getattr(data,key)
                sz = np.shape(d)
                d = np.reshape(d,(sz[0]*sz[1],))
                if len(np.where(np.isnan(d))[0]) > 0:
                    #print(key)
                    #nanError = True
                    if np.isnan(np.nanmin(d)):
                        nanError = True
                        nanKey = nanKey+key+", "
                    d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                d = np.reshape(d,(sz[0],sz[1]))
                d = im2vector(d)
                outputs.extend(d)
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        if not nanError:
            allInputs.append(inputs)#[0:5000])
            allOutputs.append(outputs)
        else:
            dataTime = data.strTime()[0]
            if debugPrint:
                print("nanError at: %s for keys: %s"%(dataTime,nanKey))
    
    print(np.shape(allInputs),np.shape(allOutputs))
    newDatas = (np.array(allInputs),np.array(allOutputs))
    return newDatas

def reconstructImg(img,k=3):
    sz = int((np.shape(img)[0]-k)/(2*k))
    print(sz,sz*k)
    u = np.reshape(img[0:sz*k],(sz,k))
    print(sz*k+1,2*sz*k)
    v = np.reshape(img[sz*k:2*sz*k],(k,sz))
    s = img[2*sz*k:]
    print(u.shape,v.shape,s.shape)
    data = np.dot(u,np.dot(np.diag(s),v))
    return data


    
def visualizeDirectory(indir,outdir,dataFile):
    if not glob.glob(dataFile):
        datas = gd.loadCandidates(indir)
        with open(dataFile,'wb') as f:
            pickle.dump(datas,f)
    else:
        with open(dataFile,'rb') as f:
            datas = pickle.load(f)
    for i in range(0,len(datas)):
        dTmp = datas[i]
        lat, lon = dTmp.getCenter(decimals=4)
        if not glob.glob(outdir+dTmp.strTime(hours=False)[0]+'_'+lat+'_'+lon+'.png'):
            dTmp.plot(saveFig=True,
                      closeFig=None,
                      saveName=outdir+dTmp.strTime(hours=False)[0]+'_'+lat+'_'+lon+'.png',
                      clim=np.linspace(0,1,10),
                      label='AF')

def visualizeDay(indir,outdir,dataFile,day):
    data = None
    if not glob.glob(dataFile):
        datas = gd.loadCandidates(indir)
        with open(dataFile,'wb') as f:
            pickle.dump(datas,f)
    else:
        with open(dataFile,'rb') as f:
            datas = pickle.load(f)
    for i in range(0,len(datas)):
        dTmp = datas[i]
        lat, lon = dTmp.getCenter(decimals=4)
        if dTmp.strTime(hours=False)[0]+'_'+lat+'_'+lon == day:
            print("Found it: ", i)
            data = dTmp
    return data

def getCoordinatesDirectory(indir,dataFile):
    if not glob.glob(dataFile):
        datas = gd.loadCandidates(indir)
        with open(dataFile,'wb') as f:
            pickle.dump(datas,f)
    else:
        with open(dataFile,'rb') as f:
            datas = pickle.load(f)
    coords = np.zeros((len(datas),3))
    for i in range(0,len(datas)):
        lat, lon = datas[i].getCenter(decimals=4)
        coords[i,1] = lat
        coords[i,2] = lon
        coords[i,0] = datas[i].inTime.timestamp()/(24*3600)
    return coords

def getClusterCenters(data,offsetFirst=True):
    import sklearn.cluster as sklc
    if offsetFirst:
        t0 = data[0,0].copy()
        data[:,0] = data[:,0]-t0
    else:
        t0 = 0
    db = sklc.DBSCAN(eps=0.3, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    #colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters)
    
    centers = np.zeros((n_clusters,3))
    variance = np.zeros((n_clusters,3))
    for k in unique_labels:
        xy = coords[(labels == k) & core_samples_mask]
        
        if k > -1 and xy.shape[0] > 0:
            centers[k,:] = np.median(xy,axis=0)
            variance[k,:] = np.std(xy,axis=0)
            print("k=%.0f\tTim=%.4f+-%.4f\tLat=%.4f+-%.4f\tLon=%.4f+-%.4f" % (k,centers[k,0],variance[k,0],
                                                                              centers[k,1],variance[k,1],
                                                                              centers[k,2],variance[k,2]))
    centers[:,0] = centers[:,0]+t0
    return centers, variance

def plotClusterCenters(centers):
    from mpl_toolkits.mplot3d import Axes3D
    n_clusters = centers.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(centers[:,1],centers[:,2],centers[:,0])
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Time [hours]')
    plt.title('Estimated number of clusters: %d' % n_clusters)    
    plt.show()

if __name__ == "__main__":
    ''' case0: Output image files visualizing all candidates in a directory
        case1: Return data for a single candidate in a directory
        case2: Determine number of unique clusters in candidates based on
            latitude, longitude, and time
    '''
    case = 0
    
    #from sklearn.cluster import DBSCAN
    #from sklearn import metrics
    if case == 0:
        #indir = ['../networkData/20180213/']
        indir = ['../output/nhood_25_25_step3/']
        outdir=indir[0]+'images/'
        #outdir = '../networkData/20180213/images/'
        dataFile = indir[0]+'dataraw.out'
        visualizeDirectory(indir,outdir,dataFile)
    elif case == 1:
        indir = ['../networkData/20180208/']
        outdir = '../networkData/images/'
        dataFile = indir[0]+'dataraw.out'
        day='201617618_42.7805_-114.642'
        data = visualizeDay(indir,outdir,dataFile,day)
    elif case == 2:
        indir = ['../networkData/20180213/']
        outdir = '../networkData/20180213/images/'
        dataFile = indir[0]+'dataraw.out'
        coords = getCoordinatesDirectory(indir,dataFile)
        centers, variance = getClusterCenters(coords,offsetFirst=True)
        plotClusterCenters(centers)
        