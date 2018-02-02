# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:47:32 2018

@author: JHodges
"""

import util_common as uc
import parse_elevation as pe
import parse_modis_file as pm
import parse_asos_file as pa
import remapSwathData as rsd

from parse_asos_file import ASOSMeasurementList, ASOSStation
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import scipy.interpolate as scpi
import psutil
import gc
import time

class GriddedMeasurement(object):
    __slots__ = ['dateTime','latitude','longitude','data','remapped','label','dataName','clim']
    
    def __init__(self,dateTime,lat,lon,data,label):
        self.dateTime = dateTime
        self.latitude = lat
        self.longitude = lon
        self.data = data
        self.label = label
        self.clim = None

    def mats2pts(self,lat,lon):
        lat = np.reshape(lat,(lat.shape[0]*lat.shape[1]))
        lon = np.reshape(lon,(lon.shape[0]*lon.shape[1]))
        pts = np.zeros((len(lat),2))
        pts[:,0] = lat
        pts[:,1] = lon
        return pts
    
    def remap(self,new_lat,new_lon,ds=10):
        oldpts = self.mats2pts(self.latitude,self.longitude)
        newpts = self.mats2pts(new_lat,new_lon)
        values = np.reshape(self.data,(self.data.shape[0]*self.data.shape[1],))
        
        remapped = scpi.griddata(oldpts[0::ds],values[0::ds],newpts,method='linear')
        
        self.data = np.reshape(remapped,(new_lat.shape[0],new_lat.shape[1]))
        self.latitude = new_lat.copy()
        self.longitude = new_lon.copy()

def loadDataByName(queryDateTime,dataName):
    modOffset = 0.0
    if dataName == 'FireMaskHourlyT':
        lat, lon, dataRaw =  rsd.queryTimeCustomHdf(queryDateTime,
                                                    datadir="E:/WildfireResearch/data/terra_hourly_activefires_jh2/",
                                                    sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
    elif dataName == 'FireMaskHourlyA':
        lat, lon, dataRaw =  rsd.queryTimeCustomHdf(queryDateTime,
                                                    datadir="E:/WildfireResearch/data/aqua_hourly_activefires_jh2/",
                                                    sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
    elif dataName == 'FireMaskT':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="E:/WildfireResearch/data/terra_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
    elif dataName == 'FireMaskA':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
    elif dataName == 'Elevation':
        lat,lon,dataRaw = pe.queryElevation()
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'m')
    elif dataName == 'WindX':
        lat, lon, speedX, speedY = pa.queryWindSpeed(
                queryDateTime,
                filename='../data-test/asos-stations.pkl',
                resolution=asosResolution,
                timeRange=asosTimeRange)
        data = GriddedMeasurement(queryDateTime,lat,lon,speedX,'u m/s')
        #speedY = GriddedMeasurement(queryDateTime,lat,lon,speedY)
    elif dataName == 'WindY':
        lat, lon, speedX, speedY = pa.queryWindSpeed(
                queryDateTime,
                filename='../data-test/asos-stations.pkl',
                resolution=asosResolution,
                timeRange=asosTimeRange)
        #speedX = GriddedMeasurement(queryDateTime,lat,lon,speedX)
        data = GriddedMeasurement(queryDateTime,lat,lon,speedY,'v m/s')
    elif dataName == 'VegetationIndexT':
        #Find vegetation index at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/terra_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'VI')
    elif dataName == 'VegetationIndexA':
        #Find vegetation index at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/aqua_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'VI')
    elif dataName == 'BurnedArea':
        #Find burned area at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/modis_burnedarea/",
                                              sdsname='burndate')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'BA')
    data.dataName = dataName
    return data

def queryDatabase(queryDateTime,dataNames,
                  asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0),
                  asosResolution = 111):
    datas = []
    for dataName in dataNames:
        data = loadDataByName(queryDateTime,dataName)
        if dataName == 'FireMaskHourlyT':
            modis_lat = data.latitude
            modis_lon = data.longitude
        datas.append(data)
    
    for i in range(0,len(datas)):
        if datas[i].dataName == 'Elevation':
            datas[i].remap(modis_lat,modis_lon,ds=10)
        elif datas[i].dataName == 'WindX' or dataName == 'WindY':
            datas[i].remap(modis_lat,modis_lon,ds=2)
    
    return datas



if __name__ == "__main__":
    ''' case 0: Find active fire, elevation, wind-x, wind-y, burned area, and
                vegetation index at a single query time and map to MODIS grid
        case 1: Generate daily active fire map for 360 days from one satellite
        case 2: Compare active fire index from aqua and terraf or a single
                query time
    '''
    
    
    case = 0
    
    if case == 0:
        tim = uc.tic()
        #queryDateTime = dt.datetime(year=2017,month=12,day=4,hour=5,minute=53)
        basetime = '2017051'
        queryTime = time.mktime(time.strptime(basetime+'03','%Y%j%H'))
        queryDateTime = dt.datetime.fromtimestamp(queryTime)
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/Database/'
        ns = "DatabaseQuery"
        
        dataNames = ['FireMaskHourlyT','Elevation','WindX','WindY','VegetationIndexA']
        datas = queryDatabase(queryDateTime,dataNames)
    
        print("Time to load data:")
        tim = uc.toc(tim)
        
        for data in datas:
            name = outdir+ns+basetime+'_'+data.dataName+'.png'
            fig = uc.plotContourWithStates(
                    data.latitude,data.longitude,data.data,
                    label=data.label,clim=data.clim,
                    saveFig=True,saveName=name)
    elif case == 1:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        queryDateTime = dt.datetime(year=2017,month=1,day=1,hour=12,minute=0)
        satellite = 'aqua'
        if satellite == 'aqua':
            dataNames = ['FireMaskA']
            ns = 'AFaqua'
        elif satellite == 'terra':
            dataNames = ['FireMaskT']
            ns = 'AFterra'
        
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/AF_images/'
        for i in range(0,360):
            mem = psutil.virtual_memory()[2]
            
            if mem < 90.0:
                datas = queryDatabase(queryDateTime,dataNames)
                data = datas[0]
                af_name = outdir+ns+queryDateTime.isoformat()[0:13]+'.png'
            
                if data is not None and mem < 90.0:
                    
                    af_fig = uc.plotContourWithStates(
                            data.latitude,data.longitude,data.data,
                            label=data.label,clim=data.clim,
                            saveFig=True,saveName=af_name)    
                    
                    data_mask = data.data.copy()
                    data_mask[data_mask < 7] = 0
                    pts = pm.extractCandidates(data.latitude,data.longitude,data_mask)
                    if i > 0:
                        match_pts = pm.compareCandidates(old_pts,pts)
                        if match_pts.shape[0] > 0:
                            print("Time %s found %.0f matches with the closest %.4f km."%(queryDateTime.isoformat(),match_pts.shape[0],np.min(match_pts[:,1])))
                    else:
                        pass
                    queryDateTime = queryDateTime + dt.timedelta(days=1)
                    old_pts = pts
                    gc.collect()
                    plt.show()
                else:
                    old_pts = np.array([])
            else:
                print("Memory usage too high")
    elif case == 2:
        ''' This case reads one data set and plots it.
        '''
        queryDateTime = dt.datetime(year=2017,month=12,day=2,hour=3,minute=0)
        dataNames = ['FireMaskHourlyT','FireMaskT']
        ns = 'AFterra'
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/AF_images/'
        for i in range(0,1):
            mem = psutil.virtual_memory()[2]
            
            if mem < 90.0:
                datas = queryDatabase(queryDateTime,dataNames)
                #aqua_data = datas[0]
                terra_data = datas[0]
                af_name = outdir+ns+queryDateTime.isoformat()[0:13]+'.png'
            
                if terra_data is not None and mem < 90.0:
                    af_fig = uc.plotContourWithStates(
                            terra_data.latitude,terra_data.longitude,terra_data.data,
                            label=terra_data.label,clim=terra_data.clim,
                            saveFig=False,saveName=af_name,
                            cmap='jet')
                    
                    #fig = plt.figure(figsize=(12,8),tight_layout=True)
                    #plt.imshow(img)
                    print(queryDateTime.isoformat())
                    plt.show()
                else:
                    print("Did not find both images")
                queryDateTime = queryDateTime + dt.timedelta(days=1)
            else:
                print("Memory usage too high")
    elif case == 3:
        ''' This case reads one hourly active fire data set and one daily
        composite data set and plots them both.
        '''
        
        basetime = '2017197'
        queryTime = time.mktime(time.strptime(basetime+'03','%Y%j%H'))
        queryDateTime = dt.datetime.fromtimestamp(queryTime)
        ns = 'HC'
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/HourlyCompositeComparison/'
            
        datas = queryDatabase(queryDateTime,['FireMaskHourlyT','FireMaskT'])
        queryDateTime = queryDateTime + dt.timedelta(hours=12)
        datas2 = queryDatabase(queryDateTime,['FireMaskHourlyT'])
        am_data = datas[0]
        cm_data = datas[1]
        pm_data = datas2[0]
        
        if am_data is not None and pm_data is not None and cm_data is not None:
            name = outdir+ns+basetime+'am'+'.png'
            data = am_data
            am_fig = uc.plotContourWithStates(
                    data.latitude,data.longitude,data.data,
                    label=data.label,clim=data.clim,
                    saveFig=True,saveName=name,cmap='jet')
            name = outdir+ns+basetime+'pm'+'.png'
            data = pm_data
            am_fig = uc.plotContourWithStates(
                    data.latitude,data.longitude,data.data,
                    label=data.label,clim=data.clim,
                    saveFig=True,saveName=name,cmap='jet')
            name = outdir+ns+basetime+'to'+'.png'
            data = cm_data
            am_fig = uc.plotContourWithStates(
                    data.latitude,data.longitude,data.data,
                    label=data.label,clim=data.clim,
                    saveFig=True,saveName=name,cmap='jet')
    elif case == 4:
        
        queryDateTime = dt.datetime(year=2017,month=3,day=31,hour=15,minute=0)
        dataNames = ['FireMaskA','FireMaskT']
        ns = 'Comparison'
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/ATComparison_images/'
        for i in range(0,360):
            mem = psutil.virtual_memory()[2]
            
            if mem < 90.0:
                datas = queryDatabase(queryDateTime,dataNames)
                aqua_data = datas[0]
                terra_data = datas[1]
                af_name = outdir+ns+queryDateTime.isoformat()[0:13]+'.png'
            
                if aqua_data is not None and terra_data is not None and mem < 90.0:
                    aqua_mask = aqua_data.data.copy()
                    aqua_mask[aqua_mask < 7] = 0
                    aqua_mask[aqua_mask != 0] = 1.0
                    
                    terra_mask = terra_data.data.copy()
                    terra_mask[terra_mask < 7] = 0
                    terra_mask[terra_mask != 0] = 1.0
                    
                    img = np.zeros((aqua_mask.shape[0],aqua_mask.shape[1]))
                    img = img+terra_mask+aqua_mask
                    #img = np.zeros((aqua_mask.shape[0],aqua_mask.shape[1],3))+1
                    #img[:,:,0] = 1-aqua_mask.copy()
                    #img[:,:,1] = 1-terra_mask.copy()
                    
                    af_fig = uc.plotContourWithStates(
                            aqua_data.latitude,aqua_data.longitude,img,
                            label=aqua_data.label,clim=[0,0.8,1,1.2,2],
                            saveFig=True,saveName=af_name,
                            cmap='satComp')
                    
                    #fig = plt.figure(figsize=(12,8),tight_layout=True)
                    #plt.imshow(img)
                    print(queryDateTime.isoformat())
                    plt.show()
                else:
                    print("Did not find both images")
                queryDateTime = queryDateTime + dt.timedelta(days=1)
            else:
                print("Memory usage too high")