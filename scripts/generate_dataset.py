# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:47:32 2018

@author: JHodges
"""

import parse_elevation as pe
import parse_modis_file as pm
import parse_asos_file as pa
import util_common as uc
from parse_asos_file import ASOSMeasurementList, ASOSStation
import datetime as dt
import numpy as np
import scipy.interpolate as scpi

class GriddedMeasurement(object):
    __slots__ = ['dateTime','latitude','longitude','data','remapped','label','dataName']
    
    def __init__(self,dateTime,lat,lon,data,label):
        self.dateTime = dateTime
        self.latitude = lat
        self.longitude = lon
        self.data = data
        self.label = label

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
    if dataName == 'FireMask':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime,lat,lon+0.5,dataRaw,'AF')
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
    elif dataName == 'VegetationIndex':
        #Find vegetation index at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/aqua_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        data = GriddedMeasurement(queryDateTime,lat,lon+0.5,dataRaw,'VI')
    elif dataName == 'BurnedArea':
        #Find burned area at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/modis_burnedarea/",
                                              sdsname='burndate')
        data = GriddedMeasurement(queryDateTime,lat,lon+0.5,dataRaw,'BA')
    data.dataName = dataName
    return data

def queryDatabase(queryDateTime,dataNames,
                  asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0),
                  asosResolution = 111):
    datas = []
    for dataName in dataNames:
        data = loadDataByName(queryDateTime,dataName)
        if dataName == 'FireMask':
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
    tim = uc.tic()
    queryDateTime = dt.datetime(year=2016,month=6,day=27,hour=5,minute=53)
    asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
    asosResolution = 111
    
    dataNames = ['FireMask','Elevation','WindX','WindY','BurnedArea','VegetationIndex']
    datas = queryDatabase(queryDateTime,dataNames)

    print("Time to load data:")
    tim = uc.toc(tim)
    
    for data in datas:
        fig = uc.plotContourWithStates(data.latitude,data.longitude,data.data,label=data.label)