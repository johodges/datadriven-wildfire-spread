# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:47:32 2018

@author: JHodges
"""

#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

import util_common as uc
import parse_elevation as pe
import parse_modis_file as pm
import parse_asos_file as pa
import remapSwathData as rsd

from parse_asos_file import ASOSMeasurementList, ASOSStation
import pyhdf.SD as phdf
#import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import scipy.interpolate as scpi
import psutil
import gc
import time
import sys
import glob
import pickle
import scipy as sp

class GriddedMeasurement(object):
    ''' This class contains a measurement on a latitude and longitude grid.
    
        Fields:
            dateTime: Time stamp of the measurement as a datetime.datetime
            latitude: Latitude of each gridded point as a numpy.ndarray
            longitude: Longitude of each gridded point as a numpy.ndarray
            data: Measurement of each gridded point as a numpy.ndarray
            label: Label to use when plotting a contour of this measurement
            dataName: Name associated with this measurement
            clim: Contour limits to use when plotting a contour of this
                measurement
                
        Functions:
            mats2pts: This will reshape the latitude and longitude matrices
                into arrays and return an Nx2 numpy array with [lat,lon]
            remap: This will remap data, latitude, and longitude to a new grid
                specified by new_lat and new_lon matrices
            computeMemory: This will calculate the memory usage by this object
            strTime: This will return a string containing the time stamp
                associated with this measurement as a string
    '''
    __slots__ = ['dateTime','latitude','longitude','data','label','dataName','clim'] # 'remapped'
    
    def __init__(self,dateTime,lat,lon,data,label):
        self.dateTime = dateTime
        self.latitude = lat
        self.longitude = lon
        self.data = data
        self.label = label
        self.clim = None
        
    def __str__(self):
        ''' This function prints summary information of the object when a
        string is requested.
        '''
        if self.dateTime is not None:
            dts = time.mktime(self.dateTime.timetuple())+self.dateTime.microsecond/1E6
            dts = time.strftime('%Y%j%H%M%S',time.localtime(dts))
        else:
            dts = 'Not Considered'
        string = "Gridded Measurement\n"
        string = string + "\tType:\t%s\n"%(self.dataName)
        string = string + "\ttime:\t%s (yyyydddhhmmssss)\n"%(dts)
        string = string + "\tgrid:\t%.0f,%.0f (Latitude,Longitude)\n"%(self.data.shape[0],self.data.shape[1])
        string = string + "\tmemory:\t%.4f MB"%(self.computeMemory())
        return string
    
    def __repr__(self):
        ''' This function prints summary information of the object when a
        string is requested.
        '''
        return self.__str__()

    def mats2pts(self,lat,lon):
        ''' This will reshape the latitude and longitude matrices into arrays
        and return an Nx2 numpy array with [lat,lon]
        '''
        lat = np.reshape(lat,(lat.shape[0]*lat.shape[1]))
        lon = np.reshape(lon,(lon.shape[0]*lon.shape[1]))
        pts = np.zeros((len(lat),2))
        pts[:,0] = lat
        pts[:,1] = lon
        return pts
    
    def removeNans(self,points,values):
        inds = np.where(~np.isnan(points[:,0]) & ~np.isnan(points[:,1]) & ~np.isnan(values))
        newPoints = points[inds]
        newValues = values[inds]
        return newPoints, newValues
    
    def remap(self,new_lat,new_lon,ds=10,method='linear'):
        ''' This will remap data, latitude, and longitude to a new grid
        specified by new_lat and new_lon matrices.
        
        NOTE: ds defines how much to downsample the original grid prior to
            remapping (scpi.griddate can use too much memory). 
        NOTE: method defines what method to use in the resampling process. If
            method='linear' bilinear interpolation will be used. If
            method='nearest' nearest neighbor value will be used.
        '''
        oldpts = self.mats2pts(self.latitude,self.longitude)
        values = np.reshape(self.data,(self.data.shape[0]*self.data.shape[1],))
        newpts = self.mats2pts(new_lat,new_lon)
        
        oldpts, values = self.removeNans(oldpts,values)
        
        remapped = scpi.griddata(oldpts[0::ds],values[0::ds],newpts,method=method)
        
        self.data = np.reshape(remapped,(new_lat.shape[0],new_lat.shape[1]))
        self.latitude = new_lat.copy()
        self.longitude = new_lon.copy()

    def computeMemory(self):
        ''' This will calculate the memory usage by this object
        '''
        mem = 0
        slots = self.__slots__
        for key in slots:
            try:
                if type(key) == list:
                    mem = mem + sys.getsizeof(getattr(self,key))/1024**2
                else:
                    mem = mem+sys.getsizeof(getattr(self,key))/1024**2
            except AttributeError:
                pass
        return mem
    
    def strTime(self,hours=True):
        ''' This will return a string containing the time stamp associated
        with this measurement as a string.
        '''
        timeFloat = time.mktime(self.dateTime.timetuple())+self.dateTime.microsecond/1E6
        timeTuple = time.localtime(timeFloat)
        if hours:
            return time.strftime('%Y%j%H%M%S',timeTuple)
        else:
            return time.strftime('%Y%j%H',timeTuple)

class GriddedMeasurementPair(object):
    ''' This class contains a pair of measurements on a latitude and longitude
    grid. The fields associated with the two measurements do not need to be
    the same.
    
        Fields:
            inTime: Time stamp of the first measurement as a datetime.datetime
            outTime: Time stamp of the second measurement as a
                datetime.datetime
            inKeys: List of fields associated with the first measurement
            outKeys: List of fields associated with the second measurement
            latitude: Latitude of each gridded point as a numpy.ndarray
            longitude: Longitude of each gridded point as a numpy.ndarray
            
            NOTE: Keys for each measurement will be added with either In_ or
                Out_ prefix to denote to which measurement they are
                associated.
                
        Functions:
            addStartData: This function will add the input data as a new input
                key
            addEndData: This function will add the input data as a new output
                key
            countData: This function will return teh numberof input and output
                keys
            getDataNames: This function will return a list of input and a list
                of output key names
            
            computeMemory: This will calculate the memory usage by this object
            strTime: This will return a string containing the time stamp
                associated with this measurement as a string
            getCenter: This will return the latitude and longitude of the
                center of the data
            plot: This will plot the object
                
    '''
    #__slots__ = ['inTime','outTime','latitude','longitude','inDatas','outDatas','inKeys','outKeys']
    
    def __init__(self,dataStart,dataEnd,inKeys,outKeys,bounds=None):
        self.inTime = dataStart.dateTime
        self.outTime = dataEnd.dateTime
        self.inKeys = inKeys
        self.outKeys = outKeys
        if bounds is None:
            self.latitude = dataStart.latitude
            self.longitude = dataStart.longitude
            #self.inData1 = dataStart.data
            #self.outDatas = [dataEnd.data]
        else:
            self.latitude = dataStart.latitude[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            self.longitude = dataStart.longitude[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            #self.inData1 = dataStart.data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            #self.outDatas = [dataEnd.data[bounds[0]:bounds[1],bounds[2]:bounds[3]]]
            
    def addStartData(self,data,bounds):
        ''' This function will add the input data as a new input key.
        
        NOTE: The key name will be 'In_' + data.dataName.
        NOTE: bounds is used when only a subset of the data is to be included.
        '''
        if data.dataName in self.inKeys:
            d = data.data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            setattr(self,'In_'+data.dataName,d)
            
    def addEndData(self,data,bounds):
        ''' This function will add the input data as a new output key.
        
        NOTE: The key name will be 'Out_' + data.dataName.
        NOTE: bounds is used when only a subset of the data is to be included.
        '''
        if data.dataName in self.outKeys:
            d = data.data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            setattr(self,'Out_'+data.dataName,d)
    
    def countData(self):
        ''' This function will return the number of input and output keys
        '''
        inCounter = 0
        outCounter = 0
        for key in self.__dict__.keys():
            if "In_" in key:
                inCounter = inCounter+1
            if "Out_" in key:
                outCounter = outCounter+1
        return inCounter, outCounter
    
    def getDataNames(self):
        ''' This function will return a list of input and a list of output key
        names
        '''
        inData = []
        outData = []
        for key in self.__dict__.keys():
            if "In_" in key:
                inData.append(key)
            if "Out_" in key:
                outData.append(key)
        return inData, outData
    
    def __str__(self):
        ''' This function prints summary information of the object when a
        string is requested.
        '''
        inTime, outTime = self.strTime()
        string = "Gridded Measurement Pair\n"
        string = string + "\tinTime:\t\t%s (yyyydddhhmmssss)\n"%(inTime)
        string = string + "\toutTime:\t%s (yyyydddhhmmssss)\n"%(outTime)
        string = string + "\tgrid:\t\t%.0f,%.0f (Latitude,Longitude)\n"%(self.latitude.shape[0],self.longitude.shape[1])
        string = string + "\tmemory:\t\t%.4f MB\n"%(self.computeMemory())
        string = string + "\tInputs:\n"
        for key in self.__dict__.keys():
            if "In_" in key:
                string = string+"\t\t"+key[3:]+"\n"
        string = string + "\tOutputs:\n"
        for key in self.__dict__.keys():
            if "Out_" in key:
                string = string+"\t\t"+key[4:]+"\n"
        return string
    
    def __repr__(self):
        ''' This function prints summary information of the object when a
        string is requested.
        '''
        return self.__str__()
    
    def computeMemory(self):
        ''' This will calculate the memory usage by this object
        '''
        mem = 0
        slots = self.__dict__.keys()
        for key in slots:
            try:
                if type(key) == list:
                    mem = mem + sys.getsizeof(getattr(self,key))/1024**2
                else:
                    mem = mem+sys.getsizeof(getattr(self,key))/1024**2
            except AttributeError:
                pass
        return mem
    
    def strTime(self,hours=True):
        ''' This will return a string containing the time stamp associated
        with this measurement as a string.
        '''
        inTimeFloat = time.mktime(self.inTime.timetuple())+self.inTime.microsecond/1E6
        inTimeTuple = time.localtime(inTimeFloat)
        outTimeFloat = time.mktime(self.outTime.timetuple())+self.outTime.microsecond/1E6
        outTimeTuple = time.localtime(outTimeFloat)
        if hours:
            return time.strftime('%Y%j%H%M%S',inTimeTuple), time.strftime('%Y%j%H%M%S',outTimeTuple)
        else:
            return time.strftime('%Y%j%H',inTimeTuple), time.strftime('%Y%j%H',outTimeTuple)

    def getCenter(self,decimals=4):
        ''' This will return the latitude and longitude of the mean of the
        latitude and longitude of the data
        '''
        lat = str(round(np.mean(self.latitude),decimals))
        lon = str(round(np.mean(self.longitude),decimals))
        return lat, lon

    def plot(self,
             saveFig=False,
             closeFig=None,
             saveName='',
             clim=None,
             cmap='jet',
             label='None'):
        ''' This function will plot the gridded measurement pair
        '''
        
        inNum, outNum = self.countData()
        inNames, outNames = self.getDataNames()
        inTime, outTime = self.strTime()
        
        
        totalPlots = np.ceil((float(inNum)+float(outNum))**0.5)
        colPlots = totalPlots
        rowPlots = np.ceil((float(inNum)+float(outNum))/colPlots)
        currentPlot = 0
        
        if saveFig:
            fntsize = 40
            lnwidth = 10
            fig = plt.figure(figsize=(colPlots*24,rowPlots*20))#,tight_layout=True)      
            if closeFig is None:
                closeFig = True
        else:
            fig = plt.figure(figsize=(colPlots*6,rowPlots*5))#,tight_layout=True)
            fntsize = 20
            lnwidth = 2
            if closeFig is None:
                closeFig = False
        
        xmin = np.round(np.min(self.longitude),1)
        xmax = np.round(np.max(self.longitude),1)
        xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/0.1)+1))
        ymin = np.round(np.min(self.latitude),1)
        ymax = np.round(np.max(self.latitude),1)
        yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/0.1)+1))



        names = inNames+outNames
        for i in range(0,len(names)):
            key = names[i]
            currentPlot = currentPlot+1

            ax = fig.add_subplot(rowPlots,colPlots,currentPlot)
            ax.tick_params(axis='both',labelsize=fntsize)
            plt.xticks(xticks)
            plt.yticks(yticks)
            plt.xlabel('Longitude',fontsize=fntsize)
            plt.ylabel('Latitude',fontsize=fntsize)
            plt.title(key,fontsize=fntsize)

            if clim is None:
                if 'FireMask' in key:
                    clim2 = np.linspace(0,9,10)
                    label = 'AF'
                elif 'Elevation' in key:
                    clim2 = np.linspace(-1000,5000,7)
                    label = 'Elev [m]'
                elif 'WindX' in key:
                    clim2 = np.linspace(-6,6,13)
                    label = 'u [m/s]'
                elif 'WindY' in key:
                    clim2 = np.linspace(-6,6,13)
                    label = 'v [m/s]'
                elif 'VegetationIndex' in key:
                    clim2 = np.linspace(-4000,10000,8)
                    label = 'NVDIx1000'
                elif 'Canopy' in key:
                    clim2 = np.linspace(0,100,11)
                    label = 'Canopy'
                else:
                    clim2 = np.linspace(0,9,10)
            else:
                label = ''
                clim2 = clim
                
            img = ax.contourf(self.longitude,self.latitude,getattr(self,key),levels=clim2,cmap=cmap)
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

        

def extractCandidates(dataStart,dataEnd,inKeys,outKeys,matches,nhood=[25,25]):
    ''' This function will extract a list of match points from a list of start
    data and a list of end data and store it as a list of
    GriddedMeasurementPairs
    '''
    
    datas = []    
    for i in range(0,len(matches)):
        rowLow = matches[i][0]-nhood[0]
        rowUp = matches[i][0]+nhood[0]
        colLow = matches[i][1]-nhood[1]
        colUp = matches[i][1]+nhood[1]
        bounds=[rowLow,rowUp,colLow,colUp]
        
        if rowLow > 0 and rowUp < dataStart[0].data.shape[0] and colLow > 0 and colUp < dataStart[0].data.shape[1]:
            data = GriddedMeasurementPair(dataStart[0],dataEnd[0],inKeys,outKeys,bounds=bounds)
        
            for d in dataStart:
                data.addStartData(d,bounds)
                    
            for d in dataEnd:
                data.addEndData(d,bounds)
                    
            datas.append(data)
    if len(datas) == 0:
        datas = None
    return datas

def compareGriddedMeasurement(data1,data2):
    ''' This function checks if 2 GriddedMeasurements have the same header
    information.
    '''
    if data1 is None or data2 is None:
        return False
    if (str(data1.dateTime) == str(data2.dateTime)) and (str(data1.dataName) == str(data2.dataName)):
        return True
    else:
        return False

def loadDataByName(queryDateTime,dataName):
    ''' This function will find the latitude, longitude, and data at a
    queryDateTime for the provided dateName. Specific cases correspond to
    different import functions.
    '''
    
    if dataName == 'FireMaskHourlyTA':
        lat, lon, dataRaw, timeStamp =  rsd.queryTimeCustomHdf(
                queryDateTime,
                datadirs=["G:/WildfireResearch/data/terra_hourly_activefires_jh/",
                          "G:/WildfireResearch/data/aqua_hourly_activefires_jh/"],
                sdsname='FireMask')
        data = []
        for i in range(0,len(dataRaw)):
            dataDateTime = dt.datetime.fromtimestamp(timeStamp[i])
            data.append(GriddedMeasurement(dataDateTime,lat[i],lon[i],dataRaw[i],'AF'))
            data[i].clim = np.linspace(0,9,10)
            data[i].dataName = dataName
    if dataName == 'FireMaskHourlyT':
        lat, lon, dataRaw, timeStamp =  rsd.queryTimeCustomHdf(
                queryDateTime,
                datadirs="G:/WildfireResearch/data/terra_hourly_activefires_jh/",
                sdsname='FireMask')
        dataDateTime = dt.datetime.fromtimestamp(timeStamp)
        data = GriddedMeasurement(dataDateTime,lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'FireMaskHourlyA':
        lat, lon, dataRaw, timeStamp =  rsd.queryTimeCustomHdf(
                queryDateTime,
                datadirs="G:/WildfireResearch/data/aqua_hourly_activefires_jh/",
                sdsname='FireMask')
        data = GriddedMeasurement(dataDateTime,lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'FireMaskT':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="G:/WildfireResearch/data/terra_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime.date(),lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'FireMaskA':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="G:/WildfireResearch/data/aqua_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime.date(),lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'Elevation':
        lat,lon,dataRaw = pe.queryElevation()
        data = GriddedMeasurement(None,lat,lon,dataRaw,'m')
        data.dataName = dataName
    elif dataName == 'WindX':
        lat, lon, speedX, speedY = pa.queryWindSpeed(
                queryDateTime,
                filename='../data-test/asos-stations.pkl',
                resolution=asosResolution,
                timeRange=asosTimeRange)
        data = GriddedMeasurement(queryDateTime,lat,lon,speedX,'u m/s')
        data.dataName = dataName
        #speedY = GriddedMeasurement(queryDateTime,lat,lon,speedY)
    elif dataName == 'WindY':
        lat, lon, speedX, speedY = pa.queryWindSpeed(
                queryDateTime,
                filename='../data-test/asos-stations.pkl',
                resolution=asosResolution,
                timeRange=asosTimeRange)
        #speedX = GriddedMeasurement(queryDateTime,lat,lon,speedX)
        data = GriddedMeasurement(queryDateTime,lat,lon,speedY,'v m/s')
        data.dataName = dataName
    elif dataName == 'VegetationIndexT':
        queryDateTime = dt.datetime(year=queryDateTime.year,month=queryDateTime.month,day=queryDateTime.day)
        #Find vegetation index at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="G:/WildfireResearch/data/terra_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'VI')
        data.dataName = dataName
    elif dataName == 'VegetationIndexA':
        queryDateTime = dt.datetime(year=queryDateTime.year,month=queryDateTime.month,day=queryDateTime.day)
        #Find vegetation index at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="G:/WildfireResearch/data/aqua_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'VI')
        data.dataName = dataName
    elif dataName == 'BurnedArea':
        queryDateTime = dt.datetime(year=queryDateTime.year,month=queryDateTime.month,day=queryDateTime.day)
        #Find burned area at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="G:/WildfireResearch/data/modis_burnedarea/",
                                              sdsname='burndate')
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'BA')
        data.dataName = dataName
    elif dataName == 'Canopy':
        lat,lon,dataRaw = uc.readPickle('E:/projects/wildfire-research/data-test/canopy.pkl')
        data = GriddedMeasurement(queryDateTime,lat,lon,dataRaw,'Canopy')
        data.dataName = dataName
        
    if type(data) is not list:
        data = [data]
    return data

def queryDatabase(queryDateTime,dataNames,
                  queryTimeRange = [0],
                  asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0),
                  asosResolution = 111,
                  closest=False):
    ''' This function will query the database for a specific queryDateTime
    for each name in dataNames.
    
    NOTE: queryTimeRange is a list of hour offsets from the queryDateTime to
        also consider.
    '''
    datas = []
    modis_lat = None
    for dataName in dataNames:
        for i in range(0,len(queryTimeRange)):
            qDT = queryDateTime+dt.timedelta(days=0,hours=queryTimeRange[i],minutes=0)
            data = loadDataByName(qDT,dataName)
            if dataName == 'FireMaskHourlyTA' and modis_lat is None and len(data) > 0:
                modis_lat = data[0].latitude
                modis_lon = data[0].longitude
            elif dataName == 'FireMaskHourlyT' or dataName == 'FireMaskHourlyA' and modis_lat is None:
                modis_lat = data.latitude
                modis_lon = data.longitude
            datas.extend(data)
    datasuq = []
    for i in range(0,len(datas)):
        uniqueCheck = True
        for j in range(0,len(datas)):
            if i != j:
                if compareGriddedMeasurement(datas[i],datas[j]):
                    if i < j:
                        pass
                    else:
                        uniqueCheck = False
                else:
                    pass
        if uniqueCheck:
            datasuq.append(datas[i])
    datas = datasuq.copy()
    if closest:
        datascl = []
        for i in range(0,len(datas)):
            closestCheck = True
            for j in range(0,len(datas)):
                if i != j and datas[i].dataName == datas[j].dataName:
                    if abs(datas[i].dateTime-queryDateTime) < abs(datas[j].dateTime-queryDateTime):
                        pass
                    else:
                        closestCheck = False
            if closestCheck:
                datascl.append(datas[i])
        datas = datascl.copy()
    
    if modis_lat is not None:
        datas = remapDatas(datas,modis_lat,modis_lon)
    else:
        lat_lmt = [30,44]
        lon_lmt = [-126,-112]
        pxPerDegree = 120
        lat_lnsp = np.linspace(np.min(lat_lmt),np.max(lat_lmt),
                               (np.max(lat_lmt)-np.min(lat_lmt)+1)*pxPerDegree)
        lon_lnsp = np.linspace(np.min(lon_lmt),np.max(lon_lmt),
                               (np.max(lon_lmt)-np.min(lon_lmt)+1)*pxPerDegree)
        modis_lon, modis_lat = np.meshgrid(lon_lnsp,lat_lnsp)
        datas = remapDatas(datas,modis_lat,modis_lon)
    
    return datas


def remapDatas(datas,modis_lat,modis_lon):
    for i in range(0,len(datas)):
        if type(datas[i]) is not list:
            if datas[i].dataName == 'Elevation':
                datas[i].remap(modis_lat,modis_lon,ds=10)
            elif datas[i].dataName == 'WindX' or datas[i].dataName == 'WindY':
                datas[i].remap(modis_lat,modis_lon,ds=2)
            elif datas[i].dataName == 'VegetationIndexA':
                datas[i].remap(modis_lat,modis_lon,ds=4)
            elif datas[i].dataName == 'VegetationIndexT':
                datas[i].remap(modis_lat,modis_lon,ds=4)
            elif datas[i].dataName == 'BurnedArea':
                datas[i].remap(modis_lat,modis_lon,ds=4)
            #elif datas[i].dataName == 'Canopy':
            #    datas[i].remap(modis_lat,modis_lon,ds=4)
    return datas

"""
def queryDatabase2(queryDateTime,dataNames,
                  queryTimeRange = [0],
                  asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0),
                  asosResolution = 111,
                  closest=False):
    ''' This function will query the database for a specific queryDateTime
    for each name in dataNames.
    
    NOTE: queryTimeRange is a list of hour offsets from the queryDateTime to
        also consider.
    '''
    datas = []
    modis_lat = None
    for dataName in dataNames:
        for i in range(0,len(queryTimeRange)):
            qDT = queryDateTime+dt.timedelta(days=0,hours=queryTimeRange[i],minutes=0)
            data = loadDataByName(qDT,dataName)
            if dataName == 'FireMaskHourlyTA' and modis_lat is None and len(data) > 0:
                modis_lat = data[0].latitude
                modis_lon = data[0].longitude
            elif dataName == 'FireMaskHourlyT' or dataName == 'FireMaskHourlyA' and modis_lat is None:
                modis_lat = data.latitude
                modis_lon = data.longitude
            datas.extend(data)
    datasuq = []
    for i in range(0,len(datas)):
        uniqueCheck = True
        for j in range(0,len(datas)):
            if i != j:
                if compareGriddedMeasurement(datas[i],datas[j]):
                    if i < j:
                        pass
                    else:
                        uniqueCheck = False
                else:
                    pass
        if uniqueCheck:
            datasuq.append(datas[i])
    datas = datasuq.copy()
    if closest:
        datascl = []
        for i in range(0,len(datas)):
            closestCheck = True
            for j in range(0,len(datas)):
                if i != j and datas[i].dataName == datas[j].dataName:
                    if abs(datas[i].dateTime-queryDateTime) < abs(datas[j].dateTime-queryDateTime):
                        pass
                    else:
                        closestCheck = False
            if closestCheck:
                datascl.append(datas[i])
        datas = datascl.copy()
    
    for i in range(0,len(datas)):
        if type(datas[i]) is not list:
            if datas[i].dataName == 'Elevation':
                datas[i].remap(modis_lat,modis_lon,ds=10)
            elif datas[i].dataName == 'WindX' or datas[i].dataName == 'WindY':
                datas[i].remap(modis_lat,modis_lon,ds=2)
            elif datas[i].dataName == 'VegetationIndexA':
                datas[i].remap(modis_lat,modis_lon,ds=4)
            elif datas[i].dataName == 'VegetationIndexT':
                datas[i].remap(modis_lat,modis_lon,ds=4)
            elif datas[i].dataName == 'BurnedArea':
                datas[i].remap(modis_lat,modis_lon,ds=4)
    
    
    return datas

"""


def getCandidates(queryDateTime,dataNames,
                  queryTimeRange=None,
                  oldData=None,
                  candidateThresh=100):
    ''' This function will query the database for a queryDateTime,
    queryDateTime + 6 hours, and queryDateTime + 12 hours. It will then look
    for data which are spatially coherent between the 3 sets. If enough
    values are found, the queryDateTime is considered a match, and the
    matches are returned.
    '''
    if queryTimeRange is None:
        queryTimeRange = np.linspace(0,12,int((12*1*1)/6+1))
    
    datas = queryDatabase(queryDateTime,dataNames,
                          asosTimeRange=asosTimeRange,
                          asosResolution=asosResolution,
                          queryTimeRange=queryTimeRange)
    if len(datas) == 0:
        return None, None, None
    
    data = datas[0]
    if not compareGriddedMeasurement(data,oldData):
        dataMesh = data.data.copy()
        dataMesh[dataMesh < 7] = 0
        pts, coords = pm.geolocateCandidates(data.latitude,data.longitude,dataMesh)
        matches = []
        candidateCheck = 0.0
    
        for i in range(1,len(datas)):
            dataNext = datas[i]
            dataNextMesh = dataNext.data.copy()
            dataNextMesh[dataNextMesh < 7] = 0
            ptsNext, coordsNext = pm.geolocateCandidates(dataNext.latitude,dataNext.longitude,dataNextMesh)
            match_pts = pm.compareCandidates(pts,ptsNext)
            if match_pts.shape[0] > 0:
                matches.extend(match_pts)
            else:
                #matches.append([])
                pass
            candidateCheck = candidateCheck+float(match_pts.shape[0])
        if candidateCheck > candidateThresh:
            coordsMatch = coords[np.array(np.squeeze(matches)[:,0],dtype=np.int),:]
        else:
            return None, None, None
    else:
        return None, None, None
    
    return datas[0], datas[1], coordsMatch

def loadCandidates(indirs,dataFile,forceRebuild=False,memThresh=90):
    ''' This function will laod all candidate pickle files from a list
    of directories.
    '''
    if not forceRebuild and glob.glob(dataFile) != []:
        print("Loading data from %s"%(dataFile))
        with open(dataFile,'rb') as f:
            datas = pickle.load(f)
        return datas
    elif not forceRebuild and glob.glob(dataFile) == []:
        print("File not found, rebuilding %s"%(dataFile))
    elif forceRebuild and glob.glob(dataFile) != []:
        print("Found file but forcing rebuild %s"%(dataFile))
    elif forceRebuild and glob.glob(dataFile) == []:
        print("File not found, rebuilding %s"%(dataFile))
    
    
    if type(indirs) is not list:
        indirs = [indirs]
    files = []
    for indir in indirs:
        if indir[-1] != '/':
            indir = indir+'/'
        files.extend(glob.glob(indir+'*.pkl'))
    
    memoryError = False
    datas = []
    for file in files:
        mem = psutil.virtual_memory()[2]
        if mem < memThresh:
            data = uc.readPickle(file)
            if data is not None:

                for i in range(0,len(data)):
                    for key in data[i].__dict__.keys():
                        d = getattr(data[i],key)
                        toMod = True
                        if 'FireMask' in key:
                            [r,c] = d.shape
                            d = d-6
                            dmin = 0
                            dmax = 3
                        elif 'Elevation' in key:
                            [r,c] = d.shape
                            centerValue = d[int(r/2),int(c/2)]
                            d = d-centerValue
                            dmin = -2000
                            dmax = 2000
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
        else:
            memoryError = True
    if memoryError:
        print("Memory is full, re-run the loading software")
    else:
        print("Saving data file to pickle %s"%(dataFile))
        with open(dataFile,'wb') as f:
                pickle.dump(datas,f)
        
    return datas

def rearrangeDataset(datas,dataFile,
                     forceRebuild=False,
                     svdInputs=False,k=25,
                     blurImage=True,kernelSz=10,stdev=1,
                     debugPrint=False,
                     memThresh=90.0):
    
    if not forceRebuild and glob.glob(dataFile) != []:
        print("Loading data from %s"%(dataFile))
        with open(dataFile,'rb') as f:
            newDatas, keys = pickle.load(f)
        return newDatas, keys
    elif not forceRebuild and glob.glob(dataFile) == []:
        print("File not found, rebuilding %s"%(dataFile))
    elif forceRebuild and glob.glob(dataFile) != []:
        print("Found file but forcing rebuild %s"%(dataFile))
    elif forceRebuild and glob.glob(dataFile) == []:
        print("File not found, rebuilding %s"%(dataFile))
    
    allInputs = []
    allOutputs = []
    nanWarningCounter = 0
    nanErrorCounter = 0
    memoryError = False
    for i in range(0,len(datas)):
        if psutil.virtual_memory()[2] < memThresh:
            data = datas[i]
            inputs = []
            outputs = []
            nanError = False
            nanWarning = False
            nanKey = ''
            for key in data.inKeys:
                d = getattr(data,'In_'+key)
                sz = np.shape(d)
                d = np.reshape(d,(sz[0]*sz[1],))
                if len(np.where(np.isnan(d))[0]) > 0:
                    #print(key)
                    #nanError = True
                    if np.isnan(np.nanmin(d)):
                        nanError = True
                        nanKey = nanKey+key+", "
                    elif len(np.where(np.isnan(d))[0]) > 0:
                        nanWarning = True
                    d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                if blurImage:
                    d = np.reshape(d,(sz[0],sz[1]))
                    d = blurImg(d,kernelSz=kernelSz,stdev=stdev)
                    d = np.reshape(d,(sz[0]*sz[1],))
                if svdInputs:
                    d = np.reshape(d,(sz[0],sz[1]))
                    d = im2vector(d,k=k)
                if d is not None:
                    inputs.extend(d)
                else:
                    nanError = True
            for key in data.outKeys:
                d = getattr(data,'Out_'+key)
                sz = np.shape(d)
                d = np.reshape(d,(sz[0]*sz[1],))
                if len(np.where(np.isnan(d))[0]) > 0:
                    #print(key)
                    #nanError = True
                    if np.isnan(np.nanmin(d)):
                        nanError = True
                        nanKey = nanKey+key+", "
                    elif np.where(np.isnan(d))[0] > 0:
                        nanWarning = True
                    d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                if blurImage:
                    d = np.reshape(d,(sz[0],sz[1]))
                    d = blurImg(d,kernelSz=kernelSz,stdev=stdev)
                    d = np.reshape(d,(sz[0]*sz[1],))
                if svdInputs:
                    d = np.reshape(d,(sz[0],sz[1]))
                    d = im2vector(d,k=k)
                outputs.extend(d)
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            if nanWarning:
                nanWarningCounter = nanWarningCounter+1
            if not nanError:
                allInputs.append(inputs)#[0:5000])
                allOutputs.append(outputs)
            else:
                nanErrorCounter = nanErrorCounter+1
                dataTime = data.strTime()[0]
                if debugPrint:
                    print("nanError at: %s for keys: %s"%(dataTime,nanKey))
        else:
            memoryError = True
            print("Not enough memory to reshape.")
    print("Number of nanWarnings: %.0f"%(nanWarningCounter))
    print("Number of nanErrors: %.0f"%(nanErrorCounter))
    print(np.shape(allInputs),np.shape(allOutputs))
    newDatas = (np.array(allInputs),np.array(allOutputs))
    keys = (datas[0].inKeys,datas[0].outKeys)
    if memoryError:
        print("Memory is full, re-run the loading software")
    else:
        print("Saving data file to pickle %s"%(dataFile))
        for d in datas:
            del d
        gc.collect()
        with open(dataFile,'wb') as f:
                pickle.dump([newDatas,keys],f)
    
    return newDatas, keys

def datasKeyRemap(datas,keys):
    szIn = datas[0].shape
    numIn = len(keys[0])
    szOut = datas[1].shape
    numOut = len(keys[1])
    
    newDatas = []
    dataSize = int(int(szIn[1]/numIn)**0.5)
    
    for i in range(0,szIn[0]):
        newData = []
        for j in range(0,numIn):
            data = datas[0][i,j*int(szIn[1]/numIn):(j+1)*int(szIn[1]/numIn)].copy()
            dataRS = np.reshape(data,(dataSize,dataSize))
            newData.append(dataRS)
        for j in range(0,numOut):
            data = datas[1][i,j*int(szOut[1]/numOut):(j+1)*int(szOut[1]/numOut)].copy()
            dataRS = np.reshape(data,(dataSize,dataSize))
            newData.append(dataRS)
        newDatas.append(newData)
    
    names = []
    for i in range(0,numIn):
        names.append('In_'+keys[0][i])
    for i in range(0,numOut):
        names.append('Val_'+keys[1][i])
    
    return newDatas, names
    

def im2vector(img,k=10):
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

def blurImg(img,kernelSz=10,stdev=1):
    blurred = sp.ndimage.filters.gaussian_filter(img, stdev, order=0)
    return blurred

def reconstructImg(img,k=10):
    sz = int((np.shape(img)[0]-k)/(2*k))
    u = np.reshape(img[0:sz*k],(sz,k))
    v = np.reshape(img[sz*k:2*sz*k],(k,sz))
    s = img[2*sz*k:]
    data = np.dot(u,np.dot(np.diag(s),v))
    return data

def datasRemoveKey(datas,key):
    
    for data in datas:
        if key[0:3] == 'In_':
            if key[3:] in data.inKeys:
                data.inKeys.remove(key[3:])
                data.__dict__.pop(key,None)
        if key[0:4] == 'Out_':
            if key[4:] in data.inKeys:
                data.inKeys.remove(key[4:])
                data.__dict__.pop(key,None)
    return datas

def generateHdfDataset(datas,lat,lon,sdsnames,timeStamp,outFile,
                       sdsdescriptions=None,sdsunits=None):
    ''' This function will generate a custom Level 3 hdf file
    '''
    hdfFile = phdf.SD(outFile,phdf.SDC.WRITE|phdf.SDC.CREATE) # Assign a few attributes at the file level
    hdfFile.author = 'JENSEN HUGHES'
    hdfFile.productionDate = time.strftime('%Y%j.%H%M',time.gmtime(time.time()))
    hdfFile.timeStamp = str('%.4f'%(timeStamp))
    hdfFile.latitudeL = str('%.8f'%(data.latitude[0]))
    hdfFile.latitudeR = str('%.8f'%(data.latitude[-1]))
    hdfFile.longitudeL = str('%.8f'%(data.longitude[0]))
    hdfFile.longitudeR = str('%.8f'%(data.longitude[-1]))
    hdfFile.priority = 2
    d1 = hdfFile.create(sdsname, phdf.SDC.FLOAT32, data.data.shape)
    d1.description = sdsdescription
    d1.units = sdsunits
    dim1 = d1.dim(0)
    dim2 = d1.dim(1)
    dim1.setname('latitude')
    dim2.setname('longitude')
    dim1.units = 'degrees'
    dim2.units = 'degrees'
    d1[:] = data.data
    d1.endaccess()
    hdfFile.end()

if __name__ == "__main__":
    ''' case 0: Find active fire, elevation, wind-x, wind-y, burned area, and
                vegetation index at a single query time and map to MODIS grid
        case 1: Generate daily active fire map for 360 days from one satellite
        case 2: Compare active fire index from aqua and terraf or a single
                query time
        case 3: This case reads one hourly active fire data set and one daily
                composite data set and plots them both.
        case 6: This case will find candidate matches in the input directory.
    '''
    
    
    case = 7
    
    if case == 0:
        tim = uc.tic()
        #queryDateTime = dt.datetime(year=2017,month=12,day=4,hour=5,minute=53)
        #basetime = '2017051'
        basetime = '2016168'
        queryTime = time.mktime(time.strptime(basetime+'03','%Y%j%H'))
        queryDateTime = dt.datetime.fromtimestamp(queryTime)
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/Database/'
        ns = "DatabaseQueryNew"
        
        dataNames = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
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
        #import matplotlib
        #matplotlib.use('agg')
        #import matplotlib.pyplot as plt
        queryDateTime = dt.datetime(year=2017,month=1,day=1,hour=12,minute=0)
        satellite = 'aqua'
        if satellite == 'aqua':
            dataNames = ['FireMaskA']
            ns = 'AFaqua'
        elif satellite == 'terra':
            dataNames = ['FireMaskT']
            ns = 'AFterra'
        
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/AF_images/'
        old_pts = np.array([])
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
                    pts = pm.geolocateCandidates(data.latitude,data.longitude,data_mask)
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
    if case == 5:
        tim = uc.tic()
        #queryDateTime = dt.datetime(year=2017,month=12,day=4,hour=5,minute=53)
        basetime = '2017195'
        queryTime = time.mktime(time.strptime(basetime+'15','%Y%j%H'))
        queryDateTime = dt.datetime.fromtimestamp(queryTime)
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/ATCompilation/'
        ns = "DatabaseQuery"
        
        dataNames = ['FireMaskHourlyTA']
        #dataNames = ['FireMaskHourlyTA','Elevation']
        #dataNames = ['Elevation']
        datas = queryDatabase(queryDateTime,dataNames,
                              asosTimeRange=asosTimeRange,
                              asosResolution=asosResolution,
                              queryTimeRange=list(np.linspace(0,24*7,int((24*7*1)/6+1))))
        
        
        
        print("Time to load data:")
        tim = uc.toc(tim)
        
        
        for data in datas:
            if data.dateTime is not None:
                dts = time.mktime(data.dateTime.timetuple())+data.dateTime.microsecond/1E6
                dts = time.strftime('%Y%j%H%M%S',time.localtime(dts))
            name = outdir+ns+dts+'_'+data.dataName+'.png'
            fig = uc.plotContourWithStates(
                    data.latitude,data.longitude,data.data,
                    label=data.label,clim=data.clim,
                    saveFig=True,saveName=name)

    if case == 6:
        #import matplotlib
        #matplotlib.use('agg')
        #import matplotlib.pyplot as plt
        tim = uc.tic()
        #queryDateTime = dt.datetime(year=2017,month=12,day=4,hour=5,minute=53)
        basetime = '2016001'
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/nhood_25_25/'
        ns = "DatabaseQuery"
        dataNames = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
        inKeys = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
        outKeys = ['FireMaskHourlyTA']
            
        candidates = []
        oldData = None
        toPlot = False
        for i in range(0,3000): #8
            mem = psutil.virtual_memory()[2]
            if mem < 90.0:
                queryTime = time.mktime(time.strptime(basetime+'15','%Y%j%H'))+(3600*6)*float(i)
                queryTimestr = time.strftime('%Y%j%H',time.localtime(queryTime))
                queryDateTime = dt.datetime.fromtimestamp(queryTime)
                if glob.glob(outdir+queryTimestr+'.pkl') == []:
                    dataIn, dataOut, coordinateMatches = getCandidates(queryDateTime,['FireMaskHourlyTA'],oldData=oldData)
                    oldData = dataIn
                    mem = psutil.virtual_memory()[2]
                    if mem < 90.0:
                        if dataIn is not None and dataOut is not None and coordinateMatches is not None:
                            print("extractCandidates on time %s"%(queryTimestr))
                            datas2= queryDatabase(queryDateTime,dataNames,
                                                  asosTimeRange=asosTimeRange,
                                                  asosResolution=asosResolution,
                                                  queryTimeRange=[0],
                                                  closest=True)
                            dataExtract = extractCandidates(datas2,[dataOut],inKeys,outKeys,coordinateMatches,nhood=[15,15])
                        else:
                            print("no Candidates on time %s"%(queryTimestr))
                            dataExtract = None
                        uc.dumpPickle(dataExtract,outdir+queryTimestr+'.pkl')
                        print("\t%s created."%(outdir+queryTimestr+'.pkl'))
                    else:
                        print("memory too high to extract i=%.0f"%(i))
                else:
                    print("Skipping queryTime: %s exists"%(outdir+queryTimestr+'.pkl'))
                    dataIn = None
                    dataOut = None
                    coordinateMatches = None
                    oldData = None
                    dataExtract = None
            else:
                print("memory too high to read i=%.0f"%(i))
                dataIn = None
                dataOut = None
                coordinateMatches = None
                oldData = None
                dataExtract = None

            if dataExtract is not None and toPlot:
                memError = False
                for j in range(0,len(dataExtract)):
                    mem = psutil.virtual_memory()[2]
                    if mem < 90.0:
                        dTmp = dataExtract[j]
                        lat, lon = dTmp.getCenter(decimals=4)
    
                        dTmp.plot(saveFig=True,
                                  closeFig=None,
                                  saveName=outdir+dTmp.strTime(hours=False)[0]+'_'+lat+'_'+lon+'.png',
                                  clim=np.linspace(0,9,10),
                                  label='AF')
                    else:
                        memError = True
                        print("memory too high to plot, j=%.0f"%(j))
                
    if case == 7:
        #import matplotlib
        #matplotlib.use('agg')
        #import matplotlib.pyplot as plt
        tim = uc.tic()
        #queryDateTime = dt.datetime(year=2017,month=12,day=4,hour=5,minute=53)
        basetime = '2016001'
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'E:/projects/wildfire-research/networkData/20180305/'
        ns = "DatabaseQuery"
        dataNames = ['Elevation','WindX','WindY','Canopy']
        inKeys = ['FireMaskHourlyTA','Elevation','WindX','WindY','Canopy']
        outKeys = ['FireMaskHourlyTA']
            
        candidates = []
        oldData = None
        toPlot = True
        for i in range(0,10):#3000): #8
            mem = psutil.virtual_memory()[2]
            if mem < 90.0:
                queryTime = time.mktime(time.strptime(basetime+'15','%Y%j%H'))+(3600*3)*float(i)
                queryTimestr = time.strftime('%Y%j%H',time.localtime(queryTime))
                queryDateTime = dt.datetime.fromtimestamp(queryTime)
                if glob.glob(outdir+queryTimestr+'.pkl') == []: #queryTimestr[0:7] == '2016177': #glob.glob(outdir+queryTimestr+'.pkl') == []:
                    datas= queryDatabase(queryDateTime,['FireMaskHourlyTA'],
                                         asosTimeRange=asosTimeRange,
                                         asosResolution=asosResolution,
                                         queryTimeRange=[0,3,6,9,12],closest=False)
                    mem = psutil.virtual_memory()[2]
                    if mem < 90.0:
                        if len(datas) > 1:
                            fileName = outdir+datas[0].strTime(hours=False)+'.pkl'
                            if glob.glob(fileName) == []:
                            
                                d = datas[0].data.copy()
                                inds = np.where(d>=7)
                                matches = np.array([inds[0],inds[1]]).T
                                if datas[0] is not None and datas[1] is not None and len(matches) != 0:
                                    print("extractCandidates on time %s"%(queryTimestr))
                                    datas2= queryDatabase(queryDateTime,dataNames,
                                                          asosTimeRange=asosTimeRange,
                                                          asosResolution=asosResolution,
                                                          queryTimeRange=[0],
                                                          closest=True)
                                    #datas2 = remapDatas(datas2,datas[0].latitude.copy(),datas[0].longitude.copy())
                                    dataIn = [datas[0]]
                                    dataIn.extend(datas2)
                                    dataOut = [datas[1]]
                                    dataExtract = extractCandidates(dataIn,dataOut,inKeys,outKeys,matches,nhood=[25,25])
                                    uc.dumpPickle(dataExtract,fileName)
                                    print("\t%s created."%(fileName+'.pkl'))
                                else:
                                    print("no Candidates on time %s"%(queryTimestr))
                                    dataExtract = None
                            else:
                                print("Skipping satelliteTime: %s exists"%(fileName))
                                dataExtract = None
                        else:
                            print("no Candidates on time %s"%(queryTimestr))
                            dataExtract = None
                    else:
                        print("memory too high to extract i=%.0f"%(i))
                else:
                    print("Skipping queryTime: %s exists"%(outdir+queryTimestr+'.pkl'))
                    dataIn = None
                    dataOut = None
                    coordinateMatches = None
                    oldData = None
                    dataExtract = None
            else:
                print("memory too high to read i=%.0f"%(i))
                dataIn = None
                dataOut = None
                coordinateMatches = None
                oldData = None
                dataExtract = None

            if dataExtract is not None and toPlot:
                memError = False
                for j in range(0,len(dataExtract)):
                    mem = psutil.virtual_memory()[2]
                    if mem < 90.0:
                        dTmp = dataExtract[j]
                        lat, lon = dTmp.getCenter(decimals=4)
    
                        dTmp.plot(saveFig=True,
                                  closeFig=None,
                                  saveName=outdir+dTmp.strTime(hours=False)[0]+'_'+lat+'_'+lon+'.png')
                                  #clim=np.linspace(0,9,10),
                                 # label='AF')
                    else:
                        memError = True
                        print("memory too high to plot, j=%.0f"%(j))
        
        
    if case == 8:
        #import matplotlib
        #matplotlib.use('agg')
        #import matplotlib.pyplot as plt
        tim = uc.tic()
        #queryDateTime = dt.datetime(year=2017,month=12,day=4,hour=5,minute=53)
        basetime = '2016001'
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/nhood_25_25_step3/'
        ns = "DatabaseQuery"
        dataNames = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
        inKeys = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
        outKeys = ['FireMaskHourlyTA']
            
        candidates = []
        oldData = None
        toPlot = False
        for i in range(0,3000): #8
            mem = psutil.virtual_memory()[2]
            if mem < 90.0:
                queryTime = time.mktime(time.strptime(basetime+'15','%Y%j%H'))+(3600*3)*float(i)
                queryTimestr = time.strftime('%Y%j%H',time.localtime(queryTime))
                queryDateTime = dt.datetime.fromtimestamp(queryTime)
                if glob.glob(outdir+queryTimestr+'.pkl') == []:
                    datas= queryDatabase(queryDateTime,['FireMaskHourlyTA'],
                                         asosTimeRange=asosTimeRange,
                                         asosResolution=asosResolution,
                                         queryTimeRange=[0,6,12],closest=False)
                    mem = psutil.virtual_memory()[2]
                    if mem < 90.0:
                        fileName = outdir+datas[0].strTime(hours=False)+'.pkl'
                        if glob.glob(fileName) == []:
                            if len(datas) > 1:
                                d = datas[0].data.copy()
                                inds = np.where(d>=7)
                                matches = np.array([inds[0],inds[1]]).T
                                if datas[0] is not None and datas[1] is not None and len(matches) != 0:
                                    print("extractCandidates on time %s"%(queryTimestr))
                                    datas2= queryDatabase(queryDateTime,dataNames,
                                                          asosTimeRange=asosTimeRange,
                                                          asosResolution=asosResolution,
                                                          queryTimeRange=[0],
                                                          closest=True)
                                    print(datas[0])
                                    datas2 = remapDatas(datas2,datas[0].latitude,datas[0].longitude)
                                    dataExtract = extractCandidates(datas2,[datas[1]],inKeys,outKeys,matches,nhood=[25,25])
                                    print(dataExtract)
                                else:
                                    print("no Candidates on time %s"%(queryTimestr))
                                    dataExtract = None
                            else:
                                print("no Candidates on time %s"%(queryTimestr))
                                dataExtract = None
                            #uc.dumpPickle(dataExtract,fileName)
                            print("\t%s created."%(outdir+queryTimestr+'.pkl'))
                        else:
                            print("Skipping satelliteTime: %s exists"%(fileName))
                    else:
                        print("memory too high to extract i=%.0f"%(i))
                else:
                    print("Skipping queryTime: %s exists"%(outdir+queryTimestr+'.pkl'))
                    dataIn = None
                    dataOut = None
                    coordinateMatches = None
                    oldData = None
                    dataExtract = None
            else:
                print("memory too high to read i=%.0f"%(i))
                dataIn = None
                dataOut = None
                coordinateMatches = None
                oldData = None
                dataExtract = None

            if dataExtract is not None and toPlot:
                memError = False
                for j in range(0,len(dataExtract)):
                    mem = psutil.virtual_memory()[2]
                    if mem < 90.0:
                        dTmp = dataExtract[j]
                        lat, lon = dTmp.getCenter(decimals=4)
    
                        dTmp.plot(saveFig=True,
                                  closeFig=None,
                                  saveName=outdir+dTmp.strTime(hours=False)[0]+'_'+lat+'_'+lon+'.png',
                                  clim=np.linspace(0,9,10),
                                  label='AF')
                    else:
                        memError = True
                        print("memory too high to plot, j=%.0f"%(j))