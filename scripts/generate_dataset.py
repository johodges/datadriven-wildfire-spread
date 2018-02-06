# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:47:32 2018

@author: JHodges
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import util_common as uc
import parse_elevation as pe
import parse_modis_file as pm
import parse_asos_file as pa
import remapSwathData as rsd

from parse_asos_file import ASOSMeasurementList, ASOSStation
#import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import scipy.interpolate as scpi
import psutil
import gc
import time
import sys

class GriddedMeasurement(object):
    __slots__ = ['dateTime','latitude','longitude','data','remapped','label','dataName','clim']
    
    def __init__(self,dateTime,lat,lon,data,label):
        self.dateTime = dateTime
        self.latitude = lat
        self.longitude = lon
        self.data = data
        self.label = label
        self.clim = None
        
    def __str__(self):
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
        return self.__str__()

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

    def computeMemory(self):
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
        timeFloat = time.mktime(dataStart.dateTime.timetuple())+self.dateTime.microsecond/1E6
        timeTuple = time.localtime(timeFloat)
        if hours:
            return time.strftime('%Y%j%H%M%S',timeTuple)
        else:
            return time.strftime('%Y%j%H',timeTuple)

class GriddedMeasurementPair(object):
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
        if data.dataName in self.inKeys:
            d = data.data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            setattr(self,'In_'+data.dataName,d)
            
    def addEndData(self,data,bounds):
        if data.dataName in self.outKeys:
            d = data.data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            setattr(self,'Out_'+data.dataName,d)
    
    def countData(self):
        inCounter = 0
        outCounter = 0
        for key in self.__dict__.keys():
            if "In_" in key:
                inCounter = inCounter+1
            if "Out_" in key:
                outCounter = outCounter+1
        return inCounter, outCounter
    
    def getDataNames(self):
        inData = []
        outData = []
        for key in self.__dict__.keys():
            if "In_" in key:
                inData.append(key)
            if "Out_" in key:
                outData.append(key)
        return inData, outData
    
    def __str__(self):
        inTime, outTime = self.strTime()
        string = "Gridded Measurement Pair\n"
        string = string + "\tinTime:\t\t%s (yyyydddhhmmssss)\n"%(inTime)
        string = string + "\toutTime:\t%s (yyyydddhhmmssss)\n"%(outTime)
        string = string + "\tgrid:\t\t%.0f,%.0f (Latitude,Longitude)\n"%(self.latitude.shape[0],self.longitude.shape[1])
        string = string + "\tmemory:\t\t%.4f MB"%(self.computeMemory())
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
        return self.__str__()
    
    def computeMemory(self):
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
        inTimeFloat = time.mktime(self.inTime.timetuple())+self.inTime.microsecond/1E6
        inTimeTuple = time.localtime(inTimeFloat)
        outTimeFloat = time.mktime(self.outTime.timetuple())+self.outTime.microsecond/1E6
        outTimeTuple = time.localtime(outTimeFloat)
        if hours:
            return time.strftime('%Y%j%H%M%S',inTimeTuple), time.strftime('%Y%j%H%M%S',outTimeTuple)
        else:
            return time.strftime('%Y%j%H',inTimeTuple), time.strftime('%Y%j%H',outTimeTuple)

    def getCenter(self,decimals=4):
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
        
        inNum, outNum = self.countData()
        inNames, outNames = self.getDataNames()
        inTime, outTime = self.strTime()
        
        #print(inNum,outNum)
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
        
        #xticks = np.round(np.linspace(np.min(self.longitude),np.max(self.longitude),5),2)
        #yticks = np.round(np.linspace(np.min(self.latitude),np.max(self.latitude),5),2)
        names = inNames+outNames
        for i in range(0,len(names)):
            key = names[i]
            currentPlot = currentPlot+1
            #print(rowPlots,colPlots,currentPlot)
            ax = fig.add_subplot(rowPlots,colPlots,currentPlot)
            ax.tick_params(axis='both',labelsize=fntsize)
            plt.xticks(xticks)
            plt.yticks(yticks)
            plt.xlabel('Longitude',fontsize=fntsize)
            plt.ylabel('Latitude',fontsize=fntsize)
            plt.title(key,fontsize=fntsize)
            #print(key)
            #print('FireMask' in key)
            if 'FireMask' in key:
                clim = np.linspace(0,9,10)
                label = 'AF'
            elif 'Elevation' in key:
                clim = np.linspace(-1000,5000,7)
                label = 'Elev [m]'
            elif 'WindX' in key:
                clim = np.linspace(-6,6,13)
                label = 'u [m/s]'
            elif 'WindY' in key:
                clim = np.linspace(-6,6,13)
                label = 'v [m/s]'
            elif 'VegetationIndex' in key:
                clim = np.linspace(-4000,10000,8)
                label = 'NVDIx1000'
            else:
                clim = None
                label = ''
            img = ax.contourf(self.longitude,self.latitude,getattr(self,key),levels=clim,cmap=cmap)
            img_cb = plt.colorbar(img,ax=ax,label=label)
            #img.set_clim(clim[0],clim[-1])
            #plt.clim(0, 9.0)
            
            #img_cb.set_clim(clim[0],clim[-1])
            
#            if clim is None:
#                pass
#                #img = ax.contourf(self.longitude,self.latitude,getattr(self,key),cmap=cmap)
#                #img_cb = plt.colorbar(img,ax=ax,label=label)
#            else:
#                img = ax.contourf(self.longitude,self.latitude,getattr(self,key),clim=clim,cmap=cmap)
#                img_cb = plt.colorbar(img,ax=ax,label=label,ticks=clim)
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

    def plot_old(self,
             saveFig=False,
             closeFig=None,
             saveName='',
             clim=None,
             cmap='jet',
             label='None'):
        
        inTime, outTime = self.strTime()
        
        if saveFig:
            fntsize = 80
            lnwidth = 10
            fig = plt.figure(figsize=(48,20),tight_layout=True)
            if closeFig is None:
                closeFig = True
        else:
            fig = plt.figure(figsize=(12,5))#,tight_layout=True)
            fntsize = 20
            lnwidth = 2
            if closeFig is None:
                closeFig = False
        
        ax1 = fig.add_subplot(1,2,1)
        plt.xlabel('Longitude',fontsize=fntsize)
        plt.ylabel('Latitude',fontsize=fntsize)
        plt.title(inTime,fontsize=fntsize)
        
        ax2 = fig.add_subplot(1,2,2)
        plt.xlabel('Longitude',fontsize=fntsize)
        plt.title(outTime,fontsize=fntsize)
        
        if clim is None:
            img1 = ax1.contourf(self.longitude,self.latitude,self.inData1,cmap=cmap)
            img2 = ax2.contourf(self.longitude,self.latitude,self.outData,cmap=cmap)
        else:
            img1 = ax1.contourf(self.longitude,self.latitude,self.inData1,clim,cmap=cmap)
            img2 = ax2.contourf(self.longitude,self.latitude,self.outData,clim,cmap=cmap)
        ax1.tick_params(axis='both',labelsize=fntsize)
        ax2.tick_params(axis='both',labelsize=fntsize)

        # Add colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.1, 0.025, 0.8])
        
        if clim is None:
            img_cb = fig.colorbar(img1,cax=cbar_ax,label=label)
        else:
            img_cb = fig.colorbar(img1,cax=cbar_ax,label=label,ticks=clim)
        
        img_cb.set_label(label=label,fontsize=fntsize)
        cbar_ax.tick_params(axis='both',labelsize=fntsize)

        if saveFig:
            ax1.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
            ax2.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
            for ln in ax1.lines:
                ln.set_linewidth(lnwidth)
            fig.savefig(saveName)
            
        if closeFig:
            plt.clf()
            plt.close(fig)
        

def extractCandidates(dataStart,dataEnd,inKeys,outKeys,matches,nhood=[50,50]):
    
    datas = []    
    for i in range(0,len(matches)):
        rowLow = matches[i][0]-nhood[0]
        rowUp = matches[i][0]+nhood[0]
        colLow = matches[i][1]-nhood[1]
        colUp = matches[i][1]+nhood[1]
        bounds=[rowLow,rowUp,colLow,colUp]    
        data = GriddedMeasurementPair(dataStart[0],dataEnd[0],inKeys,outKeys,bounds=bounds)
    
        for d in dataStart:
            data.addStartData(d,bounds)
                
        for d in dataEnd:
            data.addEndData(d,bounds)
                
        datas.append(data)

    return datas

def extractCandidates2(dataStart,dataEnd,matches,nhood=[50,50]):
    bounds = dataStart.data.shape
    
    datas = []
    for i in range(0,matches.shape[0]):
        rowLow = matches[i][0]-nhood[0]
        rowUp = matches[i][0]+nhood[0]
        colLow = matches[i][1]-nhood[1]
        colUp = matches[i][1]+nhood[1]
        if (rowLow > 0) and (rowUp < bounds[0]) and (colLow > 0) and (colUp < bounds[1]):
            data = GriddedMeasurementPair(
                    dataStart,dataEnd,inKeys,outKeys,
                    bounds=[rowLow,rowUp,colLow,colUp])
            datas.append(data)
    return datas

def compareGriddedMeasurement(data1,data2):
    if data1 is None or data2 is None:
        return False
    if (str(data1.dateTime) == str(data2.dateTime)) and (str(data1.dataName) == str(data2.dataName)):
        return True
    else:
        return False

def loadDataByName(queryDateTime,dataName):
    modOffset = 0.0
    if dataName == 'FireMaskHourlyTA':
        lat, lon, dataRaw, timeStamp =  rsd.queryTimeCustomHdf(
                queryDateTime,
                datadirs=["E:/WildfireResearch/data/terra_hourly_activefires_jh/",
                          "E:/WildfireResearch/data/aqua_hourly_activefires_jh/"],
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
                datadirs="E:/WildfireResearch/data/terra_hourly_activefires_jh/",
                sdsname='FireMask')
        dataDateTime = dt.datetime.fromtimestamp(timeStamp)
        data = GriddedMeasurement(dataDateTime,lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'FireMaskHourlyA':
        lat, lon, dataRaw, timeStamp =  rsd.queryTimeCustomHdf(
                queryDateTime,
                datadirs="E:/WildfireResearch/data/aqua_hourly_activefires_jh/",
                sdsname='FireMask')
        data = GriddedMeasurement(dataDateTime,lat,lon,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'FireMaskT':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="E:/WildfireResearch/data/terra_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime.date(),lat,lon+modOffset,dataRaw,'AF')
        data.clim = np.linspace(0,9,10)
        data.dataName = dataName
    elif dataName == 'FireMaskA':
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=False,
                                              datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
                                              sdsname='FireMask')
        data = GriddedMeasurement(queryDateTime.date(),lat,lon+modOffset,dataRaw,'AF')
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
                                              datadir="E:/WildfireResearch/data/terra_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'VI')
        data.dataName = dataName
    elif dataName == 'VegetationIndexA':
        queryDateTime = dt.datetime(year=queryDateTime.year,month=queryDateTime.month,day=queryDateTime.day)
        #Find vegetation index at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/aqua_vegetation/",
                                              sdsname='1 km 16 days NDVI')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'VI')
        data.dataName = dataName
    elif dataName == 'BurnedArea':
        queryDateTime = dt.datetime(year=queryDateTime.year,month=queryDateTime.month,day=queryDateTime.day)
        #Find burned area at queryDateTime
        lat,lon,dataRaw = pm.findQuerySdsData(queryDateTime,composite=True,
                                              datadir="E:/WildfireResearch/data/modis_burnedarea/",
                                              sdsname='burndate')
        data = GriddedMeasurement(queryDateTime,lat,lon+modOffset,dataRaw,'BA')
        data.dataName = dataName
    if type(data) is not list:
        data = [data]
    return data

def queryDatabase(queryDateTime,dataNames,
                  queryTimeRange = [0],
                  asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0),
                  asosResolution = 111,
                  closest=False):
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

def getCandidates(queryDateTime,dataNames,
                  queryTimeRange=None,
                  oldData=None,
                  candidateThresh=100):
    if queryTimeRange is None:
        queryTimeRange = np.linspace(0,12,int((12*1*1)/6+1))
    
    datas = queryDatabase(queryDateTime,dataNames,
                          asosTimeRange=asosTimeRange,
                          asosResolution=asosResolution,
                          queryTimeRange=queryTimeRange)
    
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
        basetime = '2016167'
        asosTimeRange = dt.timedelta(days=0,hours=12,minutes=0)
        asosResolution = 111
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/GoodCandidateComparison/'
        ns = "DatabaseQuery"
        dataNames = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
        inKeys = ['FireMaskHourlyTA','Elevation','WindX','WindY','VegetationIndexA']
        outKeys = ['FireMaskHourlyTA']
            
        candidates = []
        oldData = None
        for i in range(0,200): #8
            mem = psutil.virtual_memory()[2]
            if mem < 90.0:
                queryTime = time.mktime(time.strptime(basetime+'15','%Y%j%H'))+(3600*6)*float(i)
                queryDateTime = dt.datetime.fromtimestamp(queryTime)
                
                dataIn, dataOut, coordinateMatches = getCandidates(queryDateTime,['FireMaskHourlyTA'],oldData=oldData)
                oldData = dataIn
            else:
                print("memory too high, i=%.0f"%(i))
            if dataIn is not None and dataOut is not None and coordinateMatches is not None:
                datas2= queryDatabase(queryDateTime,dataNames,
                                      asosTimeRange=asosTimeRange,
                                      asosResolution=asosResolution,
                                      queryTimeRange=[0],
                                      closest=True)
                #dataExtract = extractCandidates2(dataIn,dataOut,coordinateMatches,nhood=[25,25])
                dataExtract = extractCandidates(datas2,[dataOut],inKeys,outKeys,coordinateMatches,nhood=[25,25])
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
                        print("memory too high, j=%.0f"%(j))
                oldData = dataIn
                
            
            #coordsMatch = coords[np.array(np.squeeze(match_pts)[:,0],dtype=np.int),:]
            #candidates.append([time.strftime('%Y%j%H',time.localtime(queryTime)),candidateCheck,datas[0],datas[1],coordsMatch])
        
        
        #datasExtract = []
        #for i in range(0,len(candidates)):
        #    dataStart = candidates[i][2]
        #    dataEnd = candidates[i][3]
        #    timeStart = dataStart.strTime(hours=False)
        #    timeEnd = dataEnd.strTime(hours=False)
        #    matches = candidates[i][4]
        #   dataExtract = extractCandidates(dataStart,dataEnd,matches,nhood=[25,25])
        #    datasExtract.append(dataExtract)
            
        
#        for i in range(0,len(datasExtract)):
#            for j in range(0,5):#len(datas[i])):
#                mem = psutil.virtual_memory()[2]
#                if mem < 90.0:
#                    data = datasExtract[i][j]
#                    lat, lon = data.getCenter(decimals=4)
#                    
#                    data.plot(saveFig=True,
#                              closeFig=None,
#                              saveName=outdir+data.strTime(hours=False)[0]+'_'+lat+'_'+lon+'.png',
#                              clim=np.linspace(0,9,10),
#                              label='AF')
#                else:
#                    print("memory too high.")
        

               
            
            
            
            
            
            
            
        #plt.xlim(xlim)
        #plt.ylim(ylim)
                    

        """




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
        """