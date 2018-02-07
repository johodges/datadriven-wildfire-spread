# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:44:04 2018

@author: JHodges

This file contains classes and functions to read MODIS Level 2 data and
locate multiple data tiles onto a consistent grid.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os, subprocess
import glob
#import pandas as pd
import scipy.interpolate as scpi
import pyhdf.SD as phdf
import time
import util_common as uc
import datetime as dt

class MODISHourlyMeasurement(object):
    ''' This class is an intermediate used to pass data more easily
    
    Fields:
        time: dateTime associated with the measurement
        data: data assoicated with the measurement
        name: name of the measurement
        latitude: latitude grid of the measurement
        longitude: longitude grid of the measurement
        file: filename from which the measurement was loaded
    '''
        
    __slots__ = ['time','latitude','longitude','data','name','file']
    
    def __init__(self,dtime,lat,lon,data,name,file):
        self.time = dtime
        self.latitude = lat
        self.longitude = lon
        self.data = data
        self.name = name
        self.file = file

def generateEnvironment(data_dir="/C/Users/JHodges/My\ Documents/wildfire-research/tools/MRTSwath_download_Win/MRTSwath/data",
                        home_dir="/C/Users/JHodges/My\ Documents/wildfire-research/tools/MRTSwath_download_Win/MRTSwath"):
    ''' This function will generate the environment necessary to use the
    MODIS remapping tool. Unused in the custom remapping.
    '''
    my_env = os.environ.copy()
    my_env['MRTSWATH_DATA_DIR'] = data_dir
    my_env['MRTSWATH_HOME'] = home_dir
    return my_env

def findFilePaths(inputFile,outputFile,geoFile,
                  pdir="C:/Users/JHodges/Documents/wildfire-research/tools/MRTSwath_download_Win/MRTSwath/bin",
                  indir="E:/WildfireResearch/data/terra_hourly_activefires",
                  outdir="E:/WildfireResearch/data/terra_hourly_activefires_rp",
                  geodir="E:/WildfireResearch/data/terra_geolocation"):
    ''' This function will generate system paths necessary to use the
    MODIS remapping tool. Unused in the custom remapping.
    '''
    programPath = pdir+'/swath2grid.exe'
    inputPath = indir+'/'+inputFile
    outputPath = outdir+'/'+outputFile
    geoPath = geodir+'/'+geoFile
    
    return programPath, inputPath, outputPath, geoPath

def buildCommand(programPath,inputPath,outputPath,geoPath):
    ''' This function will generate the command necessary to use the
    MODIS remapping tool. Unused in the custom remapping.
    '''
    myCommand = [programPath,
                 '-if='+inputPath,
                 '-of='+outputPath,
                 '-gf='+geoPath,
                 '-off=HDF_FMT',
                 '-sds=fire_mask',
                 '-k=NN',
                 '-oproj=SNSOID',
                 '-osp=20',
                 '-opsz=1000']
    return myCommand

def remapUsingTool(indir="E:/WildfireResearch/data/terra_hourly_activefires",
                   outdir="E:/WildfireResearch/data/terra_hourly_activefires_rp",
                   geodir="E:/WildfireResearch/data/terra_geolocation"):
    ''' This function will remap data from the input directory to the Level 3
    MODIS grid using the MODIS remapping tool.
    '''
    my_env = generateEnvironment()
    
    af_files = glob.glob(indir+'/*.hdf')
    geo_files = glob.glob(geodir+'/*.hdf')
    res_files = glob.glob(outdir+'/*.hdf')
    
    matched_files = []
    for i in range(0,len(af_files)):
        dtstr = af_files[i].split('\\MOD14.A')[1][0:12]
        matched_file = [s for s in geo_files if dtstr in s]
        if len(matched_file) > 0:
            outputFile = 'rp_'+af_files[i].split('\\')[1]
            res_match = [s for s in res_files if outputFile in s]
            if len(res_match) > 0:
                pass
            else:
                inputFile = af_files[i].split('\\')[1]
                geoFile = matched_file[0].split('\\')[1]
                matched_files.append([inputFile,geoFile,outputFile])
    for i in range(0,len(matched_files)):
        inputFile = matched_files[i][0]
        geoFile = matched_files[i][1]
        outputFile = matched_files[i][2]
        programPath, inputPath, outputPath, geoPath = findFilePaths(inputFile,outputFile,geoFile)
        myCommand = buildCommand(programPath,inputPath,outputPath,geoPath)
        
        print("Processing %s"%(myCommand[1]))
        subprocess.call(myCommand, env=my_env) #, shell=True
        print("Percent complete: %.6f"%(i/len(matched_files)))

def removeOutsideAndReshape(lat,lon,data,
                            lat_lmt = [31,44],
                            lon_lmt = [-126,-112]):
    ''' This function will reduce the matrix size of the dataset using user
    defined latitude and longitude limits.
    '''
    lat_rs = np.reshape(lat,(lat.shape[0]*lat.shape[1]))
    lon_rs = np.reshape(lon,(lon.shape[0]*lon.shape[1]))
    data_rs = np.reshape(data,(data.shape[0]*data.shape[1]))
    
    inds = np.where((lat_rs<np.max(lat_lmt)) & (lat_rs>np.min(lat_lmt)))
    lat_rs = lat_rs[inds].copy()
    lon_rs = lon_rs[inds].copy()
    data_rs = data_rs[inds].copy()
    inds = np.where((lon_rs<np.max(lon_lmt)) & (lon_rs>np.min(lon_lmt)))
    lat_rs = lat_rs[inds].copy()
    lon_rs = lon_rs[inds].copy()
    data_rs = data_rs[inds].copy()
    
    return lat_rs, lon_rs, data_rs


def gridAndResample(data,
                    lat_lmt = [31,44],
                    lon_lmt = [-126,-112],
                    pxPerDegree = 120,
                    ds=1,
                    method='nearest'):
    ''' This function will resample the raw Level 2 data from the swath grid
    to the custom Level 3 grid.
    '''
    
    lat = data.latitude
    lon = data.longitude
    
    pts = np.zeros((len(lat),2))
    pts[:,0] = lat
    pts[:,1] = lon    
    
    lat_lnsp = np.linspace(np.min(lat_lmt),np.max(lat_lmt),
                           (np.max(lat_lmt)-np.min(lat_lmt)+1)*pxPerDegree)
    lon_lnsp = np.linspace(np.min(lon_lmt),np.max(lon_lmt),
                           (np.max(lon_lmt)-np.min(lon_lmt)+1)*pxPerDegree)
    lon_grid, lat_grid = np.meshgrid(lon_lnsp,lat_lnsp)
    lon_lnsp2 = np.reshape(lon_grid,(lon_grid.shape[0]*lon_grid.shape[1]))
    lat_lnsp2 = np.reshape(lat_grid,(lat_grid.shape[0]*lat_grid.shape[1]))
    newpts = np.zeros((len(lat_lnsp2),2))
    newpts[:,0] = lat_lnsp2
    newpts[:,1] = lon_lnsp2
    if len(data.data) > 0:
        data_lnsp = scpi.griddata(pts[0::ds],data.data[0::ds],newpts,method=method)
        data_grid = np.reshape(data_lnsp,(lat_grid.shape[0],lat_grid.shape[1]))
        
        data.latitude = lat_lnsp.copy()
        data.longitude = lon_lnsp.copy()
        data.data = data_grid.copy()
    else:
        return None
    
    return data
    
def generateCustomHdf(data,outdir,sdsname,
                      sdsdescription='Active fires/thermal anomalies mask',
                      sdsunits='none'):
    ''' This function will generate a custom Level 3 hdf file
    '''
    hdfFile = phdf.SD(outdir+data.file,phdf.SDC.WRITE|phdf.SDC.CREATE) # Assign a few attributes at the file level
    hdfFile.author = 'JENSEN HUGHES'
    hdfFile.productionDate = time.strftime('%Y%j.%H%M',time.gmtime(time.time()))
    hdfFile.minTimeStamp = str('%.4f'%(np.min(data.time)))
    hdfFile.maxTimeStamp = str('%.4f'%(np.max(data.time)))
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

def matchFilesToGeo(indir="E:/WildfireResearch/data/terra_hourly_activefires/",
                    geodir = "E:/WildfireResearch/data/terra_geolocation/",
                    outdir = "E:/WildfireResearch/data/terra_hourly_activefires_jh/",
                    splitStr = '\\MOD14.A'):
    ''' This function finds all data measurements where the geolocation data
    of the satellite is also available.
    '''
    files = glob.glob(indir+'/*.hdf')
    geo_files = glob.glob(geodir+'/*.hdf')
    res_files = glob.glob(outdir+'/*.hdf')
    
    inputFiles = []
    geoFiles = []
    dates = []
    for i in range(0,len(files)):
        dtstr = files[i].split(splitStr)[1][0:12]
        matched_file = [s for s in geo_files if dtstr in s]
        res_match = [s for s in res_files if dtstr[0:7] in s]
        if len(matched_file) > 0 and len(res_match) == 0:
            inputFile = files[i].split('\\')[1]
            geoFile = matched_file[0].split('\\')[1]
            inputFiles.append(inputFile)
            geoFiles.append(geoFile)
            #matched_files.append([inputFile,geoFile,dtstr[0:7]])
            dates.append(dtstr[0:7])
        elif len(matched_file) > 0:
            pass
            #print("Found:", res_match)
        else:
            pass
    return inputFiles, geoFiles, list(set(dates))

def splitDayAndNight(lats,lons,datas,names,times,
                     timezone=0,
                     dayLowThresh=1.0,
                     dayUpThresh=13.0,
                     fileName=None):
    ''' This function splits the data into day and night sections based on
    lower and upper threshold times in hours of UTC.
    '''
    localHour = np.array([time.localtime(s+timezone*3600).tm_hour for s in times])
    
    inds = np.where((localHour<dayUpThresh) & (localHour>dayLowThresh))
    times = np.array(times)
    lats = np.array(lats)
    lons = np.array(lons)
    datas = np.array(datas)
    names = np.array(names)
    
    if fileName is not None:
        dfileName = fileName.split('.xxxx.')[0]+'.dddd.'+fileName.split('.xxxx.')[1]
        nfileName = fileName.split('.xxxx.')[0]+'.nnnn.'+fileName.split('.xxxx.')[1]
        
    day = MODISHourlyMeasurement(
            times[inds].copy(),lats[inds].copy(),lons[inds].copy(),
            datas[inds].copy(),names.copy(),dfileName)
    inds = np.where((localHour>=dayUpThresh) | (localHour<=dayLowThresh))
    night = MODISHourlyMeasurement(
            times[inds].copy(),lats[inds].copy(),lons[inds].copy(),
            datas[inds].copy(),names.copy(),nfileName)    
    return day, night

def remapUsingCustom(indir="E:/WildfireResearch/data/terra_hourly_activefires/",
                     geodir = "E:/WildfireResearch/data/terra_geolocation/",
                     outdir = "E:/WildfireResearch/data/terra_hourly_activefires_jh/",
                     sdsname_in='fire mask',
                     sdsname_out='FireMask',
                     splitStr = 'MOD14.A',
                     dayLowThresh=1.0,
                     dayUpThresh=13.0):
    ''' This function will remap Level 2 MODIS data in indir to a custom Level
    3 grid.
    '''

    inputFiles, geoFiles, dates = matchFilesToGeo(indir=indir,geodir=geodir,outdir=outdir,splitStr=splitStr)
    
    for i in range(1,len(dates)):
        files = [s for s in inputFiles if 'A'+dates[i] in s]
        
        datas = []
        lats = []
        lons = []
        times = []
        names = []
        
        for j in range(0,len(files)):
            file = files[j]
            dtstr = file[7:19]
            s = "%s%s%s%s"%(dtstr[0:4],dtstr[4:7],dtstr[8:10],dtstr[10:12])
            dateTime = time.mktime(time.strptime(s,'%Y%j%H%M'))
            gfile = [s for s in geoFiles if dtstr in s][0]
            print(file,dtstr,gfile)
            data = phdf.SD(indir+file,phdf.SDC.READ).select(sdsname_in).get()
            lat = phdf.SD(geodir+gfile,phdf.SDC.READ).select('Latitude').get()
            lon = phdf.SD(geodir+gfile,phdf.SDC.READ).select('Longitude').get()
            lat_rs,lon_rs,data_rs = removeOutsideAndReshape(lat,lon,data)
            datas.extend(data_rs)
            lats.extend(lat_rs)
            lons.extend(lon_rs)
            times.extend(np.zeros((len(data_rs),))+dateTime)
            names.append(file)
            
        name = files[0].split(splitStr)[1].split('.')
        fileName = splitStr.split('.')[0]+'JH.'+splitStr.split('.')[1]+'.'+name[0]+'.xxxx.'+name[2]+time.strftime('%Y%j.%H%M',time.gmtime(time.time()))+'.hdf'
        
        day, night = splitDayAndNight(lats,lons,datas,names,times,
                                      dayLowThresh=dayLowThresh,
                                      dayUpThresh=dayUpThresh,
                                      fileName=fileName,
                                      timezone=0)
        
        day = gridAndResample(day)
        night = gridAndResample(night)
        
        if day is not None and night is not None:
            generateCustomHdf(day,outdir,'FireMask',
                              sdsdescription='Active fires/thermal anomalies mask',
                              sdsunits='none')
            generateCustomHdf(night,outdir,'FireMask',
                              sdsdescription='Active fires/thermal anomalies mask',
                              sdsunits='none')
        else:
            if day is None:
                print("Unable to find daytime measurements.")
            if night is None:
                print("Unable to find nighttime measurements.")
        print("Percent Complete: %.4f"%((i+1)/len(dates)))
    return day, night

def loadCustomHdf(file,sdsname):
    ''' This function will load a custom Level 3 product from an hdf file.
    '''
    f = phdf.SD(file,phdf.SDC.READ)
    data = f.select(sdsname).get()
    latitudeL = float(f.attributes()['latitudeL'])
    latitudeR = float(f.attributes()['latitudeR'])
    longitudeL = float(f.attributes()['longitudeL'])
    longitudeR = float(f.attributes()['longitudeR'])
    
    lat_lnsp = np.linspace(latitudeL,latitudeR,data.shape[0])
    lon_lnsp = np.linspace(longitudeL,longitudeR,data.shape[1])
    
    lon_grid, lat_grid = np.meshgrid(lon_lnsp,lat_lnsp)
    timeStamp = (float(f.attributes()['minTimeStamp'])+float(f.attributes()['maxTimeStamp']))/2
    
    return lat_grid, lon_grid, data, timeStamp

def getTimeCustomHdf(file,timezone=0,dayLowThresh=1.0,dayUpThresh=13.0):
    ''' This function will get the time stampe from a custom Level 3
    product.
    '''
    f = phdf.SD(file,phdf.SDC.READ)
    try:
        startTime = float(f.attributes()['minTimeStamp']) +timezone*3600
        endTime = float(f.attributes()['maxTimeStamp']) +timezone*3600
    except KeyError:
        return None, None
    
    avgTime_t = (startTime+endTime)/2
    avgTime = time.localtime(avgTime_t)
    
    avgTimeToZero = avgTime.tm_sec+60*(avgTime.tm_min+60*avgTime.tm_hour)
    if (avgTime.tm_hour > dayLowThresh) & (avgTime.tm_hour < dayUpThresh):
        startTime = avgTime_t-avgTimeToZero+dayLowThresh*3600
        endTime = avgTime_t-avgTimeToZero+dayUpThresh*3600
    else:
        if avgTime.tm_hour >= dayUpThresh:
            startTime = avgTime_t-avgTimeToZero+dayUpThresh*3600
            endTime = avgTime_t-avgTimeToZero+24*3600+dayLowThresh*3600
        else:
            startTime = avgTime_t-avgTimeToZero-24*3600+dayUpThresh*3600
            endTime = avgTime_t-avgTimeToZero+dayLowThresh*3600
    
    atime = time.strftime('%Y%j%H',avgTime)
    stime = time.strftime('%Y%j%H',time.localtime(startTime))
    etime = time.strftime('%Y%j%H',time.localtime(endTime))
    
    startTime = startTime-timezone*3600
    endTime = endTime-timezone*3600
    
    #print("sTime:\t%s\naTime:\t%s\neTime:\t%s\n"%(stime,atime,etime))

    return startTime, endTime

"""
def queryTimeCustomHdf2(indir,qDT):
    queryTime = time.mktime(qDT.timetuple())+qDT.microsecond / 1E6
    files = glob.glob(indir+'/*.hdf')
    times = []
    #files = [files[0],files[1],files[2],files[3],files[4],files[5]]
    for file in files:
        startTime, endTime = getTimeCustomHdf(file)
        if startTime is not None and endTime is not None:
            times.append([startTime, endTime])
    times = np.array(times)
    
    return times, queryTime

def queryTimeCustomHdfbkup(qDT,
                       datadir="E:/WildfireResearch/data/terra_hourly_activefires_jh/",
                       sdsname='FireMask',
                       timezone=0):
    files, times = loadTimeFileCustomHdf(datadir)
    if files is None:
        _, outfile = updateTimeFileCustomHdf(indir,timezone=timezone)
        files, times = loadTimeFileCustomHdf(outfile)
    #files, times = loadTimeFileCustomHdf(outdir)
    queryTime = time.mktime(qDT.timetuple())+qDT.microsecond / 1E6
    ind = np.where(((times-queryTime)[:,0] < 0) & ((times-queryTime)[:,1] > 0))[0][0]
    file = files[ind]
    lat, lon, data = loadCustomHdf(file,sdsname)
    return lat, lon, data
"""
    
def queryTimeCustomHdf(qDT,
                       datadirs="E:/WildfireResearch/data/terra_hourly_activefires_jh/",
                       sdsname='FireMask',
                       timezone=0):
    ''' This function will query the custom Level 3 hdf database for a
    specific time.
    '''
    queryTime = time.mktime(qDT.timetuple())+qDT.microsecond / 1E6
    
    if type(datadirs) is str:
        datadirs = [datadirs]
        dataType = 'str'
    elif type(datadirs) is list:
        dataType = 'list'
    
    lat = []
    lon = []
    data = []
    timeStamp = []
    for datadir in datadirs:
        files, times = loadTimeFileCustomHdf(datadir)
        if files is None:
            _, outfile = updateTimeFileCustomHdf(indir,timezone=timezone)
            files, times = loadTimeFileCustomHdf(outfile)
        ind = np.where(((times-queryTime)[:,0] < 0) & ((times-queryTime)[:,1] > 0))[0]
        if len(ind) > 0:
            file = files[ind[0]]
            lat2, lon2, data2, timeStamp2 = loadCustomHdf(file,sdsname)
            lat.append(lat2)
            lon.append(lon2)
            data.append(data2)
            timeStamp.append(timeStamp2)
            
    if dataType == 'str':
        lat = lat[0]
        lon = lon[0]
        data = data[0]
        timeStamp = timeStamp[0]
        
    return lat, lon, data, timeStamp

def updateTimeFileCustomHdf(indir,timezone=0):
    ''' This function will generate a pickle file with all the filenames and
    corresponding timestamps of custom hdf files in a directory. Loading all
    the files to get the times takes much longer than just loading a pickle
    file.
    '''
    files = glob.glob(indir+'/*.hdf')
    times = []
    files2 = []
    for file in files:
        startTime, endTime = getTimeCustomHdf(file,timezone=0)
        if startTime is not None and endTime is not None:
            times.append([startTime, endTime])
            files2.append(file)
    times = np.array(times)
    outfile = indir+'fileTimeList.pkl'
    uc.dumpPickle([files2,times],outfile)
    return times, outfile

def loadTimeFileCustomHdf(indir):
    ''' This function will load a pickle file with all the filenames and
    corresponding timestamps of custom hdf files in a directory. If the
    file does not exist, the updateTimeFileCustomHdf function will be run
    to generate the pickle file.
    '''
    if indir[-4:] == '.pkl':
        files = [indir]
    elif indir[-1] == '/':
        files = glob.glob(indir+'*.pkl')
    else:
        files = glob.glob(indir+'/*.pkl')
    data = uc.readPickle(files[0]) if len(files) > 0 else None
    if data is not None:
        files = data[0]
        times = data[1]
    else:
        files = None
        times = None
    return files, times

if __name__ == "__main__":
    
    satellite = 'terra'
    case = 0
    
    if satellite == 'terra':
        indir="E:/WildfireResearch/data/terra_hourly_activefires/"
        geodir = "E:/WildfireResearch/data/terra_geolocation/"
        #outdir = "E:/WildfireResearch/data/terra_hourly_activefires_jh/"
        outdir = "E:/WildfireResearch/data/terra_hourly_activefires_jh/"
        dayLowThresh = 1.0
        dayUpThresh = 13.0
        splitStr = 'MOD14.A'
    elif satellite == 'aqua':
        indir="E:/WildfireResearch/data/aqua_hourly_activefires/"
        geodir = "E:/WildfireResearch/data/aqua_geolocation/"
        #outdir = "E:/WildfireResearch/data/terra_hourly_activefires_jh/"
        outdir = "E:/WildfireResearch/data/aqua_hourly_activefires_jh/"
        dayLowThresh = 4.0
        dayUpThresh = 16.0
        splitStr = 'MYD14.A'
    
    
    if case == 0:
        day, night = remapUsingCustom(indir=indir,geodir=geodir,outdir=outdir,
                                      sdsname_in='fire mask',
                                      sdsname_out='FireMask',
                                      dayLowThresh=dayLowThresh,
                                      dayUpThresh=dayUpThresh,
                                      splitStr=splitStr)
        
        plt.figure(1)
        fig1 = uc.plotContourWithStates(day.latitude,day.longitude,day.data,
                                        clim=np.linspace(0,9,10),label='AF',
                                        saveFig=False,saveName='noname')
    
        plt.figure(2)
        fig2 = uc.plotContourWithStates(night.latitude,night.longitude,night.data,
                                        clim=np.linspace(0,9,10),label='AF',
                                        saveFig=False,saveName='noname')
    elif case == 1:
        lat, lon, data = loadCustomHdf(outdir+'MOD14JH.A2017164.dddd.0062018032.2019.hdf','FireMask')
        plt.figure(1)
        fig1 = uc.plotContourWithStates(lat,lon,data,
                                        clim=np.linspace(0,9,10),label='AF',
                                        saveFig=False,saveName='noname')
        lat, lon, data = loadCustomHdf(outdir+'MOD14JH.A2017164.nnnn.0062018032.2019.hdf','FireMask')
        plt.figure(2)
        fig1 = uc.plotContourWithStates(lat,lon,data,
                                        clim=np.linspace(0,9,10),label='AF',
                                        saveFig=False,saveName='noname')
    elif case == 2:
        times, outfile = updateTimeFileCustomHdf(outdir,timezone=0)
        
    elif case == 3:
        # 6 am, 6 pm UTC time -> 10pm, 10am Cali time
        day = 4
        hour = 9
        for i in range(0,30):
            queryDateTime = dt.datetime(year=2017,month=12,day=day,hour=hour,minute=0)+dt.timedelta(hours=8)
            #files, times = loadTimeFileCustomHdf(outdir)
            lat, lon, data =  queryTimeCustomHdf(queryDateTime,datadir=outdir,sdsname='FireMask')
            plt.figure(1)
            
            if len(str(day)) == 1:
                dn = '0'+str(day)
            else:
                dn = str(day)
            if len(str(hour)) == 1:
                hn = '0'+str(hour)
            else:
                hn = str(hour)
                
            
            fig1 = uc.plotContourWithStates(lat,lon,data,
                                            clim=np.linspace(0,9,10),label='AF',
                                            saveFig=True,saveName=outdir+'exampled'+dn+'h'+hn+'.png')
            if hour == 9:
                hour = 21
            elif hour == 21:
                hour = 9
                day = day + 1
        
    elif case == 4:
        uc.makeGIF(outdir,outdir+'out.mp4')
        
    elif case == 5:
        indir = 'C:/Users/JHodges/Documents/wildfire-research/output/AF_images/terra2016'
        uc.makeGIF(indir,indir+'/out.mp4')
        
    elif case == 6:
        queryDateTime = dt.datetime(year=2017,month=12,day=18,hour=23,minute=0)
        #queryTime = time.mktime(queryDateTime.timetuple())+queryDateTime.microsecond / 1E6
        file = outdir+'MOD14JH.A2017164.nnnn.0062018032.2019.hdf'
        times, queryTime =  queryTimeCustomHdf2(outdir,queryDateTime)
        
        inds = np.where(((times-queryTime)[:,0] < 0) & ((times-queryTime)[:,1] > 0))

        #print(time.gmtime(startTime),time.gmtime(endTime))
        print(time.localtime(times[inds[0]][0][0]),time.localtime(times[inds[0]][0][1]))
        
        #startTime = dt.datetime.fromtimestamp(time.mktime(startTime))
        #endTime = dt.datetime.fromtimestamp(time.mktime(endTime))
        #print(inds)
    elif case == 7:
        files = glob.glob(outdir+'/*.hdf')
        timezone = 0
        dayLowThresh = 1 # Terra California
        dayUpThresh = 13 # Terra California
        for i in range(2,4):#len(files)):
            file = files[i]
            f = phdf.SD(file,phdf.SDC.READ)
            startTime = float(f.attributes()['minTimeStamp']) +timezone*3600
            endTime = float(f.attributes()['maxTimeStamp']) +timezone*3600
            
            avgTime_t = (startTime+endTime)/2
            avgTime = time.localtime(avgTime_t)
            
            avgTimeToZero = avgTime.tm_sec+60*(avgTime.tm_min+60*avgTime.tm_hour)
            if (avgTime.tm_hour > dayLowThresh) & (avgTime.tm_hour < dayUpThresh):
                startTime = avgTime_t-avgTimeToZero+dayLowThresh*3600
                endTime = avgTime_t-avgTimeToZero+dayUpThresh*3600
            else:
                if avgTime.tm_hour >= dayUpThresh:
                    startTime = avgTime_t-avgTimeToZero+dayUpThresh*3600
                    endTime = avgTime_t-avgTimeToZero+24*3600+dayLowThresh*3600
                else:
                    startTime = avgTime_t-avgTimeToZero-24*3600+dayUpThresh*3600
                    endTime = avgTime_t-avgTimeToZero+dayLowThresh*3600
            
            atime = time.strftime('%Y%j%H',avgTime)
            stime = time.strftime('%Y%j%H',time.localtime(startTime))
            etime = time.strftime('%Y%j%H',time.localtime(endTime))
            
            startTime = startTime-timezone*3600
            endTime = endTime-timezone*3600
            
            minTime = time.strftime('%Y%j%H%M%S',time.localtime(float(f.attributes()['minTimeStamp'])))
            maxTime = time.strftime('%Y%j%H%M%S',time.localtime(float(f.attributes()['maxTimeStamp'])))
            print("startTime:\t%s\nendTime:\t%s\n"%(stime,etime))
            
            print("minTimeStamp:\t%s\t\nmaxTimeStamp:\t%s\n"%(minTime,maxTime))
            
            
            #print("sTime:\t%s\naTime:\t%s\neTime:\t%s\n"%(stime,atime,etime))
            