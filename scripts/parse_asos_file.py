# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:05:51 2017

@author: JHodges
"""

import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import glob
import pickle
import sys
import util_common as uc

class ASOSMeasurementList(object):
    __slots__ = ['dateTime','temperatureC','relativeH','directionDeg','speedMps','gustMps']
    
    def __init__(self):
        self.dateTime = []
        self.temperatureC = []
        self.relativeH = []
        self.directionDeg = []
        self.speedMps = []
        self.gustMps = []
    
    def addTime(self,dateTime,temperatureC,relativeH,directionDeg,speedMps,gustMps):
        self.dateTime.append(dateTime)
        self.temperatureC.append(temperatureC)
        self.relativeH.append(relativeH)
        self.directionDeg.append(directionDeg)
        self.speedMps.append(speedMps)
        self.gustMps.append(gustMps)

class ASOSStation(object):
    __slots__ = ['latitude','longitude','name','call',
                 'dateTime','temperatureC','relativeH','directionDeg','speedMps','gustMps',
                 #'ncdcid','wban','coopid','aname','country','state','county','elevation','utc','stntype',
                 'dateTime']
    
    def __init__(self,info):

        self.call = info[3]
        self.name = info[4]
        self.latitude = info[9]
        self.longitude = info[10]
        
        #self.ncdcid = info[0]
        #self.wban = info[1]
        #self.coopid = info[2]
        #self.aname = info[5]
        #self.country = info[6]
        #self.state = info[7]
        #self.county = info[8]
        #self.elevation = info[11]
        #self.utc = info[12]
        #self.stntype = info[13]
        
        self.temperatureC = [] # List of temperature measurements in deg C
        self.relativeH = [] # List of relative humidity measurements
        self.directionDeg = [] # List of wind direction in degrees
        self.speedMps = [] # List of wind speed measurements in m/s
        self.gustMps = [] # List of wind gust speed measuremetns in m/s
        self.dateTime = [] # List of time stamps
        
    def computeMemory(self):
        mem = 0
        slots = self.__slots__
        for key in slots:
            if type(key) == list:
                mem = mem + sys.getsizeof(getattr(self,key))/1024**2
            else:
                mem = mem+sys.getsizeof(getattr(self,key))/1024**2
        return mem

    def __str__(self):
        string = "%s ASOS Station\n"%(self.name)
        string = string + "\tTotal measurements:\t%.0f\n"%(len(self.dateTime))
        string = string + "\tEarliest dateTime:\t%s\n"%(min(self.dateTime))
        string = string + "\tLatest dateTime:\t%s\n"%(max(self.dateTime))
        return string
    
    def __repr__(self):
        return self.__str__()

    def addTime(self,data):
        self.temperatureC.extend(data.temperatureC)
        self.relativeH.extend(data.relativeH)
        self.directionDeg.extend(data.directionDeg)
        self.speedMps.extend(data.speedMps)
        self.gustMps.extend(data.gustMps)
        self.dateTime.extend(data.dateTime)

    def timeAverage(self,timeRange):
        dateTimeNp = []
        for i in self.dateTime:
            dateTimeNp.append(np.datetime64(i))
        dateTimeNp = np.array(dateTimeNp)
        deltaTimeNp = np.array(dateTimeNp-dateTimeNp[0],dtype=np.float32)
        deltaTimeNp = deltaTimeNp/(10**6*3600)
        temperatureC = np.array(self.temperatureC,dtype=np.float32)
        relativeH = np.array(self.relativeH,dtype=np.float32)
        directionDeg = np.array(self.directionDeg,dtype=np.float32)
        speedMps = np.array(self.speedMps,dtype=np.float32)
        gustMps = np.array(self.gustMps,dtype=np.float32)
        dataNp = np.array([deltaTimeNp,temperatureC,relativeH,directionDeg,speedMps,gustMps],dtype=np.float32).T
        
        dataNp = nanInterp(dataNp.copy())
    
        deltat = 2*(timeRange.days*24+(0+(timeRange.seconds+timeRange.microseconds/10**6)/60)/60)
        maxt = np.floor(np.max(dataNp[:,0]))
        mint = np.min(dataNp[:,0])
        t = np.linspace(mint,maxt,int((maxt-mint)/deltat+1))
        dataNpI = np.zeros((len(t),dataNp.shape[1]),dtype=np.float32)
        dataNpI[:,0] = t
        dateTime = []
        basetime = min(self.dateTime)
        for i in range(1,dataNp.shape[1]):
            dataNpI[:,i] = np.interp(t,dataNp[:,0],dataNp[:,i])
        for i in range(0,dataNpI.shape[0]):
            dateTime.append(dt.timedelta(hours=int(dataNpI[i,0]))+basetime)
        self.dateTime = dateTime
        self.temperatureC = dataNpI[:,1]
        self.relativeH = dataNpI[:,2]
        self.directionDeg = dataNpI[:,3]
        self.speedMps = dataNpI[:,4]
        self.gustMps = dataNpI[:,5]        
        
        return dataNp
        
    def findTime(self,queryDateTime):
        bestMatchValue = min(self.dateTime, key=lambda d: abs(d-queryDateTime))
        bestMatchIndex = self.dateTime.index(bestMatchValue)
        return bestMatchIndex

    def extractTimeAverage(self,queryDateTime,timeRange):
        def list2avg(dataL,inds):
            dataNp = np.array(dataL)
            dataNp[dataNp == None] = np.nan
            dataNp = np.array(dataNp,dtype=np.float32)
            if not np.all(np.isnan(dataNp[inds[0]:inds[1]])):
                data = np.nanmean(dataNp[inds[0]:inds[1]])
                return data
            else:
                return np.nan
            
        bestLowValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime-timeRange)))
        bestHighValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime+timeRange)))
        bestLowIndex = self.dateTime.index(bestLowValue)
        bestHighIndex = self.dateTime.index(bestHighValue)
        bestMatchValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime)))
        bestMatchIndex = self.dateTime.index(bestMatchValue)
        
        temperatureC = list2avg(self.temperatureC,[bestLowIndex,bestHighIndex+1])
        relativeH = list2avg(self.relativeH,[bestLowIndex,bestHighIndex+1])
        directionDeg = list2avg(self.directionDeg,[bestLowIndex,bestHighIndex+1])
        speedMps = list2avg(self.speedMps,[bestLowIndex,bestHighIndex+1])
        gustMps = list2avg(self.gustMps,[bestLowIndex,bestHighIndex+1])
        
        return np.array([queryDateTime,temperatureC,relativeH,directionDeg,speedMps,gustMps])
    
    def sortMeasurements(self):
        self.temperatureC = [x for _, x in sorted(zip(self.dateTime,self.temperatureC), key=lambda pair: pair[0])]
        self.relativeH = [x for _, x in sorted(zip(self.dateTime,self.relativeH), key=lambda pair: pair[0])]
        self.directionDeg = [x for _, x in sorted(zip(self.dateTime,self.directionDeg), key=lambda pair: pair[0])]
        self.speedMps = [x for _, x in sorted(zip(self.dateTime,self.speedMps), key=lambda pair: pair[0])]
        self.gustMps = [x for _, x in sorted(zip(self.dateTime,self.gustMps), key=lambda pair: pair[0])]
        self.dateTime.sort()




def convertKnots(speedKnot):
    if speedKnot is not None:
        speedMps = speedKnot*0.514444
    return speedMps

def findGeoLimits(stations):
    minLatitude = 360
    maxLatitude = -360
    minLongitude = 360
    maxLongitude = -360
    for key, value in stations.items():
        if value.latitude < minLatitude:
            minLatitude = value.latitude
        if value.latitude > maxLatitude:
            maxLatitude = value.latitude
        if value.longitude < minLongitude:
            minLongitude = value.longitude
        if value.longitude > maxLongitude:
            maxLongitude = value.longitude
    return [minLatitude,maxLatitude,minLongitude,maxLongitude]

def stationsSortMeasurements(stations):
    for key, value in stations.items():
        value.sortMeasurements()
    return stations

def convertVector(speedMps,directionDeg):
    if directionDeg == -1:
        speedX = speedMps*2**0.5
        speedY = speedMps*2**0.5
    elif directionDeg is None:
        pass
        print("Wind direction was not set.")
    elif speedMps is None:
        pass
        print("Wind speed was not set.")
    else:
        try:
            speedX = speedMps*np.sin(directionDeg/180*math.pi)
            speedY = speedMps*np.cos(directionDeg/180*math.pi)
        except:
            assert False, "Unknown wind vector: %s Mps %s Deg" % (str(speedMps),str(directionDeg))
    return speedX, speedY

def getStationsMeasurements(stations,queryDateTime,timeRange):
    measurements = []
    for key, value in stations.items():
        data = value.extractTimeAverage(queryDateTime,timeRange)
        directionDeg = data[3]
        speedMps = data[4]
        speedX, speedY = convertVector(speedMps,directionDeg)
        measurements.append([value.latitude,value.longitude,speedX,speedY])
    measurements = np.array(measurements)
    return measurements

def defineStations(filename):
    def str2Int(s):
        s = s.strip()
        return int(s) if s else np.nan
    with open(filename) as f:
        content = f.readlines()
    stations = dict()
    for line in content:
        if "NCDCID" not in line and '-------- ----- ------ ----' not in line:
            NCDCID = str2Int(line[0:8].strip())
            WBAN = str2Int(line[9:14].strip())
            COOPID = str2Int(line[15:21].strip())
            CALL = line[22:26].strip()
            NAME = line[27:57].strip()
            ANAME = line[58:88].strip()
            COUNTRY = line[89:109].strip()
            ST = line[110:112].strip()
            COUNTY = line[113:143].strip()
            LAT = float(line[144:153].strip())
            LON = float(line[154:164].strip())
            ELEV = float(line[165:171].strip())
            UTC = float(line[172:177].strip())
            STNTYPE = line[178:-1].strip()
            
            if CALL not in stations and ST == 'CA':
                stations[CALL] = ASOSStation([NCDCID,WBAN,COOPID,CALL,NAME,ANAME,COUNTRY,ST,COUNTY,LAT,LON,ELEV,UTC,STNTYPE])
            #if CALL in stations:
            #    stations[CALL].addTime(splitLine)
            #else:
            #    stations[CALL] = ASOSStation(splitLine)
            #    stations[CALL].addTime(splitLine)
    return stations
    

def parseMETARline(line,debug=False):
    line_split = line.split(' ')
    start_index = line_split.index('5-MIN') if '5-MIN' in line_split else -1
    if start_index == -1:
        print("Unable to find 5-MIN string to start parsing:") if debug else -1
        print(line) if debug else -1
        return None
    end_index = line_split.index('RMK') if 'RMK' in line_split else -1
    line_split = line_split[start_index+1:end_index]
    
    filecall = line_split[0]
    if line_split[1][0:-1].isdigit() and len(line_split[1]) == 7:
        pass
        day = int(line_split[1][0:2])
        hour = int(line_split[1][2:4])
        minute = int(line_split[1][4:6])
    else:
        return None
    #data.auto = False
    sm = 0
    #data.sky_condition = []
    #data.weather_condition = []
    temperatureC = None
    relativeH = None
    directionDeg = None
    speedMps = None
    gustMps = None
    #data.temperature_dew = 'M'

    line_split = [x for x in line_split if x]
    for i in range(2,len(line_split)):
        if line_split[i] == 'AUTO':
            #data.auto = True
            pass
        elif 'KT' in line_split[i]:
            filewind = line_split[i].split('KT')[0]
            if 'G' in filewind:
                if filewind.split('G')[1].isdigit():
                    gustMps = convertKnots(float(filewind.split('G')[1]))
                else:
                    print("Failed to parse wind gust:") if debug else -1
                    print(line) if debug else -1
                filewind = filewind.split('G')[0]
            if 'VRB' in filewind:
                filewind = filewind.split('VRB')[1]
                directionDeg = -1
            else:
                try:
                    directionDeg = float(filewind[0:3])
                except:
                    print("Error parsing direction.") if debug else -1
                    print(line) if debug else -1
            try:
                speedMps = convertKnots(float(filewind[-2:]))
            except:
                print("Error parsing windspeed.") if debug else -1
                print(line) if debug else -1
        elif 'V' in line_split[i] and len(line_split[i]) == 7 and 'KT' in line_split[i-1]:
            #data.directionDegVar = [float(line_split[i][0:3]),float(line_split[i][4:])]
            pass
            
        elif 'SM' in line_split[i]:
            linesm = line_split[i].split('SM')[0]
            try:
                if linesm[0] == 'M':
                    linesm = linesm[1:]
            except:
                print(line_split[i]) if debug else -1
            if '/' in linesm:
                if linesm.split('/')[0].isdigit() and linesm.split('/')[1].isdigit():
                    sm += float(linesm.split('/')[0])/float(linesm.split('/')[1])
                    print("Error parsing visibility:") if debug else -1
                    print(line) if debug else -1
            else:
                try:
                    sm += float(linesm)
                except:
                    print("Error parsing visibility:") if debug else -1
                    print(line) if debug else -1

        elif line_split[i][0] == 'R' and len(line_split[i]) >= 10:
            if line_split[i][-2:] == 'FT':
                #data.rvr = line_split[i]
                pass
        elif ('BKN' in line_split[i] or 'CLR' in line_split[i]
                or 'FEW' in line_split[i] or 'SCT' in line_split[i]
                or 'OVC' in line_split[i]):
            #data.sky_condition.append([line_split[i][0:3],line_split[i][3:]])
            pass
        elif ('RA' in line_split[i] or 'SN' in line_split[i] 
                or 'UP' in line_split[i] or 'FG' in line_split[i]
                or 'FZFG' in line_split[i] or 'BR' in line_split[i]
                or 'HZ' in line_split[i] or 'SQ' in line_split[i]
                or 'FC' in line_split[i] or 'TS' in line_split[i]
                or 'GR' in line_split[i] or 'GS' in line_split[i]
                or 'FZRA' in line_split[i] or 'VA' in line_split[i]):
            #data.weather_condition.append(line_split[i])
            pass
        elif line_split[i][0] == 'A' and len(line_split[i]) == 5:
            try:
                altimeter = float(line_split[i][1:])
            except:
                print("Error parsing altitude.") if debug else -1
                print(line) if debug else -1
        elif '/' in line_split[i] and len(line_split[i]) == 5: #data.temperatureC == None:
            linetemp = line_split[i].split('/')
            temperature_air_sign = 1
            temperature_dew_sign = 1
            if 'M' in linetemp[0]:
                temperature_air_sign = -1
                linetemp[0] = linetemp[0].split('M')[1]
            if 'M' in linetemp[1]:
                temperature_dew_sign = -1
                linetemp[1] = linetemp[1].split('M')[1]
            if linetemp[0].isdigit():
                temperatureC = float(linetemp[0])*temperature_air_sign
            if linetemp[1].isdigit():
                #data.temperature_dew = float(linetemp[1])*temperature_dew_sign
                temperatureDew = float(linetemp[1])*temperature_dew_sign
                pass
            if linetemp[0].isdigit() and linetemp[1].isdigit():
                #data.relativeH = 100-5*(data.temperatureC-data.temperature_dew)
                relativeH = 100-5*(temperatureC-temperatureDew)
        else:
            if i < len(line_split)-1:
                if 'SM' in line_split[i+1] and '/' in line_split[i+1] and line_split[i].isdigit():
                    try:
                        sm += float(line_split[i])
                    except:
                        print(line) if debug else -1
                        print(line_split) if debug else -1
                else:
                    pass
                    #print('Unknown argument %s at %.0f.' % (line_split[i],0))
            else:
                pass
                #print('Unknown argument %s at %.0f.' % (line_split[i],1))
    if sm == 0:
        #data.sm = None
        pass
    else:
        #data.sm = sm
        pass
    
    return [temperatureC,relativeH,directionDeg,speedMps,gustMps], [day,hour,minute]







def parseMETARfile(file):
    dateTimes = []
    datas = ASOSMeasurementList()
    with open(file) as f:
        old_day = 0
        content = f.readlines()
        if content is not None:
            #print(len(content))
            i = 0
            for line in content:
                data = None
                try:
                    data, times = parseMETARline(line)
                except:
                    print("Failed to parse the METAR line in file %s line %.0f."%(file,i))
                    pass
                day = times[0]
                hour = times[1]
                minute = times[2]
                if data is not None:
                    year = int(file[-10:-6])
                    month = int(file[-6:-4])
                    if day < old_day:
                        month = month + 1
                    if month > 12:
                        month = 1
                        year = year + 1
                    old_day = day
                    dateTime = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)
                    datas.addTime(dateTime,data[0],data[1],data[2],data[3],data[4])
                i = i+1
    return datas, dateTimes

def readStationsFromText(filename='../data-test/asos-stations.txt',
                         datadir='E:/WildfireResearch/data/asos-fivemin/6401-2016/',
                         timeRange = dt.timedelta(days=0,hours=0,minutes=30)):
    stations = defineStations(filename)
    empty_stations = []
    totalMem = 0
    for key in stations.keys():
        call = stations[key].call
        files = glob.glob(datadir+'*'+call+'*')
        if len(files) != 0:# and key == 'WVI':
            for file in files:
                data, dateTime = parseMETARfile(file)
                stations[key].addTime(data)
                #stations[key].addTime(data)
                stations[key].dateTime.extend(dateTime)
            localMem = stations[key].computeMemory()
            _ = stations[key].timeAverage(timeRange)
            reducMem = stations[key].computeMemory()
            totalMem = totalMem+reducMem
            print("Station %s\n\tRaw memory:\t%.04f MB\n\tReduced Memory:\t%0.4f MB\n\tTotal Memory:\t%.04f MB"%(key,localMem,reducMem,totalMem))
    
        else:
            empty_stations.append(key)  
            print("%s was empty."%(key))
    for key in empty_stations:
        stations.pop(key,None)
        
    for key in stations:
        stations[key].sortMeasurements()
    print("Finished %s, total Memory: %0.4f MB"%(key,computeStationsMemory(stations,printSummary=False)))
    return stations

def dumpPickleStations(stations,filename='../data-test/asos-stations.pkl'):
    with open(filename,'wb') as f:
        pickle.dump(stations,f)
        
def readPickleStations(filename='../data-test/asos-stations.pkl'):
    with open(filename,'rb') as f:
        stations = pickle.load(f)
    return stations

def computeStationsMemory(stations,printSummary=True):
    mem = 0
    for station in stations:
        mem2 = stations[station].computeMemory()
        print("Station %s Memory %.4f"%(station,mem2))
        mem = mem+mem2
    print("Total Memory: %0.4f MB"%(mem)) if printSummary else -1
    return mem

def nanInterp(data):
    x = data[:,0]
    for i in range(1,len(data[0,:])):
        y = data[:,i]
        nans = np.where(~np.isfinite(y))[0]
        y[nans] = np.nan
        data[nans,i] = np.interp(x[nans],x[~nans],y[~nans])
    return data

def buildCoordinateGrid(stations,resolution=111):
    ''' This function will build a latitude and longitude grid using the
    limits of the station file at the resolution specified in pixels per
    degree of latitude and longitude.
    
    NOTE: 1 degree is approximately 69 miles, or 111 km
    NOTE: Modis resolution is approximately 1km
    NOTE: Thus, 111 pixels per degree will match modis resolution
    '''
    geoLimits = findGeoLimits(stations)
    latGrid = np.linspace(geoLimits[0]-1,geoLimits[1]+1,int((geoLimits[1]-geoLimits[0]+2)*resolution+1))
    lonGrid = np.linspace(geoLimits[2]-1,geoLimits[3]+1,int((geoLimits[3]-geoLimits[2]+2)*resolution+1))
    return latGrid, lonGrid

def getSpeedContours(measurements,lat,lon):
    ''' This function will build a contour map of measurements using point
    measurements at known latitude and longitudes.
    '''
    speedXcontour = griddata(measurements[:,0],measurements[:,1],measurements[:,2],lat,lon,interp='linear').T
    speedYcontour = griddata(measurements[:,0],measurements[:,1],measurements[:,3],lat,lon,interp='linear').T
    return speedXcontour, speedYcontour

def queryWindSpeed(queryDateTime,
                   filename='../data-test/asos-stations.pkl',
                   resolution=111,
                   timeRange = dt.timedelta(days=0,hours=0,minutes=30)):
    stations = readPickleStations(filename=filename)
    lat, lon = buildCoordinateGrid(stations,resolution=resolution)
    measurements = getStationsMeasurements(stations,queryDateTime,timeRange)
    speedX, speedY = getSpeedContours(measurements,lat,lon)
    lon, lat = np.meshgrid(lon,lat)
    return lat, lon, speedX, speedY

if __name__ == "__main__":
    ''' Example cases:
        case 0: Load raw data, dump raw data, plot data at query time
        case 1: Load pickled data, compute memory requirements
        case 2: Load pickled data, plot data at query time
    '''
    case = 2
    filename = '../data-test/asos-stations.txt'
    datadir='E:/WildfireResearch/data/asos-fivemin/6401-2016/'
    resolution = 111 # pixels per degree
    queryDateTime = dt.datetime(year=2016,month=6,day=17,hour=5,minute=53)
    timeRange = dt.timedelta(days=0,hours=0,minutes=30)
    
    if case == 0:
        stations = readStationsFromText(filename=filename,datadir=datadir)
        dumpPickleStations(stations,filename=filename[0:-4]+'.pkl')
        #computeStationsMemory(stations)
        lat, lon = buildCoordinateGrid(stations,resolution=resolution)
        measurements = getStationsMeasurements(stations,queryDateTime,timeRange)
        speedX, speedY = getSpeedContours(measurements,lat,lon)
        
        speedX_fig = uc.plotContourWithStates(lat,lon,speedX,label='m/s',
                                              states=None,
                                              clim=None,xlim=None,ylim=None)
        speedY_fig = uc.plotContourWithStates(lat,lon,speedY,label='m/s',
                                              states=None,
                                              clim=None,xlim=None,ylim=None)
    if case == 1:
        stations = readPickleStations(filename=filename[0:-4]+'.pkl')
        computeStationsMemory(stations)
    elif case == 2:
        stations = readPickleStations(filename=filename[0:-4]+'.pkl')
        #computeStationsMemory(stations)
        lat, lon = buildCoordinateGrid(stations,resolution=resolution)
        measurements = getStationsMeasurements(stations,queryDateTime,timeRange)
        speedX, speedY = getSpeedContours(measurements,lat,lon)
        
        speedX_fig = uc.plotContourWithStates(lat,lon,speedX,label='m/s',
                                              states=None,
                                              clim=None,xlim=None,ylim=None)
        speedY_fig = uc.plotContourWithStates(lat,lon,speedY,label='m/s',
                                              states=None,
                                              clim=None,xlim=None,ylim=None)