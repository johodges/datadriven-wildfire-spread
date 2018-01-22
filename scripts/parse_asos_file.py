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
import pandas as pd

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

"""
class ASOSMeasurement(object):
    __slots__ = ['dateTime','temperatureC','relativeH','directionDeg','speedMps','gustMps']
                 #'day','hour','minute','year','month']
                 #'auto','sky_condition','weather_condition','temperature_dew','sm','directionDegVar','rvr',
                 
    def __init__(self,dateTime=None):
        self.dateTime = dateTime
        self.temperatureC = None
        self.relativeH = None
        self.directionDeg = None
        self.speedMps = None
        
        self.gustMps = None
        #self.day = None
        #self.hour = None
        #self.minute = None
        #self.year = None
        #self.month = None
    
    def computeMemory(self):
        slots = self.__slots__
        #slots = ['dateTime','temperatureC','relativeH','directionDeg','speedKnot','speedMps','gustKnot','gustMps',
        #     'speedX','speedY','day','hour','minute','year','month',
             #'auto','sky_condition','weather_condition','temperature_dew','sm','directionDegVar','rvr',
        #     'speedMph','gustMph']
        mem = 0
        for key in slots:
            mem = mem+sys.getsizeof(getattr(self,key))/1024**2
        return mem
    
    def convertKnots(self,speedKnot):
        if speedKnot is not None:
            speedMps = speedKnot*0.514444
        return speedMps
        
    def convertVector(self):
        if self.directionDeg == -1:
            speedX = self.speedMps*2**0.5
            speedY = self.speedMps*2**0.5
        elif self.directionDeg is None:
            pass
            #print("Wind direction was not set.")
        elif self.speedMps is None:
            pass
            #print("Wind speed was not set.")
        else:
            try:
                speedX = self.speedMps*np.sin(self.directionDeg/180*math.pi)
                speedY = self.speedMps*np.cos(self.directionDeg/180*math.pi)
            except:
                assert False, "Unknown wind vector: %s Mps %s Deg" % (str(self.speedMps),str(self.directionDeg))
        return speedX, speedY
    
    def __str__(self):
        try:
            string = "dateTime:\t\t%s\n"%self.dateTime
        except:
            pass
        try:
            string += "temperatureC:\t%.2f degC\n"%self.temperatureC
        except:
            pass
        try:
            string += "relativeH:\t\t%.2f \%\n"%self.relativeH
        except:
            pass
        try:
            string += "WindSpeed:\t\t\t%.2f m/s\n"%self.speedMps
        except:
            pass
        try:
            string += "WindDir:\t\t\t%.2f deg\n"%self.directionDeg
        except:
            pass
        return string
    
    def __repr__(self):
        return self.__str__()
"""

class ASOSStation(object):
    __slots__ = ['latitude','longitude','name','call',
                 'dateTime','temperatureC','relativeH','directionDeg','speedMps','gustMps',
                 #'ncdcid','wban','coopid','aname','country','state','county','elevation','utc','stntype',
                 'dateTime']
    
    def __init__(self,info):
        #self.ncdcid = info[0]
        #self.wban = info[1]
        #self.coopid = info[2]
        self.call = info[3]
        self.name = info[4]
        #self.aname = info[5]
        #self.country = info[6]
        #self.state = info[7]
        #self.county = info[8]
        self.latitude = info[9]
        self.longitude = info[10]
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
        #slots = ['latitude','longitude','name','call',
                 #'ncdcid','wban','coopid','aname','country','state','county','elevation','utc','stntype',
        #         'data','dateTime']
        for key in slots:
            if type(key) == list:
                mem = mem + sys.getsizeof(getattr(self,key))/1024**2
            else:
                mem = mem+sys.getsizeof(getattr(self,key))/1024**2
        return mem

    def convertKnots(self,speedKnot):
        if speedKnot is not None:
            speedMps = speedKnot*0.514444
        return speedMps
    
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
    
    def timeAverage2(self,timeRange):
        minDateTime = min(self.dateTime)
        maxDateTime = max(self.dateTime)
        
        currentDateTime = dt.datetime(year=minDateTime.year,month=minDateTime.month,day=minDateTime.day,hour=minDateTime.hour)
        data = []
        i = 0
        while currentDateTime < maxDateTime:
            data.append(self.extractTimeAverage(currentDateTime,timeRange))
            currentDateTime = currentDateTime+timeRange*2
            if i % 100 == 0:
                print('currentDateTime: %s\nMaxDateTime: %s'%(currentDateTime,maxDateTime))
            i = i+1
        data = np.array(data)
        print(data.shape)
        self.dateTime = data[:,0]
        self.temperatureC = data[:,1]
        self.relativeH = data[:,2]
        self.directionDeg = data[:,3]
        self.speedMps = data[:,4]
        self.gustMps = data[:,5]

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
        #dataPd = pd.DataFrame([dateTimeNp,temperatureC,relativeH,directionDeg,speedMps,gustMps]).T
        dataNp = np.array([deltaTimeNp,temperatureC,relativeH,directionDeg,speedMps,gustMps],dtype=np.float32).T
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

    """
    def extractTimeAverage2(self,queryDateTime,timeRange):
        data = ASOSMeasurement(queryDateTime)
        bestMatchValue = min(self.dateTime, key=lambda d: abs(d-queryDateTime))
        bestLowValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime-timeRange)))
        bestHighValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime+timeRange)))
        bestLowIndex = self.dateTime.index(bestLowValue)
        bestHighIndex = self.dateTime.index(bestHighValue)
        bestMatchIndex = self.dateTime.index(bestMatchValue)
        
        temperatureC = []
        relativeH = []
        directionDeg = []
        speedMps = []
        gustMps = []
        
        for i in range(bestLowIndex,bestHighIndex+1):
            temperatureC.append(self.data[i].temperatureC) if self.data[i].temperatureC is not None else temperatureC.append(np.nan)
            relativeH.append(self.data[i].relativeH) if self.data[i].relativeH is not None else relativeH.append(np.nan)
            directionDeg.append(self.data[i].directionDeg) if (self.data[i].directionDeg is not None and self.data[i].directionDeg != -1) else directionDeg.append(np.nan)
            speedMps.append(self.data[i].speedMps) if self.data[i].speedMps is not None else speedMps.append(np.nan)
            gustMps.append(self.data[i].gustMps) if self.data[i].gustMps is not None else gustMps.append(np.nan)
            
        temperatureC = np.nanmean(temperatureC) if (temperatureC and all(v is not None for v in temperatureC)) else None
        relativeH = np.nanmean(relativeH) if (relativeH and all(v is not None for v in relativeH)) else None
        directionDeg = np.nanmean(directionDeg) if (directionDeg and all(v is not None for v in directionDeg)) else None
        speedMps = np.nanmean(speedMps) if (speedMps and all(v is not None for v in speedMps)) else None
        gustMps = np.nanmean(gustMps) if (gustMps and all(v is not None for v in gustMps)) else None
        
        data.temperatureC = temperatureC
        data.relativeH = relativeH
        data.directionDeg = directionDeg
        data.speedMps = speedMps
        
        #data.convertVector()

        return data
    """
    
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





def parseAsosFile(filename):

    with open(filename) as f:
        content = f.readlines()
    stations = dict()
    for line in content:
        if "#DEBUG" not in line and 'station,valid' not in line:
            splitLine = line.split(',')
            station = splitLine[0]
            if splitLine[0] in stations:
                stations[station].addTime(splitLine)
            else:
                stations[station] = ASOSStation(splitLine)
                stations[station].addTime(splitLine)
    
    #writeAsosFile('./data/asos_reduced.txt',content)
    stations = stationsSortMeasurements(stations)
    return stations

def writeAsosFile(filename,contents,headerLines=6,lines2write=2000):
    with open(filename, 'w') as f:
        for i in range(0,headerLines):
            f.write("%s"%contents[i])
        for i in range(headerLines,lines2write+headerLines):
            f.write("%s"%contents[i])

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

def stationsGetMeasurements(stations,queryDateTime,timeRange):
    measurements = []
    for key, value in stations.items():
        data = value.timeAverage(queryDateTime,timeRange)
        measurements.append([value.latitude,value.longitude,data.speedX,data.speedY])
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

def readStationsFromText(filename='../data-test/asos-stations.txt',datadir='E:/WildfireResearch/data/asos-fivemin/6401-2016/'):
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
            totalMem = totalMem+localMem
            print("Finished %s, Memory: %0.4f MB Total Memory: %.04f MB"%(key,stations[key].computeMemory(),totalMem))
        else:
            empty_stations.append(key)  
            print("%s was empty."%(key))
    for key in empty_stations:
        stations.pop(key,None)
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

if __name__ == "__main__":

    if True: #'stations' not in locals():
        #datadir='../data-test/asos/asos-fivemin/6401-2016/'
        datadir='../data-test/asos/asos-fivemin/smallset/'
        #datadir='../data-test/asos/asos-fivemin/breaks/'
        stations = readStationsFromText(datadir=datadir)
        
        #stations = readStationsFromText()
    
    for key in stations:
        stations[key].sortMeasurements()
    
    geoLimits = findGeoLimits(stations)
    
    # 1 degree is approximately 69 miles, or 111 km
    # Modis resolution is approximately 1km, so we need 111 pixels per degree
    # to match modis resolution
    resolution = 111 # pixels per degree
    
    latGrid = np.linspace(geoLimits[0]-1,geoLimits[1]+1,int((geoLimits[1]-geoLimits[0]+2)*resolution+1))
    lonGrid = np.linspace(geoLimits[2]-1,geoLimits[3]+1,int((geoLimits[3]-geoLimits[2]+2)*resolution+1))
    
    queryDateTime = dt.datetime(year=2007,month=6,day=15,hour=5,minute=53)
    timeRange = dt.timedelta(days=0,hours=0,minutes=30)
    dataNp = stations['AAT'].timeAverage(timeRange)
    deltat = 1
    maxt = np.floor(np.max(dataNp[:,0]))
    mint = np.min(dataNp[:,0])
    t = np.linspace(mint,maxt,(maxt-mint)/deltat+1)
    dataNpI = np.interp(t,dataNp[:,0],dataNp[:,3])
    
    """
    measurements = stationsGetMeasurements(stations,queryDateTime,timeRange)
    

    speedXcontour = griddata(measurements[:,0],measurements[:,1],measurements[:,2],latGrid,lonGrid,interp='linear')
    speedYcontour = griddata(measurements[:,0],measurements[:,1],measurements[:,3],latGrid,lonGrid,interp='linear')
    
    plt.figure(figsize=(12,8))
    CS = plt.contourf(lonGrid,latGrid,speedXcontour.T)
    plt.colorbar()
    plt.xlim([-126,-114])
    plt.ylim([32,42])
    
    plt.figure(figsize=(12,8))
    CS = plt.contourf(lonGrid,latGrid,speedYcontour.T)
    plt.colorbar()
    plt.xlim([-126,-114])
    plt.ylim([32,42])
    """

    
    

"""
remove_comments = []
for i in range(0,len(content)):
    if "*" not in content[i]:
        remove_comments.append(content[i])
        if "IBWR " in content[i]:
            print(i,len(remove_comments))
        if "IFCCHPOWER " in content[i]:
            print(i,len(remove_comments))
print("There are a total of %.0f lines in the file."%(len(content)))
print("There are a total of %.0f parameter lines."%(len(remove_comments)))

with open('./PEACH5_stripped.par','w') as f:
    for i in range(0,len(remove_comments)):
        f.write("%s"%remove_comments[i])
"""