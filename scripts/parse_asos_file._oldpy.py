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


class ASOSMeasurement(object):
    def __init__(self, dateTime=None,temperatureC=None,relativeH=None,
                 directionDeg=None,speedKnot=None,speedMps=None,
                 gustKnot=None,gustMph=None):
        self.dateTime = dateTime
        self.temperatureC = temperatureC
        self.relativeH = relativeH
        self.directionDeg = directionDeg
        self.speedKnot = speedKnot
        self.speedMps = speedMps
        self.gustKnot = gustKnot
        self.gustMph = gustMph
        if speedMps is not None and directionDeg is not None:
            self.speedX = speedMps*np.sin(directionDeg/180*math.pi)
            self.speedY = speedMps*np.cos(directionDeg/180*math.pi)
        else:
            self.speedX = None
            self.speedY = None
    
    def convertKnots(self):
        if self.speedKnot is not None:
            self.speedMps = self.speedKnot*0.514444
            self.speedMph = self.speedKnot*1.15078
        if self.gustKnot is not None:
            self.gustMps = self.gustKnot*0.514444
            self.gustMph = self.gustKnot*1.15078
        
    def convertVector(self):
        if self.speedMps is None:
            self.convertKnots()
        if self.directionDeg == 'VRB':
            self.speedX = self.speedMps*2**0.5
            self.speedY = self.speedMps*2**0.5
        elif self.directionDeg is None:
            pass
            #print("Wind direction was not set.")
        elif self.speedMps is None:
            pass
            #print("Wind speed was not set.")
        else:
            try:
                self.speedX = self.speedMps*np.sin(self.directionDeg/180*math.pi)
                self.speedY = self.speedMps*np.cos(self.directionDeg/180*math.pi)
            except:
                assert False, "Unknown wind vector: %s Mps %s Deg" % (str(self.speedMps),str(self.directionDeg))
    
    def __str__(self):
        string = "dateTime:\t\t%s\n"%self.dateTime
        string += "temperatureC:\t%.2f\n"%self.temperatureC
        string += "relativeH:\t\t%.2f\n"%self.relativeH
        string += "speedX:\t\t\t%.2f\n"%self.speedX
        string += "speedY:\t\t\t%.2f\n"%self.speedY
        return string
    
    def __repr__(self):
        return self.__str__()

class ASOSStation(object):
    def __init__(self,info):
        self.ncdcid = info[0]
        self.wban = info[1]
        self.coopid = info[2]
        self.call = info[3]
        self.name = info[4]
        self.aname = info[5]
        self.country = info[6]
        self.state = info[7]
        self.county = info[8]
        self.latitude = info[9]
        self.longitude = info[10]
        self.elevation = info[11]
        self.utc = info[12]
        self.stntype = info[13]
        
        self.data = [] # List of measurements
        self.dateTime = [] # List of time stamps
        
    def old_init2(self, splitLine):
        self.name = splitLine[0] # Station name
        self.longitude = float(splitLine[2]) # Station longitude
        self.latitude = float(splitLine[3]) # Station latitude
        self.data = [] # List of measurements
        self.dateTime = [] # List of time stamps
    
    def __str__(self):
        string = "%s ASOS Station\n"%(self.name)
        string = string + "\tTotal measurements:\t%.0f\n"%(len(self.data))
        string = string + "\tEarliest dateTime:\t%s\n"%(min(self.dateTime))
        string = string + "\tLatest dateTime:\t%s\n"%(max(self.dateTime))
        return string
    
    def __repr__(self):
        return self.__str__()
    
    def old_init(self):
        self.dateTime = [] # Time stamp
        self.temperatureC = [] # Air temperature in Celcius, typicall @ 2m
        self.relativeH = [] # Relative humidity percentage
        self.directionDeg = [] # Wind direction in degrees from north
        self.speedKnot = [] # Wind speed in knots
        self.speedMps = [] # Wind speed in m/s
        self.gustKnot = [] # Wind gust in knots
        self.gustMph = [] # Wind gust in miles per hour
        self.speedX = [] # X-component wind speed in m/s (East positive)
        self.speedY = [] # Y-component wind speed in m/s (North positive)
    
    def addTime(self,splitLine):
        year = int(splitLine[1][0:4])
        month = int(splitLine[1][5:7])
        day = int(splitLine[1][8:10])
        hour = int(splitLine[1][11:13])
        minute = int(splitLine[1][14:16])
        dateTime = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)
        for i in range(4,11):
            try:
                splitLine[i] = float(splitLine[i])
            except ValueError:
                splitLine[i] = np.nan
        temperatureC = splitLine[4]
        relativeH = splitLine[5]
        directionDeg = splitLine[6]
        speedKnot = splitLine[7]
        speedMps = splitLine[8]
        gustKnot = splitLine[9]
        gustMph = splitLine[10]
        data = ASOSMeasurement(dateTime,temperatureC,relativeH,
                               directionDeg,speedKnot,speedMps,gustKnot,gustMph)
        self.dateTime.append(dateTime)
        self.data.append(data)
    
    def addTime_old(self,splitLine):
        year = int(splitLine[1][0:4])
        month = int(splitLine[1][5:7])
        day = int(splitLine[1][8:10])
        hour = int(splitLine[1][11:13])
        minute = int(splitLine[1][14:16])
        self.dateTime.append(dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute))
        for i in range(4,11):
            try:
                splitLine[i] = float(splitLine[i])
            except ValueError:
                splitLine[i] = np.nan
        self.temperatureC.append(splitLine[4])
        self.relativeH.append(splitLine[5])
        self.directionDeg.append(splitLine[6])
        self.speedKnot.append(splitLine[7])
        self.speedMps.append(splitLine[8])
        self.gustKnot.append(splitLine[9])
        self.gustMph.append(splitLine[10])
        self.speedX.append(splitLine[8]*np.sin(splitLine[6]/180*math.pi))
        self.speedY.append(splitLine[8]*np.cos(splitLine[6]/180*math.pi))
        
    def findTime(self,queryDateTime):
        bestMatchValue = min(self.dateTime, key=lambda d: abs(d-queryDateTime))
        bestMatchIndex = self.dateTime.index(bestMatchValue)
        return bestMatchIndex
    
    def timeAverage(self,queryDateTime,timeRange):
        data = ASOSMeasurement()
        bestMatchValue = min(self.dateTime, key=lambda d: abs(d-queryDateTime))
        bestLowValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime-timeRange)))
        bestHighValue = min(self.dateTime, key=lambda d: abs(d-(queryDateTime+timeRange)))
        bestLowIndex = self.dateTime.index(bestLowValue)
        bestHighIndex = self.dateTime.index(bestHighValue)
        bestMatchIndex = self.dateTime.index(bestMatchValue)
        
        temperatureC = []
        relativeH = []
        directionDeg = []
        speedKnot = []
        speedMps = []
        gustKnot = []
        gustMph = []
        
        for i in range(bestLowIndex,bestHighIndex+1):
            temperatureC.append(self.data[i].temperatureC) if self.data[i].temperatureC is not None else temperatureC.append(np.nan)
            relativeH.append(self.data[i].relativeH) if self.data[i].relativeH is not None else relativeH.append(np.nan)
            directionDeg.append(self.data[i].directionDeg) if (self.data[i].directionDeg is not None and self.data[i].directionDeg != 'VRB') else directionDeg.append(np.nan)
            speedKnot.append(self.data[i].speedKnot) if self.data[i].speedKnot is not None else speedKnot.append(np.nan)
            gustKnot.append(self.data[i].gustKnot) if self.data[i].gustKnot is not None else gustKnot.append(np.nan)
            
        dateTime = queryDateTime
        temperatureC = np.nanmean(temperatureC) if (temperatureC and all(v is not None for v in temperatureC)) else None
        relativeH = np.nanmean(relativeH) if (relativeH and all(v is not None for v in relativeH)) else None
        directionDeg = np.nanmean(directionDeg) if (directionDeg and all(v is not None for v in directionDeg)) else None
        speedKnot = np.nanmean(speedKnot) if (speedKnot and all(v is not None for v in speedKnot)) else None
        gustKnot = np.nanmean(gustKnot) if (gustKnot and all(v is not None for v in gustKnot)) else None
        
        data.temperatureC = temperatureC
        data.relativeH = relativeH
        data.directionDeg = directionDeg
        data.speedKnot = speedKnot
        
        data.convertKnots()
        data.convertVector()

        return data
    
    def sortMeasurements(self):
        sorted_data = [x for _, x in sorted(zip(self.dateTime,self.data), key=lambda pair: pair[0])]
        self.data = sorted_data.copy()
        self.dateTime.sort()
        
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
    
    #return stations

def parseMETARline(line):
    line_split = line.split(' ')
    start_index = line_split.index('5-MIN') if '5-MIN' in line_split else -1
    if start_index == -1:
        print("Unable to find 5-MIN string to start parsing:")
        print(line)
        return None
    end_index = line_split.index('RMK') if 'RMK' in line_split else -1
    line_split = line_split[start_index+1:end_index]
    """
    try:
        if 'RMK' in line:
            line_split = line_split[line_split.index('5-MIN')+1:line_split.index('RMK')]
            rmk = True
            rmk_text = line.split('RMK')[1].strip()
        else:
            line_split = line_split[line_split.index('5-MIN')+1:]
    except:
       print(line_split)
    """
    data = ASOSMeasurement()
    
    filecall = line_split[0]
    if line_split[1][0:-1].isdigit() and len(line_split[1]) == 7:
        data.day = int(line_split[1][0:2])
        data.hour = int(line_split[1][2:4])
        data.minute = int(line_split[1][4:6])
    else:
        return None
    data.auto = False
    sm = 0
    data.sky_condition = []
    data.weather_condition = []
    data.temperatureC = None
    data.temperature_dew = 'M'

    line_split = [x for x in line_split if x]
    for i in range(2,len(line_split)):
        if line_split[i] == 'AUTO':
            data.auto = True
        elif 'KT' in line_split[i]:
            filewind = line_split[i].split('KT')[0]
            if 'G' in filewind:
                if filewind.split('G')[1].isdigit():
                    data.gustKnot = float(filewind.split('G')[1])
                else:
                    print("Failed to parse wind gust:")
                    print(line)
                filewind = filewind.split('G')[0]
            if 'VRB' in filewind:
                filewind = filewind.split('VRB')[1]
                data.directionDeg = 'VRB'
            else:
                try:
                    data.directionDeg = float(filewind[0:3])
                except:
                    print("Error parsing direction.")
                    print(line)
            try:
                data.speedKnot = float(filewind[-2:])
            except:
                print("Error parsing windspeed.")
                print(line)
        elif 'V' in line_split[i] and len(line_split[i]) == 7 and 'KT' in line_split[i-1]:
            data.directionDegVar = [float(line_split[i][0:3]),float(line_split[i][4:])]
            
        elif 'SM' in line_split[i]:
            linesm = line_split[i].split('SM')[0]
            try:
                if linesm[0] == 'M':
                    linesm = linesm[1:]
            except:
                print(line_split[i])
            if '/' in linesm:
                if linesm.split('/')[0].isdigit() and linesm.split('/')[1].isdigit():
                    sm += float(linesm.split('/')[0])/float(linesm.split('/')[1])
                    print("Error parsing visibility:")
                    print(line)
            else:
                try:
                    sm += float(linesm)
                except:
                    print("Error parsing visibility:")
                    print(line)

        elif line_split[i][0] == 'R' and len(line_split[i]) >= 10:
            if line_split[i][-2:] == 'FT':
                data.rvr = line_split[i]
        elif ('BKN' in line_split[i] or 'CLR' in line_split[i]
                or 'FEW' in line_split[i] or 'SCT' in line_split[i]
                or 'OVC' in line_split[i]):
            data.sky_condition.append([line_split[i][0:3],line_split[i][3:]])
        elif ('RA' in line_split[i] or 'SN' in line_split[i] 
                or 'UP' in line_split[i] or 'FG' in line_split[i]
                or 'FZFG' in line_split[i] or 'BR' in line_split[i]
                or 'HZ' in line_split[i] or 'SQ' in line_split[i]
                or 'FC' in line_split[i] or 'TS' in line_split[i]
                or 'GR' in line_split[i] or 'GS' in line_split[i]
                or 'FZRA' in line_split[i] or 'VA' in line_split[i]):
            data.weather_condition.append(line_split[i])
        elif line_split[i][0] == 'A' and len(line_split[i]) == 5:
            try:
                data.altimeter = float(line_split[i][1:])
            except:
                print("Error parsing altitude.")
                print(line)
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
                data.temperatureC = float(linetemp[0])*temperature_air_sign
            if linetemp[1].isdigit():
                data.temperature_dew = float(linetemp[1])*temperature_dew_sign
            if linetemp[0].isdigit() and linetemp[1].isdigit():
                data.relativeH = 100-5*(data.temperatureC-data.temperature_dew)
        else:
            if i < len(line_split)-1:
                if 'SM' in line_split[i+1] and '/' in line_split[i+1] and line_split[i].isdigit():
                    try:
                        sm += float(line_split[i])
                    except:
                        print(line)
                        print(line_split)
                else:
                    pass
                    #print('Unknown argument %s at %.0f.' % (line_split[i],0))
            else:
                pass
                #print('Unknown argument %s at %.0f.' % (line_split[i],1))
    if sm == 0:
        data.sm = None
    else:
        data.sm = sm
    
    data.convertKnots()
    data.convertVector()
    return data
  
  
def parseMETARfile(file):
    datas = []
    dateTimes = []
    with open(file) as f:
        old_day = 0
        content = f.readlines()
        for line in content:
            data = parseMETARline(line)
            if data is not None:
                data.year = int(file[-10:-6])
                data.month = int(file[-6:-4])
                if data.day < old_day:
                    data.month = data.month + 1
                if data.month > 12:
                    data.month = 1
                    data.year = data.year + 1
                old_day = data.day
                data.dateTime = dt.datetime(year=data.year,month=data.month,day=data.day,hour=data.hour,minute=data.minute)
                datas.append(data)
                dateTimes.append(data.dateTime)
    return datas, dateTimes

if __name__ == "__main__":

    if 'stations' not in locals():
        #stations = parseAsosFile('./data/asos.txt')
        stations = defineStations('../data/asos-stations.txt')
        empty_stations = []
        for key in stations.keys():
            call = stations[key].call
            files = glob.glob('../data/asos/asos-fivemin/6401-2016/*'+call+'*')
            if len(files) != 0:# and key == 'WVI':
                for file in files:
                    data, dateTime = parseMETARfile(file)
                    stations[key].data.extend(data)
                    stations[key].dateTime.extend(dateTime)
            else:
                empty_stations.append(key)  
            print("Finished %s"%key)
        for key in empty_stations:
            stations.pop(key,None)
        
    geoLimits = findGeoLimits(stations)
    
    latGrid = np.linspace(geoLimits[0]*0.95,geoLimits[1]*1.05,101)
    lonGrid = np.linspace(geoLimits[2]*0.95,geoLimits[3]*1.05,101)
    
    queryDateTime = dt.datetime(year=2016,month=6,day=30,hour=5,minute=53)
    timeRange = dt.timedelta(days=0,hours=24,minutes=0)
    data = stations['WVI'].timeAverage(queryDateTime,timeRange)
    measurements = stationsGetMeasurements(stations,queryDateTime,timeRange)
    
    """
    speedXcontour = griddata(measurements[:,0],measurements[:,1],measurements[:,2],latGrid,lonGrid,interp='linear')
    speedYcontour = griddata(measurements[:,0],measurements[:,1],measurements[:,3],latGrid,lonGrid,interp='linear')
    
    plt.figure(figsize=(12,8))
    CS = plt.contourf(lonGrid,latGrid,speedXcontour)
    plt.colorbar()
    plt.xlim([-126,-114])
    plt.ylim([32,42])
    
    plt.figure(figsize=(12,8))
    CS = plt.contourf(lonGrid,latGrid,speedYcontour)
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