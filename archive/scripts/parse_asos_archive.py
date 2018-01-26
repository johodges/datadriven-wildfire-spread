# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:18:39 2018

@author: JHodges
"""


"""
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
    
    stations = stationsSortMeasurements(stations)
    return stations
"""

"""
def writeAsosFile(filename,contents,headerLines=6,lines2write=2000):
    with open(filename, 'w') as f:
        for i in range(0,headerLines):
            f.write("%s"%contents[i])
        for i in range(headerLines,lines2write+headerLines):
            f.write("%s"%contents[i])
"""







    """
    def convertKnots(self,speedKnot):
        if speedKnot is not None:
            speedMps = speedKnot*0.514444
        return speedMps
    """
    
    """
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
    """
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