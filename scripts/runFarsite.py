# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:41:16 2018

@author: JHodges
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import behavePlus as bp
import glob
from shapely.geometry import Polygon, Point
import datetime
import subprocess
import queryLandFire as qlf
import multiprocessing as mp
import sys
from runFarsiteMp import runFarsite

"""
def runFarsite(commandFile):
    dockerStart = 'docker run -it -v E:\\projects\\wildfire-research\\farsite\\:/commonDir/ farsite'
    dockerCmd = './commonDir/farsite/src/TestFARSITE %s'%(commandFile)
    
    p = subprocess.Popen('winpty '+dockerStart+' '+dockerCmd,shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE)
    p_status = p.wait()
"""

def getFuelMoistureData(string,params,fuelModels=np.linspace(0,1,2)):
    string = string+'FUEL_MOISTURES_DATA: %.0f\n'%(fuelModels.shape[0])
    m1h = params['m1h']
    m10h = params['m10h']
    m100h = params['m100h']
    lhm = params['lhm']
    lwm = params['lwm']
    
    for i in range(0,fuelModels.shape[0]):
        #string = string+'%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n'%(fuelModels[i],m1h,m10h,m100h,lhm,lwm)
        string = string+'%.0f %.1f %.1f %.1f %.1f %.1f\n'%(fuelModels[i],m1h,m10h,m100h,lhm,lwm)
    return string

def getMaxDay(Mth):
    maxDay = 31
    if Mth == 4 or Mth == 6 or Mth == 9 or Mth == 11:
        maxDay = 30
    if Mth == 2:
        maxDay = 28
    return maxDay

def incrementDay(Day,Mth):
    Day = Day + 1
    maxDay = getMaxDay(Mth)
    if Day > maxDay:
        Day = 1
        Mth = Mth+1
    if Mth > 12:
        Mth = 1
    return Day, Mth

def getWeatherData(string,params,Elv,totalDays=2):
    string = string+"WEATHER_DATA: %.0f\n"%(totalDays)
    
    Mth = round(params['Mth'])
    Day = round(params['Day'])
    maxDay = getMaxDay(Mth)
    Day = min(Day,maxDay)
    
    Pcp = params['Pcp']
    mTH = round(params['mTH'],-2)
    xTH = round(params['xTH'],-2)
    mT = params['mT']
    xT = params['xT']
    xH = params['xH']
    mH = params['mH']
    PST = round(params['PST'],-2)
    PET = round(params['PET'],-2)
    
    for i in range(0,totalDays):
        # Mth  Day  Pcp  mTH  xTH   mT xT   xH mH   Elv   PST  PET
        Day, Mth = incrementDay(Day,Mth)
        string = string+'%.0f %.0f %.1f %.0f %.0f %.1f %.1f %.1f %.1f %.0f %.0f %.0f\n'%(
                Mth,Day,0,mTH,xTH,mT,xT,xH,xH,Elv,0,0)
    string = string+"WEATHER_DATA_UNITS: Metric\n"
    return string
    
def getWindData(string,params,totalDays=2):
    string = string+"WIND_DATA: %.0f\n"%(totalDays*24)
    windSpeed = params['windSpeed'] # mph
    windDir = params['windDir']
    
    windSpeed = windSpeed*5280*12*25.4/(1000*1000) # km/h
    
    Mth = round(params['Mth'])
    Day = round(params['Day'])
    maxDay = getMaxDay(Mth)
    Day = min(Day,maxDay)

    for i in range(0,totalDays):
        # Mth  Day  Hour   Speed Direction CloudCover
        Day, Mth = incrementDay(Day,Mth)
        for Hour in range(0,2400,100):
            string = string+'%.0f %.0f %.0f %.1f %.1f %.1f\n'%(
                    Mth,Day,Hour,windSpeed,windDir,0)
    string = string+"WIND_DATA_UNITS: Metric\n"
    return string
    
def getMiscData(string):
    string = string + "FOLIAR_MOISTURE_CONTENT: 100\n"
    string = string + "CROWN_FIRE_METHOD: Finney\n" # Either Finney or ScottRhienhardt
    string = string + "FARSITE_SPOT_PROBABILITY: 0.01\n"
    string = string + "FARSITE_SPOT_IGNITION_DELAY: 15\n"
    string = string + "FARSITE_MINIMUM_SPOT_DISTANCE: 30\n"
    string = string + "FARSITE_ACCELERATION_ON: 1\n"
    return string

def getSimulationData(string,params,totalDays=2):
    Mth = round(params['Mth'])
    Day = round(params['Day'])    
    maxDay = getMaxDay(Mth)
    Day = min(Day,maxDay)
    startTime = params['startTime']
    startHour = np.floor(startTime)
    startMin = min(int(round((startTime-startHour)*60,0)),59)
    startHour = int(startHour)
    
    Day, Mth = incrementDay(Day,Mth)
    Day, Mth = incrementDay(Day,Mth)
    
    sTime = datetime.datetime(year=2016,month=Mth,day=Day,hour=startHour,minute=startMin)
    eTime = datetime.datetime(year=2016,month=Mth,day=Day)+datetime.timedelta(days=totalDays-2)
    #eTime = sTime+datetime.timedelta(days=totalDays-3)
    sTimeString = sTime.strftime('%m %d %H%M')
    eTimeString = eTime.strftime('%m %d %H%M')
    
    string = string + "FARSITE_START_TIME: %s\n"%(sTimeString)
    string = string + "FARSITE_END_TIME: %s\n"%(eTimeString)
    string = string + "FARSITE_TIME_STEP: 60\n"
    string = string + "FARSITE_DISTANCE_RES: 30.0\n"
    string = string + "FARSITE_PERIMETER_RES: 60.0\n"
    return string

def getIgnitionData(string,ignitionFile):
    string = string + "FARSITE_IGNITION_FILE: %s\n"%(ignitionFile)
    return string

def saveFarsiteInput(string,file):
    with open(file,'w') as f:
        f.write(string)
    

def generateFarsiteInput(file,elevation,ignitionFile,totalDays=5):
    
    paramsInput = bp.getStandardParamsInput()
    params = bp.getRandomConditions(paramsInput,allowDynamicModels=True)
    string = ''
    string = getFuelMoistureData(string,params)
    string = getWeatherData(string,params,elevation,totalDays=totalDays)
    string = getWindData(string,params,totalDays=totalDays)
    string = getSimulationData(string,params,totalDays=totalDays)
    string = getMiscData(string)
    string = getIgnitionData(string,ignitionFile)
    saveFarsiteInput(string,file+'.input')
    #print(string)
    
    return params

def getLcpElevation(file):
    imgs, names, header = qlf.readLcpFile(file)
    elevation = np.median(imgs[0])
    return elevation

def makeCenterIgnition(file,N=5):
    imgs, names, header = qlf.readLcpFile(file)
    eastUtm = header[1039]; westUtm = header[1040]
    northUtm = header[1041]; southUtm = header[1042]
    xResol = header[1044]; yResol = header[1045];
    
    centerX = (eastUtm+westUtm)/2
    centerY = (northUtm+southUtm)/2
    
    corners = [[centerX-N*xResol,centerY-N*yResol],
               [centerX-N*xResol,centerY+N*yResol],
               [centerX+N*xResol,centerY+N*yResol],
               [centerX+N*xResol,centerY-N*yResol]]
    points = [Point(xyz) for xyz in corners]
    geometry = Polygon([[p.x,p.y] for p in points])
    data = gpd.GeoDataFrame([[0,0]],columns=['ENTITY','VALUE'],geometry=[geometry]) # ENTITY=0, VALUE = 0
    file = file.replace('.LCP','_ignite.SHP')
    data.to_file(driver = 'ESRI Shapefile', filename=file)
    return data

def generateCmdFile(lcps,inputs,ignites,outputs,cmdFile):
    cDir = 'commonDir/data/'
    string = ''
    for lcp, Input, ignite, output in zip(lcps,inputs,ignites,outputs):
        string = string+'%s%s %s%s %s%s 0 %s%s 0\n'%(cDir,lcp,cDir,Input,cDir,ignite,cDir,output)
    with open(cmdFile,'w') as f:
        f.write(string)
    

if __name__ == "__main__":

    distance = 25000

    commandFile = 'commonDir/farsite/example/Panther/runPanther.txt'
    inDir = 'E:/projects/wildfire-research/farsite/data/'
    cDir = 'commonDir/data/'
    #namespace = 'n117-9343_36-5782_3000'
    lcpNames = glob.glob(inDir+"*_25000.LCP")
    
    totalDays = 5
    
    cmdFileDockers = []
    for j in range(0,100):
        cmdFile = inDir+'toRun_'+str(j)+'.txt'
        cmdFileDocker = cDir+'toRun_'+str(j)+'.txt'
        lcpFiles = []
        inputFiles = []
        igniteFiles = []
        outputFiles = []
        for i in range(0,50):
            lcpName = np.random.choice(lcpNames).split('\\')[1].split('.LCP')[0]
            namespace = "run_"+str(j)+"_"+(str(i))+"_"+lcpName
            lcpFile = lcpName+'.LCP' # Only one landscape file per location
            igniteFile = lcpName+'_ignite.SHP' # Only one ignition file per location
            
            inputFile = namespace+'.input' # Different input file per simulation
            outputFile = namespace+'_out' # Different output file per simulation
            elevation = getLcpElevation(inDir+lcpFile)
            if len(glob.glob(igniteFile)) == 0:
                ignitionShape = makeCenterIgnition(inDir+lcpFile)
            fileName = inDir+namespace
            params = generateFarsiteInput(fileName,elevation,cDir+igniteFile,totalDays=totalDays)
            lcpFiles.append(lcpFile)
            inputFiles.append(inputFile)
            igniteFiles.append(igniteFile)
            outputFiles.append(outputFile)
        generateCmdFile(lcpFiles,inputFiles,igniteFiles,outputFiles,cmdFile)
        cmdFileDockers.append(cmdFileDocker)
    
    totalSamples = 5
    with mp.Pool(processes=6) as pool:
        q = 0
        for i in pool.imap_unordered(runFarsite,cmdFileDockers):
            q += 1
            if q % 2 == 0:
                print("Percent complete: %.2f"%(q/totalSamples*100))
                sys.stdout.flush()
    
    #runFarsite(cmdFileDocker)
    
    #dataOut = gpd.GeoDataFrame.from_file(inDir+outputFile+'_Perimeters.shp')
    #dataIn = gpd.GeoDataFrame.from_file(inDir+igniteFile)
    
    
    
    
    #filename = 'E:/projects/wildfire-research/farsite/data/californiaRaw.LCP'
    #inDir = 'E:/projects/wildfire-research/farsite/data/'
    
    #print(checkPoint([[lon,lat]]))
    #datas, headers, names = generateLcpFile(lats,lons,distance,inDir)
    
    