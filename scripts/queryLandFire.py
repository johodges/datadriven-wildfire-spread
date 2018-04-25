# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:41:16 2018

@author: JHodges
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import gdal
import skimage.transform as skt
import numpy as np
from generate_dataset import GriddedMeasurement
import scipy.interpolate as scpi
import behavePlus as bp
import os
import glob
import struct
import matplotlib.path as mpltPath
from shapely.geometry import Polygon, Point
import datetime
import subprocess

def readImgFile(imgFile):
    img = gdal.Open(imgFile)
    band = np.array(img.ReadAsArray(),dtype=np.float32)
    band[band<0] = np.nan
    
    old_cs = gdal.osr.SpatialReference()
    old_cs.ImportFromWkt(img.GetProjectionRef())
    wgs84_wkt = """
    GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
    
    new_cs = gdal.osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    transform = gdal.osr.CoordinateTransformation(old_cs,new_cs)
    width = img.RasterXSize
    height = img.RasterYSize
    gt = img.GetGeoTransform()
    
    x = np.linspace(0,width,width+1)
    y = np.linspace(0,height,height+1)
    x = gt[0] + x*gt[1]
    y = gt[3] + y*gt[5]
    xGrid, yGrid = np.meshgrid(x,y)
    
    ds = 5
    xGridDs = xGrid[::ds,::ds]
    yGridDs = yGrid[::ds,::ds]
    bandDs = band[::ds,::ds]
    sz = xGridDs.shape
    
    xGrid_rs = np.reshape(xGridDs,(xGridDs.shape[0]*xGridDs.shape[1],))
    yGrid_rs = np.reshape(yGridDs,(yGridDs.shape[0]*yGridDs.shape[1],))
    points = np.array([xGrid_rs,yGrid_rs]).T

    latlong = np.array(transform.TransformPoints(points))
    lat = np.reshape(latlong[:,1],(sz[0],sz[1]))
    lon = np.reshape(latlong[:,0],(sz[0],sz[1]))
    
    return bandDs, lat, lon
    #data, lat, lon = gridAndResample(bandDs,lat,lon)

    #data2 = GriddedMeasurement(None,lat,lon,data,'FuelModel')
    #data2.dataName = 'FuelModel'

    #return data2


def getExtents(imgFile):
    img = gdal.Open(imgFile)
    #band = np.array(img.ReadAsArray(),dtype=np.float32)
    #band[band<0] = np.nan
    
    old_cs = gdal.osr.SpatialReference()
    old_cs.ImportFromWkt(img.GetProjectionRef())
    wgs84_wkt = """
    GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
    
    new_cs = gdal.osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    transform = gdal.osr.CoordinateTransformation(old_cs,new_cs)
    width = img.RasterXSize
    height = img.RasterYSize
    gt = img.GetGeoTransform()
    
    x = np.linspace(0,width,width+1)
    y = np.linspace(0,height,height+1)
    x = gt[0] + x*gt[1]
    y = gt[3] + y*gt[5]
    xGrid, yGrid = np.meshgrid(x,y)
    
    ds = 5
    xGridDs = xGrid[::ds,::ds]
    yGridDs = yGrid[::ds,::ds]
    bandDs = band[::ds,::ds]
    sz = xGridDs.shape
    
    xGrid_rs = np.reshape(xGridDs,(xGridDs.shape[0]*xGridDs.shape[1],))
    yGrid_rs = np.reshape(yGridDs,(yGridDs.shape[0]*yGridDs.shape[1],))
    points = np.array([xGrid_rs,yGrid_rs]).T

    latlong = np.array(transform.TransformPoints(points))
    lat = np.reshape(latlong[:,1],(sz[0],sz[1]))
    lon = np.reshape(latlong[:,0],(sz[0],sz[1]))
    
    return bandDs, lat, lon

def splitImages(inDirs=None,namespace=None):
    if inDirs is None:
        inDirs = ['E:/projects/wildfire-research/landfireData/US_140FBFM40/1/',
                  'E:/projects/wildfire-research/landfireData/US_140FBFM40/2/',
                  'E:/projects/wildfire-research/landfireData/US_140FBFM40/3/',
                  'E:/projects/wildfire-research/landfireData/US_140FBFM40/4/']
    if namespace is None:
        namespace = 'US_140FBFM40'
    for inDir in inDirs:
        files = glob.glob(inDir+namespace+'.tif')
        for file in files:
            img = gdal.Open(file)
            old_cs = gdal.osr.SpatialReference()
            old_cs.ImportFromWkt(img.GetProjectionRef())
            gt = img.GetGeoTransform()
            width = img.RasterXSize
            height = img.RasterYSize
            minX = gt[0]
            maxX = gt[0] + width*gt[1]
            minY = gt[3]
            maxY = gt[3] + height*gt[5]
            
            halfX = (maxX+minX)/2
            halfY = (maxY+minY)/2
            
            cmd = "gdal_translate.exe %s%s.tif %s%s_ul.tif -projwin %.3f %.3f %.3f %.3f"%(inDir,namespace,inDir,namespace,minX,minY,halfX,halfY)
            os.system(cmd)
            cmd = "gdal_translate.exe %s%s.tif %s%s_ur.tif -projwin %.3f %.3f %.3f %.3f"%(inDir,namespace,inDir,namespace,halfX,minY,maxX,halfY)
            os.system(cmd)
            cmd = "gdal_translate.exe %s%s.tif %s%s_dl.tif -projwin %.3f %.3f %.3f %.3f"%(inDir,namespace,inDir,namespace,minX,halfY,halfX,maxY)
            os.system(cmd)
            cmd = "gdal_translate.exe %s%s.tif %s%s_dr.tif -projwin %.3f %.3f %.3f %.3f"%(inDir,namespace,inDir,namespace,halfX,halfY,maxX,maxY)
            os.system(cmd)

def generateLatLonImgs(file,outDir,namespace,debug=False):
    img = gdal.Open(file)
    band = np.array(img.ReadAsArray(),dtype=np.float32)
    #band[band<0] = -1
    
    old_cs = gdal.osr.SpatialReference()
    old_cs.ImportFromWkt(img.GetProjectionRef())
    wgs84_wkt = """
    GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
    print(img.GetProjectionRef())
    assert False, "Stopped"
    new_cs = gdal.osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    transform = gdal.osr.CoordinateTransformation(old_cs,new_cs)
    iTf = gdal.osr.CoordinateTransformation(new_cs,old_cs)
    width = img.RasterXSize
    height = img.RasterYSize
    gt = img.GetGeoTransform()
    
    x = np.linspace(0,width,width)
    y = np.linspace(0,height,height)
    x = gt[0] + x*gt[1]
    y = gt[3] + y*gt[5]
    
    if debug:
        xGrid, yGrid = np.meshgrid(x,y)
        
        sz = xGrid.shape
        xGrid_rs = np.reshape(xGrid,(xGrid.shape[0]*xGrid.shape[1],))
        yGrid_rs = np.reshape(yGrid,(yGrid.shape[0]*yGrid.shape[1],))
        points = np.array([xGrid_rs,yGrid_rs]).T
        
        latlong = np.array(transform.TransformPoints(points))[:,:-1]
        lat = np.reshape(latlong[:,1],(sz[0],sz[1]))
        lon = np.reshape(latlong[:,0],(sz[0],sz[1]))
    
    points = np.array(transform.TransformPoints([[x.min(),y.min()],[x.min(),y.max()],[x.max(),y.min()],[x.max(),y.max()]]))
    
    lat_b = np.ceil(np.min(points[:,1]))
    lat_u = np.floor(np.max(points[:,1]))
    lon_l = np.ceil(np.min(points[:,0]))
    lon_r = np.floor(np.max(points[:,0]))
    
    resX = 3700
    resY = 3700
    
    interpFunction = scpi.RegularGridInterpolator((-1*y,x),band,bounds_error=False,fill_value=-9999,method='nearest')
    
    driver = gdal.GetDriverByName('GTiff')
    
    for i in range(0,int(lat_u-lat_b)):
        for j in range(0,int(lon_r-lon_l)):
            latMin = lat_b+float(i)
            lonMin = lon_l+float(j)
            yNew = np.linspace(latMin,latMin+1,resY)
            xNew = np.linspace(lonMin,lonMin+1,resX)
            yNewGrid, xNewGrid = np.meshgrid(yNew,xNew)
            xNewGrid_rs = np.reshape(xNewGrid,(xNewGrid.shape[0]*xNewGrid.shape[1],))
            yNewGrid_rs = np.reshape(yNewGrid,(yNewGrid.shape[0]*yNewGrid.shape[1],))
            points2 = np.array([xNewGrid_rs,yNewGrid_rs]).T
            ipoints2 = np.array(iTf.TransformPoints(points2))[:,:-1]
            ipoints2swap = np.array([ipoints2[:,1],ipoints2[:,0]]).T
            
            dataPoints_rs = interpFunction((-1*ipoints2swap[:,0],ipoints2swap[:,1]))
            dataPoints = np.reshape(dataPoints_rs,(resX,resY)).T
            
            if latMin < 0:
                latStr = 'n%.0f'%(abs(latMin))
            else:
                latStr = '%.0f'%(abs(latMin))
            if lonMin < 0:
                lonStr = 'n%.0f'%(abs(lonMin))
            else:
                lonStr = '%.0f'%(abs(lonMin))
            
            name = outDir+namespace+'_%s_%s'%(latStr,lonStr)
            
            if len(glob.glob(name+'.tif')) > 0:
                img2 = gdal.Open(name+'.tif')
                band2 = np.array(img2.ReadAsArray(),dtype=np.float32)
                dataPoints = dataPoints[::-1]
                band2[band2 == -9999] = dataPoints[band2 == -9999]
                dataPoints = band2[::-1]
                print("Found: %s, Merging."%(name))
            else:
                print("\tMaking %s"%(name))
            
            dataset = driver.Create(
                    name+'.tif',
                    resX,
                    resY,
                    1,
                    gdal.GDT_Float32, )
            dataset.SetGeoTransform((
                    lonMin,
                    1/resX,
                    0,
                    latMin+1,
                    0,
                    -1/resY))
            dataset.SetProjection(wgs84_wkt)
            dataset.GetRasterBand(1).WriteArray(dataPoints[::-1,:])
            dataset.FlushCache()
            if debug:
                plt.figure(figsize=(12,8))
                plt.subplot(1,2,1)
                plt.contourf(xNew,yNew,dataPoints); plt.clim([90,210]); plt.colorbar(); 
                plt.subplot(1,2,2)
                plt.contourf(lon[::10,::10],lat[::10,::10],band[::10,::10]); plt.clim([90,210]); plt.colorbar(); plt.xlim([lon_l+float(j),lon_l+float(j)+1]); plt.ylim([lat_b+float(i),lat_b+float(i)+1]); 
            
    return dataPoints

def getHistogram(file):
    img = gdal.Open(file)
    band = np.array(img.ReadAsArray(),dtype=np.float32)
    return band

def getTransformFromMtoD():
    wgs84_wkt = """
    GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
    
    nad83_wkt = """
        PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",
            GEOGCS["NAD83",
                DATUM["North_American_Datum_1983",
                    SPHEROID["GRS 1980",6378137,298.2572221010042,
                        AUTHORITY["EPSG","7019"]],
                    AUTHORITY["EPSG","6269"]],
                PRIMEM["Greenwich",0],
                UNIT["degree",0.0174532925199433],
                AUTHORITY["EPSG","4269"]],
           PROJECTION["Albers_Conic_Equal_Area"],
           PARAMETER["standard_parallel_1",29.5],
           PARAMETER["standard_parallel_2",45.5],
           PARAMETER["latitude_of_center",23],
           PARAMETER["longitude_of_center",-96],
           PARAMETER["false_easting",0],
           PARAMETER["false_northing",0],
           UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"""
    new_cs = gdal.osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    old_cs = gdal.osr.SpatialReference()
    old_cs.ImportFromWkt(nad83_wkt)
    Tf = gdal.osr.CoordinateTransformation(old_cs,new_cs)
    iTf = gdal.osr.CoordinateTransformation(new_cs,old_cs)
    return Tf, iTf

def readAscFile(file,lat,lon,distance,debug=False):
    with open(file,'r') as f:
        lines = f.readlines()
    nCols = float(lines[0].split(' ')[-1].split('\n')[0])
    nRows = float(lines[1].split(' ')[-1].split('\n')[0])
    xll = float(lines[2].split(' ')[-1].split('\n')[0])
    yll = float(lines[3].split(' ')[-1].split('\n')[0])
    fileDx = float(lines[4].split(' ')[-1].split('\n')[0])
    noDataValue = float(lines[5].split(' ')[-1].split('\n')[0])
    xur = xll + fileDx*nCols
    yur = yll + fileDx*nRows
    
    Tf, iTf = getTransformFromMtoD()
    
    queryPoint = iTf.TransformPoints([[lon,lat]])[0]
    bounds = Tf.TransformPoints([[xll,yll],[xll,yur],[xur,yll],[xur,yur]])
    
    centerX = int(np.round(nCols*(queryPoint[0]-xll)/(xur-xll)))
    centerY = int(nRows)-int(np.round(nRows*(queryPoint[1]-yll)/(yur-yll)))+1
    N = int(distance/fileDx)
    if debug:
        print("\tpoint\tx\t\ty")
        print("\tll\t%.0f\t%.0f"%(xll,yll))
        print("\tqp\t%.0f\t%.0f"%(queryPoint[0],queryPoint[1]))
        print("\tur\t%.0f\t%.0f"%(xur,yur))
        print("\tfull\t%.0f\t%.0f"%(nCols,nRows))
        print('cen\t%.0f\t%.0f'%(centerX,centerY))
    if centerY-N-2 > 0 and centerY+N+1 < nRows and centerX-N > 0 and centerX+N+1 < nCols:
        #if xll < queryPoint[0] and xur > queryPoint[0] and yll < queryPoint[1] and yur > queryPoint[1]:
        
        data = []
        for i in range(6+centerY-N,6+centerY+N+1):
            line = lines[i]
            lineSplit = line.split(' ')[1:]
            lineSplit[-1] = lineSplit[-1].split('\n')[0]
            data.append(lineSplit)
        data = np.array(data,dtype=np.float32)[:,centerX-N:centerX+N+1]
        #data[data == noDataValue] = np.nan
        header = ['ncols        %.0f\n'%(2*N+1),
                  'nrows        %.0f\n'%(2*N+1),
                  'xllcorner    %.12f\n'%(queryPoint[0]-N*fileDx),
                  'yllcorner    %.12f\n'%(yll+(nRows-(centerY+N))*fileDx),
                  'cellsize     %.12f\n'%(fileDx),
                  'NODATA_value %.0f'%(noDataValue)]
        stringHeader = ''
        for line in header:
            stringHeader = stringHeader+line
        
    else:
        print("Point is not contained in file.")
        print("\tpoint\tx\t\ty")
        print("\tll\t%.0f\t%.0f"%(xll,yll))
        print("\tqp\t%.0f\t%.0f"%(queryPoint[0],queryPoint[1]))
        print("\tur\t%.0f\t%.0f"%(xur,yur))
        data = None
        stringHeader = None
        
    return data,stringHeader

def readFullAscFile(file,lat=None,lon=None,debug=False):
    with open(file,'r') as f:
        lines = f.readlines()
    nCols = float(lines[0].split(' ')[-1].split('\n')[0])
    nRows = float(lines[1].split(' ')[-1].split('\n')[0])
    xll = float(lines[2].split(' ')[-1].split('\n')[0])
    yll = float(lines[3].split(' ')[-1].split('\n')[0])
    fileDx = float(lines[4].split(' ')[-1].split('\n')[0])
    noDataValue = float(lines[5].split(' ')[-1].split('\n')[0])
    xur = xll + fileDx*nCols
    yur = yll + fileDx*nRows
    
    if lat is not None or lon is not None:
        Tf, iTf = getTransformFromMtoD()
        queryPoint = iTf.TransformPoints([[lon,lat]])[0]
        centerX = int(np.round(nCols*(queryPoint[0]-xll)/(xur-xll)))
        centerY = int(nRows)-int(np.round(nRows*(queryPoint[1]-yll)/(yur-yll)))
        if debug:
            print("\tpoint\tx\t\ty")
            print("\tll\t%.0f\t%.0f"%(xll,yll))
            print("\tqp\t%.0f\t%.0f"%(queryPoint[0],queryPoint[1]))
            print("\tur\t%.0f\t%.0f"%(xur,yur))
            print("\tfull\t%.0f\t%.0f"%(nCols,nRows))
            print('cen\t%.0f\t%.0f'%(centerX,centerY))
        if centerY > 0 and centerY < nRows and centerX > 0 and centerX < nCols:
            check = True
        else:
            check = False
    else:
        check = True
    data = []
    if check:
        for i in range(6,int(nRows+6)):
            line = lines[i]
            lineSplit = line.split(' ')
            if lineSplit[0] == ' ':
                lineSplit = lineSplit[1:]
            lineSplit[-1] = lineSplit[-1].split('\n')[0]
            
            if lineSplit[-1] == '':
                lineSplit = lineSplit[:-2]
            
            data.append(lineSplit)
            
        data = np.array(data,dtype=np.float32)
        #data[data == noDataValue] = np.nan
        header = ['ncols        %.0f\n'%(nCols),
                  'nrows        %.0f\n'%(nRows),
                  'xllcorner    %.12f\n'%(xll),
                  'yllcorner    %.12f\n'%(yll),
                  'cellsize     %.12f\n'%(fileDx),
                  'NODATA_value %.0f'%(noDataValue)]
        stringHeader = ''
        for line in header:
            stringHeader = stringHeader+line
    else:
        print("Point is not contained in file.")
        print("\tpoint\tx\t\ty")
        print("\tll\t%.0f\t%.0f"%(xll,yll))
        print("\tqp\t%.0f\t%.0f"%(queryPoint[0],queryPoint[1]))
        print("\tur\t%.0f\t%.0f"%(xur,yur))
        data = None
        stringHeader = None
        
    return data,stringHeader

def queryAsciiFiles(lat,lon,distance,skipMissing=False):
    inDir = 'E:/projects/wildfire-research/farsiteData/'
    #names = ['US_SLP2010','US_140CBD','US_140CBH','US_140CC','US_140CH','US_140FBFM40','US_ASP2010','US_DEM2010']
    names = ['US_DEM2010','US_SLP2010','US_ASP2010','US_140FBFM40','US_140CC']
    canopyNames = ['US_140CH','US_140CBH','US_140CBD']
    groundNames = []
    namespace = getNamespace(lat,lon,distance)
    print(namespace)
    fullDatas = []
    fullNames = []
    filenames = []
    datas = []
    headers = []
    for name in names:
        filename = inDir+name+'.asc'
        extractedFile = inDir+namespace+'_'+name+'.asc'
        if len(glob.glob(extractedFile)) == 0:
            return None, None, None
            data, header = readAscFile(filename,lat,lon,distance)
            if data is not None and header is not None:
                extractedFile = inDir+namespace+'_'+name+'.asc'
                np.savetxt(extractedFile,data, fmt='%.1f', delimiter=' ',newline='\n', header=header,comments='')
        else:
            data, header = readFullAscFile(extractedFile,lat,lon)
        datas.append(data)
        headers.append(header)
        filenames.append(filename)
    fullDatas.append(datas)
    fullNames.append(filenames)
    datas = []
    filenames = []
    for name in canopyNames:
        filename = inDir+name+'.asc'
        extractedFile = inDir+namespace+'_'+name+'.asc'
        if len(glob.glob(extractedFile)) == 0:
            data, header = readAscFile(filename,lat,lon,distance)
            if data is not None and header is not None:
                extractedFile = inDir+namespace+'_'+name+'.asc'
                np.savetxt(extractedFile,data, fmt='%.1f', delimiter=' ',newline='\n', header=header,comments='')
        else:
            data, header = readFullAscFile(extractedFile,lat,lon)
        datas.append(data)
        headers.append(header)
        filenames.append(filename)
    fullDatas.append(datas)
    fullNames.append(filenames)
    datas = []
    filenames = []
    for name in groundNames:
        filename = inDir+name+'.asc'
        extractedFile = inDir+namespace+'_'+name+'.asc'
        if len(glob.glob(extractedFile)) == 0:
            data, header = readAscFile(filename,lat,lon,distance)
            if data is not None and header is not None:
                extractedFile = inDir+namespace+'_'+name+'.asc'
                np.savetxt(extractedFile,data, fmt='%.1f', delimiter=' ',newline='\n', header=header,comments='')
        else:
            data, header = readFullAscFile(extractedFile,lat,lon)
        datas.append(data)
        headers.append(header)
        filenames.append(filename)
    fullDatas.append(datas)
    fullNames.append(filenames)
    
    return fullDatas, headers, fullNames

def getAscFile(file):
    with open(file,'r') as f:
        lines = f.readlines()
    nCols = float(lines[0].split(' ')[-1].split('\n')[0])
    nRows = float(lines[1].split(' ')[-1].split('\n')[0])
    xll = float(lines[2].split(' ')[-1].split('\n')[0])
    yll = float(lines[3].split(' ')[-1].split('\n')[0])
    fileDx = float(lines[4].split(' ')[-1].split('\n')[0])
    noDataValue = float(lines[5].split(' ')[-1].split('\n')[0])
    params = (nCols,nRows,xll,yll,fileDx,noDataValue)
    return lines, params

def getNamespace(lat,lon,distance):
    namespace = '%.4f_%.4f_%.0f'%(lon,lat,distance)
    namespace = namespace.replace('-','n')
    namespace = namespace.replace('.','-')
    return namespace

def getCoordinateData(lines,params,lat,lon):

    (nCols,nRows,xll,yll,fileDx,noDataValue) = params
    xur = xll + fileDx*nCols
    yur = yll + fileDx*nRows
    Tf, iTf = getTransformFromMtoD()
    
    queryPoint = iTf.TransformPoints([[lon,lat]])[0]
    
    if checkPoint([[lon,lat]]):
        centerX = int(np.round(nCols*(queryPoint[0]-xll)/(xur-xll)))
        centerY = int(nRows)-int(np.round(nRows*(queryPoint[1]-yll)/(yur-yll)))+1
        N = int(distance/fileDx)

        data = []
        try:
            for i in range(6+centerY-N,6+centerY+N+1):
                line = lines[i]
                lineSplit = line.split(' ')[1:]
                lineSplit[-1] = lineSplit[-1].split('\n')[0]
                data.append(lineSplit)
        except:
            return None, None
        data = np.array(data,dtype=np.float32)[:,centerX-N:centerX+N+1]
        
        header = ['ncols        %.0f\n'%(2*N+1),
                  'nrows        %.0f\n'%(2*N+1),
                  'xllcorner    %.12f\n'%(queryPoint[0]-N*fileDx),
                  'yllcorner    %.12f\n'%(yll+(nRows-(centerY+N))*fileDx),
                  'cellsize     %.12f\n'%(fileDx),
                  'NODATA_value %.0f'%(noDataValue)]
        stringHeader = ''
        for line in header:
            stringHeader = stringHeader+line
    else:
        data = None
        stringHeader = None
    return data, stringHeader

def extractListOfCoordinates(lats,lons,distance):
    inDir = 'E:/projects/wildfire-research/farsiteData/'
    names = ['US_DEM2010','US_SLP2010','US_ASP2010','US_140FBFM40','US_140CC',
             'US_140CH','US_140CBH','US_140CBD']
    #names = ['US_140CC','US_140CH','US_140CBH','US_140CBD']
    for name in names:
        filename = inDir+name+'.asc'
        lines, params = getAscFile(filename)
        for lat, lon in zip(lats,lons):
            data, header = getCoordinateData(lines,params,lat,lon)
            if data is not None and header is not None:
                namespace = getNamespace(lat,lon,distance)
                extractedFile = inDir+namespace+'_'+name+'.asc'
                np.savetxt(extractedFile,data, fmt='%.1f', delimiter=' ',newline='\n', header=header,comments='')

def limitAscLimits():
    inDir = 'E:/projects/wildfire-research/farsiteData/'
    names = ['US_DEM2010','US_SLP2010','US_ASP2010','US_140FBFM40']#,'US_140CC',
             #'US_140CH','US_140CBH','US_140CBD']
    #names = ['US_140CC','US_140CH','US_140CBH','US_140CBD']
    limitXN = 30384
    limitYN = 32938
    limitXll = -2362425
    limitYll = 1590585
    limitYur = 2578725
    limitXur = -1450905
    for name in names:
        filename = inDir+name+'.asc'
        newFilename = inDir+name+'_limited.asc'
        lines, params = getAscFile(filename)
        
        nCols = float(lines[0].split(' ')[-1].split('\n')[0])
        nRows = float(lines[1].split(' ')[-1].split('\n')[0])
        xll = float(lines[2].split(' ')[-1].split('\n')[0])
        yll = float(lines[3].split(' ')[-1].split('\n')[0])
        fileDx = float(lines[4].split(' ')[-1].split('\n')[0])
        noDataValue = float(lines[5].split(' ')[-1].split('\n')[0])
        xur = xll + fileDx*nCols
        yur = yll + fileDx*nRows
        
        yOff = int((yur-limitYur)/fileDx)
        xOff = int((xur-limitXur)/fileDx)
        print(yOff,xOff)
        
        header = ['ncols        %.0f\n'%(limitXN),
                  'nrows        %.0f\n'%(limitYN),
                  'xllcorner    %.12f\n'%(limitXll),
                  'yllcorner    %.12f\n'%(limitYll),
                  'cellsize     %.12f\n'%(fileDx),
                  'NODATA_value %.0f\n'%(noDataValue)]
        stringHeader = ''
        for line in header:
            stringHeader = stringHeader+line
        
        with open(newFilename,'w+') as f:
            f.write(stringHeader)
            for i in range(6+yOff,6+yOff+limitYN):
                line = lines[i]
                lineSplit = line.split(' ')[1:]
                lineSplit[-1] = lineSplit[-1].split('\n')[0]
                data = np.char.mod('%.1f',np.array(lineSplit,dtype=np.float32))
                data = data[xOff:xOff+limitXN]
                dataStr = ",".join(data)
                f.write(dataStr+'\n')
        
        #np.savetxt(newFilename,data, fmt='%.1f', delimiter=' ',newline='\n', header=stringHeader,comments='')
        
        
        
        
        
        
        
        
        
        
        
        
        
        #for lat, lon in zip(lats,lons):
        #    data, header = getCoordinateData(lines,params,lat,lon)
        #    if data is not None and header is not None:
        #        namespace = getNamespace(lat,lon,distance)
        #        extractedFile = inDir+namespace+'_'+name+'.asc'
        #        np.savetxt(extractedFile,data, fmt='%.1f', delimiter=' ',newline='\n', header=header,comments='')

def parseLcpHeader(header):
    headerDict = dict()
    headerDict['nX'] = header[1037]; headerDict['nY'] = header[1038]
    headerDict['eastUtm'] = header[1039]; headerDict['westUtm'] = header[1040]
    headerDict['northUtm'] = header[1041]; headerDict['southUtm'] = header[1042]
    headerDict['gridUnits'] = header[1043];
    headerDict['xResol'] = header[1044]; headerDict['yResol'] = header[1045];
    headerDict['eUnits'] = header[1046]; headerDict['sUnits'] = header[1047];
    headerDict['aUnits'] = header[1048]; headerDict['fOptions'] = header[1049];
    headerDict['cUnits'] = header[1050]; headerDict['hUnits'] = header[1051];
    headerDict['bUnits'] = header[1052]; headerDict['pUnits'] = header[1053];
    headerDict['dUnits'] = header[1054]; headerDict['wUnits'] = header[1055];
    headerDict['elevFile'] = header[1056]; headerDict['slopeFile'] = header[1057];
    headerDict['aspectFile'] = header[1058]; headerDict['fuelFile'] = header[1059];
    headerDict['coverFile'] = header[1060]; headerDict['heightFile'] = header[1061];
    headerDict['baseFile'] = header[1062]; headerDict['densityFile'] = header[1063];
    headerDict['duffFile'] = header[1064]; headerDict['woodyFile'] = header[1065];
    headerDict['description'] = header[1066]
    return headerDict


def readLcpFile(filename):
    with open(filename,'rb') as f:
        data = f.read()
        
    dataFormat = '=llldddd'
    for i in range(0,10):
        dataFormat = dataFormat+'lll%.0fl'%(100)
    dataFormat = dataFormat+'llddddlddhhhhhhhhhh256s256s256s256s256s256s256s256s256s256s512s'
    #print(dataFormat)
    los = []
    his = []
    nums = []
    values = []
    names = []
    header = struct.unpack(dataFormat,data[:7316])
    crownFuels = header[0]; groundFuels = header[1]; latitude = header[2];
    loEast = header[3]; hiEast = header[4]
    loNorth = header[5]; hiNorth = header[6]
    #print(crownFuels,groundFuels,latitude,loEast,hiEast,loNorth,hiNorth)
    loElev = header[7]; hiElev = header[8]; numElev = header[9]; elevationValues = header[10:110]; los.append(loElev); his.append(hiElev); nums.append(numElev); values.append(elevationValues); names.append('Elevation')
    loSlope = header[110]; hiSlope = header[111]; numSlope = header[112]; slopeValues = header[113:213]; los.append(loSlope); his.append(hiSlope); nums.append(numSlope); values.append(slopeValues); names.append('Slope')
    loAspect = header[213]; hiAspect = header[214]; numAspect = header[215]; aspectValues = header[216:316]; los.append(loAspect); his.append(hiAspect); nums.append(numAspect); values.append(aspectValues); names.append('Aspect')
    loFuel = header[316]; hiFuel = header[317]; numFuel = header[318]; fuelValues = header[319:419]; los.append(loFuel); his.append(hiFuel); nums.append(numFuel); values.append(fuelValues); names.append('Fuel')
    loCover = header[419]; hiCover = header[420]; numCover = header[421]; coverValues = header[422:522]; los.append(loCover); his.append(hiCover); nums.append(numCover); values.append(coverValues); names.append('Cover')
    #print(loElev,hiElev,numElev,elevationValues)
    #print(loSlope,hiSlope,numSlope,slopeValues)
    #print(loAspect,hiAspect,numAspect,aspectValues)
    #print(loFuel,hiFuel,numFuel,fuelValues)
    #print(loCover,hiCover,numCover,coverValues)
    
    if crownFuels == 21 and groundFuels == 21:
        loHeight = header[522]; hiHeight = header[523]; numHeight = header[524]; heightValues = header[525:625]; los.append(loHeight); his.append(hiHeight); nums.append(numHeight); values.append(heightValues); names.append('Canopy Height')
        loBase = header[625]; hiBase = header[626]; numBase = header[627]; baseValues = header[628:728]; los.append(loBase); his.append(hiBase); nums.append(numBase); values.append(baseValues); names.append('Canopy Base Height')
        loDensity = header[728]; hiDensity = header[729]; numDensity = header[730]; densityValues = header[731:831]; los.append(loDensity); his.append(hiDensity); nums.append(numDensity); values.append(densityValues); names.append('Canopy Density')
        loDuff = header[831]; hiDuff = header[832]; numDuff = header[833]; duffValues = header[834:934]; los.append(loDuff); his.append(hiDuff); nums.append(numDuff); values.append(duffValues); names.append('Duff')
        loWoody = header[934]; hiWoody = header[935]; numWoody = header[936]; woodyValues = header[937:1037]; los.append(loWoody); his.append(hiWoody); nums.append(numWoody); values.append(woodyValues); names.append('Coarse Woody')
        numImgs = 10
    elif crownFuels == 21 and groundFuels == 20:
        loHeight = header[522]; hiHeight = header[523]; numHeight = header[524]; heightValues = header[525:625]; los.append(loHeight); his.append(hiHeight); nums.append(numHeight); values.append(heightValues); names.append('Canopy Height')
        loBase = header[625]; hiBase = header[626]; numBase = header[627]; baseValues = header[628:728]; los.append(loBase); his.append(hiBase); nums.append(numBase); values.append(baseValues); names.append('Canopy Base Height')
        loDensity = header[728]; hiDensity = header[729]; numDensity = header[730]; densityValues = header[731:831]; los.append(loDensity); his.append(hiDensity); nums.append(numDensity); values.append(densityValues); names.append('Canopy Density')
        numImgs = 8
        #print(loHeight,hiHeight,numHeight,heightValues)
        #print(loBase,hiBase,numBase,baseValues)
        #print(loDensity,hiDensity,numDensity,densityValues)
    elif crownFuels == 20 and groundFuels == 21:
        loDuff = header[831]; hiDuff = header[832]; numDuff = header[833]; duffValues = header[834:934]; los.append(loDuff); his.append(hiDuff); nums.append(numDuff); values.append(duffValues); names.append('Duff')
        loWoody = header[934]; hiWoody = header[935]; numWoody = header[936]; woodyValues = header[937:1037]; los.append(loWoody); his.append(hiWoody); nums.append(numWoody); values.append(woodyValues); names.append('Coarse Woody')
        numImgs = 7
    else:
        numImgs = 5
    
    nX = header[1037]; nY = header[1038]
    eastUtm = header[1039]; westUtm = header[1040]
    northUtm = header[1041]; southUtm = header[1042]
    gridUnits = header[1043];
    xResol = header[1044]; yResol = header[1045];
    eUnits = header[1046]; sUnits = header[1047];
    aUnits = header[1048]; fOptions = header[1049];
    cUnits = header[1050]; hUnits = header[1051];
    bUnits = header[1052]; pUnits = header[1053];
    dUnits = header[1054]; wUnits = header[1055];
    elevFile = header[1056]; slopeFile = header[1057];
    aspectFile = header[1058]; fuelFile = header[1059];
    coverFile = header[1060]; heightFile = header[1061];
    baseFile = header[1062]; densityFile = header[1063];
    duffFile = header[1064]; woodyFile = header[1065];
    description = header[1066]
    
    #print(eastUtm,westUtm,northUtm,southUtm,gridUnits,xResol,yResol)
    #print(eUnits,sUnits,aUnits,fOptions,cUnits,hUnits,bUnits,pUnits,dUnits,wUnits)
    
    bodyFormat = ''
    for i in range(0,numImgs):
        bodyFormat = bodyFormat+'%.0fh'%(nX*nY)
        
    body = np.array(struct.unpack(bodyFormat,data[7316:]))
    
    imgs = np.split(body,numImgs)
    
    for i in range(0,numImgs):
        img = body[i::numImgs]
        img = np.array(img,dtype=np.float32)
        img[img == -9999] = np.nan
        imgs[i] = np.reshape(img,(nY,nX),order='C')
    return imgs, names, header

def checkLcpFile(lcpFile=None,rawFiles=None,case=0):
    if lcpFile is None or rawFiles is None:
        if case == 0: # all data files
            rawFiles = ['C:/FARSITE 4/Ashley/input/ash_elev.asc',
            'C:/FARSITE 4/Ashley/input/ash_slope.asc',
            'C:/FARSITE 4/Ashley/input/ash_aspect.asc',
            'C:/FARSITE 4/Ashley/input/ash_fuel.asc',
            'C:/FARSITE 4/Ashley/input/ash_canopy.asc',
            'C:/FARSITE 4/Ashley/input/ash_height.asc',
            'C:/FARSITE 4/Ashley/input/ash_cbh.asc',
            'C:/FARSITE 4/Ashley/input/ash_cbd.asc',
            'C:/FARSITE 4/Ashley/input/ash_duff.asc',
            'C:/FARSITE 4/Ashley/input/ash_cwd.asc']
            lcpFile = 'C:/FARSITE 4/Ashley/input/ashleyFull.lcp'
        elif case == 1: # first five data files
            rawFiles = ['C:/FARSITE 4/Ashley/input/ash_elev.asc',
            'C:/FARSITE 4/Ashley/input/ash_slope.asc',
            'C:/FARSITE 4/Ashley/input/ash_aspect.asc',
            'C:/FARSITE 4/Ashley/input/ash_fuel.asc',
            'C:/FARSITE 4/Ashley/input/ash_canopy.asc']
            lcpFile = 'C:/FARSITE 4/Ashley/input/ashleyFive.lcp'
        elif case == 2: # first five + 3 canopy files
            rawFiles = ['C:/FARSITE 4/Ashley/input/ash_elev.asc',
            'C:/FARSITE 4/Ashley/input/ash_slope.asc',
            'C:/FARSITE 4/Ashley/input/ash_aspect.asc',
            'C:/FARSITE 4/Ashley/input/ash_fuel.asc',
            'C:/FARSITE 4/Ashley/input/ash_canopy.asc',
            'C:/FARSITE 4/Ashley/input/ash_height.asc',
            'C:/FARSITE 4/Ashley/input/ash_cbh.asc',
            'C:/FARSITE 4/Ashley/input/ash_cbd.asc']
            lcpFile = 'C:/FARSITE 4/Ashley/input/ashleyCanopy.lcp'
        elif case == 3: # first five + 2 ground files
            rawFiles = ['C:/FARSITE 4/Ashley/input/ash_elev.asc',
            'C:/FARSITE 4/Ashley/input/ash_slope.asc',
            'C:/FARSITE 4/Ashley/input/ash_aspect.asc',
            'C:/FARSITE 4/Ashley/input/ash_fuel.asc',
            'C:/FARSITE 4/Ashley/input/ash_canopy.asc',
            'C:/FARSITE 4/Ashley/input/ash_duff.asc',
            'C:/FARSITE 4/Ashley/input/ash_cwd.asc']
            lcpFile = 'C:/FARSITE 4/Ashley/input/ashleyGround.lcp'
    imgs, names = readLcpFile(lcpFile)
    for i in range(0,len(imgs)):
        plt.figure(figsize=(20,4))
        plt.subplot(1,2,1)
        plt.imshow(imgs[i],cmap='jet'); plt.colorbar();
        plt.subplot(1,2,2)
        imgRaw, headerRaw = readFullAscFile(rawFiles[i])
        imgRaw[imgRaw<-1000] = np.nan
        plt.imshow(imgRaw,cmap='jet'); plt.colorbar();

def generateLcpFile(lat,lon,distance,indir,
                    gridUnits=0,
                    eUnits=0,
                    sUnits=0,
                    aUnits=2,
                    fOptions=0,
                    cUnits=1,
                    hUnits=3,
                    bUnits=3,
                    pUnits=1,
                    dUnits=0,
                    wUnits=0):
    
    #datas, headers, names = queryAsciiFiles(lats,lons,distance)
    if checkPoint([[lon,lat]]):
        datas, headers, names = queryAsciiFiles(lat,lon,distance,skipMissing=True)
        if datas is not None and headers is not None and names is not None:
            sH = [float(x.split(' ')[-1]) for x in headers[0].split('\n')]
            
            crownFuels = 21 if len(datas[1]) > 0 else 20
            groundFuels = 21 if len(datas[2]) > 0 else 20
            latitude = lat
            
            nCols = sH[0]
            nRows = sH[1]
            westUtm = sH[2]
            southUtm = sH[3]
            xResol = sH[4]
            yResol = sH[4]
            
            eastUtm = westUtm + xResol*nCols
            northUtm = southUtm + yResol*nRows
            
            loEast = westUtm - round(westUtm,-3)
            hiEast = loEast + xResol*nCols
            
            loNorth = southUtm - round(southUtm,-3)
            hiNorth = loNorth + yResol*nRows
            
            #print(crownFuels,groundFuels,latitude,loEast,hiEast,loNorth,hiNorth)
            dataFormat = '=llldddd'
            header = struct.pack(dataFormat,crownFuels,groundFuels,int(latitude),loEast,hiEast,loNorth,hiNorth)
        
            for i in range(0,5):
                data = datas[0][i]
                name = names[0][i]
                if 'US_ASP2010' in name:
                    data[data < 0] = -9999
                packed, lo, hi, num, values = getHeaderInfo(data)
                header = header + packed
            if crownFuels == 21:
                for data in datas[1]:
                    packed, lo, hi, num, values = getHeaderInfo(data)
                    header = header + packed
            else:
                header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
                header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
                header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
            if groundFuels == 21:
                for data in datas[2]:
                    packed, lo, hi, num, values = getHeaderInfo(data)
                    header = header + packed
            else:
                header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
                header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
        
            header = header+struct.pack('=ll',int(nCols),int(nRows))
            header = header+struct.pack('=ddddldd',eastUtm,westUtm,northUtm,southUtm,gridUnits,xResol,yResol)
            header = header+struct.pack('=hhhhhhhhhh',eUnits,sUnits,aUnits,fOptions,cUnits,hUnits,bUnits,pUnits,dUnits,wUnits)
            
            #print("Base five names:")
            for name in names[0]:
                #print(name)
                header = header + struct.pack('=256s',str.encode(name))
            if crownFuels == 21:
                #print("crownFuel names:")
                for name in names[1]:
                    #print(name)
                    header = header + struct.pack('=256s',str.encode(name))
            else:
                header = header + struct.pack('=256s',str.encode(''))
                header = header + struct.pack('=256s',str.encode(''))
                header = header + struct.pack('=256s',str.encode(''))
            if groundFuels == 21:
                #print("groundFuel names:")
                for name in names[2]:
                    #print(name)
                    header = header + struct.pack('=256s',str.encode(name))
            else:
                header = header + struct.pack('=256s',str.encode(''))
                header = header + struct.pack('=256s',str.encode(''))
            description = 'Automatically generated. lat = %.4f, lon = %.4f, dist = %.0f'%(lon,lat,distance)
            header = header + struct.pack('=512s',str.encode(description))
            
            #print(len(header))
            
            imgSize = int(nCols*nRows)
            numImgs = int(len(datas[0])+len(datas[1])+len(datas[2]))
            totalSize = int(imgSize*numImgs)
            
            allImgs = np.zeros((totalSize))
            ct = 0
            for data in datas[0]:
                allImgs[ct::numImgs] = np.reshape(data,(imgSize))
                ct = ct+1
            for data in datas[1]:
                allImgs[ct::numImgs] = np.reshape(data,(imgSize))
                ct = ct+1
            for data in datas[2]:
                allImgs[ct::numImgs] = np.reshape(data,(imgSize))
                ct = ct+1
            allImgs = np.array(allImgs,dtype=np.int16)
            dataFormat = '=%.0fh'%(totalSize)
            
            body = struct.pack(dataFormat,*allImgs)
            #print(len(body),totalSize)
            
            #print(len(header)+len(body))
            namespace = "%.4f_%.4f_%.0f"%(lon,lat,distance)
            namespace = namespace.replace('-','n')
            namespace = namespace.replace('.','-')
            with open(indir+namespace+'.LCP','wb') as f:
                f.write(header+body)
            
            
            return datas, headers, names
        else:
            return None, None, None



def generateLcpFileTif(indir,names,outname,
                    gridUnits=0,
                    eUnits=0,
                    sUnits=0,
                    aUnits=2,
                    fOptions=0,
                    cUnits=1,
                    hUnits=3,
                    bUnits=3,
                    pUnits=1,
                    dUnits=0,
                    wUnits=0):
    
    datas = []
    headers = []
    for name in names:
        print("Reading %s"%(name))
        file = gdal.Open(indir+name+'.tif')
        band = file.GetRasterBand(1)
        noDataValue = band.GetNoDataValue() if band.GetNoDataValue() is not None else -9999
        nCols = file.RasterXSize
        nRows = file.RasterYSize
        xll = file.GetGeoTransform()[0]
        yll = file.GetGeoTransform()[3]+file.GetGeoTransform()[5]*file.RasterYSize
        fileDx = file.GetGeoTransform()[1]
        
        header = ['ncols        %.0f\n'%(nCols),
          'nrows        %.0f\n'%(nRows),
          'xllcorner    %.12f\n'%(xll),
          'yllcorner    %.12f\n'%(yll),
          'cellsize     %.12f\n'%(fileDx),
          'NODATA_value %.0f'%(noDataValue)]
        stringHeader = ''
        for line in header:
            stringHeader = stringHeader+line
        headers.append(stringHeader)
        datas.append(np.reshape(band.ReadAsArray(),(nCols*nRows,)))
        
    #assert False, "Stopped"
    Tf, iTf = getTransformFromMtoD()
    centerCoordinates = Tf.TransformPoints([[xll+fileDx*nCols/2,yll+fileDx*nRows/2]])[0][:-1]
    
    #datas, headers, names = queryAsciiFiles(lat,lon,distance,skipMissing=True)
    #if datas is not None and headers is not None and names is not None:
    sH = [float(x.split(' ')[-1]) for x in headers[0].split('\n')]
    
    crownFuels = 21 if len(datas) == 8 or len(datas) == 10 else 20
    groundFuels = 21 if len(datas) == 7 or len(datas) == 10 else 20
    print(crownFuels,groundFuels)
    latitude = centerCoordinates[1]
    
    nCols = sH[0]
    nRows = sH[1]
    westUtm = sH[2]
    southUtm = sH[3]
    xResol = sH[4]
    yResol = sH[4]
    
    eastUtm = westUtm + xResol*nCols
    northUtm = southUtm + yResol*nRows
    
    loEast = westUtm - round(westUtm,-3)
    hiEast = loEast + xResol*nCols
    
    loNorth = southUtm - round(southUtm,-3)
    hiNorth = loNorth + yResol*nRows
    
    #print(crownFuels,groundFuels,latitude,loEast,hiEast,loNorth,hiNorth)
    dataFormat = '=llldddd'
    header = struct.pack(dataFormat,crownFuels,groundFuels,int(latitude),loEast,hiEast,loNorth,hiNorth)
    
    for i in range(0,5):
        data = datas[0][i]
        name = names[0][i]
        if 'US_ASP2010' in name:
            data[data < 0] = -9999
        packed, lo, hi, num, values = getHeaderInfo(data)
        header = header + packed
    if crownFuels == 21:
        for j in range(i,i+3):
            packed, lo, hi, num, values = getHeaderInfo(datas[j])
            header = header + packed
    else:
        header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
        header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
        header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
        j = i
    if groundFuels == 21:
        for i in range(j,j+2):
            packed, lo, hi, num, values = getHeaderInfo(datas[i])
            header = header + packed
    else:
        header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))
        header = header + struct.pack('=lll100l',0,0,0,*np.array(np.zeros((100,)),dtype=np.int16))

    header = header+struct.pack('=ll',int(nCols),int(nRows))
    header = header+struct.pack('=ddddldd',eastUtm,westUtm,northUtm,southUtm,gridUnits,xResol,yResol)
    header = header+struct.pack('=hhhhhhhhhh',eUnits,sUnits,aUnits,fOptions,cUnits,hUnits,bUnits,pUnits,dUnits,wUnits)
    
    #print("Base five names:")
    for name in names[0]:
        #print(name)
        header = header + struct.pack('=256s',str.encode(name))
    if crownFuels == 21:
        #print("crownFuel names:")
        for name in names[1]:
            #print(name)
            header = header + struct.pack('=256s',str.encode(name))
    else:
        header = header + struct.pack('=256s',str.encode(''))
        header = header + struct.pack('=256s',str.encode(''))
        header = header + struct.pack('=256s',str.encode(''))
    if groundFuels == 21:
        #print("groundFuel names:")
        for name in names[2]:
            #print(name)
            header = header + struct.pack('=256s',str.encode(name))
    else:
        header = header + struct.pack('=256s',str.encode(''))
        header = header + struct.pack('=256s',str.encode(''))
    description = 'Automatically generated california landscape file.'
    header = header + struct.pack('=512s',str.encode(description))
    
    #print(len(header))
    
    imgSize = int(nCols*nRows)
    numImgs = int(len(datas[0])+len(datas[1])+len(datas[2]))
    totalSize = int(imgSize*numImgs)
    print("Starting to write binary file.")
    with open(outname,'wb') as f:
        f.write(header)
        for i in range(0,totalSize):
            print(i)
            for data in datas:
                tmp = struct.pack('=h',int(data[i]))
                f.write(tmp)
    
    #allImgs = np.zeros((totalSize))
    #ct = 0
    #for data in datas:
    #    allImgs[ct::numImgs] = np.reshape(data,(imgSize))
    #    ct = ct+1
    #allImgs = np.array(allImgs,dtype=np.int16)
    
    
    #body = struct.pack(dataFormat,*allImgs)
    
    #with open(outname,'wb') as f:
    #    f.write(header+body)
        
    return datas, headers, names






def generateListLcpFiles(inDir):
    files = glob.glob(inDir+'*US_DEM2010.asc')
    for i in range(0,len(files)):#file in files:
        file = files[i]
        fSplit = file.split('\\')[1].split('_')
        
        lon = fSplit[0]
        lat = fSplit[1]
        distance = float(fSplit[2])
        if lon[0] == 'n':
            lon = -1*(float(lon.split('-')[0][1:])+float(lon.split('-')[1])/10000)
        else:
            lon = (float(lon.split('-')[0])+float(lon.split('-')[1])/10000)
        if lat[0] == 'n':
            lat = -1*(float(lat.split('-')[0][1:])+float(lat.split('-')[1])/10000)
        else:
            lat = (float(lat.split('-')[0])+float(lat.split('-')[1])/10000)
        try:
            datas, headers, names = generateLcpFile(lat,lon,distance,inDir)
        except:
            pass
    

def getHeaderInfo(data):
    _,idx = np.unique(data.flatten(),return_index=True)
    values = data.flatten()[np.sort(idx)]

    if len(values) > 100:
        values = values[0:100]
        values[-1] = data.flatten()[-1]
        num = -1
    else:
        num = len(values)
        tmpData = np.zeros((100,))
        tmpData[1:num+1] = np.sort(values)
        values = tmpData
    values = np.array(values,dtype=np.int16)
    lo = int(data[data>-9999].min())
    hi = int(data.max())
    
    header = struct.pack('=lll100l',lo,hi,num,*values)
    return header, lo, hi, num, values
    

def checkPoint(query,polygon=[[-125,42],[-122,34],[-112,36],[-114.5,44]]):
    path = mpltPath.Path(polygon)
    inside = path.contains_points(query)[0]
    return inside






def runFarsite(commandFile):
    dockerStart = 'docker run -it -v E:\\projects\\wildfire-research\\farsite\\:/commonDir/ farsite'
    dockerCmd = './commonDir/farsite/src/TestFARSITE %s'%(commandFile)
    
    p = subprocess.Popen('winpty '+dockerStart+' '+dockerCmd,shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE)
    p_status = p.wait()

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
    startMin = int(round((startTime-startHour)*60,0))
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
    print(string)
    
    return params

def getLcpElevation(file):
    imgs, names, header = readLcpFile(file)
    elevation = np.median(imgs[0])
    return elevation

def makeCenterIgnition(file,N=5):
    imgs, names, header = readLcpFile(file)
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
    #imgFile = 'E:/projects/wildfire-research/landfireData/US_140FBFM40/1/US_140FBFM40.tif'
    #imgFile = 'E:/projects/wildfire-research/landfireData/US_140FBFM40/1/test2.tif'
    #data, lat, lon = readImgFile(imgFile)
    #plt.contourf(lon,lat,data)
    
    
    #inDirs = ['E:/projects/wildfire-research/landfireData/US_140FBFM40/1/',
    #          'E:/projects/wildfire-research/landfireData/US_140FBFM40/2/',
    #          'E:/projects/wildfire-research/landfireData/US_140FBFM40/3/',
    #          'E:/projects/wildfire-research/landfireData/US_140FBFM40/4/']
    #outDir = 'E:/projects/wildfire-research/farsiteData/fireModel40/'
    #namespace = 'US_140FBFM40'
    
    #inDirs = ['E:/projects/wildfire-research/landfireData/US_DEM2010/1/',
    #          'E:/projects/wildfire-research/landfireData/US_DEM2010/2/',
    #          'E:/projects/wildfire-research/landfireData/US_DEM2010/3/',
    #          'E:/projects/wildfire-research/landfireData/US_DEM2010/4/',
    #          'E:/projects/wildfire-research/landfireData/US_DEM2010/5/',
    #          'E:/projects/wildfire-research/landfireData/US_DEM2010/6/',
    #          'E:/projects/wildfire-research/landfireData/US_DEM2010/7/']
    #outDir = 'E:/projects/wildfire-research/farsiteData/dem2010/'
    #namespace = 'US_DEM2010'
    
    #inDirs = ['E:/projects/wildfire-research/landfireData/US_140FBFM40/1/']
    #namespace = 'US_140FBFM40'
    
    #splitImages(inDirs,namespace)
    #Tf, iTf = getTransformFromMtoD()
    
    lats = np.random.random((1000,))*(44-34)+34
    lons = np.random.random((1000,))*(-114+121)-121
    
    distance = 25000
    
    #extractListOfCoordinates(lats,lons,distance)
    #limitAscLimits()
    indir = 'E:/projects/wildfire-research/farsiteData/'
    generateListLcpFiles(indir)
    
    #indir = "E:/projects/wildfire-research/landfireData/Processed/"
    #names = ['US_DEM2010','US_SLP2010','US_ASP2010','US_140FBFM40',
    #         'US_140CC','US_140CH','US_140CBH','US_140CBD']
    #namespace = indir+'california.LCP'
    #generateLcpFileTif(indir,names,namespace)

    #names = ['US_140CC','US_140CH','US_140CBH','US_140CBD']
    
    
    
    """
    commandFile = 'commonDir/farsite/example/Panther/runPanther.txt'
    inDir = 'E:/projects/wildfire-research/farsite/data/'
    cDir = 'commonDir/data/'
    #namespace = 'n117-9343_36-5782_3000'
    namespace = 'n114-0177_38-3883_25000'
    totalDays = 5
    lcpFile = namespace+'.LCP'
    inputFile = namespace+'.input'
    igniteFile = namespace+'_ignite.SHP'
    outputFile = namespace+'_out'
    cmdFile = inDir+'toRun.txt'
    cmdFileDocker = cDir+'toRun.txt'
    elevation = getLcpElevation(inDir+lcpFile)
    ignitionShape = makeCenterIgnition(inDir+lcpFile)

    
    fileName = inDir+namespace
    params = generateFarsiteInput(fileName,elevation,cDir+igniteFile,totalDays=totalDays)
    generateCmdFile([lcpFile],[inputFile],[igniteFile],[outputFile],cmdFile)
    
    runFarsite(cmdFileDocker)
    """
    
    
    #dataOut = gpd.GeoDataFrame.from_file(inDir+outputFile+'_Perimeters.shp')
    #dataIn = gpd.GeoDataFrame.from_file(inDir+igniteFile)
    
    
    
    
    #filename = 'E:/projects/wildfire-research/farsite/data/californiaRaw.LCP'
    #inDir = 'E:/projects/wildfire-research/farsite/data/'
    
    #print(checkPoint([[lon,lat]]))
    #datas, headers, names = generateLcpFile(lats,lons,distance,inDir)
    
    