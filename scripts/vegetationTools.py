# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:41:16 2018

@author: JHodges
"""

import matplotlib.pyplot as plt
import osgeo.gdal
import numpy as np
import struct
import matplotlib.path as mpltPath
from collections import defaultdict
import os

def getHistogram(file):
    img = osgeo.gdal.Open(file)
    band = np.array(img.ReadAsArray(),dtype=np.float32)
    return band

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
    
    los = []
    his = []
    nums = []
    values = []
    names = []
    header = struct.unpack(dataFormat,data[:7316])
    header2 = parseLcpHeader(header)
    
    crownFuels = header[0]; groundFuels = header[1]; latitude = header[2];
    loEast = header[3]; hiEast = header[4]
    loNorth = header[5]; hiNorth = header[6]
    
    loElev = header[7]; hiElev = header[8]; numElev = header[9]; elevationValues = header[10:110]; los.append(loElev); his.append(hiElev); nums.append(numElev); values.append(elevationValues); names.append('Elevation')
    loSlope = header[110]; hiSlope = header[111]; numSlope = header[112]; slopeValues = header[113:213]; los.append(loSlope); his.append(hiSlope); nums.append(numSlope); values.append(slopeValues); names.append('Slope')
    loAspect = header[213]; hiAspect = header[214]; numAspect = header[215]; aspectValues = header[216:316]; los.append(loAspect); his.append(hiAspect); nums.append(numAspect); values.append(aspectValues); names.append('Aspect')
    loFuel = header[316]; hiFuel = header[317]; numFuel = header[318]; fuelValues = header[319:419]; los.append(loFuel); his.append(hiFuel); nums.append(numFuel); values.append(fuelValues); names.append('Fuel')
    loCover = header[419]; hiCover = header[420]; numCover = header[421]; coverValues = header[422:522]; los.append(loCover); his.append(hiCover); nums.append(numCover); values.append(coverValues); names.append('Cover')
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

def checkPoint(query,polygon=[[-125,42],[-122,34],[-112,36],[-114.5,44]]):
    path = mpltPath.Path(polygon)
    inside = path.contains_points(query)[0]
    return inside

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

def generateLcpFile_v2(datas, headers, names, lat=43.0,
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
        description = 'Automatically generated.'
        header = header + struct.pack('=512s',str.encode(description))
        
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
        
        return header+body

def queryLandfireFile(file, queryPoint=None, resolution=1500, buildMesh=False):
    dataset = osgeo.gdal.Open(file)
    band = dataset.GetRasterBand(1)
    
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    
    transform = dataset.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    
    xMax = xOrigin + pixelWidth*cols
    yMax = yOrigin + pixelHeight*rows
    
    if queryPoint is None:
        queryPoint = ((xOrigin+xMax)/2, (yOrigin+yMax)/2)
    
    x = np.linspace(xOrigin, xMax, int((xMax-xOrigin)/pixelWidth+1))
    y = np.linspace(yOrigin, yMax, int((yMax-yOrigin)/pixelHeight+1))
    
    xind = np.argmin(abs(x-queryPoint[0]))
    yind = np.argmin(abs(y-queryPoint[1]))

    xind01 = int(xind-int((resolution/2)))
    yind01 = int(yind-int((resolution/2)))
    
    data = band.ReadAsArray(xind01, yind01, resolution, resolution)
    
    if (data.dtype == np.int16):
        data = np.array(data, np.float)
        data[data == -9999] = np.nan
    else:
        data = np.array(data, np.float)
    
    noDataValue = band.GetNoDataValue()
    data[np.isclose(data, noDataValue)] = np.nan
    
    if buildMesh:
        xind02 = int(xind+int((resolution/2)))
        yind02 = int(yind+int((resolution/2)))
        xData = x[xind01:xind02]
        yData = y[yind01:yind02]
        xGrid, yGrid = np.meshgrid(xData, yData)
    else:
        xGrid = False
        yGrid = False
    return data, xGrid, yGrid

def getLandfireWkt():
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
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]]]"""
    cs = osgeo.gdal.osr.SpatialReference()
    cs.ImportFromWkt(nad83_wkt)
    return cs

def getModisWkt():
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
    
    cs = osgeo.gdal.osr.SpatialReference()
    cs.ImportFromWkt(wgs84_wkt)
    return cs

def getTransformLandfireToModis():
    old_cs = getLandfireWkt()
    new_cs = getModisWkt()
    transform = osgeo.gdal.osr.CoordinateTransformation(old_cs,new_cs)
    return transform

def getTransformModisToLandfire():
    new_cs = getLandfireWkt()
    old_cs = getModisWkt()
    transform = osgeo.gdal.osr.CoordinateTransformation(old_cs,new_cs)
    return transform

if __name__ == "__main__":
    ###########################################################################
    # USER INPUTS
    ###########################################################################
    # Data naming
    prefix = "us_"
    yearStr = "130"
    indir = "G:\\WildfireResearch\\landfireData\\"
    names = ['Canopy Base Height', 'Canopy Base Density', 'Canopy Height',
             'Canopy Cover', 'Fuel Model 13', 'Fuel Model 40',
             'Aspect', 'Elevation', 'Slope']
    ids = ['cbh', 'cbd', 'ch', 'cc', 'fbfm13', 'fbfm40', 'asp_2016', 'dem_2016', 'slpd_2016']
    
    # Ouptut options
    resolution = 1000
    #queryPoint = np.array([-2144805, 1565987, 0.0], dtype=np.float) # Same point in meters
    queryPoint = np.array([-121.896571, 38.030451, 0.0], dtype=np.float) # Same point in degrees
    queryPoint = np.array([-116.680072, 32.808536, 0.0], dtype=np.float) # Same point in degrees
    queryPointUnits = 'degree' # either degree or meter
    axisType = 'kilometer' # either degree, meter, or kilometer
    displayType = 'contour' # either contour or image
    ###########################################################################
    # END USER INPUTS
    ###########################################################################
    
    # Example transformation from LANDFIRE coordinate system to standard latitude/longitude
    if queryPointUnits == 'meter':
        transform1 = getTransformLandfireToModis()
        qp_lon, qp_lat, _ = transform1.TransformPoint(queryPoint[0], queryPoint[1], queryPoint[2])
        qp = queryPoint
    elif queryPointUnits == 'degree':
        transform1 = getTransformLandfireToModis()
        transform2 = getTransformModisToLandfire()
        qp_x, qp_y, qp_z = transform2.TransformPoint(queryPoint[0], queryPoint[1], queryPoint[2])
        qp = np.array([qp_x, qp_y, qp_z], dtype=np.float)
    
    # Initialize parameters for looping over data queries
    datas = defaultdict(bool)
    buildMesh = True
    
    # Read rest of data query
    for i in range(0, len(ids)):
        did = ids[i]
        file = "%s%s%s%sGrid%s%s%s%sw001000.adf"%(indir, prefix, did, os.sep, os.sep, prefix, did, os.sep)
        if not os.path.exists(file):
            file = "%s%s%s%s%sGrid2%s%s%s%s%sw001000.adf"%(indir, prefix, yearStr, did, os.sep, os.sep, prefix, yearStr, did, os.sep)
        if not os.path.exists(file):
            print("Unable to find file: %s"%(file))
        print(file)
        data, tmp1, tmp2 = queryLandfireFile(file, queryPoint=qp, resolution=resolution, buildMesh=buildMesh)
        if buildMesh:
            # Only build grid on first data query
            (xGrid, yGrid) = (tmp1, tmp2)
            buildMesh = False
        datas[names[i]] = data
    
    # Build latitude and longitude arrays if plotting in degrees
    if axisType == 'degree':
        sz = xGrid.shape
        xGrid_rs = np.reshape(xGrid,(xGrid.shape[0]*xGrid.shape[1],))
        yGrid_rs = np.reshape(yGrid,(yGrid.shape[0]*yGrid.shape[1],))
        points = np.array([xGrid_rs,yGrid_rs]).T
        latlong = np.array(transform1.TransformPoints(points))
        lat = np.reshape(latlong[:,1],(sz[0],sz[1]))
        lon = np.reshape(latlong[:,0],(sz[0],sz[1]))
    
    # Visualize data
    totR = 3
    totC = 3
    fs = 16
    fig, ax = plt.subplots(totR, totC, figsize=(18,12))
    for i in range(0, len(ids)):
        plotC = (i % totC)
        plotR = int((i-(i % totC))/totC)
        
        if (axisType == 'degree') and (displayType == 'contour'):
            ax[plotR][plotC].set_xlabel('Longitude $\mathrm{(^{\circ})}$', fontsize=fs)
            ax[plotR][plotC].set_ylabel('Latitude $\mathrm{(^{\circ})}$', fontsize=fs)
            (xPlot, yPlot, decimals) = (lon, lat, 2)
        elif axisType == 'meter' and (displayType == 'contour'):
            ax[plotR][plotC].set_xlabel('X-Position (m)', fontsize=fs)
            ax[plotR][plotC].set_ylabel('Y-Position (m)', fontsize=fs)
            (xPlot, yPlot, decimals) = (xGrid, yGrid, 0)
        elif axisType == 'kilometer' and (displayType == 'contour'):
            ax[plotR][plotC].set_xlabel('X-Position (km)', fontsize=fs)
            ax[plotR][plotC].set_ylabel('Y-Position (km)', fontsize=fs)
            (xPlot, yPlot, decimals) = (xGrid/1000, yGrid/1000, 1)
        else:
            (xPlot, yPlot, decimals) = (xGrid, yGrid, 0)
        (xmn, xmx, xavg) = (xPlot.min(), xPlot.max(), (xPlot.max() + xPlot.min())/2)
        (ymn, ymx, yavg) = (yPlot.min(), yPlot.max(), (yPlot.max() + yPlot.min())/2)
        
        (xmn, xmx, xavg) = (np.round(xmn, decimals), np.round(xmx, decimals), np.round(xavg, decimals))
        (ymn, ymx, yavg) = (np.round(ymn, decimals), np.round(ymx, decimals), np.round(yavg, decimals))
        
        if displayType == 'contour':
            im = ax[plotR][plotC].contourf(xPlot, yPlot, datas[names[i]], 25, cmap='jet_r')
            ax[plotR][plotC].set_xticks([xmn, xavg, xmx])
            ax[plotR][plotC].set_yticks([ymn, yavg, ymx])
            ax[plotR][plotC].tick_params(labelsize=fs)
            cbar = plt.colorbar(im, ax=ax[plotR][plotC])
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=fs)
        elif displayType == 'image':
            im = ax[plotR][plotC].imshow(datas[names[i]], cmap='jet_r')
        ax[plotR][plotC].set_title(names[i], fontsize=fs)
        ax[plotR][plotC].ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.show()
    
    # Build LCP file for FARSITE
    dictNames1 = ['Elevation', 'Slope', 'Aspect', 'Fuel Model 40', 'Canopy Cover']
    lcpNames1 = ['Elevation', 'Slope', 'Aspect', 'Fuel', 'Cover']
    
    dictNames2 = ['Canopy Height', 'Canopy Base Height', 'Canopy Base Density']
    lcpNames2 = ['Canopy Height', 'Canopy Base Height', 'Canopy Density']
    
    imgs1 = []
    for dictName, lcpName in zip(dictNames1, lcpNames1):
        imgs1.append(np.array(datas[dictName], dtype=np.int16))
    imgs2 = []
    for dictName, lcpName in zip(dictNames2, lcpNames2):
        imgs2.append(np.array(datas[dictName], dtype=np.int16))
    
    nCols = imgs1[0].shape[1]
    nRows = imgs1[0].shape[0]
    xll = xGrid.min()
    yll = yGrid.min()
    fileDx = xGrid[0,1]-xGrid[0,0]
    
    headers = []
    for name in lcpNames1:
        header = ['ncols        %.0f\n'%(nCols),
                  'nrows        %.0f\n'%(nRows),
                  'xllcorner    %.12f\n'%(xll),
                  'yllcorner    %.12f\n'%(yll),
                  'cellsize     %.12f\n'%(fileDx),
                  'NODATA_value %.0f'%(0)]
        stringHeader = ''
        for line in header:
            stringHeader = stringHeader+line
        headers.append(stringHeader)
    
    text = generateLcpFile_v2([imgs1, imgs2, []], headers, [lcpNames1, lcpNames2, []])
    
    with open('test.LCP', 'wb') as f:
        f.write(text)
    
    imgs, names, header = readLcpFile("test.LCP")