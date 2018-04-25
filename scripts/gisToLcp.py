# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:48:18 2018

@author: JHodges
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import gdal
import skimage.transform as skt
import numpy as np
from generate_dataset import GriddedMeasurement
import scipy.interpolate as scpi

def modifyAscFile(inAscFile,outAscFile):

    with open(inAscFile,'r') as f:
        content = f.readlines()
    
    header = content[0:5]

    '''
    aspectHeader
    ['ncols        31644\n',
     'nrows        52342\n',
     'xllcorner    -2362845.000000000000\n',
     'yllcorner    1056075.000000000000\n',
     'cellsize     30.000000000000\n']
    
    elevationHeader
    ['ncols        31644\n',
     'nrows        52342\n',
     'xllcorner    -2362845.000000000000\n',
     'yllcorner    1056075.000000000000\n',
     'cellsize     30.000000000000\n']
    
    slopeHeader
    ['ncols        31644\n',
     'nrows        52342\n',
     'xllcorner    -2362845.000000000000\n',
     'yllcorner    1056075.000000000000\n',
     'cellsize     30.000000000000\n']
    
    fuelHeader
    ['ncols        27530\n',
     'nrows        52342\n',
     'xllcorner    -2239425.000000000000\n',
     'yllcorner    1056075.000000000000\n',
     'cellsize     30.000000000000\n']
    
    canopyHeader
    ['ncols        27530\n',
     'nrows        52342\n',
     'xllcorner    -2239425.000000000000\n',
     'yllcorner    1056075.000000000000\n',
     'cellsize     30.000000000000\n']
    '''
    
    cols = float(header[0].split('\n')[0].split(' ')[-1])
    rows = float(header[1].split('\n')[0].split(' ')[-1])
    Xll = float(header[2].split('\n')[0].split(' ')[-1])
    Yll = float(header[3].split('\n')[0].split(' ')[-1])
    CellSize = float(header[4].split('\n')[0].split(' ')[-1])

    
    
    for i in range(5,len(content)):
        tmp = content[i].split(' ')[1:][int(cols-27530-1):]
        tmp2 = ' '.join(tmp)
        content[i] = tmp2
    content[0]='ncols        27530\n'
    content[1]='nrows        52342\n'
    content[2]='xllcorner    -2239425.000000000000\n'
    content[3]='yllcorner    1056075.000000000000\n'
    content[4]='cellsize     30.000000000000\n'
    with open(outAscFile,'w') as f:
        for line in content:
            f.write(line)
    
    print(len(content[6].split(' ')))
    
    


def readModifyAscFile(ascFile):
    img = np.loadtxt(ascFile,skiprows=5)
    """
    with open(outAscFile,'r') as f:
        content = f.readlines()
    header = content[0:5]
    cols = float(header[0].split('\n')[0].split(' ')[-1])
    for i in range(5,len(content)):
        tmp = content[i].split(' ')
        tmp[-1] = tmp[-1].split('\n')[0]
        content[i] = tmp
    """
    return img

def readImgFile2(imgFile):
    img = gdal.Open(imgFile)
    
    band = np.array(img.ReadAsArray(),dtype=np.float32)
    band[band<0] = np.nan
    
    return band

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
    
    ds = 15
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

    data, lat, lon = gridAndResample(bandDs,lat,lon)

    data2 = GriddedMeasurement(None,lat,lon,data,'FuelModel')
    data2.dataName = 'FuelModel'

    return data2

def gridAndResample(data,lat,lon,
                    lat_lmt = [30,44],
                    lon_lmt = [-126,-112],
                    pxPerDegree = 120,
                    ds=1,
                    method='nearest'):
    ''' This function will resample the raw Level 2 data from the swath grid
    to the custom Level 3 grid.
    '''

    pts = np.zeros((lat.shape[0]*lat.shape[1],2))    
    pts[:,0] = np.reshape(lat,(lat.shape[0]*lat.shape[1],))
    pts[:,1] = np.reshape(lon,(lon.shape[0]*lon.shape[1],))    
    
    dataRs = np.reshape(data,(data.shape[0]*data.shape[1],))
    
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
    if len(data) > 0:
        data_lnsp = scpi.griddata(pts[0::ds],dataRs[0::ds],newpts,method=method)
        data_grid = np.reshape(data_lnsp,(lat_grid.shape[0],lat_grid.shape[1]))
        
    else:
        return None
    
    return data_grid, lat_grid, lon_grid

def visualizeImg(band):
    img = skt.resize(band, (int(band.shape[0]/8),int(band.shape[1]/8)))
    fig = plt.figure(figsize=(8,12))
    plt.imshow(img,cmap='jet')
    plt.colorbar()
    return fig

def cpGM(data):
    data2 = GriddedMeasurement(data.dateTime,data.latitude,data.longitude,data.data,data.label)
    data2.dataName = data.dataName
    return data2

if __name__ == "__main__":
    
    basedir = 'E:/projects/wildfire-research/landfireData/'
    elevationName = 'US_DEM2010'
    aspectName = 'US_ASP2010'
    slopeName = 'US_SLP2010'
    fuelName = 'PM_F0SU17'
    canopyName = 'PM_EVCSU17'
    
    name = elevationName
    
    inAscFile = basedir+name+'/'+name+'.asc'
    outAscFile = inAscFile+'.rmap'
    imgFile = basedir+name+'/merged.tif'
  
  
    inAscFile = basedir+name+'/'+name+'.asc'
    outAscFile = inAscFile+'.rmap'
    imgFile = 'E:/projects/wildfire-research/landfireData/US_140FBFM40/merged.tif'
    
    data = readImgFile(imgFile)
    data2 = cpGM(data)
    data2.data[data2.data <= 50] = 0
    dataTmp = data2.data.copy()
    dataTmp[dataTmp == 91] = 1
    dataTmp[dataTmp == 92] = 2
    dataTmp[dataTmp == 93] = 3
    dataTmp[dataTmp == 98] = 4
    dataTmp[dataTmp == 99] = 5
    dataTmp[dataTmp ==101] = 6
    dataTmp[dataTmp ==102] = 7
    dataTmp[dataTmp ==103] = 8
    dataTmp[dataTmp ==104] = 9
    dataTmp[dataTmp ==105] = 10
    dataTmp[dataTmp ==106] = 11
    dataTmp[dataTmp ==107] = 12
    dataTmp[dataTmp ==108] = 13
    dataTmp[dataTmp ==109] = 14
    dataTmp[dataTmp ==121] = 15
    dataTmp[dataTmp ==122] = 16
    dataTmp[dataTmp ==123] = 17
    dataTmp[dataTmp ==124] = 18
    dataTmp[dataTmp ==141] = 19
    dataTmp[dataTmp ==142] = 20
    dataTmp[dataTmp ==143] = 21
    dataTmp[dataTmp ==144] = 22
    dataTmp[dataTmp ==145] = 23
    dataTmp[dataTmp ==146] = 24
    dataTmp[dataTmp ==147] = 25
    dataTmp[dataTmp ==148] = 26
    dataTmp[dataTmp ==149] = 27
    dataTmp[dataTmp ==161] = 28
    dataTmp[dataTmp ==162] = 29
    dataTmp[dataTmp ==163] = 30
    dataTmp[dataTmp ==164] = 31
    dataTmp[dataTmp ==165] = 32
    dataTmp[dataTmp ==181] = 33
    dataTmp[dataTmp ==182] = 34
    dataTmp[dataTmp ==183] = 35
    dataTmp[dataTmp ==184] = 36
    dataTmp[dataTmp ==185] = 37
    dataTmp[dataTmp ==186] = 38
    dataTmp[dataTmp ==187] = 39
    dataTmp[dataTmp ==188] = 40
    dataTmp[dataTmp ==189] = 41
    dataTmp[dataTmp ==201] = 42
    dataTmp[dataTmp ==202] = 43
    dataTmp[dataTmp ==203] = 44
    dataTmp[dataTmp ==204] = 45
    data2.data = dataTmp.copy()
    
    fig = plt.figure(figsize=(12,8))
    plt.contourf(data2.longitude,data2.latitude,data2.data,cmap='jet')
    plt.colorbar()

    
    #set(np.reshape(data2.data,(data2.data.shape[0]*data2.data.shape[1])))
    
    #data = GriddedMeasurement(None,lat,lon,data,'FuelModel')
    #latlong = readImgCoords(imgFile)
    
    #modifyAscFile(inAscFile,outAscFile)
    #readModifyAscFile(outAscFile)

#revisedElevationContent = []
#revisedElevationContent.extend(elevationHeader)
#for i in range(5,len(elevationContent)):
#    revisedElevationContent.append(elevationContent[i].split(' ')[1:])
    
#if fuelCols < elevationCols and fuelXll > elevationXll:
#    tmp = revisedElevationContent[5][(elevationCols-fuelCols):]
    #fuelXll + (fuelCols-elevationCols)*fuelCellSize = elevationXll
# -2239425+(27530-31644)*30
#fuel_file = fuel_dir+'Spatial_Metadata/pm_0k_su17.shp'
#data = gpd.read_file(fuel_file)


#file = 'E:/projects/wildfire-research/landfireData/US_140EVT/US_140EVT_1.asc'

#with open(file,'r') as f:
#    content = f.readlines()
