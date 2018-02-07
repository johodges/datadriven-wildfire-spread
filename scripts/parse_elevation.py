# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 06:58:26 2018

@author: JHodges
"""

import gdal
import numpy as np
import glob
from scipy.ndimage.interpolation import zoom
import util_common as uc
import os

def coordinatesFromTile(tile):
    ''' This function will return the longitude and latitude from a USGS
    tilt in the format 'n000e000'
    '''
    lat = float(tile[1:3])
    lon = float(tile[4:7])
    if tile[0] == 's':
        lat = lat*-1
    if tile[3] == 'w':
        lon = lon*-1
    return lat, lon

def restrictContour(data,target_sz=250):
    ''' This function will downsample a contour to the target size
    '''
    current_sz = data.shape[0]
    restrictedData = zoom(data,target_sz/current_sz,order=1)
    return restrictedData

def coordinatesToContour(data,lat,lon):
    ''' This function will generate a meshgrid for latitude and longitude
    '''
    [r,c] = data.shape
    latAxis = np.linspace(lat,lat-1.0,r)
    lonAxis = np.linspace(lon,lon+1.0,r)
    lonGrid, latGrid = np.meshgrid(lonAxis,latAxis)
    return latGrid, lonGrid

def loadRawData(datadir,forceRead=False):
    ''' This function will load the raw elevation data from USGS
    '''
    files = [fn for fn in glob.glob(datadir+'*grd*') if not os.path.basename(fn).endswith('.pkl')]
    lats = []
    lons = []
    datas = []
    tiles = []
    for i in range(0,len(files)):
        file = files[i]
        tile = files[i].split('grd')[1].split('_')[0]
        tiles.append(tile)
        if len(glob.glob(files[i]+'.pkl')) == 0 or forceRead:
            print("Starting %s"%file)
            filetim = uc.tic()
            tim = uc.tic()
            geo = gdal.Open(file)
            data = geo.ReadAsArray()
            data[data<= -99999] = 0.0
            data[data== 0.0] = 0.0
            lat, lon = coordinatesFromTile(tile)
            data = restrictContour(data)
            latGrid, lonGrid = coordinatesToContour(data,lat,lon)
            tim = uc.toc(tim)
            datas.append(data)
            lats.append(latGrid)
            lons.append(lonGrid)
            print("\tFile time:")
            tim = uc.toc(filetim)
            print("\tCreating pickle file.")
            uc.dumpPickle([data,latGrid,lonGrid],files[i]+'.pkl')
            print("\tPickle file created.")
        else:
            dataLoad = uc.readPickle(files[i]+'.pkl')
            datas.append(dataLoad[0])
            lats.append(dataLoad[1])
            lons.append(dataLoad[2])
    return lats, lons, datas

def rawData2Map(datadir,lats,lons,datas):
    ''' This function will generate a single contour map from a set of contour
    maps.
    '''
    files = [fn for fn in glob.glob(datadir+'*grd*') if not os.path.basename(fn).endswith('.pkl')]
    tiles = [ti.split('grd')[1].split('_')[0] for ti in files]
    pixels = int(datas[0].shape[0])
    tiles_grid_dict, tiles_grid = uc.mapTileGrid(tiles,pixels,coordinatesFromTile)
    tiles_data = tiles_grid.copy()
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    for i in range(0,len(files)):
        tile = files[i].split('grd')[1].split('_')[0]
        data = np.flip(datas[i].copy(),axis=0)
        lat = np.flip(lats[i].copy(),axis=0)
        lon = np.flip(lons[i].copy(),axis=0)
        tiles_data = uc.fillTileGrid(tiles_data,tiles_grid_dict,tile,data,pixels)
        tiles_lat = uc.fillTileGrid(tiles_lat,tiles_grid_dict,tile,lat,pixels)
        tiles_lon = uc.fillTileGrid(tiles_lon,tiles_grid_dict,tile,lon,pixels)
    tiles_lat = uc.fillEmptyCoordinatesRectilinear(tiles_lat)
    tiles_lon = uc.fillEmptyCoordinatesRectilinear(tiles_lon)
    return tiles_lat, tiles_lon, tiles_data

def dumpMapData(lat,lon,data,name):
    ''' This function will dump an elevation map to a pickle file
    '''
    uc.dumpPickle([data,lat,lon],name+'.pkl')
    return name+'.pkl'

def loadMapData(name):
    ''' This function will read an elevation map froma pickle file
    '''
    dataLoad = uc.readPickle(name)
    data = dataLoad[0]
    lat = dataLoad[1]
    lon = dataLoad[2]
    return data, lat, lon

def queryElevation(
        filename='E:/WildfireResearch/data/usgs_elevation_30m/California.pkl'):
    ''' This is the function which is used to query the database.
    
    NOTE: Since there is no dependence on time for elevation in this database,
        the same values will always be returned.
    '''
    data, lat, lon = loadMapData(filename)
    return lat, lon, data
    

if __name__ == "__main__":
    ''' Example cases:
        case 0: Load raw data, plot raw data
        case 1: Load raw data, build map, plot map, dump map
        case 2: Load map, plot map
    '''
    datadir = 'E:/WildfireResearch/data/usgs_elevation_30m/'
    clim = [-1000,-500,0,500,1000,1500,2000,2500,3000,3500,4000]
    case = 1
    totaltim = uc.tic()
    if case == 0:
        tim = uc.tic()
        lats, lons, datas = loadRawData(datadir)
        print("Time to load:")
        tim = uc.toc(tim)
        print("Time to plot raw:")
        fig = uc.plotContourWithStates(lats,lons,datas,clim=clim)
        tim = uc.toc(tim)
    elif case == 1:
        tim = uc.tic()
        lats, lons, datas = loadRawData(datadir)
        print("Time to load:")
        tim = uc.toc(tim)
        print("Time to build map:")
        lat, lon, data = rawData2Map(datadir,lats,lons,datas)
        tim = uc.toc(tim)
        print("Time to plot map:")
        fig = uc.plotContourWithStates(lat,lon,data,clim=clim)
        tim = uc.toc(tim)
        print("Time to dump map:")
        dumpMapData(lat,lon,data,datadir+'California')
        tim = uc.toc(tim)
    elif case == 2:
        tim = uc.tic()
        data, lat, lon = loadMapData(datadir+'California.pkl')
        print("Time to load:")
        tim = uc.toc(tim)
        print("Time to plot map:")
        fig = uc.plotContourWithStates(lat,lon,data,clim=clim)
        tim = uc.toc(tim)
    print("Total time:")
    uc.toc(totaltim)
