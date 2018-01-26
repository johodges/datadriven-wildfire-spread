# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 06:58:26 2018

@author: JHodges
"""

import gdal
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage.interpolation import zoom
import parse_asos_file as paf
import sys
from util_common import tic, toc, dumpPickle, readPickle
import os

def coordinatesFromFile(file):
    parsedFile = file.split('grd')[1]
    lat = float(file.split('grd')[1][1:3])
    lon = float(file.split('grd')[1][4:7])
    if parsedFile[0] == 's':
        lat = lat*-1
    if parsedFile[3] == 'w':
        lon = lon*-1
    return lat, lon

def restrictContour(data,target_sz=250):
    current_sz = data.shape[0]
    restrictedData = zoom(data,target_sz/current_sz,order=1)
    return restrictedData

def coordinatesToContour(data,lat,lon):
    [r,c] = data.shape
    latAxis = np.linspace(lat,lat-1.0,r)
    lonAxis = np.linspace(lon,lon+1.0,r)
    #latAxis = np.linspace(lat+1.0,lat,r)
    #lonAxis = np.linspace(lon-1.0,lon,r)
    lonGrid, latGrid = np.meshgrid(lonAxis,latAxis)
    return latGrid, lonGrid

def mapTileGrid(files,pixels):
    h_min = 361
    h_max = -361
    v_min = 361
    v_max = -361
    tiles_grid_dict = dict()
    for file in files:
        v_tile,h_tile = coordinatesFromFile(file)
        tile = file.split('grd')[1].split('_')[0]
        tiles_grid_dict[tile] = [h_tile,v_tile]
        h_min = int(h_tile) if h_tile < h_min else h_min
        h_max = int(h_tile) if h_tile > h_max else h_max
        v_min = int(v_tile) if v_tile < v_min else v_min
        v_max = int(v_tile) if v_tile > v_max else v_max
    for key in tiles_grid_dict:
        tiles_grid_dict[key] = list(np.array(tiles_grid_dict[key])-np.array([h_min,v_min]))
    tiles_grid = np.zeros((pixels*(v_max-v_min+1),pixels*(h_max-h_min+1)),dtype=np.float32)
    return tiles_grid_dict, tiles_grid

def fillTileGrid(tile_grid,tile_grid_dict,tile,data,pixels):
    h,v = tile_grid_dict[tile]
    start_h = (h*pixels)
    end_h = ((h+1)*pixels)
    start_v = (v*pixels)
    end_v = ((v+1)*pixels)
    #print(tile,start_v,end_v,start_h,end_h)
    (start_h,end_h,start_v,end_v) = (int(start_h),int(end_h),int(start_v),int(end_v))
    tile_grid[start_v:end_v,start_h:end_h] = data.copy()
    return tile_grid

def fillEmptyCoordinates(data):
    [r,c] = data.shape
    dataNan = data.copy()
    dataNan[data == 0] = np.nan
    r0mn = np.nanmin(dataNan[0,:])
    r0mx = np.nanmax(dataNan[0,:])
    c0mn = np.nanmin(dataNan[:,0])
    c0mx = np.nanmax(dataNan[:,0])
    if abs(c0mx-c0mn) > 0 and abs(r0mx-r0mn) > 0:
        assert False, "Both column and row values are changing."
    elif abs(r0mx-r0mn) == 0:
        for i in range(0,r):
            rowValue = np.nanmax(dataNan[i,:])
            dataNan[i,:] = rowValue.copy()
    elif abs(c0mx-c0mn) == 0:
        for i in range(0,c):
            colValue = np.nanmax(dataNan[:,i])
            dataNan[:,i] = colValue.copy()
    else:
        assert False, "Did not find a valid region."
    return dataNan

def plotListContourWithStates(lats,lons,datas,states=None,clim=None,label=''):
    if states is None:
        states = paf.getStateBoundaries(state='California')
    elif type(states) is str:
        states=paf.getStateBoundaries(state=states)
    fig = plt.figure(figsize=(12,8))
    
    for i in range(0,len(datas)):
        data = datas[i]
        lat = lats[i]
        lon = lons[i]
        if clim is None:
            plt.contourf(lon,lat,data,cmap='jet')
            #plt.colorbar(label=label)
        else:
            plt.contourf(lon,lat,data,clim,cmap='jet')
            #plt.colorbar(label=label,ticks=clim)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
        
    if type(states) is dict:
        for state in states:
            plt.plot(states[state][:,1],states[state][:,0],'-k')
        plt.xlim([-128,-66])
        plt.ylim([24,50])
    else:
        plt.plot(states[:,1],states[:,0],'-k')
        plt.xlim([-126,-113])
        plt.ylim([32,43])
    if clim is None:
        plt.colorbar(label=label)
    else:
        plt.colorbar(label=label,ticks=clim)
    return fig

def plotContourWithStates(lat,lon,data,states=None,clim=None,label=''):
    if states is None:
        states = paf.getStateBoundaries(state='California')
    elif type(states) is str:
        states=paf.getStateBoundaries(state=states)
    fig = plt.figure(figsize=(12,8))
    
    plt.contourf(lon,lat,data,clim,cmap='jet')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
        
    if type(states) is dict:
        for state in states:
            plt.plot(states[state][:,1],states[state][:,0],'-k')
        plt.xlim([-128,-66])
        plt.ylim([24,50])
    else:
        plt.plot(states[:,1],states[:,0],'-k')
        plt.xlim([-126,-113])
        plt.ylim([32,43])
    if clim is None:
        plt.colorbar(label=label)
    else:
        plt.colorbar(label=label,ticks=clim)
    return fig

def loadRawData(datadir):
    files = [fn for fn in glob.glob(datadir+'*grd*') if not os.path.basename(fn).endswith('.pkl')]
    lats = []
    lons = []
    datas = []
    tiles = []
    for i in range(0,len(files)):
        file = files[i]
        tile = files[i].split('grd')[1].split('_')[0]
        tiles.append(tile)
        #print("Starting %s"%file)
        if len(glob.glob(files[i]+'.pkl')) == 0:
            filetim = tic()
            tim = tic()
            geo = gdal.Open(file)
            data = geo.ReadAsArray()
            data[data<= -99999] = 0.0
            data[data== 0.0] = 0.0
            lat, lon = coordinatesFromFile(file)
            data = restrictContour(data)
            latGrid, lonGrid = coordinatesToContour(data,lat,lon)
            tim = toc(tim)
            datas.append(data)
            lats.append(latGrid)
            lons.append(lonGrid)
            print("\tFile time:")
            tim = toc(filetim)
            print("\tCreating pickle file.")
            dumpPickle([data,latGrid,lonGrid],files[i]+'.pkl')
            print("\tPickle file created.")
        else:
            dataLoad = readPickle(files[i]+'.pkl')
            pass
            #dumpPickle([data,latGrid-0.5,lonGrid+0.5],files[i]+'.pkl')
            #print("\tPickle file already exists.\tPickle file loaded.")
            datas.append(dataLoad[0])
            lats.append(dataLoad[1])
            lons.append(dataLoad[2])
    return lats, lons, datas

def rawData2Map(datadir,lats,lons,datas):
    files = [fn for fn in glob.glob(datadir+'*grd*') if not os.path.basename(fn).endswith('.pkl')]
    pixels = int(datas[0].shape[0])
    tiles_grid_dict, tiles_grid = mapTileGrid(files,pixels)
    tiles_data = tiles_grid.copy()
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    for i in range(0,len(files)):
        tile = files[i].split('grd')[1].split('_')[0]
        data = np.flip(datas[i].copy(),axis=0)
        lat = np.flip(lats[i].copy(),axis=0)
        lon = np.flip(lons[i].copy(),axis=0)
        tiles_data = fillTileGrid(tiles_data,tiles_grid_dict,tile,data,pixels)
        tiles_lat = fillTileGrid(tiles_lat,tiles_grid_dict,tile,lat,pixels)
        tiles_lon = fillTileGrid(tiles_lon,tiles_grid_dict,tile,lon,pixels)
    tiles_lat = fillEmptyCoordinates(tiles_lat)
    tiles_lon = fillEmptyCoordinates(tiles_lon)
    return tiles_lat, tiles_lon, tiles_data

def dumpMapData(lat,lon,data,name):
    dumpPickle([data,lat,lon],name+'.pkl')
    return name+'.pkl'

def loadMapData(name):
    dataLoad = readPickle(name)
    data = dataLoad[0]
    lat = dataLoad[1]
    lon = dataLoad[2]
    return data, lat, lon
    
if __name__ == "__main__":
    ''' Example cases:
        case 0: Load raw data, plot raw data
        case 1: Load raw data, build map, plot map, dump map
        case 2: Load map, plot map
    '''
    datadir = 'E:/WildfireResearch/data/usgs_elevation_30m/'
    clim = [-1000,-500,0,500,1000,1500,2000,2500,3000,3500,4000]
    case = 2
    totaltim = tic()
    if case == 0:
        tim = tic()
        lats, lons, datas = loadRawData(datadir)
        print("Time to load:")
        tim = toc(tim)
        print("Time to plot raw:")
        fig = plotListContourWithStates(lats,lons,datas,clim=clim)
        tim = toc(tim)
    elif case == 1:
        tim = tic()
        lats, lons, datas = loadRawData(datadir)
        print("Time to load:")
        tim = toc(tim)
        print("Time to build map:")
        lat, lon, data = rawData2Map(datadir,lats,lons,datas)
        tim = toc(tim)
        print("Time to plot map:")
        fig = plotContourWithStates(lat,lon,data,clim=clim)
        tim = toc(tim)
        print("Time to dump map:")
        dumpMapData(lat,lon,data,datadir+'California')
        tim = toc(tim)
    elif case == 2:
        tim = tic()
        data, lat, lon = loadMapData(datadir+'California.pkl')
        print("Time to load:")
        tim = toc(tim)
        print("Time to plot map:")
        fig = plotContourWithStates(lat,lon,data,clim=clim)
        tim = toc(tim)
    print("Total time:")
    toc(totaltim)
