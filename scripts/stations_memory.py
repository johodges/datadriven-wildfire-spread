# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:05:33 2018

@author: jhodges
"""


import glob
from pyhdf.SD import SD, SDC
import xml.etree.ElementTree as ET
import datetime as dt
import matplotlib.path as mpltPath
from scipy.ndimage.interpolation import zoom
import numpy as np
import matplotlib.pyplot as plt
import parse_asos_file as paf
import sys
import pickle

def loadGPolygon(file):
    tree = ET.parse(file)
    root = tree.getroot()
    ps = root[2][9][0][0][0]
    p = []
    for i in range(0,4):
        p.append([float(ps[i][0].text),float(ps[i][1].text)])
        #path = mpltPath.Path(p)
    return p

def loadXmlDate(file):
    tree = ET.parse(file)
    root = tree.getroot()
    DT = root[2][8]
    fmt = '%Y-%m-%d-%H:%M:%S'
    enddate = DT[1].text+'-'+DT[0].text.split('.')[0]
    startdate = DT[3].text+'-'+DT[2].text.split('.')[0]
    enddate = dt.datetime.strptime(enddate,fmt)
    startdate = dt.datetime.strptime(startdate,fmt)
    return startdate, enddate
    
def arrangeGPolygon(p,topleft=1,topright=2,botleft=0,botright=3):
    plat = np.array([[p[topleft][1],p[topright][1]],[p[botleft][1],p[botright][1]]])
    plon = np.array([[p[topleft][0],p[topright][0]],[p[botleft][0],p[botright][0]]])
    return plat, plon

def interpGPolygon(plat,plon,pixels=1200):
    lat = zoom(plat,pixels/2,order=1)
    lon = zoom(plon,pixels/2,order=1)
    return lat, lon

def loadSdsData(file,sdsname):
    f = SD(file,SDC.READ)
    sds_obj = f.select(sdsname)
    data = sds_obj.get()
    return data

def mapTileGrid(tiles,pixels):
    h_min = 20
    h_max = 0
    v_min = 20
    v_max = 0
    tiles_grid_dict = dict()
    for tile in tiles:
        h_tile = int(tile[1:3])
        v_tile = int(tile[4:])
        tiles_grid_dict[tile] = [h_tile,v_tile]
        h_min = h_tile if h_tile < h_min else h_min
        h_max = h_tile if h_tile > h_max else h_max
        v_min = v_tile if v_tile < v_min else v_min
        v_max = v_tile if v_tile > v_max else v_max
    for key in tiles_grid_dict:
        tiles_grid_dict[key] = list(np.array(tiles_grid_dict[key])-np.array([h_min,v_min]))
    tiles_grid = np.zeros((pixels*(v_max-v_min+1),pixels*(h_max-h_min+1)),dtype=np.float32)
    return tiles_grid_dict, tiles_grid

def fillTileGrid(tile_grid,tile_grid_dict,tile,data,pixels):
    h,v = tile_grid_dict[tile]
    start_h = h*pixels
    end_h = (h+1)*pixels
    start_v = v*pixels
    end_v = (v+1)*pixels
    tile_grid[start_v:end_v,start_h:end_h] = data.copy()
    return tile_grid
    

case = 4
#datename = '2017177.h'
#datename = '2016001.h'
#datename = '2016033.h'
#datename = '2016025.h'

if case == 0:
    datadir = "G:/WildfireResearch/data/aqua_activefires/"
    datename = '2016033.h'
    sdsname = 'FireMask'
    pixels = 1200
elif case == 1:
    datadir = "G:/WildfireResearch/data/aqua_temperature/"
    datename = '2016033.h'
    sdsname = 'LST_Day_1km'
    pixels = 1200
elif case == 2:
    datadir = "G:/WildfireResearch/data/aqua_vegetation/"
    datename = '2016025.h'
    sdsname = '1 km 16 days NDVI'
    pixels = 1200
elif case == 3:
    datadir = "G:/WildfireResearch/data/modis_burnedarea/"
    datename = '2016032.h'
    sdsname = 'burndate'
    pixels = 2400
elif case == 4:
    datadir = "G:/WildfireResearch/data/terra_daily_activefires/"
    datename = '2017137.h'
    sdsname = 'FireMask'
    pixels = 1200
    #MOD14A1.A2017137.h09v05.006.2017145230951
    
for k in range(0,8):
    tiles = ['h08v04','h08v05','h09v04']
    tiles_grid_dict, tiles_grid = mapTileGrid(tiles,pixels)
    tiles_data = np.array(tiles_grid.copy(),dtype=np.int8)
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    
    
    files = glob.glob(datadir+'*'+datename+'*.hdf')
    #files = glob.glob(datadir+'*'+'2016033.h*.hdf')
    #files = [x for x in os.listdir(idir) if x.endswith(".hdf")]
    
    #plt.figure(figsize=(12,8))
    lats = []
    lons = []
    datas = []
    for j in range(0,len(files)):
        use_file = False
        for tile in tiles:
            if tile in files[j]:
                use_file = True
                use_tile = tile
        if use_file:
            p = loadGPolygon(files[j]+'.xml')
            startdate, enddate = loadXmlDate(files[j]+'.xml')
            plat, plon = arrangeGPolygon(p)
            lat, lon = interpGPolygon(plat,plon,pixels=pixels)
            data = loadSdsData(files[j],sdsname)[k,:,:]
            #lats.append(lat)
            #lons.append(lon)
            #datas.append(data)
            #CS = plt.contourf(lon,lat,data,cmap='jet')
            tiles_data = fillTileGrid(tiles_data,tiles_grid_dict,use_tile,data,pixels)
            tiles_lat = fillTileGrid(tiles_lat,tiles_grid_dict,use_tile,lat,pixels)
            tiles_lon = fillTileGrid(tiles_lon,tiles_grid_dict,use_tile,lon,pixels)
    #tiles_lon[tiles_lon == 0] = np.nan
    #tiles_lat[tiles_lat == 0] = np.nan
    #tiles_data[tiles_data == 0] = np.nan
    states = paf.getStateBoundaries(state='California')
    #plt.colorbar(label='AF')
    #for state in states:
    #    plt.plot(states[state][:,1],states[state][:,0],'-k')
    #plt.plot(states[:,1],states[:,0],'-k')
    #plt.xlim([-126,-113])
    #plt.ylim([32,43])
    
    plt.figure(figsize=(12,8))
    plt.contourf(tiles_lon,tiles_lat,tiles_data,np.linspace(0,9,10),cmap='jet')
    plt.plot(states[:,1],states[:,0],'-k')
    plt.colorbar(label='AF',ticks=np.linspace(0,9,10))
    plt.xlim([-126,-113])
    plt.ylim([32,43])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    mem = (sys.getsizeof(tiles_data)+sys.getsizeof(tiles_lat)+sys.getsizeof(tiles_lon))/1024**2
    
    print("File Size: %.4f MB"%mem)