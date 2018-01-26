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
import re

class MODISmeasurement(object):
    __slots__ = ['dateTime','latitude','longitude','firemask']
    
    def __init__(self):
        self.dateTime = []
        self.latitude = []
        self.longitude = []
        self.firemask = []

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
    
def findXmlTimes(datadir,tiles):
    ''' This function finds the start and end times of each .hdf.xml file
    in datadir within the first tile.
    '''
    files = glob.glob(datadir+'*'+tiles[0]+'*'+'.hdf')
    startdates = []
    enddates = []
    for file in files:
        startdate, enddate = loadXmlDate(file+'.xml')
        startdates.append(startdate)
        enddates.append(enddate)
    return [startdates, enddates], files

def daysInYear(year):
    ''' daysInYear: This function will return the number of days in a year by
    determining if the year number is a leap year.
    '''
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                days = 366
            else:
                days = 365
        else:
            days = 366
    else:
        days = 365
    return days

def findQueryDateTime(files,dates,queryDateTime):
    ''' findQueryDateTime: This function takes a list containing start and end
    datetimes returns the index of the list which contains a queryDateTime.
    If no match is found, returns None.
    
    Using timedeltas from datetime.datetime would have been better.
    Unfortunately, that gave an error when the day was the same and the hour
    difference was negative since the negative was stored in the day part of
    the structure.
    '''
    index = None
    queryDay = queryDateTime.timetuple().tm_yday+((queryDateTime.hour*60+queryDateTime.minute)*60+queryDateTime.second)/(24*60*60)
    
    for i in range(0,len(dates[0])):
        lowYearDiff = queryDateTime.year-dates[0][i].year
        highYearDiff = dates[1][i].year-queryDateTime.year
        
        lowDay = dates[0][i].timetuple().tm_yday+((dates[0][i].hour*60+dates[0][i].minute)*60+dates[0][i].second)/(24*60*60)
        highDay = dates[1][i].timetuple().tm_yday+((dates[1][i].hour*60+dates[1][i].minute)*60+dates[1][i].second)/(24*60*60)
        
        if lowYearDiff < 0:
            lowDay = 367
        elif lowYearDiff > 0:
            lowDay = lowDay-daysInYear(dates[0][i].year)
        if highYearDiff < 0:
            highDay = 0
        elif highYearDiff > 0:
            highDay = highDay+daysInYear(dates[0][i].year-1)
        if queryDay >= lowDay and queryDay <= highDay:
            index = i
            #print(dates[0][i],dates[1][i])
    if index is not None:
        tile = extractTileFromFile(files[index])
        datename = files[index].split(tile)[0][-8:-1]
    else:
        print("Did not find queryDateTime.")
        datename = None
    return datename

def removeUnlistedTilesFromFiles(datadir,datename,tiles,use_all=False):
    ''' This will remove tiles which were not included in the list from the
    list of files. If the use_all argument is active, it will instead
    update the list of tiles to include all files found in the file names.
    '''
    files = glob.glob(datadir+'*'+datename+'*'+'.hdf')
    if use_all:
        tiles = findAllTilesFromFiles(files)
    updated_files = []
    for file in files:
        use_file = False
        for tile in tiles:
            if tile in file:
                use_file = True
        if use_file:
            updated_files.append(file)
    return updated_files, tiles

def extractTileFromFile(file):
    ''' This function uses regular expressions to find .h00v00. in a filename
    to extract the MODIS tile.
    '''
    m = re.search('\.h\d\dv\d\d\.',file)
    tile = m.group(0)[1:-1]
    return tile

def findAllTilesFromFiles(files):
    ''' This function finds all MODIS tiles in a list of file names
    '''
    tiles = []
    for file in files:
        tile = extractTileFromFile(file)
        tiles.append(tile)
    return list(set(tiles))

def findAllTilesFromDir(datadir):
    ''' This function finds all MODIS tiles in a list of file names
    '''
    files = glob.glob(datadir+'*.hdf')
    tiles = []
    for file in files:
        tile = extractTileFromFile(file)
        tiles.append(tile)
    return list(set(tiles))

def activeFireDayIndex(dates,queryDateTime):
    ''' This function finds the index of the queryDateTime within the range
    of dates of the (.hdf) file.
    '''
    index = None
    queryDay = queryDateTime.timetuple().tm_yday
    lowDay = dates[0].timetuple().tm_yday
    highDay = dates[1].timetuple().tm_yday
    
    lowYearDiff = queryDateTime.year-dates[0].year
    highYearDiff = dates[1].year-queryDateTime.year
    
    if lowYearDiff == 0:
        index = queryDay-lowDay
    elif highYearDiff == 0:
        index = 8-(highDay-queryDay)
    else:
        print("Is query within range for the file?")
    return index

def buildContour(files,queryDateTime,sdsname='FireMask',composite=True):
    ''' This function will combine measurements from multiple
    MODIS tiles into a single dataset. The list of file names should
    correspond to the same time and be for different tiles. The file names
    should reference the (.hdf) files.
    '''
    pixels = loadSdsData(files[0],sdsname).shape[1]
    tiles = findAllTilesFromFiles(files)
    tiles_grid_dict, tiles_grid = mapTileGrid(tiles,pixels)
    tiles_data = tiles_grid.copy()
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    for file in files:
        p = loadGPolygon(file+'.xml')
        startdate, enddate = loadXmlDate(file+'.xml')
        plat, plon = arrangeGPolygon(p)
        
        if not composite:
            day_index = activeFireDayIndex([startdate,enddate],queryDateTime)
            data = loadSdsData(file,sdsname)[day_index,:,:]
        else:
            data = loadSdsData(file,sdsname)
        tile = extractTileFromFile(file)
        lat, lon = interpGPolygon(plat,plon,pixels=pixels)
        #lats.append(lat)
        #lons.append(lon)
        #datas.append(data)
        #CS = plt.contourf(lon,lat,data,cmap='jet')
        tiles_data = fillTileGrid(tiles_data,tiles_grid_dict,tile,data,pixels)
        tiles_lat = fillTileGrid(tiles_lat,tiles_grid_dict,tile,lat,pixels)
        tiles_lon = fillTileGrid(tiles_lon,tiles_grid_dict,tile,lon,pixels)
    return tiles_lat, tiles_lon, tiles_data

def plotContourWithStates(lat,lon,data,states=None,clim=None,label=''):
    if states is None:
        states = paf.getStateBoundaries(state='California')
    elif type(states) is str:
        states=paf.getStateBoundaries(state=states)
    
    fig = plt.figure(figsize=(12,8))
    if clim is None:
        plt.contourf(lon,lat,data,cmap='jet')
        plt.colorbar(label=label)
    else:
        plt.contourf(lon,lat,data,clim,cmap='jet')
        plt.colorbar(label=label,ticks=clim)
    
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
    return fig
    
def findQuerySdsData(queryDateTime,
                     datadir="E:/WildfireResearch/data/aqua_vegetation/",
                     tiles=None,
                     composite=False,
                     use_all=False,
                     sdsname='1 km 16 days NDVI'):
    # Arrange files and tiles
    if tiles is None:
        tiles = findAllTilesFromDir(datadir)
    dates, files = findXmlTimes(datadir,tiles)
    datename = findQueryDateTime(files,dates,queryDateTime)
    files, tiles = removeUnlistedTilesFromFiles(datadir,datename,tiles,use_all=use_all)
    
    # Load all vegetation tiles at the queryDateTime
    lat,lon,data = buildContour(files,queryDateTime,sdsname=sdsname,composite=composite)
    
    return lat, lon, data

def dumpPickleModis(data,filename='../data-test/modis.pkl'):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
        
def readPickleStations(filename='../data-test/modis.pkl'):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    # User inputs
    #tiles = None
    tiles = ['h08v04','h08v05','h09v04']
    queryDateTime = dt.datetime(year=2016,month=6,day=27,hour=5,minute=53)
    
    # Find activefires at queryDateTime
#    fa_lat,fa_lon,fa_data = findQuerySdsData(queryDateTime,
#                                             tiles=tiles,
#                                             datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
#                                             sdsname='FireMask',
#                                             composite=False)
    
    # Find vegetation index at queryDateTime
#    vi_lat,vi_lon,vi_data = findQuerySdsData(queryDateTime,
#                                             tiles=tiles,
#                                             datadir="E:/WildfireResearch/data/aqua_vegetation/",
#                                             sdsname='1 km 16 days NDVI',
#                                             composite=True)
    # Find burned area at queryDateTime
#    ba_lat,ba_lon,ba_data = findQuerySdsData(queryDateTime,
#                                             tiles=tiles,
#                                             datadir="E:/WildfireResearch/data/modis_burnedarea/",
#                                             sdsname='burndate',
#                                             composite=True)

    # Find elevation data
    
    

    # Plot all activefire tiles at the queryDateTime
#    if tiles is None:
#        states = 'All'
#    else:
#        states = 'California'
#    fig = plotContourWithStates(fa_lat,fa_lon,fa_data,states=states,clim=np.linspace(0,9,10),label='AF')
#    fig = plotContourWithStates(vi_lat,vi_lon,vi_data,states=states,label='VI')
#    fig = plotContourWithStates(ba_lat,ba_lon,ba_data,states=states,label='BA')
    
    # Print memory requirements for the queryDateTime
#    mem = (sys.getsizeof(fa_data)+sys.getsizeof(fa_lat)+sys.getsizeof(fa_lon))/1024**2
#    print("VI File Size: %.4f MB"%mem)
#    
#    mem = (sys.getsizeof(vi_data)+sys.getsizeof(vi_lat)+sys.getsizeof(vi_lon))/1024**2
#    print("VI File Size: %.4f MB"%mem)
    
    
    
    