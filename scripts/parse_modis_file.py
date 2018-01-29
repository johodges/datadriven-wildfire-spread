# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:05:33 2018

@author: jhodges
"""


import glob
from pyhdf.SD import SD, SDC
import xml.etree.ElementTree as ET
import datetime as dt
from scipy.ndimage.interpolation import zoom
import numpy as np
import util_common as uc
import re
import sys

def coordinatesFromTile(tile):
    lon = int(tile[1:3])
    lat = int(tile[4:])
    return lat, lon

def loadGPolygon(file):
    tree = ET.parse(file)
    root = tree.getroot()
    ps = root[2][9][0][0][0]
    p = []
    for i in range(0,4):
        p.append([float(ps[i][0].text),float(ps[i][1].text)])
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
            lowDay = lowDay-uc.daysInYear(dates[0][i].year)
        if highYearDiff < 0:
            highDay = 0
        elif highYearDiff > 0:
            highDay = highDay+uc.daysInYear(dates[0][i].year-1)
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
    tiles_grid_dict, tiles_grid = uc.mapTileGrid(tiles,pixels,coordinatesFromTile)
    tiles_data = tiles_grid.copy()
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    for file in files:
        p = loadGPolygon(file+'.xml')
        startdate, enddate = loadXmlDate(file+'.xml')
        plat, plon = arrangeGPolygon(p)
        
        if not composite:
            day_index = activeFireDayIndex([startdate,enddate],queryDateTime)
            data = loadSdsData(file,sdsname)
            if day_index < data.shape[0]:
                data = data[day_index,:,:]
            else:
                print("Required day index does not have data included.")
                print("\tdata.shape:\t",data.shape)
                print("\tday_index:\t",day_index)
                data = None
        else:
            data = loadSdsData(file,sdsname)
        tile = extractTileFromFile(file)
        lat, lon = interpGPolygon(plat,plon,pixels=pixels)
        if data is not None:
            tiles_data = uc.fillTileGrid(tiles_data,tiles_grid_dict,tile,data,pixels)
        tiles_lat = uc.fillTileGrid(tiles_lat,tiles_grid_dict,tile,lat,pixels)
        tiles_lon = uc.fillTileGrid(tiles_lon,tiles_grid_dict,tile,lon,pixels)
    #tiles_lat = uc.fillEmptyCoordinates(tiles_lat,tiles,pixels,coordinatesFromTile)
    #tiles_lon = uc.fillEmptyCoordinates(tiles_lon,tiles,pixels,coordinatesFromTile)
    return tiles_lat, tiles_lon, tiles_data
    
def findQuerySdsData(queryDateTime,
                     datadir="E:/WildfireResearch/data/aqua_vegetation/",
                     tiles=['h08v04','h08v05','h09v04'],
                     composite=False,
                     use_all=False,
                     sdsname='1 km 16 days NDVI'):
    # Arrange files and tiles
    if tiles is None:
        tiles = findAllTilesFromDir(datadir)
    dates, files = findXmlTimes(datadir,tiles)
    datename = findQueryDateTime(files,dates,queryDateTime)
    files, tiles = removeUnlistedTilesFromFiles(datadir,datename,tiles,use_all=use_all)
    
    # Load all tiles at the queryDateTime
    lat,lon,data = buildContour(files,queryDateTime,sdsname=sdsname,composite=composite)
    
    return lat, lon, data

def extractCandidates(lat,lon,data):
    ''' This function extracts latitude and longitude corresponding to points
    in the binary mask data.
    '''
    r,c = np.where(data > 0)
    pts = []
    for i in range(0,len(r)):
        ptlat = lat[r[i],c[i]]
        ptlon = lon[r[i],c[i]]
        ptdat = data[r[i],c[i]]
        pts.append([ptlat,ptlon,ptdat])
    pts = np.array(pts)
    return pts

def compareCandidates(old_pts,new_pts,dist_thresh=0.5):
    ''' This function compares two sets of points to return minimum distance
    to a point in the new_pts set from an old_pt. dist_thresh is the minimum
    distance away for two points to be considered a match in degrees.
    NOTE: 1 degree is approximately 69 miles, or 111 km
    NOTE: Modis resolution is approximately 1km    
    '''
    
    matched_pts = []
    if old_pts.shape[0] != 0 and new_pts.shape[0] != 0:
        for i in range(0,old_pts.shape[0]):
            squared = np.power(new_pts[:,0:2]-old_pts[i,0:2],2)
            summed = np.sum(squared,axis=1)
            rooted = np.power(summed,0.5)
            min_dist = np.min(rooted)
            if min_dist <= dist_thresh:
                matched_pts.append([i,min_dist*111,np.argmin(rooted)])
                
    matched_pts = np.array(matched_pts)
    return matched_pts

if __name__ == '__main__':
    ''' case 0: loads modis vegetation index at queryDateTime and plots for
                the whole United states
        case 1: Loads modis active fires at queryDateTime and plots for
                California
        case 2: Loads modis vegetation index, active fires, and burned area
                at queryDateTime for California.
    '''
    
    # User inputs
    queryDateTime = dt.datetime(year=2016,month=6,day=27,hour=5,minute=53)
    case = 3
    
    if case == 0:
        tiles = None
        states = 'All'
        #Find vegetation index at queryDateTime
        vi_lat,vi_lon,vi_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=True,
                                                 datadir="E:/WildfireResearch/data/aqua_vegetation/",
                                                 sdsname='1 km 16 days NDVI')
        vi_fig = uc.plotContourWithStates(vi_lat,vi_lon,vi_data,states=states,label='VI')
        
        vi_mem = (sys.getsizeof(vi_data)+sys.getsizeof(vi_lat)+sys.getsizeof(vi_lon))/1024**2
        
        print("VI File Size: %.4f MB"%(vi_mem))
        
    if case == 1:
        tiles = ['h08v04','h08v05','h09v04']
        states = 'California'
        # Find activefires at queryDateTime
        af_lat,af_lon,af_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=False,
                                                 datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
                                                 sdsname='FireMask')
        af_fig = uc.plotContourWithStates(af_lat,af_lon,af_data,states=states,
                                          clim=np.linspace(0,9,10),label='AF')
        
        af_mem = (sys.getsizeof(af_data)+sys.getsizeof(af_lat)+sys.getsizeof(af_lon))/1024**2
        
        print("AF File Size: %.4f MB"%(af_mem))
        
    if case == 2:
        tiles = ['h08v04','h08v05','h09v04']
        states = 'California'
        # Find activefires at queryDateTime
        af_lat,af_lon,af_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=False,
                                                 datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
                                                 sdsname='FireMask')
        #Find vegetation index at queryDateTime
        vi_lat,vi_lon,vi_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=True,
                                                 datadir="E:/WildfireResearch/data/aqua_vegetation/",
                                                 sdsname='1 km 16 days NDVI')
        #Find burned area at queryDateTime
        ba_lat,ba_lon,ba_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=True,
                                                 datadir="E:/WildfireResearch/data/modis_burnedarea/",
                                                 sdsname='burndate')
        af_fig = uc.plotContourWithStates(af_lat,af_lon,af_data,states=states,
                                          clim=np.linspace(0,9,10),label='AF')
        vi_fig = uc.plotContourWithStates(vi_lat,vi_lon,vi_data,states=states,label='VI')
        ba_fig = uc.plotContourWithStates(ba_lat,ba_lon,ba_data,states=states,label='BA')
        
        vi_mem = (sys.getsizeof(vi_data)+sys.getsizeof(vi_lat)+sys.getsizeof(vi_lon))/1024**2
        af_mem = (sys.getsizeof(af_data)+sys.getsizeof(af_lat)+sys.getsizeof(af_lon))/1024**2
        ba_mem = (sys.getsizeof(ba_data)+sys.getsizeof(ba_lat)+sys.getsizeof(ba_lon))/1024**2
        total_mem = vi_mem+af_mem+ba_mem
        
        print("VI, AF, BA, Total File Size: %.4f,%.4f,%.4f,%.4f MB"%(vi_mem,af_mem,ba_mem,total_mem))
        
    if case == 3:
        tiles = ['h08v04','h08v05','h09v04']
        states = 'California'
        # Find activefires at queryDateTime
        queryDateTime = dt.datetime(year=2016,month=1,day=1,hour=12,minute=0)
        outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/AF_images/'
        for i in range(0,365):
            af_name = outdir+'AF2_'+queryDateTime.isoformat()[0:13]+'.png'
        
            af_lat,af_lon,af_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=False,
                                                     datadir="E:/WildfireResearch/data/terra_daily_activefires/",
                                                     sdsname='FireMask')
            if af_data is not None:
                af_fig = uc.plotContourWithStates(af_lat,af_lon,af_data,states=states,
                                                  clim=np.linspace(0,9,10),label='AF',
                                                  saveFig=True,saveName=af_name)
            
                af_mem = (sys.getsizeof(af_data)+sys.getsizeof(af_lat)+sys.getsizeof(af_lon))/1024**2
                
                data_mask = af_data.copy()
                data_mask[data_mask < 7] = 0
                pts = extractCandidates(af_lat,af_lon,data_mask)
                if i > 0:
                    match_pts = compareCandidates(old_pts,pts)
                    if match_pts.shape[0] > 0:
                        print("Time %s found %.0f matches with the closest %.4f km."%(queryDateTime.isoformat(),match_pts.shape[0],np.min(match_pts[:,1])))
                else:
                    pass
                queryDateTime = queryDateTime + dt.timedelta(days=1)
                old_pts = pts
            else:
                old_pts = np.array([])
        #print(match_pts)
        print("AF File Size: %.4f MB"%(af_mem))
    
    