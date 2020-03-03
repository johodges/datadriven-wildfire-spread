# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:05:33 2018

@author: jhodges

This file contains classes and functions to read MODIS Level 3 data and
locate multiple data tiles onto a single larger grid.

Results can be queried from the database or a specific time. If a static query
time is given, the best estimated value at that time will be returned. If a
time range is given, the average value across the time interval will be
returned.
"""


import glob
import pyhdf.SD as phdf
import xml.etree.ElementTree as ET
import datetime as dt
from scipy.ndimage.interpolation import zoom
import numpy as np
import util_common as uc
import re
import sys
import math
import scipy.interpolate as scpi

def coordinatesFromTile(tile):
    ''' This function will return the longitude and latitude MODIS Level 3
    tile coordinate from the tile name in the format 'h00v00'
    '''
    lon = int(tile[1:3])
    lat = int(tile[4:])
    return lat, lon

def loadGPolygon(file):
    ''' This function will return the corner latitude and longitudes from a
    MODIS Level 3 metadata xml file.
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    ps = root[2][9][0][0][0]
    p = []
    for i in range(0,4):
        p.append([float(ps[i][0].text),float(ps[i][1].text)])
    return p

def loadXmlDate(file):
    ''' This function will return the start and end dates from a MODIS Level 3
    metadata xml file.
    '''
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
    ''' This function will rearrange GPolygon points into a human readable
    format.
    '''
    plat = np.array([[p[topleft][1],p[topright][1]],[p[botleft][1],p[botright][1]]])
    plon = np.array([[p[topleft][0],p[topright][0]],[p[botleft][0],p[botright][0]]])
    return plat, plon

def interpGPolygon(plat,plon,pixels=1200):
    ''' This function will interpolate the 2x2 coordinate matricies to
    pixel x pixel matricies using bilinear interpolation. Note, this function
    should not be used with MODIS Level 3 data as the grid is non-linear. Use
    invertModisTile instead.
    '''
    lat = zoom(plat,pixels/2,order=1)
    lon = zoom(plon,pixels/2,order=1)
    return lat, lon

def loadSdsData(file,sdsname):
    ''' This function will open an hdf4 file and return the data stored in
    the sdsname attribute.
    '''
    f = phdf.SD(file,phdf.SDC.READ)
    sds_obj = f.select(sdsname)
    data = sds_obj.get()
    return data

def returnDataFile(file):
    f = phdf.SD(file,phdf.SDC.READ)
    return f    

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

def invertModisTile(tile,pixels=1200):
    ''' This function will create a pixel x pixel matrix for latitude and
    longitude using the tile name. This algorithm is presented in the
    Active Fire Index User Guide.
    '''
    R=6371007.181
    T=1111950
    xmin=-20015109
    ymax=10007555
    w=T/pixels
    lat_lnsp = np.linspace(0,pixels-1,pixels)
    lon_lnsp = np.linspace(0,pixels-1,pixels)
    lon_grid, lat_grid = np.meshgrid(lon_lnsp,lat_lnsp)
    H = float(tile[1:3])
    V = float(tile[4:])
    
    lat = (ymax-(lat_grid+0.5)*w-V*T)/R*(180/math.pi)
    lon = ((lon_grid+0.5)*w+H*T+xmin)/(R*np.cos(lat/180*math.pi))*(180/math.pi)
    return lat, lon

def buildContour(files,queryDateTime,
                 sdsname='FireMask',
                 composite=True,
                 greedyMethod=False):
    ''' This function will combine measurements from multiple
    MODIS tiles into a single dataset. The list of file names should
    correspond to the same time and be for different tiles. The file names
    should reference the (.hdf) files.
    '''
    #print(files[0])
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
        if greedyMethod:
            lat, lon = interpGPolygon(plat,plon,pixels=pixels)
        else:
            lat, lon = invertModisTile(tile)
        if data is not None:
            tiles_data = uc.fillTileGrid(tiles_data,tiles_grid_dict,tile,data,pixels)
        tiles_lat = uc.fillTileGrid(tiles_lat,tiles_grid_dict,tile,lat,pixels)
        tiles_lon = uc.fillTileGrid(tiles_lon,tiles_grid_dict,tile,lon,pixels)
    #tiles_lat = uc.fillEmptyCoordinates(tiles_lat,tiles,pixels,coordinatesFromTile)
    #tiles_lon = uc.fillEmptyCoordinates(tiles_lon,tiles,pixels,coordinatesFromTile)
    return tiles_lat, tiles_lon, tiles_data
    
def findQuerySdsData(queryDateTime,
                     datadir="G:/WildfireResearch/data/aqua_vegetation/",
                     tiles=['h08v04','h08v05','h09v04'],
                     composite=False,
                     use_all=False,
                     sdsname='1 km 16 days NDVI'):
    ''' This function will find the specified sdsname for each tile in tiles
    within the datadir and find the closest to the queryDateTime. Matrices
    of the latitutde, longitude, and data are returned.
    '''
    
    # Arrange files and tiles
    if tiles is None:
        tiles = findAllTilesFromDir(datadir)
    dates, files = findXmlTimes(datadir,tiles)
    datename = findQueryDateTime(files,dates,queryDateTime)
    files, tiles = removeUnlistedTilesFromFiles(datadir,datename,tiles,use_all=use_all)
    
    # Load all tiles at the queryDateTime
    lat,lon,data = buildContour(files,queryDateTime,sdsname=sdsname,composite=composite)
    
    return lat, lon, data

def geolocateCandidates(lat,lon,data):
    ''' This function extracts latitude and longitude corresponding to points
    in the binary mask data.
    '''
    r,c = np.where(data > 0)
    pts = []
    coords = []
    for i in range(0,len(r)):
        ptlat = lat[r[i],c[i]]
        ptlon = lon[r[i],c[i]]
        ptdat = data[r[i],c[i]]
        pts.append([ptlat,ptlon,ptdat])
        coords.append([r[i],c[i]])
    coords = np.array(np.squeeze(coords),dtype=np.int)
    pts = np.array(pts)
    return pts, coords

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





def buildOneDayContour(files,sdsname='sur_refl_b01',targetPixels=1200):
    pixels = loadSdsData(files[0],sdsname).shape[1]
    zoomLevel = targetPixels/pixels
    tiles = findAllTilesFromFiles(files)
    tiles_grid_dict, tiles_grid = uc.mapTileGrid(tiles,targetPixels,coordinatesFromTile)
    tiles_data = tiles_grid.copy()
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    for file in files:
        data = loadSdsData(file,sdsname)
        data = zoom(data,zoomLevel)
        tile = extractTileFromFile(file)
        lat, lon = invertModisTile(tile,pixels=targetPixels)
        if data is not None:
            tiles_data = uc.fillTileGrid(tiles_data,tiles_grid_dict,tile,data,targetPixels)
        tiles_lat = uc.fillTileGrid(tiles_lat,tiles_grid_dict,tile,lat,targetPixels)
        tiles_lon = uc.fillTileGrid(tiles_lon,tiles_grid_dict,tile,lon,targetPixels)
    return tiles_lat, tiles_lon, tiles_data

def list2stats(datas,name=''):
    dataMedian = np.median(datas,axis=0)
    dataMean = np.nanmean(datas,axis=0)
    dataMin = np.nanmin(datas,axis=0)
    dataMax = np.nanmax(datas,axis=0)
    uc.dumpPickle([dataMin,dataMax,dataMedian,dataMean],name)
    return dataMin, dataMax, dataMedian, dataMean

def generateVegetationStats(datadir="G:/WildfireResearch/data/aqua_reflectance/",
                            outdir="E:/projects/wildfire-research/data-test/",
                            tiles=['h08v04','h08v05','h09v04']):
    ''' This function will store out images with the min, max, median, and mean
    values of VIGR, NDVI, VARI, and NDI16. These are needed for moisture
    content estimation.
    '''
    files = glob.glob(datadir+'*.hdf')
    dates = []
    for file in files:
        dates.append(file.split("//")[1].split('.')[1])
    dates = list(set(dates))
    ndvis = []
    varis = []
    ndi16s = []
    vigrs = []
    for i in range(0,len(dates)):#date in dates:
        date = dates[i]
        files = glob.glob(datadir+'/*'+date+'*.hdf')
        goodFiles = []
        for file in files:
            tileCheck = False
            for tile in tiles:
                if tile in file:
                    tileCheck = True
            if tileCheck:
                goodFiles.append(file)
        lat,lon,rho1 = buildOneDayContour(goodFiles,sdsname='sur_refl_b01')
        lat,lon,rho2 = buildOneDayContour(goodFiles,sdsname='sur_refl_b02')
        lat,lon,rho3 = buildOneDayContour(goodFiles,sdsname='sur_refl_b03')
        lat,lon,rho4 = buildOneDayContour(goodFiles,sdsname='sur_refl_b04')
        lat,lon,rho6 = buildOneDayContour(goodFiles,sdsname='sur_refl_b06')
        
        num_ndvi = np.array(rho2-rho1,dtype=np.float32)
        den_ndvi = np.array(rho2+rho1,dtype=np.float32)
        ndvi = np.zeros(num_ndvi.shape)
        ndvi[den_ndvi > 0] = num_ndvi[den_ndvi > 0]/den_ndvi[den_ndvi > 0]
        ndvis.append(ndvi)
        
        num_vari = rho4-rho1
        den_vari = rho4+rho1-rho3
        vari = np.zeros(num_vari.shape)
        vari[den_vari > 0] = num_vari[den_vari > 0]/den_vari[den_vari > 0]
        varis.append(vari)
        
        num_ndi16 = rho2-rho6
        den_ndi16 = rho2+rho6
        ndi16 = np.zeros(num_ndi16.shape)
        ndi16[den_ndi16 > 0] = num_ndi16[den_ndi16 > 0]/den_ndi16[den_ndi16 > 0]
        ndi16s.append(ndi16)
        
        num_vigr = rho4-rho1
        den_vigr = rho4+rho1
        vigr = np.zeros(num_vigr.shape)
        vigr[den_vigr > 0] = num_vigr[den_vigr > 0]/den_vigr[den_vigr > 0]
        vigrs.append(vigr)
    
    vigrMin, vigrMax, vigrMedian, vigrMean = list2stats(vigrs,name=outdir+'vigrStats2016.pkl')
    ndviMin, ndviMax, ndviMedian, ndviMean = list2stats(ndvis,name=outdir+'ndviStats2016.pkl')
    variMin, variMax, variMedian, variMean = list2stats(varis,name=outdir+'variStats2016.pkl')
    ndi16Min, ndi16Max, ndi16Median, ndi16Mean = list2stats(ndi16s,name=outdir+'ndi16Stats2016.pkl')
    
    uc.dumpPickle([dates,lat,lon,vigrs],outdir+'vigrAll.pkl')
    uc.dumpPickle([dates,lat,lon,ndvis],outdir+'ndvisAll.pkl')
    uc.dumpPickle([dates,lat,lon,varis],outdir+'varisAll.pkl')
    uc.dumpPickle([dates,lat,lon,ndi16s],outdir+'ndi16sAll.pkl')
    
    return dates, ndvis, varis, ndi16s, vigrs

def getLfmChap(vari,lfmLowerThresh=0,lfmUpperThresh=200,
               vigrFile="E:/projects/wildfire-research/data-test/vigrStats2016.pkl"):
    ''' This function will return chapperal moisture estimation based on
    VARI measurement.
    '''
    vigrMin, vigrMax, vigrMedian, vigrMean = uc.readPickle(vigrFile)
    lfm = 97.8+471.6*vari-293.9*vigrMedian-816.2*vari*(vigrMax-vigrMin)
    lfm[lfm<lfmLowerThresh] = lfmLowerThresh
    lfm[lfm>lfmUpperThresh] = lfmUpperThresh
    return lfm

def getLfmCss(vari,lfmLowerThresh=0,lfmUpperThresh=200,
              ndi16File="E:/projects/wildfire-research/data-test/ndi16Stats2016.pkl",
              ndviFile="E:/projects/wildfire-research/data-test/ndviStats2016.pkl"):
    ''' This function will return coastal ss moisture estimation beased on
    VARI measurement.
    '''
    ndi16Min, ndi16Max, ndi16Median, ndi16Mean = uc.readPickle(ndi16File)
    ndviMin, ndviMax, ndviMedian, ndviMean = uc.readPickle(ndviFile)
    lfm = 179.2 + 1413.9*vari-450.5*ndi16Median-1825.2*vari*(ndviMax-ndviMin)
    lfm[lfm<lfmLowerThresh] = lfmLowerThresh
    lfm[lfm>lfmUpperThresh] = lfmUpperThresh
    return lfm

def buildCanopyData(datadir='G:/WildfireResearch/data/terra_canopy/',
                    outdir = "E:/projects/wildfire-research/data-test/",
                    sdsname='Percent_Tree_Cover',
                    outname='canopy.pkl'):
    ds = 1
    method='linear'
    files = glob.glob(datadir+'/*.hdf')
    #f = returnDataFile(files[0])
    lat,lon,data = buildOneDayContour(files,sdsname=sdsname,targetPixels=1200)
    data[lat==0] = np.nan
    lat[lat == 0] = np.nan
    lon[lon == 0] = np.nan
    data[data > 100] = 100
    
    lat = np.reshape(lat,(lat.shape[0]*lat.shape[1]))
    lon = np.reshape(lon,(lon.shape[0]*lon.shape[1]))
    values = np.reshape(data,(data.shape[0]*data.shape[1]))

    inds = np.where(~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(values))
    lat = lat[inds]
    lon = lon[inds]
    values = values[inds]
    
    pts = np.zeros((len(lat),2))
    pts[:,0] = lat
    pts[:,1] = lon
    
    newpts, sz = getCustomGrid(reshape=True)

    remapped = scpi.griddata(pts[0::ds],values[0::ds],newpts,method=method)
    
    data = np.reshape(remapped,(sz[0],sz[1]))
    latitude, longitude = getCustomGrid(reshape=False)
    
    uc.dumpPickle([latitude,longitude,data],outdir+outname)
    return latitude, longitude, data


def getCustomGrid(lat_lmt = [30,44],
                  lon_lmt = [-126,-112],
                  pxPerDegree = 120,
                  ds=1,
                  method='nearest',
                  reshape=False):
    ''' This function will generate custom MODIS grid
    ''' 
    
    lat_lnsp = np.linspace(np.min(lat_lmt),np.max(lat_lmt),
                           (np.max(lat_lmt)-np.min(lat_lmt)+1)*pxPerDegree)
    lon_lnsp = np.linspace(np.min(lon_lmt),np.max(lon_lmt),
                           (np.max(lon_lmt)-np.min(lon_lmt)+1)*pxPerDegree)
    lon_grid, lat_grid = np.meshgrid(lon_lnsp,lat_lnsp)
    
    if reshape:
        lon_lnsp2 = np.reshape(lon_grid,(lon_grid.shape[0]*lon_grid.shape[1]))
        lat_lnsp2 = np.reshape(lat_grid,(lat_grid.shape[0]*lat_grid.shape[1]))
        newpts = np.zeros((len(lat_lnsp2),2))
        newpts[:,0] = lat_lnsp2
        newpts[:,1] = lon_lnsp2
        sz = lat_grid.shape
        return newpts, sz
    else:
        return lat_grid, lon_grid

if __name__ == '__main__':
    ''' case 0: loads modis vegetation index at queryDateTime and plots for
                the whole United states
        case 1: Loads modis active fires at queryDateTime and plots for
                California
        case 2: Loads modis vegetation index, active fires, and burned area
                at queryDateTime for California.
        case 3: Loads modis active fires at 365 consecuitive queryDateTimes
                and saves the results.
    '''
    
    # User inputs
    queryDateTime = dt.datetime(year=2017,month=7,day=9,hour=6,minute=00)
    case = 1
    
    if case == 0:
        tiles = None
        states = 'All'
        #Find vegetation index at queryDateTime
        vi_lat,vi_lon,vi_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=True,
                                                 datadir="G:/WildfireResearch/data/aqua_vegetation/",
                                                 sdsname='1 km 16 days NDVI')
        vi_fig = uc.plotContourWithStates(vi_lat,vi_lon,vi_data,states=states,label='VI')
        
        vi_mem = (sys.getsizeof(vi_data)+sys.getsizeof(vi_lat)+sys.getsizeof(vi_lon))/1024**2
        
        print("VI File Size: %.4f MB"%(vi_mem))
        
    if case == 1:
        tiles = ['h08v04','h08v05','h09v04']
        states = 'California'
        # Find activefires at queryDateTime
        af_lat,af_lon,af_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=False,
                                                 datadir="G:/WildfireResearch/data/aqua_daily_activefires/",
                                                 sdsname='FireMask')
        af_fig = uc.plotContourWithStates(af_lat,af_lon,af_data,states=states,
                                          clim=np.linspace(0,9,10),label='AF',
                                          xlim=[-121.5, -118.5], ylim=[33.5, 36.5], saveFig=True)
        
        af_mem = (sys.getsizeof(af_data)+sys.getsizeof(af_lat)+sys.getsizeof(af_lon))/1024**2
        
        print("AF File Size: %.4f MB"%(af_mem))
        
    if case == 2:
        tiles = ['h08v04','h08v05','h09v04']
        states = 'California'
        # Find activefires at queryDateTime
        af_lat,af_lon,af_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=False,
                                                 datadir="G:/WildfireResearch/data/aqua_daily_activefires/",
                                                 sdsname='FireMask')
        #Find vegetation index at queryDateTime
        vi_lat,vi_lon,vi_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=True,
                                                 datadir="G:/WildfireResearch/data/aqua_vegetation/",
                                                 sdsname='1 km 16 days NDVI')
        #Find burned area at queryDateTime
        ba_lat,ba_lon,ba_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=True,
                                                 datadir="G:/WildfireResearch/data/modis_burnedarea/",
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
        #queryDateTime = dt.datetime(year=2016,month=1,day=1,hour=12,minute=0)
        outdir = 'E:\\projects\\forensics\\parkfield\\'
        for i in range(0,365):
            af_name = outdir+'AF2_'+queryDateTime.isoformat()[0:13]+'.png'
        
            af_lat,af_lon,af_data = findQuerySdsData(queryDateTime,tiles=tiles,composite=False,
                                                     datadir="G:/WildfireResearch/data/terra_daily_activefires/",
                                                     sdsname='FireMask')
            if af_data is not None:
                af_fig = uc.plotContourWithStates(af_lat,af_lon,af_data,states=states,
                                                  clim=np.linspace(0,9,10),label='AF',
                                                  saveFig=True,saveName=af_name)
            
                af_mem = (sys.getsizeof(af_data)+sys.getsizeof(af_lat)+sys.getsizeof(af_lon))/1024**2
                
                data_mask = af_data.copy()
                data_mask[data_mask < 7] = 0
                pts = geolocateCandidates(af_lat,af_lon,data_mask)
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
        
        
        
    if case == 4:
        datadir = "E:/projects/wildfire-research/data-test/"
    
        dates, lat, lon, varis = uc.readPickle(datadir+'varisAll.pkl')
        
        for i in range(0,1):#len(varis)):
            lfm_chap = getLfmChap(varis[i])
            #lfm_css = getLfmCss(varis[i])
            uc.plotContourWithStates(lat,lon,lfm_chap,
                                     clim=np.linspace(0,200,11))
                                     #saveFig=True,saveName=datadir+"lfmCss_"+dates[i]+".png",)
    
    if case == 5:
        lat, lon, data = buildCanopyData()
        uc.plotContourWithStates(lat,lon,data,clim=np.linspace(0,100,11))
        """
        datadir = 'G:/WildfireResearch/data/terra_canopy/'
        outdir = "E:/projects/wildfire-research/data-test/"
        files = glob.glob(datadir+'/*.hdf')
        #f = returnDataFile(files[0])
        lat,lon,data = buildOneDayContour(files,sdsname='Percent_Tree_Cover',targetPixels=1200)
        data[lat==0] = np.nan
        lat[lat == 0] = np.nan
        lon[lon == 0] = np.nan
        data[data > 100] = 100
        uc.plotContourWithStates(lat,lon,data,clim=np.linspace(0,100,11))
        uc.dumpPickle([lat,lon,data],outdir+'canopy.pkl')
        """
        
