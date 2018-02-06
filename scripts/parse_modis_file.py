# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:05:33 2018

@author: jhodges
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
    f = phdf.SD(file,phdf.SDC.READ)
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

def invertModisTile(tile,pixels=1200):
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
    print(files[0])
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

"""

def removeOutsideAndReshape(lat,lon,data,
                            lat_lmt = [31,44],
                            lon_lmt = [-126,-112]):
    lat_rs = np.reshape(lat,(lat.shape[0]*lat.shape[1]))
    lon_rs = np.reshape(lon,(lon.shape[0]*lon.shape[1]))
    data_rs = np.reshape(data,(data.shape[0]*data.shape[1]))
    
    inds = np.where((lat_rs<np.max(lat_lmt)) & (lat_rs>np.min(lat_lmt)))
    lat_rs = lat_rs[inds].copy()
    lon_rs = lon_rs[inds].copy()
    data_rs = data_rs[inds].copy()
    inds = np.where((lon_rs<np.max(lon_lmt)) & (lon_rs>np.min(lon_lmt)))
    lat_rs = lat_rs[inds].copy()
    lon_rs = lon_rs[inds].copy()
    data_rs = data_rs[inds].copy()
    
    return lat_rs, lon_rs, data_rs

def gridAndResample(lat,lon,data,
                    lat_lmt = [31,44],
                    lon_lmt = [-126,-112],
                    pxPerDegree = 120,
                    ds=1,
                    method='nearest'):
    lat = np.array(lat)
    lon = np.array(lon)
    data = np.array(data)
    pts = np.zeros((len(lat),2))
    pts[:,0] = lat
    pts[:,1] = lon    
    
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
    
    data_lnsp = scpi.griddata(pts[0::ds],data[0::ds],newpts,method=method)
    data_grid = np.reshape(data_lnsp,(lat_grid.shape[0],lat_grid.shape[1]))
    
    lat_grid = np.array(lat_grid,dtype=np.float32)
    lon_grid = np.array(lon_grid,dtype=np.float32)
    
    return lat_lnsp, lon_lnsp, data_grid

def generateCustomHdf(file,sdsname,lat,lon,data,times,
                      sdsdescription='none',
                      sdsunits='none'):
    hdfFile = SD(file,SDC.WRITE|SDC.CREATE) # Assign a few attributes at the file level
    hdfFile.author = 'JENSEN HUGHES'
    hdfFile.productionDate = time.strftime('%Y%j.%H%M',time.gmtime(time.time()))
    hdfFile.minTimeStamp = str('%.4f'%(np.min(times)))
    hdfFile.maxTimeStamp = str('%.4f'%(np.max(times)))
    hdfFile.latitudeL = str('%.8f'%(lat[0]))
    hdfFile.latitudeR = str('%.8f'%(lat[-1]))
    hdfFile.longitudeL = str('%.8f'%(lon[0]))
    hdfFile.longitudeR = str('%.8f'%(lon[-1]))
    hdfFile.priority = 2
    d1 = hdfFile.create(sdsname, SDC.FLOAT32, data.shape)
    d1.description = sdsdescription
    d1.units = sdsunits
    dim1 = d1.dim(0)
    dim2 = d1.dim(1)
    dim1.setname('latitude')
    dim2.setname('longitude')
    dim1.units = 'degrees'
    dim2.units = 'degrees'
    d1[:] = data
    d1.endaccess()
    hdfFile.end()

def gridFromCustomHdf(file,sdsname):
    data = file.select(sdsname).get()
    latitudeL = file.attributes()['latitudeL']
    latitudeR = file.attributes()['latitudeR']
    longitudeL = file.attributes()['longitudeL']
    longitudeR = file.attributes()['longitudeR']
    
    lat_lnsp = np.linspace(latitudeL,latitudeR,data.shape[0])
    lon_lnsp = np.linspace(longitudeL,longitudeR,data.shape[1])
    
    lon_grid, lat_grid = np.meshgrid(lon_lnsp,lat_lnsp)
    
    return lat_grid, lon_grid, data

"""

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
    case = 1
    
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
    
    """
    if case == 4:
        import matplotlib.pyplot as plt
        import pandas as pd
        import scipy.interpolate as scpi
        import time
        #indir = "E:/WildfireResearch/data/terra_hourly_activefires_rp/"
        indir = "E:/WildfireResearch/data/terra_hourly_activefires/"
        geodir = "E:/WildfireResearch/data/terra_geolocation/"
        outdir = "E:/WildfireResearch/data/terra_hourly_activefires_jh/"
        files = glob.glob(indir+'/*2017227*.hdf')
        geofiles = glob.glob(geodir+'/*2017227*.hdf')
        
        for i in range(0,len(files)):
            
            datas_day = []
            lats_day = []
            lons_day = []
            times_day = []
            names_day = []
            datas_night = []
            lats_night = []
            lons_night = []
            times_night = []
            names_night = []
        
        for i in range(0,len(files)): #file in files:
            file = files[i]
            basename = file.split('\\rp_')[1] #[:-4]
            dtstr = basename[7:19]
            s = "%s %s %s %s"%(dtstr[0:4],dtstr[4:7],dtstr[8:10],dtstr[10:12])
            dateTime = time.mktime(time.strptime(s,'%Y %j %H %M'))/(3600*24)
            gfile = [s for s in geofiles if dtstr in s][0]
            

            sdsname="fire mask"
            #f = SD(file,SDC.READ)
            f = SD(xmldir+basename,SDC.READ)
            sds_obj = f.select(sdsname)
            data = sds_obj.get()
            
            f2 = SD(gfile,SDC.READ)
            lat = f2.select('Latitude').get()
            lon = f2.select('Longitude').get()
            
            lat_rs,lon_rs,data_rs = removeOutsideAndReshape(lat,lon,data)
            
            if int(basename[15:19]) > 800 and int(basename[15:19]) < 2200:
                #plt.figure(1)
                datas_day.extend(data_rs)
                lats_day.extend(lat_rs)
                lons_day.extend(lon_rs)
                times_day.append(dateTime)
                names_day.append(basename)
            else:
                #plt.figure(2)
                datas_night.extend(data_rs)
                lats_night.extend(lat_rs)
                lons_night.extend(lon_rs)
                times_night.append(dateTime)
                names_night.append(basename)
            #plt.contourf(lon,lat,data,[0,1,2,3,4,5,6,7,8,9],cmap='jet')
        lats_day, lons_day, datas_day = gridAndResample(lats_day,lons_day,datas_day)
        lats_night, lons_night, datas_night = gridAndResample(lats_night,lons_night,datas_night)
        

        plt.figure(1)
        fig1 = uc.plotContourWithStates(lats_day,lons_day,datas_day,
                                        clim=np.linspace(0,9,10),label='AF',
                                        saveFig=False,saveName='noname')

        plt.figure(2)
        fig2 = uc.plotContourWithStates(lats_night,lons_night,datas_night,
                                        clim=np.linspace(0,9,10),label='AF',
                                        saveFig=False,saveName='noname')

        name = names_day[0].split('MOD14.A')[1].split('.')
        file = outdir+'MOD14JH.A'+name[0]+'.dddd.'+name[2]+time.strftime('%Y%j.%H%M',time.gmtime(time.time()))+'.hdf'
        generateCustomHdf(file,'FireMask',lats_day,lons_day,datas_day,times_day,
                          sdsdescription='Active fires/thermal anomalies mask',
                          sdsunits='none')
        
        name = names_night[0].split('MOD14.A')[1].split('.')
        file = outdir+'MOD14JH.A'+name[0]+'.nnnn.'+name[2]+time.strftime('%Y%j.%H%M',time.gmtime(time.time()))+'.hdf'
        generateCustomHdf(file,'FireMask',lats_night,lons_night,datas_night,times_night,
                          sdsdescription='Active fires/thermal anomalies mask',
                          sdsunits='none')
        """
    