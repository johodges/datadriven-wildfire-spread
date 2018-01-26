# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:33:15 2018

@author: JHodges
"""

"""
def buildActiveFireContour(files,queryDateTime,sdsname='FireMask'):
    ''' This function will combine active fire measurements from multiple
    MODIS tiles into a single dataset. The list of file names should
    correspond to the same time and be for different tiles. The file names
    should reference the (.hdf) files. This differs from buildCompositeContour
    in that there is daily data present within the (.hdf) file.
    '''
    pixels = loadSdsData(files[0],sdsname).shape[1]
    tiles_grid_dict, tiles_grid = mapTileGrid(tiles,pixels)
    tiles_data = tiles_grid.copy()
    tiles_lat = tiles_grid.copy()
    tiles_lon = tiles_grid.copy()
    for file in files:
        p = loadGPolygon(file+'.xml')
        startdate, enddate = loadXmlDate(file+'.xml')
        plat, plon = arrangeGPolygon(p)
        
        day_index = activeFireDayIndex([startdate,enddate],queryDateTime)
        data = loadSdsData(file,sdsname)[day_index,:,:]
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
    
def testcase(case=4):
    #datename = '2017177.h'
    #datename = '2016001.h'
    #datename = '2016033.h'
    #datename = '2016025.h'
    
    if case == 0:
        datadir = "E:/WildfireResearch/data/aqua_activefires/"
        datename = '2016033.h'
        sdsname = 'FireMask'
        pixels = 1200
    elif case == 1:
        datadir = "E:/WildfireResearch/data/aqua_temperature/"
        datename = '2016033.h'
        sdsname = 'LST_Day_1km'
        pixels = 1200
    elif case == 2:
        datadir = "E:/WildfireResearch/data/aqua_vegetation/"
        datename = '2016025.h'
        sdsname = '1 km 16 days NDVI'
        pixels = 1200
    elif case == 3:
        datadir = "E:/WildfireResearch/data/modis_burnedarea/"
        datename = '2016032.h'
        sdsname = 'burndate'
        pixels = 2400
    elif case == 4:
        datadir = "E:/WildfireResearch/data/terra_daily_activefires/"
        datename = '2017137.h'
        sdsname = 'FireMask'
        pixels = 1200
        #MOD14A1.A2017137.h09v05.006.2017145230951
        
    for k in range(0,1):
        tiles = ['h08v04','h08v05','h09v04']
        tiles_grid_dict, tiles_grid = mapTileGrid(tiles,pixels)
        tiles_data = np.array(tiles_grid.copy(),dtype=np.int8)
        tiles_lat = tiles_grid.copy()
        tiles_lon = tiles_grid.copy()
        
        
        files = glob.glob(datadir+'*'+datename+'*.hdf')
        files = removeUnlistedTilesFromFiles(files,tiles,use_all=False)
        #files = glob.glob(datadir+'*'+'2016033.h*.hdf')
        #files = [x for x in os.listdir(idir) if x.endswith(".hdf")]
        
        #plt.figure(figsize=(12,8))
        lats = []
        lons = []
        datas = []
        for j in range(0,len(files)):
            for tile in tiles:
                if tile in files[j]:
                    use_tile = tile
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

def findQueryActiveFire(queryDateTime,
                        datadir="E:/WildfireResearch/data/aqua_daily_activefires/",
                        tiles=None,
                        use_all=False,
                        sdsname='FireMask'):
    # Arrange files and tiles
    if tiles is None:
        tiles = findAllTilesFromDir(datadir)
    dates, files = findXmlTimes(datadir,tiles)
    datename = findQueryDateTime(files,dates,queryDateTime)
    files, tiles = removeUnlistedTilesFromFiles(datadir,datename,tiles,use_all=False)
    
    # Load all activefire tiles at the queryDateTime
    lat,lon,data = buildContour(files,queryDateTime,sdsname=sdsname,composite=False)
    #lat,lon,data = buildActiveFireContour(files,queryDateTime,sdsname=sdsname)
    
    return lat, lon, data

def findQueryVegetation(queryDateTime,
                        datadir="E:/WildfireResearch/data/aqua_vegetation/",
                        tiles=None,
                        use_all=False,
                        sdsname='1 km 16 days NDVI'):
    # Arrange files and tiles
    if tiles is None:
        tiles = findAllTilesFromDir(datadir)
    dates, files = findXmlTimes(datadir,tiles)
    datename = findQueryDateTime(files,dates,queryDateTime)
    files, tiles = removeUnlistedTilesFromFiles(datadir,datename,tiles,use_all=False)
    
    # Load all vegetation tiles at the queryDateTime
    lat,lon,data = buildContour(files,queryDateTime,sdsname=sdsname,composite=True)
    #lat,lon,data = buildActiveFireContour(files,queryDateTime,sdsname=sdsname)
    
    return lat, lon, data
    
"""


"""
class ASOSMeasurement(object):
    __slots__ = ['dateTime','temperatureC','relativeH','directionDeg','speedMps','gustMps']
                 #'day','hour','minute','year','month']
                 #'auto','sky_condition','weather_condition','temperature_dew','sm','directionDegVar','rvr',
                 
    def __init__(self,dateTime=None):
        self.dateTime = dateTime
        self.temperatureC = None
        self.relativeH = None
        self.directionDeg = None
        self.speedMps = None
        
        self.gustMps = None
        #self.day = None
        #self.hour = None
        #self.minute = None
        #self.year = None
        #self.month = None
    
    def computeMemory(self):
        slots = self.__slots__
        #slots = ['dateTime','temperatureC','relativeH','directionDeg','speedKnot','speedMps','gustKnot','gustMps',
        #     'speedX','speedY','day','hour','minute','year','month',
             #'auto','sky_condition','weather_condition','temperature_dew','sm','directionDegVar','rvr',
        #     'speedMph','gustMph']
        mem = 0
        for key in slots:
            mem = mem+sys.getsizeof(getattr(self,key))/1024**2
        return mem
    
    def convertKnots(self,speedKnot):
        if speedKnot is not None:
            speedMps = speedKnot*0.514444
        return speedMps
        
    def convertVector(self):
        if self.directionDeg == -1:
            speedX = self.speedMps*2**0.5
            speedY = self.speedMps*2**0.5
        elif self.directionDeg is None:
            pass
            #print("Wind direction was not set.")
        elif self.speedMps is None:
            pass
            #print("Wind speed was not set.")
        else:
            try:
                speedX = self.speedMps*np.sin(self.directionDeg/180*math.pi)
                speedY = self.speedMps*np.cos(self.directionDeg/180*math.pi)
            except:
                assert False, "Unknown wind vector: %s Mps %s Deg" % (str(self.speedMps),str(self.directionDeg))
        return speedX, speedY
    
    def __str__(self):
        try:
            string = "dateTime:\t\t%s\n"%self.dateTime
        except:
            pass
        try:
            string += "temperatureC:\t%.2f degC\n"%self.temperatureC
        except:
            pass
        try:
            string += "relativeH:\t\t%.2f \%\n"%self.relativeH
        except:
            pass
        try:
            string += "WindSpeed:\t\t\t%.2f m/s\n"%self.speedMps
        except:
            pass
        try:
            string += "WindDir:\t\t\t%.2f deg\n"%self.directionDeg
        except:
            pass
        return string
    
    def __repr__(self):
        return self.__str__()
"""
