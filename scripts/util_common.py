# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:14 2018

@author: JHodges
"""

import time
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import scipy.interpolate as scpi
from matplotlib.colors import LinearSegmentedColormap
import imageio
import glob
import psutil

def tic():
    ''' This function returns the current time.
    '''
    return time.time()

def toc(tim):
    ''' This function prints the time since the input time and returns the
    current time.
    '''
    tom = time.time()
    print("%.4f sec Elapsed"%(tom-tim))
    return tom

def dumpPickle(data,filename):
    ''' This function will dump data to filename for future reuse.
    '''
    with open(filename,'wb') as f:
        pickle.dump(data,f)
        
def readPickle(filename):
    ''' This function will read data from filename and return it
    '''
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

def mapTileGrid(tiles,pixels,coordFromNameFnc):
    ''' This function will create a larger matrix which encompasses multiple
    MODIS Level 3 tiles. Returns a dictionary with key = tile name and value =
    grid coordinates in larger matrix and the larger grid.
    '''
    h_min = 361
    h_max = -361
    v_min = 361
    v_max = -361
    tiles_grid_dict = dict()
    for tile in tiles:
        v_tile,h_tile = coordFromNameFnc(tile)
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
    ''' This function will fill measurements associated with a single MODIS
    Level 3 tile into a previously generated larger tile grid encompassing
    multiple tiles.
    '''
    
    h,v = tile_grid_dict[tile]
    start_h = (h*pixels)
    end_h = ((h+1)*pixels)
    start_v = (v*pixels)
    end_v = ((v+1)*pixels)
    (start_h,end_h,start_v,end_v) = (int(start_h),int(end_h),int(start_v),int(end_v))
    tile_grid[start_v:end_v,start_h:end_h] = data.copy()
    return tile_grid

def getStateBoundaries(state='California',filename='../data-sample/states.xml'):
    ''' This function will load the latitude/longitude boundaries of USA
    states from an xml file. the xml file used here was taken from:
        http://econym.org.uk/gmap/states.xml
    '''
    if state == 'All':
        all_states = True
        all_pts = dict()
    else:
        all_states = False
    tree = ET.parse(filename)
    root = tree.getroot()
    for actor in root.findall('state'):
        if actor.attrib['name'] == state or all_states:   
            pts = []             
            for point in actor.findall('point'):
                pts.append([float(point.attrib['lat']),float(point.attrib['lng'])])
            pts = np.array(pts,dtype=np.float32)
            if all_states:
                state_name = actor.attrib['name']
                all_pts[state_name] = pts
    if all_states:
        return all_pts
    else:
        return pts
    
def plotContourWithStates(lat,lon,data,
                          states=None,clim=None,xlim=None,ylim=None,
                          saveFig=False,saveName='',
                          label='',cmap='jet'):
    ''' This function will plot a contour map using latitude, longitude, and
    data matricies. State boundary lines will be overlaid on the contour.
    If a list of matrices are passed instead, each contour in the list will
    be plotted.
    '''
    if states is None:
        states = getStateBoundaries(state='California')
    elif type(states) is str:
        states=getStateBoundaries(state=states)
    
    if cmap == 'satComp':
        colors = [(1, 1, 1), (0, 0, 1), (1, 0, 0)] 
        cmap = LinearSegmentedColormap.from_list('satComp', colors, N=3)
    
    if saveFig:
        fntsize = 80
        lnwidth = 10
        fig = plt.figure(figsize=(48,38),tight_layout=True)
        #fig = plt.figure(figsize=(12,8),tight_layout=True)

    else:
        fig = plt.figure(figsize=(12,8),tight_layout=True)
        fntsize = 20
        lnwidth = 2
    ax1 = fig.add_subplot(1,1,1)
    
    if type(data) is list:
        for i in range(0,len(data)):
            da = data[i]
            la = lat[i]
            lo = lon[i]
            if clim is None:
                ax1.contourf(lo,la,da,cmap=cmap)
            else:
                ax1.contourf(lo,la,da,clim,cmap=cmap)
    else:
        if clim is None:
            img = ax1.contourf(lon,lat,data,cmap=cmap)
        else:
            img = ax1.contourf(lon,lat,data,clim,cmap=cmap)
    
    if clim is None:
        img_cb = fig.colorbar(img,label=label)
    else:
        img_cb = fig.colorbar(img,label=label,ticks=clim)
    
    img_cb.set_label(label=label,fontsize=fntsize)
    
    plt.xlabel('Longitude',fontsize=fntsize)
    plt.ylabel('Latitude',fontsize=fntsize)
    
    if type(states) is dict:
        for state in states:
            ax1.plot(states[state][:,1],states[state][:,0],'-k')
        if xlim is None:
            xlim = [-125,-66]
        if ylim is None:
            ylim = [24,50]
    else:
        #ax1.plot(states[:,1],states[:,0],'-k')
        if xlim is None:
            xlim = [-125,-113]
        if ylim is None:
            ylim = [32,43]
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    if saveFig:
        img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        ax1.tick_params(axis='both',labelsize=fntsize)
        ax1.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        for ln in ax1.lines:
            ln.set_linewidth(lnwidth)
        
        fig.savefig(saveName)
        
        plt.clf()
        plt.close(fig)
        print(saveName)
    
    return fig

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

def fillEmptyCoordinatesRectilinear(data):
    ''' This function will fill empty values in a rectilinear grid with only
    the corners defined.
    '''
    [r,c] = data.shape
    dataNan = data.copy()
    dataNan[data == 0] = np.nan
    r0mn = np.nanmin(dataNan[0,:])
    r0mx = np.nanmax(dataNan[0,:])
    c0mn = np.nanmin(dataNan[:,0])
    c0mx = np.nanmax(dataNan[:,0])
    #print(r0mn,r0mx,c0mn,c0mx)
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

def makeGIF(indir,outfile,ext='.png',fps=None):
    ''' This function will take all files with extension ext from an input
    director indir and create a video file named outfile.
    
    NOTE: If outfile ends in (.gif) the file will be a gif, if it ends in
        (.mp4) the file will be an mp4.
    NOTE: (.mp4) is preferred to reduce file size.
    NOTE: Specifying a custom fps with (.mp4) will cause issues in some media
        players.
    '''
    if indir[-1] == '/':
        files = glob.glob(indir+'*'+ext)
    else:
        files = glob.glob(indir+'/*'+ext)
    if len(files) > 0:
        if fps is None:
            writer = imageio.get_writer(outfile)
        else:
            writer = imageio.get_writer(outfile,fps=fps)
        for file in files:
            mem = psutil.virtual_memory()[2]
            if mem < 90.0:
                print(file)
                writer.append_data(imageio.imread(file))
            else:
                print("Memory full.")
        writer.close()

def findTensorflowGPUs():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def plotImageList(datas,names,
                     clims=None,closeFig=None,
                     saveFig=False,saveName=''):
    totalPlots = np.ceil(float(len(datas))**0.5)
    colPlots = totalPlots
    rowPlots = np.ceil((float(len(datas)))/colPlots)
    currentPlot = 0
    
    if saveFig:
        fntsize = 20
        lnwidth = 5
        fig = plt.figure(figsize=(colPlots*12,rowPlots*10))#,tight_layout=True)      
        if closeFig is None:
            closeFig = True
    else:
        fig = plt.figure(figsize=(colPlots*6,rowPlots*5))#,tight_layout=True)
        fntsize = 20
        lnwidth = 2
        if closeFig is None:
            closeFig = False
        
    xmin = 0
    xmax = datas[0].shape[1]
    xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/10)+1))
    ymin = 0
    ymax = datas[0].shape[0]
    yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/10)+1))

    for i in range(0,len(names)):
        key = names[i]
        currentPlot = currentPlot+1

        ax = fig.add_subplot(rowPlots,colPlots,currentPlot)
        ax.tick_params(axis='both',labelsize=fntsize)
        plt.xticks(xticks)
        plt.yticks(yticks)
        #plt.xlabel('Longitude',fontsize=fntsize)
        #plt.ylabel('Latitude',fontsize=fntsize)
        plt.title(key,fontsize=fntsize)

        if clims is None:
            clim = np.linspace(0,1,10)
            label = ''
        else:
            clim = clims[i]
            label = ''
        img = ax.imshow(datas[i],cmap='jet',vmin=clim[0],vmax=clim[-1])#,vmin=0,vmax=1)
        #img = ax.contourf(self.longitude,self.latitude,getattr(self,key),levels=clim,cmap=cmap)
        img_cb = plt.colorbar(img,ax=ax,label=label)

        img_cb.set_label(label=label,fontsize=fntsize)
        img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        ax.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        for ln in ax.lines:
            ln.set_linewidth(lnwidth)
    if saveFig:
        fig.savefig(saveName)
        
    if closeFig:
        plt.clf()
        plt.close(fig)

def plotGriddedList(datas,
         saveFig=False,
         closeFig=None,
         saveName='',
         clim=None,
         cmap='jet',
         label='None'):
    ''' This function will plot the gridded measurement pair
    '''
    
    inNum, outNum = datas.countData()
    inNames, outNames = datas.getDataNames()
    inTime, outTime = datas.strTime()
    
    
    totalPlots = np.ceil((float(inNum)+float(outNum))**0.5)
    colPlots = totalPlots
    rowPlots = np.ceil((float(inNum)+float(outNum))/colPlots)
    currentPlot = 0
    
    if saveFig:
        fntsize = 40
        lnwidth = 10
        fig = plt.figure(figsize=(colPlots*24,rowPlots*20))#,tight_layout=True)      
        if closeFig is None:
            closeFig = True
    else:
        fig = plt.figure(figsize=(colPlots*6,rowPlots*5))#,tight_layout=True)
        fntsize = 20
        lnwidth = 2
        if closeFig is None:
            closeFig = False
    
    xmin = np.round(np.min(datas.longitude),1)
    xmax = np.round(np.max(datas.longitude),1)
    xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/0.1)+1))
    ymin = np.round(np.min(datas.latitude),1)
    ymax = np.round(np.max(datas.latitude),1)
    yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/0.1)+1))



    names = inNames+outNames
    for i in range(0,len(names)):
        key = names[i]
        currentPlot = currentPlot+1

        ax = fig.add_subplot(rowPlots,colPlots,currentPlot)
        ax.tick_params(axis='both',labelsize=fntsize)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('Longitude',fontsize=fntsize)
        plt.ylabel('Latitude',fontsize=fntsize)
        plt.title(key,fontsize=fntsize)
        print(clim)
        if clim is None:
            if 'FireMask' in key:
                clim2 = np.linspace(0,9,10)
                label = 'AF'
            elif 'Elevation' in key:
                clim2 = np.linspace(-1000,5000,7)
                label = 'Elev [m]'
            elif 'WindX' in key:
                clim2 = np.linspace(-6,6,13)
                label = 'u [m/s]'
            elif 'WindY' in key:
                clim2 = np.linspace(-6,6,13)
                label = 'v [m/s]'
            elif 'VegetationIndex' in key:
                clim2 = np.linspace(-4000,10000,8)
                label = 'NVDIx1000'
            elif 'Canopy' in key:
                clim2 = np.linspace(0,100,11)
                label = 'Canopy'
            else:
                clim2 = np.linspace(0,9,10)
        else:
            label = ''
            clim2 = clim
            
        img = ax.contourf(datas.longitude,datas.latitude,getattr(datas,key),levels=clim2,cmap=cmap)
        img_cb = plt.colorbar(img,ax=ax,label=label)

        img_cb.set_label(label=label,fontsize=fntsize)
        img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        ax.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        for ln in ax.lines:
            ln.set_linewidth(lnwidth)
            
    if saveFig:
        fig.savefig(saveName)
        
    if closeFig:
        plt.clf()
        plt.close(fig)

if __name__ == "__main__":
    devices = findTensorflowGPUs()
    print(devices)