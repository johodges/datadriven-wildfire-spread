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

def tic():
    return time.time()

def toc(tim):
    tom = time.time()
    print("%.4f sec Elapsed"%(tom-tim))
    return tom

def dumpPickle(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
        
def readPickle(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

def mapTileGrid(tiles,pixels,coordFromNameFnc):
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
    h,v = tile_grid_dict[tile]
    start_h = (h*pixels)
    end_h = ((h+1)*pixels)
    start_v = (v*pixels)
    end_v = ((v+1)*pixels)
    (start_h,end_h,start_v,end_v) = (int(start_h),int(end_h),int(start_v),int(end_v))
    tile_grid[start_v:end_v,start_h:end_h] = data.copy()
    return tile_grid

def getStateBoundaries(state='California',filename='../data-sample/states.xml'):
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
                          label=''):
    ''' This function will plot a contour map using latitude, longitude, and
    data matricies. State boundary lines will be overlaid on the contour.
    If a list of matrices are passed instead, each contour in the list will
    be plotted.
    '''
    if states is None:
        states = getStateBoundaries(state='California')
    elif type(states) is str:
        states=getStateBoundaries(state=states)
    
    if saveFig:
        fig = plt.figure(figsize=(96,64))
        fntsize = 160
    else:
        fig = plt.figure(figsize=(12,8))
        fntsize = 20
    
    if type(data) is list:
        for i in range(0,len(data)):
            da = data[i]
            la = lat[i]
            lo = lon[i]
            if clim is None:
                plt.contourf(lo,la,da,cmap='jet')
            else:
                plt.contourf(lo,la,da,clim,cmap='jet')
    else:
        if clim is None:
            plt.contourf(lon,lat,data,cmap='jet')
        else:
            plt.contourf(lon,lat,data,clim,cmap='jet')
    
    if clim is None:
        plt.colorbar(label=label)
    else:
        plt.colorbar(label=label,ticks=clim)
    
    plt.xlabel('Longitude',fontsize=fntsize)
    plt.ylabel('Latitude',fontsize=fntsize)
    
    if type(states) is dict:
        for state in states:
            plt.plot(states[state][:,1],states[state][:,0],'-k')
        if xlim is None:
            xlim = [-128,-66]
        if ylim is None:
            ylim = [24,50]
    else:
        plt.plot(states[:,1],states[:,0],'-k')
        if xlim is None:
            xlim = [-126,-113]
        if ylim is None:
            ylim = [32,43]
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    if saveFig:
        
        for tick in fig.xaxis.get_major_ticks():
            tick.label.set_fontsize(fntsize)
        for tick in fig.yaxis.get_major_ticks():
            tick.label.set_fontsize(fntsize)
        
        fig.savefig(saveName)
        plt.clf()
        plt.close(fig)
    
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

"""
def fillEmptyCoordinates(data,tiles,pixels,coordFromNameFnc):
    tiles_grid_dict, tiles_grid = mapTileGrid(tiles,pixels,coordFromNameFnc)
    [r,c] = data.shape
    print(data.shape)
    rtiles = int(r/pixels)
    ctiles = int(c/pixels)
    bt_r, bt_c = tiles_grid_dict[tiles[0]]
    print(bt_r,bt_c)
    bg_ls_y = np.linspace(bt_r*pixels,(bt_r+1)*pixels,pixels)
    bg_ls_x = np.linspace(bt_c*pixels,(bt_c+1)*pixels,pixels)
    bg_x, bg_y = np.meshgrid(bg_ls_y,bg_ls_x)
    print(bg_x.shape,bg_y.shape)
    bg_x = np.reshape(bg_x,(pixels*pixels,))
    bg_y = np.reshape(bg_y,(pixels*pixels,))
    print(bg_x.shape,bg_y.shape)
    print(bt_r*pixels,(bt_r+1)*pixels,bt_c*pixels,(bt_c+1)*pixels)
    bg_d = data[bt_r*pixels:(bt_r+1)*pixels,bt_c*pixels:(bt_c+1)*pixels]
    print(bg_d.shape)
    bd = np.reshape(bg_d,(pixels*pixels,))
    print(bd.shape)
    
    for i in range(0,rtiles):
        for j in range(0,ctiles):
            start_c = (j*pixels)
            end_c = ((j+1)*pixels)
            start_r = (i*pixels)
            end_r = ((i+1)*pixels)
            (start_c,end_c,start_r,end_r) = (int(start_c),int(end_c),int(start_r),int(end_r))
            if np.sum(data[start_r:end_r,start_c:end_c]) == 0:
                grid_x, grid_y = np.meshgrid(np.linspace(start_r,end_r,pixels),np.linspace(start_c,end_c,pixels))
                grid_rs_x = np.reshape(grid_x,(pixels*pixels,))-bg_x
                grid_rs_y = np.reshape(grid_y,(pixels*pixels,))-bg_y
                interp = scpi.interp2d(bg_x,bg_y,bd)
                data_ip = interp(grid_rs_x,grid_rs_y)
                data[start_r:end_r,start_c:end_c] = np.reshape(data_ip,(pixels,pixels))
            

    interp = scpi.interp2d(colsRemoved[0::ds],rowsRemoved[0::ds],dataRemoved[0::ds])
    print(len(naninds))
    print(rows[naninds].shape,rows[naninds].shape,cols[naninds].shape)
    dataNan[naninds] = interp(rows[naninds][:,0],cols[naninds][:,0])
            
    tile_grid[start_v:end_v,start_h:end_h] = data.copy()
"""
    
    
    
"""
    dataNan = data.copy()
    dataNan[data == 0] = np.nan
    
    dataNanGradX = dataNan.copy()
    dataNanGradY = dataNan.copy()
    dataNanGradX[1:-2,:] = dataNanGradX[0:-3,:]-dataNanGradX[2:-1,:]
    dataNanGradY[1:-2,:] = dataNanGradY[0:-3,:]-dataNanGradY[2:-1,:]
    dx = np.nanmean(dataNanGradX,axis=1)
    dy = np.nanmean(dataNanGradY,axis=0)
    
    
    
"""
"""
    dataNan = np.reshape(dataNan,(r*c,))
    rows = np.zeros((r,c))
    cols = np.zeros((r,c))
    for i in range(0,r):
        rows[i,:] = i
    for j in range(0,c):
        cols[:,i] = i
    dataNan = np.reshape(dataNan,(r*c,))
    rows = np.reshape(rows,(r*c,))
    cols = np.reshape(cols,(r*c,))
    naninds = np.argwhere(np.isnan(dataNan))
    rowsRemoved = rows[~naninds]
    colsRemoved = cols[~naninds]
    dataRemoved = dataNan.copy()[~naninds]
    
    ds = 1000
    interp = scpi.interp2d(colsRemoved[0::ds],rowsRemoved[0::ds],dataRemoved[0::ds])
    print(len(naninds))
    print(rows[naninds].shape,rows[naninds].shape,cols[naninds].shape)
    dataNan[naninds] = interp(rows[naninds][:,0],cols[naninds][:,0])
    data = np.reshape(dataNan,(r,c))
    return data
"""