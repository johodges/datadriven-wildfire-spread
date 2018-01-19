# -*- coding: utf-8 -*-
"""
MODIS datareader

@author: JHodges

Note: If you have trouble importing phydf.SD, try this command in terminal:
    conda install -c conda-forge pyhdf

"""

import xml.etree.ElementTree as ET
import datetime
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import os
from pyhdf.SD import SD, SDC
import numpy as np

# User inputs
idir = 'xml/'
target_lat = 38.440429
target_lon = -122.7140548

# Find all header files
files = [x for x in os.listdir(idir) if x.endswith(".hdf.xml")]

# Read first hdf file to determine available fields
f = SD(idir+files[0][:-4],SDC.READ)
datasets_dic = f.datasets()
print("Available data fields:")
for idx,sds in enumerate(datasets_dic.keys()):
    print(idx,sds)

# Loop through files and determine if target is contained within area
print("Files containing coordinates:")
alldatas = []
for file in files:

    tree = ET.parse(idir+file)
    root = tree.getroot()
    dt = root[2][8]
    fmt = '%Y-%m-%d-%H:%M:%S'
    enddate = dt[1].text+'-'+dt[0].text.split('.')[0]
    startdate = dt[3].text+'-'+dt[2].text.split('.')[0]
    enddate = datetime.datetime.strptime(enddate,fmt)
    startdate = datetime.datetime.strptime(startdate,fmt)
    
    ps = root[2][9][0][0][0]
    p = []
    for i in range(0,4):
        p.append([float(ps[i][0].text),float(ps[i][1].text)])
    path = mpltPath.Path(p)
    if path.contains_point([target_lon,target_lat]):
        #print(pd.DataFrame(p))
        print(file,startdate,enddate)
        f = SD(idir+file[:-4],SDC.READ)

        sds_obj = f.select('FireMask')
        data = sds_obj.get()
        alldatas.append(data)

# Visualize map
sz = alldatas[0].shape
plt.imshow(alldatas[0][0,:,:])