# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:14 2018

@author: JHodges
"""

import time
import pickle

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