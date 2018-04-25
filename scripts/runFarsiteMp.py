# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:41:16 2018

@author: JHodges
"""

import subprocess

def runFarsite(commandFile):
    dockerStart = 'docker run -it -v E:\\projects\\wildfire-research\\farsite\\:/commonDir/ farsite'
    dockerCmd = './commonDir/farsite/src/TestFARSITE %s'%(commandFile)
    
    p = subprocess.Popen('winpty '+dockerStart+' '+dockerCmd,shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE)
    p_status = p.wait()
    