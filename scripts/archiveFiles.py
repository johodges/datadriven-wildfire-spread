# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:48:25 2019

@author: jhodges
"""

import numpy as np
import os
import glob
import subprocess

if __name__ == "__main__":
    
    indir = "F:\\WildfireResearch\\farsiteResults\\"
    outdir = "/c/projects/wildfire-research/toMove/"
    
    files = glob.glob("%srun_*"%(indir))
    
    prefixes = ['run_%s_%s'%(x.split('_')[1], x.split('_')[2]) for x in files]
    
    uniques = list(set(prefixes))
    uniques.sort()
    
    indir = "/f/WildfireResearch/farsiteResults/"
    
    commands = ['7z a %s%s.zip %s%s_*'%(outdir, x, indir, x) for x in uniques]
    
    fileToWrite = "C:\\projects\\wildfire-research\\toMove\\moveFiles.sh"
    with open(fileToWrite, 'w') as f:
        for command in commands:
            f.write("%s\n"%(command))
        