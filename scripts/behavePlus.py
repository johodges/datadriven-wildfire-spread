# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:17:47 2018

@author: JHodges
"""

import numpy as np
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage.interpolation import zoom
import pandas as pd
import util_common as uc

class FuelModel(object):
    ''' This class contains a fuel model for wildfire spread.
    
        Fields:
                
        Functions:

    '''
    __slots__ = ['id','code','name',
                 'fuelBedDepth','moistureOfExtinctionDeadFuel','heatOfCombustionDeadFuel','heatOfCombustionLiveFuel',
                 'fuelLoad1Hour','fuelLoad10Hour','fuelLoad100Hour','fuelLoadLiveHerb','fuelLoadLiveWood',
                 'savr1HourFuel','savrLiveHerb','savrLiveWood',
                 'isDynamic','isReserved']
    
    def __init__(self,idNumber,idCode,idName,
                 fuelBedDepth,moistureOfExtinctionDeadFuel,heatOfCombustionDeadFuel,heatOfCombustionLiveFuel,
                 fuelLoad1Hour,fuelLoad10Hour,fuelLoad100Hour,fuelLoadLiveHerb,fuelLoadLiveWood,
                 savr1HourFuel,savrLiveHerb,savrLiveWood,
                 isDynamic, isReserved):
        self.id = idNumber
        self.code = idCode
        self.name = idName
        self.fuelBedDepth = fuelBedDepth
        self.moistureOfExtinctionDeadFuel = moistureOfExtinctionDeadFuel
        self.heatOfCombustionDeadFuel = heatOfCombustionDeadFuel
        self.heatOfCombustionLiveFuel = heatOfCombustionLiveFuel
        self.fuelLoad1Hour = fuelLoad1Hour
        self.fuelLoad10Hour = fuelLoad10Hour
        self.fuelLoad100Hour = fuelLoad100Hour
        self.fuelLoadLiveHerb = fuelLoadLiveHerb
        self.fuelLoadLiveWood = fuelLoadLiveWood
        self.savr1HourFuel = savr1HourFuel
        self.savrLiveHerb = savrLiveHerb
        self.savrLiveWood = savrLiveWood
        self.isDynamic = isDynamic
        self.isReserved = isReserved
        
    def __str__(self):
        ''' This function prints summary information of the object when a
        string is requested.
        '''
        string = "Fuel Model\n"
        string = string + "\tID:\t\t%s\n"%(str(self.id))
        string = string + "\tCode:\t%s\n"%(str(self.code))
        string = string + "\tName:\t%s\n"%(str(self.name))
        return string
    
    def __repr__(self):
        ''' This function prints summary information of the object when a
        string is requested.
        '''
        return self.__str__()

class lifeStateIntermediate(object):
    __slots__ = ['dead','live']
    
    def __init__(self,fuelModel,moistureDead,moistureLive):
        
        savrDead, savrLive = getSAV(fuelModel)
        deadFraction, liveFraction, deadFractionTotal, liveFractionTotal = getDLFraction(fuelModel,moistureLive)
        loadDead, loadLive = getFuelLoad(fuelModel,moistureLive)
        heatCDead, heatCLive = getHeatOfCombustion(fuelModel)
        heatDead = np.zeros((len(savrDead),))+heatCDead
        heatLive = np.zeros((len(savrLive),))+heatCLive
        heatLive[liveFraction == 0] = 0
        moistureLive.extend([0,0])
        self.dead = self.calculateIntermediates(fuelModel,savrDead,loadDead,deadFraction,heatDead,moistureDead)
        self.live = self.calculateIntermediates(fuelModel,savrLive,loadLive,liveFraction,heatLive,moistureLive)
        
    def calculateIntermediates(self,fuelModel,savr,load,fraction,heat,moisture):
        totalSilicaContent = 0.0555 # Rothermel 1972
        silicaEffective = np.zeros((len(savr),))+0.01 # From behavePlus source, should be 0 if no fuel
        wn = np.zeros((len(savr),))
        weightedHeat = 0.0
        weightedSilica = 0.0
        weightedMoisture = 0.0
        weightedSavr = 0.0
        totalLoadForLifeState = 0.0
        """
        for i in range(0,len(moisture)):
            wn[i] = load[i]*(1.0-totalSilicaContent)
            weightedHeat = weightedHeat + fraction[i] * heat[i]
            weightedSilica = weightedSilica + fraction[i] * silicaEffective
            weightedMoisture = weightedMoisture + fraction[i]*moisture[i]
            weightedSavr = weightedSavr + fraction[i] * savr[i]
            totalLoadForLifeState = totalLoadForLifeState + load[i]
        """
        #print(fraction,moisture)
        
        wn = [x*(1.0-totalSilicaContent) for x in load] #wn[i] = load[i]*(1.0-totalSilicaContent)
        weightedHeat = np.dot(fraction,heat) #weightedHeat = weightedHeat + fraction[i] * heat[i]
        weightedSilica = np.dot(fraction,silicaEffective) #weightedSilica = weightedSilica + fraction[i] * silicaEffective
        weightedMoisture = np.dot(fraction,moisture) #weightedMoisture = weightedMoisture + fraction[i]*moisture[i]
        weightedSavr = np.dot(fraction,savr) #weightedSavr = weightedSavr + fraction[i] * savr[i]
        totalLoadForLifeState = np.sum(load) #totalLoadForLifeState = totalLoadForLifeState + load[i]
        if fuelModel.isDynamic and False:
            weightedFuelLoad = np.sum(wn)   # This gives better agreement with
                                            # behavePlus for dynamic fuel models;
                                            # however, the source code for
                                            # BehavePlus shows this should be 
                                            # weightedFuelLoad=np.dot(fraction,wn)
        else:
            weightedFuelLoad = np.dot(wn,fraction)
        return [weightedHeat,weightedSilica,weightedMoisture,weightedSavr,totalLoadForLifeState,weightedFuelLoad]

def buildFuelModels(allowDynamicModels=True,allowNonBurningModels=False):
    """
        fuelModelNumber, code, name
        fuelBedDepth, moistureOfExtinctionDeadFuel, heatOfCombustionDeadFuel, heatOfCombustionLiveFuel,
        fuelLoad1Hour, fuelLoad10Hour, fuelLoad100Hour, fuelLoadLiveHerb, fuelLoadLiveWood,
        savr1HourFuel, savrLiveHerb, savrLiveWood,
        isDynamic, isReserved
        - WMC 10/2015
    """
    fuelModels = dict()
    
    # Code FMx: Original 13 Fuel Models
    fuelModels["FM1"] = FuelModel(
            1, "FM1", "Short grass [1]",
            1.0, 0.12, 8000, 8000,
            0.034, 0, 0, 0, 0,
            3500, 1500, 1500,
            False, True)
    fuelModels["FM2"] = FuelModel(
            2, "FM2", "Timber grass and understory [2]",
            1.0, 0.15, 8000, 8000,
            0.092, 0.046, 0.023, 0.023,
            0,3000, 1500, 1500,
            False, True)
    fuelModels["FM3"] = FuelModel(
            3, "FM3", "Tall grass [3]",
            2.5, 0.25, 8000, 8000,
            0.138, 0, 0, 0, 0,
            1500, 1500, 1500,
            False, True)
    fuelModels["FM4"] = FuelModel(
            4, "FM4", "Chaparral [4]",
            6.0, 0.2, 8000, 8000,
            0.230, 0.184, 0.092, 0, 0.230,
            2000, 1500, 1500,
            False, True)
    fuelModels["FM5"] = FuelModel(
            5, "FM5", "Brush [5]",
            2.0, 0.20, 8000, 8000,
            0.046, 0.023, 0, 0, 0.092,
            2000, 1500, 1500,
            False, True)
    fuelModels["FM6"] = FuelModel(
            6, "FM6", "Dormant brush, hardwood slash [6]",
            2.5, 0.25, 8000, 8000,
            0.069, 0.115, 0.092, 0, 0,
            1750, 1500, 1500,
            False, True)
    fuelModels["FM7"] = FuelModel(
            7, "FM7", "Southern rough [7]",
            2.5, 0.40, 8000, 8000,
            0.052, 0.086, 0.069, 0, 0.017,
            1750, 1500, 1500,
            False, True)
    fuelModels["FM8"] = FuelModel(
            8, "FM8", "Short needle litter [8]",
            0.2, 0.3, 8000, 8000,
            0.069, 0.046, 0.115, 0, 0,
            2000, 1500, 1500,
            False, True)
    fuelModels["FM9"] = FuelModel(
            9, "FM9", "Long needle or hardwood litter [9]",
            0.2, 0.25, 8000, 8000,
            0.134, 0.019, 0.007, 0, 0,
            2500, 1500, 1500,
            False, True)
    fuelModels["FM10"] = FuelModel(
            10, "FM10", "Timber litter & understory [10]",
            1.0, 0.25, 8000, 8000,
            0.138, 0.092, 0.230, 0, 0.092,
            2000, 1500, 1500,
            False, True)
    fuelModels["FM11"] = FuelModel(
            11, "FM11", "Light logging slash [11]",
            1.0, 0.15, 8000, 8000,
            0.069, 0.207, 0.253, 0, 0,
            1500, 1500, 1500,
            False, True)
    fuelModels["FM12"] = FuelModel(
            12, "FM12", "Medium logging slash [12]",
            2.3, 0.20, 8000, 8000,
            0.184, 0.644, 0.759, 0, 0,
            1500, 1500, 1500,
            False, True)
    fuelModels["FM13"] = FuelModel(
            13, "FM13", "Heavy logging slash [13]",
            3.0, 0.25, 8000, 8000,
            0.322, 1.058, 1.288, 0, 0,
            1500, 1500, 1500,
            False, True)
    
    if not allowDynamicModels:
        return fuelModels
    else:
        pass
    # 14-89 Available for custom models

    if allowNonBurningModels:
        # Code NBx: Non-burnable
        # 90 Available for custom NB model  
        fuelModels["NB1"] = FuelModel(
                91, "NB1", "Urban, developed [91]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)
        fuelModels["NB2"] = FuelModel(
                92, "NB2", "Snow, ice [92]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)
        fuelModels["NB3"] = FuelModel(
                93, "NB3", "Agricultural [93]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)
    
        # Indices 94-95 Reserved for future standard non-burnable models
    
        fuelModels["NB4"] = FuelModel(
                94, "NB4", "Future standard non-burnable [94]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)
        fuelModels["NB5"] = FuelModel(
                95, "NB5", "Future standard non-burnable [95]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)
    
        # Indices 96-97 Available for custom NB model
    
        fuelModels["NB8"] = FuelModel(
                98, "NB8", "Open water [98]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)
        fuelModels["NB9"] = FuelModel(
                99, "NB9", "Bare ground [99]",
                1.0, 0.10, 8000, 8000,
                0, 0, 0, 0, 0,
                1500, 1500, 1500,
                False, True)

    # Code GRx: Grass
    # Index 100 Available for custom GR model
    f = 2000.0 / 43560.0
    fuelModels["GR1"] = FuelModel(
            101, "GR1", "Short, sparse, dry climate grass (D)",
            0.4, 0.15, 8000, 8000,
            0.10*f, 0, 0, 0.30*f, 0,
            2200, 2000, 1500,
            True, True)
    fuelModels["GR2"] = FuelModel(
            102, "GR2", "Low load, dry climate grass (D)",
            1.0, 0.15, 8000, 8000,
            0.10*f, 0, 0, 1.0*f, 0,
            2000, 1800, 1500,
            True, True)
    fuelModels["GR3"] = FuelModel(
            103, "GR3", "Low load, very coarse, humid climate grass (D)",
            2.0, 0.30, 8000, 8000,
            0.10*f, 0.40*f, 0, 1.50*f, 0,
            1500, 1300, 1500,
            True, True)
    fuelModels["GR4"] = FuelModel(
            104, "GR4", "Moderate load, dry climate grass (D)",
            2.0, 0.15, 8000, 8000,
            0.25*f, 0, 0, 1.9*f, 0,
            2000, 1800, 1500,
            True, True)
    fuelModels["GR5"] = FuelModel(
            105, "GR5", "Low load, humid climate grass (D)",
            1.5, 0.40, 8000, 8000,
            0.40*f, 0.0, 0.0, 2.50*f, 0.0,
            1800, 1600, 1500,
            True, True)
    fuelModels["GR6"] = FuelModel(
            106, "GR6", "Moderate load, humid climate grass (D)",
            1.5, 0.40, 9000, 9000,
            0.10*f, 0, 0, 3.4*f, 0,
            2200, 2000, 1500,
            True, True)
    fuelModels["GR7"] = FuelModel(
            107, "GR7", "High load, dry climate grass (D)",
            3.0, 0.15, 8000, 8000,
            1.0*f, 0, 0, 5.4*f, 0,
            2000, 1800, 1500,
            True, True)
    fuelModels["GR8"] = FuelModel(
            108, "GR8", "High load, very coarse, humid climate grass (D)",
            4.0, 0.30, 8000, 8000,
            0.5*f, 1.0*f, 0, 7.3*f, 0,
            1500, 1300, 1500,
            True, True)
    fuelModels["GR9"] = FuelModel(
            109, "GR9", "Very high load, humid climate grass (D)",
            5.0, 0.40, 8000, 8000,
            1.0*f, 1.0*f, 0, 9.0*f, 0,
            1800, 1600, 1500,
            True, True)
    # 110-112 are reserved for future standard grass models
    # 113-119 are available for custom grass models

    # Code GSx: Grass and shrub
    # 120 available for custom grass and shrub model
    fuelModels["GS1"] = FuelModel(
            121, "GS1", "Low load, dry climate grass-shrub (D)",
            0.9, 0.15, 8000, 8000,
            0.2*f, 0, 0, 0.5*f, 0.65*f,
            2000, 1800, 1800,
            True, True)
    fuelModels["GS2"] = FuelModel(
            122, "GS2", "Moderate load, dry climate grass-shrub (D)",
            1.5, 0.15, 8000, 8000,
            0.5*f, 0.5*f, 0, 0.6*f, 1.0*f,
            2000, 1800, 1800,
            True, True)
    fuelModels["GS3"] = FuelModel(
            123, "GS3", "Moderate load, humid climate grass-shrub (D)",
            1.8, 0.40, 8000, 8000,
            0.3*f, 0.25*f, 0, 1.45*f, 1.25*f,
            1800, 1600, 1600,
            True, True)
    fuelModels["GS4"] = FuelModel(
            124, "GS4", "High load, humid climate grass-shrub (D)",
            2.1, 0.40, 8000, 8000,
            1.9*f, 0.3*f, 0.1*f, 3.4*f, 7.1*f,
            1800, 1600, 1600,
            True, True)
    # 125-130 reserved for future standard grass and shrub models
    # 131-139 available for custom grass and shrub models

    # Shrub
    # 140 available for custom shrub model
    fuelModels["SH1"] = FuelModel(
            141, "SH1", "Low load, dry climate shrub (D)",
            1.0, 0.15, 8000, 8000,
            0.25*f, 0.25*f, 0, 0.15*f, 1.3*f,
            2000, 1800, 1600,
            True, True)
    fuelModels["SH2"] = FuelModel(
            142, "SH2", "Moderate load, dry climate shrub (S)",
            1.0, 0.15, 8000, 8000,
            1.35*f, 2.4*f, 0.75*f, 0, 3.85*f,
            2000, 1800, 1600,
            True, True)
    fuelModels["SH3"] = FuelModel(
            143, "SH3", "Moderate load, humid climate shrub (S)",
            2.4, 0.40, 8000., 8000.,
            0.45*f, 3.0*f, 0, 0, 6.2*f,
            1600, 1800, 1400,
            True, True)
    fuelModels["SH4"] = FuelModel(
            144, "SH4", "Low load, humid climate timber-shrub (S)",
            3.0, 0.30, 8000, 8000,
            0.85*f, 1.15*f, 0.2*f, 0, 2.55*f,
            2000, 1800, 1600,
            True, True)
    fuelModels["SH5"] = FuelModel(
            145, "SH5", "High load, dry climate shrub (S)",
            6.0, 0.15, 8000, 8000,
            3.6*f, 2.1*f, 0, 0, 2.9*f,
            750, 1800, 1600,
            True, True)
    fuelModels["SH6"] = FuelModel(
            146, "SH6", "Low load, humid climate shrub (S)",
            2.0, 0.30, 8000, 8000,
            2.9*f, 1.45*f, 0, 0, 1.4*f,
            750, 1800, 1600,
            True, True)
    fuelModels["SH7"] = FuelModel(
            147, "SH7", "Very high load, dry climate shrub (S)",
            6.0, 0.15, 8000, 8000,
            3.5*f, 5.3*f, 2.2*f, 0, 3.4*f,
            750, 1800, 1600,
            True, True)
    fuelModels["SH8"] = FuelModel(
            148, "SH8", "High load, humid climate shrub (S)",
            3.0, 0.40, 8000, 8000,
            2.05*f, 3.4*f, 0.85*f, 0, 4.35*f,
            750, 1800, 1600,
            True, True)
    fuelModels["SH9"] = FuelModel(
            149, "SH9", "Very high load, humid climate shrub (D)",
            4.4, 0.40, 8000, 8000,
            4.5*f, 2.45*f, 0, 1.55*f, 7.0*f,
            750, 1800, 1500,
            True, True)
    # 150-152 reserved for future standard shrub models
    # 153-159 available for custom shrub models

    # Timber and understory
    # 160 available for custom timber and understory model
    fuelModels["TU1"] = FuelModel(
            161, "TU1", "Light load, dry climate timber-grass-shrub (D)",
            0.6, 0.20, 8000, 8000,
            0.2*f, 0.9*f, 1.5*f, 0.2*f, 0.9*f,
            2000, 1800, 1600,
            True, True)
    fuelModels["TU2"] = FuelModel(
            162, "TU2", "Moderate load, humid climate timber-shrub (S)",
            1.0, 0.30, 8000, 8000,
            0.95*f, 1.8*f, 1.25*f, 0, 0.2*f,
            2000, 1800, 1600,
            True, True)
    fuelModels["TU3"] = FuelModel(
            163, "TU3", "Moderate load, humid climate timber-grass-shrub (D)",
            1.3, 0.30, 8000, 8000,
            1.1*f, 0.15*f, 0.25*f, 0.65*f, 1.1*f,
            1800, 1600, 1400,
            True, True)
    fuelModels["TU4"] = FuelModel(
            164, "TU4", "Dwarf conifer understory (S)",
            0.5, 0.12, 8000, 8000,
            4.5*f, 0, 0, 0, 2.0*f,
            2300, 1800, 2000,
            True, True)
    fuelModels["TU5"] = FuelModel(
            165, "TU5", "Very high load, dry climate timber-shrub (S)",
            1.0, 0.25, 8000, 8000,
            4.0*f, 4.0*f, 3.0*f, 0, 3.0*f,
            1500, 1800, 750,
            True, True)
    # 166-170 reserved for future standard timber and understory models
    # 171-179 available for custom timber and understory models
    # Timber and litter
    # 180 available for custom timber and litter models
    fuelModels["TL1"] = FuelModel(
            181, "TL1", "Low load, compact conifer litter (S)",
            0.2, 0.30, 8000, 8000,
            1.0*f, 2.2*f, 3.6*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["TL2"] = FuelModel(
            182, "TL2", "Low load broadleaf litter (S)",
            0.2, 0.25, 8000, 8000,
            1.4*f, 2.3*f, 2.2*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["TL3"] = FuelModel(
            183, "TL3", "Moderate load conifer litter (S)",
            0.3, 0.20, 8000, 8000,
            0.5*f, 2.2*f, 2.8*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["TL4"] = FuelModel(
            184, "TL4", "Small downed logs (S)",
            0.4, 0.25, 8000, 8000,
            0.5*f, 1.5*f, 4.2*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["TL5"] = FuelModel(
            185, "TL5", "High load conifer litter (S)",
            0.6, 0.25, 8000, 8000,
            1.15*f, 2.5*f, 4.4*f, 0, 0,
            2000, 1800, 160,
            True, True)
    fuelModels["TL6"] = FuelModel(
            186, "TL6", "High load broadleaf litter (S)",
            0.3, 0.25, 8000, 8000,
            2.4*f, 1.2*f, 1.2*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["TL7"] = FuelModel(
            187, "TL7", "Large downed logs (S)",
            0.4, 0.25, 8000, 8000,
            0.3*f, 1.4*f, 8.1*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["TL8"] = FuelModel(
            188, "TL8", "Long-needle litter (S)",
            0.3, 0.35, 8000, 8000,
            5.8*f, 1.4*f, 1.1*f, 0, 0,
            1800, 1800, 1600,
            True, True)
    fuelModels["TL9"] = FuelModel(
            189, "TL9", "Very high load broadleaf litter (S)",
            0.6, 0.35, 8000, 8000,
            6.65*f, 3.30*f, 4.15*f, 0, 0,
            1800, 1800, 1600,
            True, True)
    # 190-192 reserved for future standard timber and litter models
    # 193-199 available for custom timber and litter models
    # Slash and blowdown
    # 200 available for custom slash and blowdown model
    fuelModels["SB1"] = FuelModel(
            201, "SB1", "Low load activity fuel (S)",
            1.0, 0.25, 8000, 8000,
            1.5*f, 3.0*f, 11.0*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["SB2"] = FuelModel(
            202, "SB2", "Moderate load activity or low load blowdown (S)",
            1.0, 0.25, 8000, 8000,
            4.5*f, 4.25*f, 4.0*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["SB3"] = FuelModel(
            203, "SB3", "High load activity fuel or moderate load blowdown (S)",
            1.2, 0.25, 8000, 8000,
            5.5*f, 2.75*f, 3.0*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    fuelModels["SB4"] = FuelModel(
            204, "SB4", "High load blowdown (S)",
            2.7, 0.25, 8000, 8000,
            5.25*f, 3.5*f, 5.25*f, 0, 0,
            2000, 1800, 1600,
            True, True)
    
    return fuelModels

def buildFuelModelsIdx():
    fuelModels = np.empty((256,),dtype=object)
    fuelModels[1] = 'FM1'
    fuelModels[2] = 'FM2'
    fuelModels[3] = 'FM3'
    fuelModels[4] = 'FM4'
    fuelModels[5] = 'FM5'
    fuelModels[6] = 'FM6'
    fuelModels[7] = 'FM7'
    fuelModels[8] = 'FM8'
    fuelModels[9] = 'FM9'
    fuelModels[10] = 'FM10'
    fuelModels[11] = 'FM11'
    fuelModels[12] = 'FM12'
    fuelModels[13] = 'FM13'
    # 14-89 Available for custom models
    fuelModels[91] = 'NB1'
    fuelModels[92] = 'NB2'
    fuelModels[93] = 'NB3'
    # Indices 94-95 Reserved for future standard non-burnable models
    # Indices 96-97 Available for custom NB model
    fuelModels[98] = 'NB8'
    fuelModels[99] = 'NB9'
    # Index 100 Available for custom GR model
    fuelModels[101] = 'GR1'
    fuelModels[102] = 'GR2'
    fuelModels[103] = 'GR3'
    fuelModels[104] = 'GR4'
    fuelModels[105] = 'GR5'
    fuelModels[106] = 'GR6'
    fuelModels[107] = 'GR7'
    fuelModels[108] = 'GR8'
    fuelModels[109] = 'GR9'
    # 110-112 are reserved for future standard grass models
    # 113-119 are available for custom grass models
    # 120 available for custom grass and shrub model
    fuelModels[121] = 'GS1'
    fuelModels[122] = 'GS2'
    fuelModels[123] = 'GS3'
    fuelModels[124] = 'GS4'
    # 125-130 reserved for future standard grass and shrub models
    # 131-139 available for custom grass and shrub models
    # 140 available for custom shrub model
    fuelModels[141] = 'SH1'
    fuelModels[142] = 'SH2'
    fuelModels[143] = 'SH3'
    fuelModels[144] = 'SH4'
    fuelModels[145] = 'SH5'
    fuelModels[146] = 'SH6'
    fuelModels[147] = 'SH7'
    fuelModels[148] = 'SH8'
    fuelModels[149] = 'SH9'
    # 150-152 reserved for future standard shrub models
    # 153-159 available for custom shrub models
    # 160 available for custom timber and understory model
    fuelModels[161] = 'TU1'
    fuelModels[162] = 'TU2'
    fuelModels[163] = 'TU3'
    fuelModels[164] = 'TU4'
    fuelModels[165] = 'TU5'
    # 166-170 reserved for future standard timber and understory models
    # 171-179 available for custom timber and understory models
    # 180 available for custom timber and litter models
    fuelModels[181] = 'TL1'
    fuelModels[182] = 'TL2'
    fuelModels[183] = 'TL3'
    fuelModels[184] = 'TL4'
    fuelModels[185] = 'TL5'
    fuelModels[186] = 'TL6'
    fuelModels[187] = 'TL7'
    fuelModels[188] = 'TL8'
    fuelModels[189] = 'TL9'
    # 190-192 reserved for future standard timber and litter models
    # 193-199 available for custom timber and litter models
    # 200 available for custom slash and blowdown model
    fuelModels[201] = 'SB1'
    fuelModels[202] = 'SB2'
    fuelModels[203] = 'SB3'
    fuelModels[204] = 'SB4'
    return fuelModels

def getFuelModel(fuelModel):
    fuelModels = buildFuelModels(allowDynamicModels=True,allowNonBurningModels=True)
    return fuelModels[fuelModel]

def getMoistureContent(m1h,m10h,m100h,lhm,lwm):
    moistureDead = [m1h,m10h,m100h,m1h]
    moistureLive = [lhm,lwm]
    
    moistureDead = [x/100 for x in moistureDead]
    moistureLive = [x/100 for x in moistureLive]
    
    return moistureDead, moistureLive
    
def getSAV(fuelModel):
    # In behavePlus, there is a conversion to surfaceAreaToVolumeUnits
    savrDead = [fuelModel.savr1HourFuel, 109.0, 30.0, fuelModel.savrLiveHerb]
    savrLive = [fuelModel.savrLiveHerb, fuelModel.savrLiveWood, 0.0, 0.0]
    return savrDead, savrLive

def getFuelLoad(fuelModel,moistureLive):
    loadDead = [fuelModel.fuelLoad1Hour,
                fuelModel.fuelLoad10Hour,
                fuelModel.fuelLoad100Hour,
                0.0]
    loadLive = [fuelModel.fuelLoadLiveHerb,
                fuelModel.fuelLoadLiveWood,
                0.0,
                0.0]
    #print(loadDead)
    #print(loadLive)
    if fuelModel.isDynamic:
        if moistureLive[0] < 0.30:
            loadDead[3] = loadLive[0]
            loadLive[0] = 0.0
        elif moistureLive[0] <= 1.20:
            #print(loadLive[0] * (1.333 - 1.11 * moistureLive[0]))
            loadDead[3] = loadLive[0] * (1.333 - 1.11 * moistureLive[0])
            #loadDead[3] = loadLive[0] * (1.20 - moistureLive[0])/0.9
            loadLive[0] = loadLive[0] - loadDead[3]
        #print(loadLive)
        #print(loadDead)
    #print(loadDead)
    #print(loadLive)
    
    return loadDead, loadLive

def getHeatOfCombustion(fuelModel):
    heatOfCombustionDead = fuelModel.heatOfCombustionDeadFuel
    heatOfCombustionLive = fuelModel.heatOfCombustionLiveFuel
    return heatOfCombustionDead, heatOfCombustionLive

def getDLFraction(fuelModel,moistureLive):
    fuelDensity = 32.0 # Rothermel 1972
    savrDead, savrLive = getSAV(fuelModel)
    loadDead, loadLive = getFuelLoad(fuelModel,moistureLive)
    #print(loadDead)
    #print(savrDead)
    surfaceAreaDead = [x*y/fuelDensity for x,y in zip(loadDead,savrDead)]
    surfaceAreaLive = [x*y/fuelDensity for x,y in zip(loadLive,savrLive)]
    #print(surfaceAreaDead)
        
    totalSurfaceAreaDead = np.sum(surfaceAreaDead)
    totalSurfaceAreaLive = np.sum(surfaceAreaLive)
    
    fractionOfTotalSurfaceAreaDead = totalSurfaceAreaDead/(totalSurfaceAreaDead+totalSurfaceAreaLive)
    fractionOfTotalSurfaceAreaLive = 1.0 - fractionOfTotalSurfaceAreaDead
    
    if totalSurfaceAreaDead > 1.0e-7:
        deadFraction = [x/totalSurfaceAreaDead for x in surfaceAreaDead]
    else:
        deadFraction= [0 for x in surfaceAreaDead]
    
    if totalSurfaceAreaLive > 1.0e-7:
        liveFraction = [x/totalSurfaceAreaLive for x in surfaceAreaLive]
    else:
        liveFraction= [0 for x in surfaceAreaLive]
        
    return deadFraction, liveFraction, fractionOfTotalSurfaceAreaDead, fractionOfTotalSurfaceAreaLive

def getMoistOfExt(fuelModel,moistureDead,moistureLive):
    loadDead, loadLive = getFuelLoad(fuelModel,moistureLive)
    savrDead, savrLive = getSAV(fuelModel)
    
    moistOfExtDead = fuelModel.moistureOfExtinctionDeadFuel
    
    fineDead = 0.0
    fineLive = 0.0
    fineFuelsWeightingFactor = 0.0
    weightedMoistureFineDead = 0.0
    fineDeadMoisture = 0.0
    fineDeadOverFineLive = 0.0
    for i in range(0,len(loadDead)):
        if savrDead[i] > 1.0e-7:
            fineFuelsWeightingFactor = loadDead[i] * np.exp(-138.0/savrDead[i])
        fineDead = fineDead + fineFuelsWeightingFactor
        weightedMoistureFineDead = weightedMoistureFineDead + fineFuelsWeightingFactor * moistureDead[i]
    if fineDead > 1.0e-7:
        fineDeadMoisture = weightedMoistureFineDead / fineDead
    
    for i in range(0,len(loadLive)):
        if savrLive[i] > 1.0e-7:
            fineLive = fineLive + loadLive[i]*np.exp(-500.0/savrLive[i])
    
    if fineLive > 1.0e-7:
        fineDeadOverFineLive = fineDead / fineLive
    
    moistOfExtLive = (2.9 * fineDeadOverFineLive * (1.0 - (fineDeadMoisture) / moistOfExtDead)) - 0.226
    #print("MoEL:",moistOfExtLive)
    
    if moistOfExtLive < moistOfExtDead:
        moistOfExtLive = moistOfExtDead
    
    return moistOfExtDead, moistOfExtLive

def getCharacteristicSAVR(fuelModel,intermediates,moistureLive):
    deadFraction, liveFraction, deadFractionTotal, liveFractionTotal = getDLFraction(fuelModel,moistureLive)
    
    weightedSavrLive = intermediates.live[3]
    weightedSavrDead = intermediates.dead[3]
    
    sigma = deadFractionTotal * weightedSavrDead + liveFractionTotal * weightedSavrLive
    
    return sigma

def getPackingRatios(fuelModel,intermediates,moistureLive):
    fuelDensity = 32.0 # Rothermel 1972
    
    sigma = getCharacteristicSAVR(fuelModel,intermediates,moistureLive)
    
    totalLoadForLifeStateLive = intermediates.live[4]
    totalLoadForLifeStateDead = intermediates.dead[4]
    
    totalLoad = totalLoadForLifeStateLive + totalLoadForLifeStateDead
    
    depth = fuelModel.fuelBedDepth
    bulkDensity = totalLoad / depth
    
    packingRatio = totalLoad / (depth * fuelDensity)
    sigma = round(sigma,0)
    optimumPackingRatio = 3.348 / (sigma**0.8189)
    #packingRatio = round(packingRatio,4)
    relativePackingRatio = packingRatio / optimumPackingRatio
    
    return packingRatio, relativePackingRatio, bulkDensity

def getWeightedFuelLoads(fuelModel,intermediates):
    weightedFuelLoadDead = intermediates.dead[5]
    weightedFuelLoadLive = intermediates.live[5]
    return weightedFuelLoadDead, weightedFuelLoadLive

def getWeightedHeats(fuelModel,intermediates):
    weightedHeatDead = intermediates.dead[0]
    weightedHeatLive = intermediates.live[0]
    return weightedHeatDead, weightedHeatLive
    
def getWeightedSilicas(fuelModel,intermediates):
    weightedSilicaDead = intermediates.dead[1]
    weightedSilicaLive = intermediates.live[1]
    return weightedSilicaDead, weightedSilicaLive

def getHeatSink(fuelModel,moistureDead,moistureLive,bulkDensity):
    savrDead, savrLive = getSAV(fuelModel)
    qigDead = np.zeros((len(savrDead),))
    qigLive = np.zeros((len(savrLive),))
    deadFraction, liveFraction, deadFractionTotal, liveFractionTotal = getDLFraction(fuelModel,moistureLive)
    heatSink = 0
    
    for i in range(0,len(savrDead)):
        if savrDead[i] > 1.0e-7:
            qigDead[i] = 250 + 1116.0 * (moistureDead[i])
            heatSink = heatSink + deadFractionTotal*deadFraction[i]*qigDead[i]*np.exp(-138.0/savrDead[i])
        if savrLive[i] > 1.0e-7:
            qigLive[i] = 250 + 1116.0 * (moistureLive[i])
            heatSink = heatSink + liveFractionTotal*liveFraction[i]*qigLive[i]*np.exp(-138.0/savrLive[i])
    heatSink = heatSink * bulkDensity
    return heatSink

def getHeatFlux(fuelModel,moistureDead,moistureLive,sigma,packingRatio):
    if sigma < 1.0e-7:
        heatFlux = 0.0
    else:
        heatFlux = np.exp((0.792 + 0.681 * sigma**0.5)*(packingRatio + 0.1)) / (192 + 0.2595 * sigma)
    return heatFlux

def getWeightedMoistures(fuelModel,intermediates):
    weightedMoistureDead = intermediates.dead[2]
    weightedMoistureLive = intermediates.live[2]
    return weightedMoistureDead, weightedMoistureLive

def getEtaM(fuelModel,intermediates,MoED,MoEL):
    weightedMoistureDead, weightedMoistureLive = getWeightedMoistures(fuelModel,intermediates)
    
    def calculateEtaM(weightedMoisture,MoE):
        relativeMoisture = 0.0
        if MoE > 0.0:
            relativeMoisture = weightedMoisture / MoE
        if weightedMoisture > MoE or relativeMoisture > 1.0:
            etaM = 0
        else:
            etaM = 1.0 - (2.59 * relativeMoisture) + (5.11 * (relativeMoisture**2))-(3.52*(relativeMoisture**3))
        return etaM
    etaMDead = calculateEtaM(weightedMoistureDead,MoED)
    etaMLive = calculateEtaM(weightedMoistureLive,MoEL)
    
    return etaMDead, etaMLive

def getEtaS(fuelModel,intermediates):
    weightedSilicaDead, weightedSilicaLive = getWeightedSilicas(fuelModel,intermediates)
    
    def calculateEtaS(weightedSilica):
        etaSDen = weightedSilica ** 0.19
        if etaSDen < 1e-6:
            etaS = 0.0
        else:
            etaS = 0.174 / etaSDen
        return min([etaS,1.0])
    
    etaSDead = calculateEtaS(weightedSilicaDead)
    etaSLive = calculateEtaS(weightedSilicaLive)
    
    return etaSDead, etaSLive
    
def getSurfaceFireReactionIntensity(fuelModel,sigma,relativePackingRatio,MoED,MoEL,intermediates):
    aa = 133.0 / (sigma ** 0.7913) # Albini 1976
    gammaMax = (sigma ** 1.5) / (495.0+(0.0594*(sigma**1.5)))
    gamma = gammaMax * (relativePackingRatio**aa) * np.exp(aa * (1.0-relativePackingRatio))
    weightedFuelLoadDead, weightedFuelLoadLive = getWeightedFuelLoads(fuelModel,intermediates)
    weightedHeatDead, weightedHeatLive = getWeightedHeats(fuelModel,intermediates)
    #MoEL = 1.99
    etaMDead, etaMLive = getEtaM(fuelModel,intermediates,MoED,MoEL)
    etaSDead, etaSLive = getEtaS(fuelModel,intermediates)
    
    #print("gamma:",gamma)
    #print("weightedFuelLoadDead/Live:",weightedFuelLoadDead,weightedFuelLoadLive)
    #print("weightedHeatDead/Live:",weightedHeatDead,weightedHeatLive)
    #print("etaMDead/Live",etaMDead,etaMLive)
    #print("etaSDead/Live",etaSDead,etaSLive)
    """
    print("gamma",gamma)
    print("weightedFuelLoadDead",weightedFuelLoadDead)
    print("weightedHeatDead",weightedHeatDead,weightedHeatLive)
    print("etaMDead",etaMDead,etaMLive)
    print("etaSDead",etaSDead,etaSLive)
    """
    reactionIntensityDead = gamma * weightedFuelLoadDead * weightedHeatDead * etaMDead * etaSDead
    reactionIntensityLive = gamma * weightedFuelLoadLive * weightedHeatLive * etaMLive * etaSLive
    
    #reactionIntensityDead = 7505
    
    reactionIntensity = reactionIntensityDead+reactionIntensityLive
    
    #print("Dead Fuel Reaction Intensity: %.0f"%(reactionIntensityDead))
    #print("Live Fuel Reaction Intensity: %.0f"%(reactionIntensityLive))
    #print("Reaction Intensity: %.0f"%(reactionIntensity))
    
    return reactionIntensity, reactionIntensityDead, reactionIntensityLive

def getNoWindNoSlopeSpreadRate(reactionIntensity,heatFlux,heatSink):
    if heatSink < 1.0e-7:
        Rstar = 0.0
    else:
        Rstar = reactionIntensity*heatFlux/heatSink
    #print("HeatSource:",reactionIntensity*heatFlux)
    #print("NoWindNoSlopeSpredRate:",Rstar)
    
    return Rstar

def convertFtMinToChHr(R):
    R = R/1.100
    return R

def calculateMidflameWindSpeed(fuelModel,windSpeed,canopyCover,canopyHeight,crownRatio,
                               windHeightInputMode='TwentyFoot'):
    if windHeightInputMode == 'TenMeter':
        windSpeed = windSpeed/ 1.15
    depth = fuelModel.fuelBedDepth
    canopyCrownFraction = crownRatio * canopyCover / 3.0
    
    if canopyCover < 1.0e-7 or canopyCrownFraction < 0.05 or canopyHeight < 6.0:
        sheltered = False
    else:
        sheltered = True
    
    if sheltered:
        waf = 0.555 / (((canopyCrownFraction * canopyHeight)**0.5)*np.log((20.0+0.36*canopyHeight) / (0.13 * canopyHeight)))
    elif depth > 1.0e-7:
        waf = 1.83 / np.log((20.0+0.36 * depth) / (0.13 * depth))
    else:
        waf = 1.0
    midflameWindSpeed = waf * windSpeed
    return midflameWindSpeed, waf
        
def calculateWindFactor(sigma,relativePackingRatio,mfWindSpeed):
    windC, windB, windE = getWindIntermediates(sigma)
    mfWindSpeed = mfWindSpeed*88 # Convert mph to ft/min
    if mfWindSpeed < 1.0e-7:
        phiW = 0.0
    else:
        phiW = (mfWindSpeed**windB) * windC * (relativePackingRatio**(-windE))
    return phiW

def getWindIntermediates(sigma):
    windC = 7.47 * np.exp(-0.133 * (sigma**0.55))
    windB = 0.02526 * (sigma ** 0.54)
    windE = 0.715 * np.exp(-0.000359*sigma)
    return windC, windB, windE

def calculateSlopeFactor(slope,packingRatio,isAngle=False,isDegree=True):
    if isAngle:
        if isDegree:
            slope = slope/180.0*3.1415926535
        slopex = np.tan(slope)
    else:
        slopex = slope
    phiS = 5.275 * (packingRatio**(-0.3)) * (slopex**2)
    return phiS

def calculateROS(Rstar,phiW,phiS):
    R = Rstar * (1+phiW+phiS)
    return R

def calculateDirectionOfMaxSpread(windDir,aspect,Rstar,phiS,phiW):
    correctedWindDir = windDir-aspect
    windDirRadians = correctedWindDir * 3.1415926535 / 180.0
    slopeRate = Rstar*phiS
    windRate = Rstar*phiW
    
    x = slopeRate + (windRate * np.cos(windDirRadians))
    y = windRate * np.sin(windDirRadians)
    
    rateVector = ((x**2)+(y**2))**0.5
    
    forwardSpreadRate = Rstar + rateVector
    
    azimuth = np.arctan2(y,x) * 180.0 / 3.1415926535
    if azimuth < -1.0e-20:
        azimuth = azimuth + 360
        
    azimuth = azimuth + aspect + 180.0
    
    if azimuth >= 360.0:
        azimuth = azimuth - 360.0
    
    return azimuth, forwardSpreadRate

def calculateWindSpeedLimit(reactionIntensity,phiS):
    windSpeedLimit = 0.9 * reactionIntensity
    if phiS > 0.0:
        if phiS > windSpeedLimit:
            phiS = windSpeedLimit
    return windSpeedLimit

def calculateEffectiveWindSpeed(forwardSpreadRate,Rstar,relativePackingRatio,sigma,windSpeedLimit=9001):
    windC, windB, windE = getWindIntermediates(sigma)
    phiEffectiveWind = forwardSpreadRate/Rstar - 1.0
    effectiveWindSpeed = ((phiEffectiveWind*(relativePackingRatio**windE)) / windC)**(1/windB)
    effectiveWindSpeed = effectiveWindSpeed / 88
    if effectiveWindSpeed > windSpeedLimit/88:
        effectiveWindSpeed = windSpeedLimit/88
    return effectiveWindSpeed
    
def getResidenceTime(sigma):
    if sigma < 1.0e-7:
        residenceTime = 0.0
    else:
        residenceTime = 384.0/sigma
    return residenceTime

def calculateFireBasicDimensions(effectiveWindSpeed,forwardSpreadRate):
    #print("***EFF,",effectiveWindSpeed)#*88/60)
    if effectiveWindSpeed > 1.0e-7:
        fireLengthToWidthRatio = 1.0 + (0.25 * effectiveWindSpeed)#*88/60)
    else:
        fireLengthToWidthRatio = 1.0
    #print("default fl2wr:",fireLengthToWidthRatio)
    #print("default ecc:",(1-(1/fireLengthToWidthRatio)**2)**0.5)
    #fireLengthToWidthRatio = 1.174
    #fireLengthToWidthRatio = 2.25
    #fireLengthToWidthRatio = 1.174 # with effective wind speed 15.7 mi/h
    #fireLengthToWidthRatio = 1.161 # with effective wind speed 8.5 mi/h
    #fireLengthToWidthRatio = 1.145 # with effective wind speed 5.0 mi/h
    x = (fireLengthToWidthRatio**2) - 1.0
    if x > 0.0:
        eccentricity = (x**0.5) / fireLengthToWidthRatio
        #eccentricity = (1-(1/fireLengthToWidthRatio)**2)**0.5
    else:
        eccentricity = 0.0
    #eccentricity = 0.9045
    #print("modded fl2wr:",fireLengthToWidthRatio)
    #print("modded ecc:",eccentricity)
    backingSpreadRate = forwardSpreadRate * (1.0-eccentricity) / (1.0+eccentricity)
    
    ellipticalB = (forwardSpreadRate + backingSpreadRate) / 2.0
    ellipticalC = ellipticalB - backingSpreadRate
    if fireLengthToWidthRatio > 1e-7:
        ellipticalA = ellipticalB / fireLengthToWidthRatio
    else:
        ellipticalA = 0.0
    
    return fireLengthToWidthRatio, eccentricity, backingSpreadRate, ellipticalA, ellipticalB, ellipticalC

def calculateFireFirelineIntensity(forwardSpreadRate,reactionIntensity,residenceTime):
    firelineIntensity = forwardSpreadRate * reactionIntensity * residenceTime / 60.0
    return firelineIntensity

def calculateFlameLength(firelineIntensity):
    flameLength = max([0.0,0.45*(firelineIntensity**0.46)])
    return flameLength

def calculateSpreadRateAtVector(forwardSpreadRate,eccentricity,dirRmax,dirOfInterest):
    if forwardSpreadRate > 0.0:
        beta = abs(dirRmax - dirOfInterest)
        #print("%.1f,%.1f,%.1f"%(dirRmax,dirOfInterest,beta))
        if beta > 180.0:
            beta = (360-beta)
        betaRad = beta * 3.1415926535/180.0
        dirFactor = ((np.cos(betaRad)+1)/2)
        # This is the equation according to the BehavePlus source code:
        rosVector = forwardSpreadRate * (1.0-eccentricity) / (1.0-eccentricity* np.cos(betaRad))
        
        # This is the equaiton I have found to match BehavePlus results:
        rosVector = ((1-abs(betaRad)/3.1415926535)*forwardSpreadRate * dirFactor)
        
        # Combining the two smooths out the peak
        rosVector = ((1-abs(betaRad)/3.1415926535)*forwardSpreadRate * dirFactor + (abs(betaRad)/3.1415926535)*rosVector)
        
        #eccentricity = 0.9
        #rosVector = forwardSpreadRate * (1.0-eccentricity) / (1.0-eccentricity*dirFactor)
        #if beta < 30:
        
        #rosVector = ((1-abs(betaRad)/3.1415926535)*forwardSpreadRate * dirFactor + (betaRad/3.1415926535)*rosVector)
        #print(dirOfInterest,betaRad,rosVector)
    else:
        rosVector = 0.0
    return rosVector

def calculateSpreadRateAtVector2(forwardSpreadRate,backSpreadRate,eccentricity,dirRmax,dirOfInterest):
    if forwardSpreadRate > 0.0:
        beta = abs(dirRmax - dirOfInterest)
        #print("%.1f,%.1f,%.1f"%(dirRmax,dirOfInterest,beta))
        if beta > 180.0:
            beta = (360-beta)
        if abs(beta) > 0.1:
            betaRad = beta * 3.1415926535/180.0
            dirFactor = ((np.cos(betaRad)+1)/2)
            
            # This is the equation according to the BehavePlus source code:
            rosVector = forwardSpreadRate * (1.0-eccentricity) / (1.0-eccentricity* np.cos(betaRad))
            
            # This is the equaiton I have found to match BehavePlus results:
            #rosVector = ((1-abs(betaRad)/3.1415926535)*forwardSpreadRate * dirFactor)
            #rosVector = ((1-abs(betaRad)/3.1415926535)*(forwardSpreadRate-backSpreadRate) * dirFactor)+backSpreadRate
            
            # Combining the two smooths out the peak
            #rosVector = ((1-abs(betaRad)/3.1415926535)*forwardSpreadRate * dirFactor + (abs(betaRad)/3.1415926535)*rosVector)
            
            #eccentricity = 0.9
            #rosVector = forwardSpreadRate * (1.0-eccentricity) / (1.0-eccentricity*dirFactor)
            #if beta < 30:
            
            #rosVector = ((1-abs(betaRad)/3.1415926535)*forwardSpreadRate * dirFactor + (betaRad/3.1415926535)*rosVector)
            #print(dirOfInterest,betaRad,rosVector)
        else:
            rosVector = forwardSpreadRate
        if rosVector < backSpreadRate:
            rosVector = backSpreadRate
    else:
        rosVector = 0.0
    return rosVector
    
    
def scaleRandomValue(mn,mx):
    value = np.random.random()*(mx-mn)+mn
    return value

def getRandomConditions(params,allowDynamicModels=True):
    paramsRand = dict()
    for key in params.keys():
        if params[key][0] == None:
            minValue = params[key][1]
            maxValue = params[key][2]
            if key == 'model':
                fuelModels = list(buildFuelModels(allowDynamicModels=minValue,allowNonBurningModels=maxValue).keys())
                value = fuelModels[np.random.randint(0,len(fuelModels))]
            else:
                value = scaleRandomValue(minValue,maxValue)
        else:
            value = params[key][0]
        paramsRand[key] = value
    return paramsRand

def orderParams(params,toPrint=False):
    model = params['model']
    canopyCover = params['canopyCover']*100
    canopyHeight = params['canopyHeight']
    crownRatio = params['crownRatio']
    m1h = params['m1h']
    m10h = params['m10h']
    m100h = params['m100h']
    lhm = params['lhm']
    lwm = params['lwm']
    windSpeed = params['windSpeed']
    windDir = params['windDir']
    slope = params['slope']*100
    aspect = params['aspect']
    
    orderedParams = [model,canopyCover,canopyHeight,crownRatio,m1h,m10h,m100h,lhm,lwm,windSpeed,windDir,slope,aspect]
    
    if toPrint:
        print("************************************************************")
        print("Starting simulation")
        print("model:\t\t\t%s"%(model))
        print("canopyCover:\t\t%.2f"%(canopyCover))
        print("canopyHeight:\t\t%.2f"%(canopyHeight))
        print("crownRatio:\t\t%.2f"%(crownRatio))
        print("m1h:\t\t\t%.2f"%(m1h))
        print("m10h:\t\t\t%.2f"%(m10h))
        print("m100h:\t\t\t%.2f"%(m100h))
        print("lhm:\t\t\t%.2f"%(lhm))
        print("lwm:\t\t\t%.2f"%(lwm))
        print("windSpeed:\t\t%.2f"%(windSpeed))
        print("windDir:\t\t%.2f"%(windDir))
        print("slope:\t\t\t%.2f"%(slope))
        print("aspect:\t\t\t%.2f"%(aspect))
        
    return orderedParams

def getROSfromParams(params,toPrint=False,maxOnly=False):
    model = params['model']
    canopyCover = params['canopyCover']
    canopyHeight = params['canopyHeight']
    crownRatio = params['crownRatio']
    m1h = params['m1h']
    m10h = params['m10h']
    m100h = params['m100h']
    lhm = params['lhm']
    lwm = params['lwm']
    windSpeed = params['windSpeed']
    windDir = params['windDir']
    slope = params['slope']
    aspect = params['aspect']
    
    orderParams(params,toPrint=toPrint)
    
    directions = np.linspace(0,360,361)

    fuelModel = getFuelModel(model)
    
    moistureDead, moistureLive = getMoistureContent(m1h,m10h,m100h,lhm,lwm)
    loadDead, loadLive = getFuelLoad(fuelModel,moistureLive)
    savrDead, savrLive = getSAV(fuelModel)
    deadFraction, liveFraction, deadFractionTotal, liveFractionTotal = getDLFraction(fuelModel,moistureLive)
    if toPrint:
        print(deadFraction)
        print(liveFraction)
    moistOfExtDead, moistOfExtLive = getMoistOfExt(fuelModel,moistureDead,moistureLive)
    
    heatDead, heatLive = getHeatOfCombustion(fuelModel)
    
    intermediates = lifeStateIntermediate(fuelModel,moistureDead,moistureLive)
    
    sigma = getCharacteristicSAVR(fuelModel,intermediates,moistureLive)
    packingRatio, relativePackingRatio, bulkDensity = getPackingRatios(fuelModel,intermediates,moistureLive)
    heatSink = getHeatSink(fuelModel,moistureDead,moistureLive,bulkDensity)
    heatFlux = getHeatFlux(fuelModel,moistureDead,moistureLive,sigma,packingRatio)
    reactionIntensity, reactionIntensityDead, reactionIntensityLive  = getSurfaceFireReactionIntensity(fuelModel,sigma,relativePackingRatio,moistOfExtDead,moistOfExtLive,intermediates)
    Rstar = getNoWindNoSlopeSpreadRate(reactionIntensity,heatFlux,heatSink)
    
    mfWindSpeed, waf = calculateMidflameWindSpeed(fuelModel,windSpeed,canopyCover,canopyHeight,crownRatio)
    phiW = calculateWindFactor(sigma,relativePackingRatio,mfWindSpeed)
    phiS = calculateSlopeFactor(slope,packingRatio)
    dirRmax, forwardSpreadRate = calculateDirectionOfMaxSpread(windDir,aspect,Rstar,phiS,phiW)
    windSpeedLimit = calculateWindSpeedLimit(reactionIntensity,phiS)
    
    effectiveWindSpeed = calculateEffectiveWindSpeed(forwardSpreadRate,Rstar,relativePackingRatio,sigma,windSpeedLimit=windSpeedLimit)
    residenceTime = getResidenceTime(sigma)
    #effectiveWindSpeed = 3.9
    
    fireLengthToWidthRatio, eccentricity, backingSpreadRate, eA, eB, eC = calculateFireBasicDimensions(effectiveWindSpeed,forwardSpreadRate)
    firelineIntensity = calculateFireFirelineIntensity(forwardSpreadRate,reactionIntensity,residenceTime)
    flameLength = calculateFlameLength(firelineIntensity)
    rosVectors = []
    R = calculateSpreadRateAtVector(forwardSpreadRate,eccentricity,dirRmax,dirRmax)
    R = convertFtMinToChHr(R)

    if toPrint:
        print("************************************************************")
        print("Rate of Spread:\t\t\t\t\t%.1f\tch/h"%(R))
        print("Reaction Intensity:\t\t\t\t%.0f\tBtu/ft2/min"%(reactionIntensity))
        print("Surface Fire Dir of Max Spread (from north):\t%.0f\tdeg"%(dirRmax))
        print("Midflame Wind Speed:\t\t\t\t%.1f\tmi/h"%(mfWindSpeed))
        print("Wind Adjustment Factor:\t\t\t\t%.2f"%(waf))
        print("Effective Wind Speed:\t\t\t\t%.1f\tmi/h"%(effectiveWindSpeed))
        print("Live Fuel Moisture of Extinction:\t\t%.0f"%(moistOfExtLive*100))
        print("Characteristic SA/V:\t\t\t\t%s\tft2/ft3"%(int(sigma)))
        print("Bulk Density:\t\t\t\t\t%.4f\tlbs/ft3"%(bulkDensity))
        print("Packing Ratio:\t\t\t\t\t%.4f"%(packingRatio))
        print("Relative Packing Ratio:\t\t\t\t%.4f"%(relativePackingRatio))
        print("Dead Fuel Reaction Intensity:\t\t\t%.0f\tBtu/ft2/min"%(reactionIntensityDead))
        print("Live Fuel Reaction Intensity:\t\t\t%.0f\tBtu/ft2/min"%(reactionIntensityLive))
        print("Surface Fire Wind Factor:\t\t\t%.1f"%(phiW))
        print("Slope Factor:\t\t\t\t\t%.1f"%(phiS))
        print("Heat Source:\t\t\t\t\t%.0f\tBtu/ft2/min"%(heatFlux*reactionIntensity*(1+phiS+phiW)))
        print("Heat Sink:\t\t\t\t\t%.1f\tBtu/ft3"%(heatSink))
        print("Dead Herbaceous Fuel Load:\t\t\t%.2f\tton/ac"%(loadDead[3]*21.78))
        print("Live Fuel Load Remainder:\t\t\t%.2f\tton/ac"%(loadLive[0]*21.78))
        print("Total Dead Fuel Load:\t\t\t\t%.2f\tton/ac"%(np.sum(loadDead)*21.78))
        print("Total Live Fuel Load:\t\t\t\t%.2f\tton/ac"%(np.sum(loadLive)*21.78))
        print("Dead Fuel Load Portion:\t\t\t\t%.2f"%(np.sum(loadDead)/(np.sum(loadDead)+np.sum(loadLive))*100))
        print("Live Fuel Load Portion:\t\t\t\t%.2f"%(np.sum(loadLive)/(np.sum(loadDead)+np.sum(loadLive))*100))
        print("************************************************************")

    if maxOnly:
        return dirRmax, R
    
    for dirOfInterest in directions:
        rosVector = calculateSpreadRateAtVector2(forwardSpreadRate,backingSpreadRate,eccentricity,dirRmax,dirOfInterest)
        rosVector = convertFtMinToChHr(rosVector)
        rosVectors.append(rosVector)
    
    
    rosVectors = np.array(rosVectors)
    #R = calculateROS(Rstar, phiW, phiS)

    return directions, rosVectors

def cartCoords(thetaRad,rosVectorsKmHr):
    coords = np.zeros((len(rosVectorsKmHr),2))
    x = -1*np.array(rosVectorsKmHr)*np.sin(thetaRad)
    y = -1*np.array(rosVectorsKmHr)*np.cos(thetaRad)
    coords[:,0] = x
    coords[:,1] = y

    return coords

def rothermelOuputToImg(theta,R,resX=50,resY=50):
    coords = cartCoords(theta,R.copy())
       
    (coords[:,0],coords[:,1]) = (coords[:,0]+resX/2,coords[:,1]+resY/2)
    coords = np.array(coords,dtype=np.int32)
    
    coordsTuple = []
    for c in coords:
        coordsTuple.append((c[0],c[1]))
    
    img = Image.new('LA',(resX,resY))
    draw = ImageDraw.Draw(img)
    draw.polygon(coordsTuple,fill='black',outline=None)
    img = np.copy(np.asarray(img)[:,:,1])
    #img[int(resX/2),int(resY/2)] = 125
    
    return img

def rothermelOuputToImgMulti(theta,Rbase,times,resX=50,resY=50):
    img = Image.new('LA',(resX,resY))
    draw = ImageDraw.Draw(img)
    for t in times:
        coords = cartCoords(theta,Rbase.copy()*t)
           
        (coords[:,0],coords[:,1]) = (coords[:,0]+resX/2,coords[:,1]+resY/2)
        coords = np.array(coords,dtype=np.int32)
        
        coordsTuple = []
        for c in coords:
            coordsTuple.append((c[0],c[1]))
        draw.polygon(coordsTuple,fill=(t,t),outline=(t,t))
    img = np.copy(np.asarray(img)[:,:,1])
    #img[int(resX/2),int(resY/2)] = 125
    
    return img
    

def convertChHrToKmHour(R):
    if R is list:
        for r in R:
            r = r*(1.1)*(60.0/5280.0)*(1.60934)
    else:
        R = R*1.1 # Convert to ft/min
        R = R*60.0/5280.0 # Convert to mi/hour
        R = R*1.60934 # Convert to km/hour
    return R

def convertDegToRad(theta):
    if theta is list:
        for r in theta:
            r = r*3.1415926535/180.0
    else:
        theta = theta*3.1415926535/180.0
    return theta

def slopeToElevImg(phi,phiDir,resX=50,resY=50):
    phiDirRad = phiDir*3.1415926535/180.0
    slopeX = phi*np.sin(phiDirRad)
    slopeY = -phi*np.cos(phiDirRad)
    img = np.zeros((2,2))
    #img[img == 0] = np.nan
    img[0,0] = -resX/2*slopeX+resY/2*slopeY
    img[0,-1]= resX/2*slopeX+resY/2*slopeY
    img[-1,0] = -resX/2*slopeX-resY/2*slopeY
    img[-1,-1] = resX/2*slopeX-resY/2*slopeY
    img = zoom(img,resX/2,order=1)
    return img

def visualizeInputImgs(directions,rosVectors,params,resX=50,resY=50,toPlot=True):
    rosVectorsKmHr = convertChHrToKmHour(rosVectors)
    directionsRad = convertDegToRad(directions)
    
    x = -1*np.array(rosVectorsKmHr)*np.sin(directionsRad)
    y = np.array(rosVectorsKmHr)*np.cos(directionsRad)
    
    img6 = rothermelOuputToImg(directionsRad,rosVectorsKmHr*6.0,resX=resX,resY=resY)
    img12 = rothermelOuputToImg(directionsRad,rosVectorsKmHr*12.0,resX=resX,resY=resY)
    img18 = rothermelOuputToImg(directionsRad,rosVectorsKmHr*18.0,resX=resX,resY=resY)
    img24 = rothermelOuputToImg(directionsRad,rosVectorsKmHr*24.0,resX=resX,resY=resY)
    
    elevImg = slopeToElevImg(params['slope'],params['aspect'],resX=resX,resY=resY)
    windDirRad = params['windDir']*3.1415926536/180.0
    windX = np.zeros((resX,resY))+params['windSpeed']*np.sin(windDirRad)
    windY = np.zeros((resX,resY))-params['windSpeed']*np.cos(windDirRad)
    lhmImg = np.zeros((resX,resY))+params['lhm']
    lwmImg = np.zeros((resX,resY))+params['lwm']
    m1hImg = np.zeros((resX,resY))+params['m1h']
    m10hImg = np.zeros((resX,resY))+params['m10h']
    m100hImg = np.zeros((resX,resY))+params['m100h']
    
    canopyCoverImg = np.zeros((resX,resY))+params['canopyCover']
    canopyHeightImg = np.zeros((resX,resY))+params['canopyHeight']
    crownRatioImg = np.zeros((resX,resY))+params['crownRatio']
    
    modelImg = np.zeros((resX,resY))+params['modelInd']
    
    fireImages = [img6,img12,img18,img24]
    modelInputs = [elevImg,windX,windY,lhmImg,lwmImg,m1hImg,m10hImg,m100hImg,canopyCoverImg,canopyHeightImg,crownRatioImg,modelImg]
    
    if toPlot:
        plt.figure(figsize=(12,12))
        plt.suptitle('Fuel Model:%s'%(params['model']))
        plt.subplot(4,4,1)
        plt.imshow(img12,cmap='jet')
        plt.colorbar()
        plt.title('Fire at 12 hours')
        
        plt.subplot(4,4,2)
        plt.imshow(img24,cmap='jet')
        plt.colorbar()
        plt.title('Fire at 24 hours')
        
        plt.subplot(4,4,3)
        plt.imshow(elevImg,cmap='jet')
        plt.colorbar()
        plt.title('Elevation')
        
        plt.subplot(4,4,4)
        plt.imshow(windX,cmap='jet',vmin=-20,vmax=20)
        plt.colorbar()
        plt.title('WindX')
    
        plt.subplot(4,4,5)
        plt.imshow(windY,cmap='jet',vmin=-20,vmax=20)
        plt.colorbar()
        plt.title('WindY')
        
        plt.subplot(4,4,6)
        plt.imshow(lhmImg,cmap='jet',vmin=30,vmax=150)
        plt.colorbar()
        plt.title('Live Herbaceous Moisture')
        
        plt.subplot(4,4,7)
        plt.imshow(lwmImg,cmap='jet',vmin=30,vmax=150)
        plt.colorbar()
        plt.title('Live Woody Moisture')
        
        plt.subplot(4,4,8)
        plt.imshow(m1hImg,cmap='jet',vmin=0,vmax=40)
        plt.colorbar()
        plt.title('1-hour Moisture')
        
        plt.subplot(4,4,9)
        plt.imshow(canopyCoverImg,cmap='jet',vmin=0,vmax=1)
        plt.colorbar()
        plt.title('Canopy Cover')
        
        plt.subplot(4,4,10)
        plt.imshow(canopyHeightImg,cmap='jet',vmin=1,vmax=20)
        plt.colorbar()
        plt.title('Canopy Height')
        
        plt.subplot(4,4,11)
        plt.imshow(crownRatioImg,cmap='jet',vmin=0,vmax=1)
        plt.colorbar()
        plt.title('Crown Ratio')
        
        plt.subplot(4,4,12)
        plt.imshow(modelImg,cmap='jet',vmin=0,vmax=52)
        plt.colorbar()
        plt.title('Model')
        #plt.plot(x,y)
        #plt.plot(0,0,'ok')
        #xRange = x.max()-x.min()
        #yRange = y.max()-y.min()
        #plt.xlim([x.min()-xRange/2,x.max()+xRange/2])
        #plt.ylim([y.min()-yRange/2,y.max()+yRange/2])
        #plt.title('Rate of Spread')

    return fireImages, modelInputs
    


def visualizeInputValues(directions,rosVectors,params,resX=50,resY=50):
    rosVectorsKmHr = convertChHrToKmHour(rosVectors)
    directionsRad = convertDegToRad(directions)
    
    x = -1*np.array(rosVectorsKmHr)*np.sin(directionsRad)
    y = np.array(rosVectorsKmHr)*np.cos(directionsRad)
    
    imgFire = rothermelOuputToImgMulti(directionsRad,rosVectorsKmHr,[48,42,36,30,24,18,12,6],resX=resX,resY=resY)
    imgFire[25,25] = 0
    elevImg = slopeToElevImg(params['slope'],params['aspect'],resX=resX,resY=resY)
    
    windDirRad = params['windDir']*3.1415926536/180.0
    windSpeed = params['windSpeed']
    windX = 1.0*windSpeed*np.sin(windDirRad)
    windY = -1.0*windSpeed*np.cos(windDirRad)
    windYs = [windX,windY]
    windXs = np.arange(len(windYs))
    windNames = ('E+','N+')
    windLimits = [-20,20]
    
    moistYs = [params['m1h'],params['m10h'],params['m100h'],params['lhm']/5,params['lwm']/5]
    moistXs = np.arange(len(moistYs))
    moistNames = ('m1h','m10h','m100h','lhm/5','lwm/5')
    moistLimits = [0,60]
    
    canopyYs = [params['canopyCover'],params['canopyHeight']/20,params['crownRatio']]
    canopyXs = np.arange(len(canopyYs))
    canopyNames = ('Cover (%)','Height (ft/20)','Ratio (%)')
    canopyLimits = [0,1]
    
    modelYs = [params['modelInd'],0]
    modelXs = np.arange(len(modelYs))
    modelNames = (str(params['model']),'')
    modelLimits = [0,52]
    
    plt.figure(figsize=(10,14))
    plt.suptitle('Fuel Model:%s'%(params['model']))
    plt.subplot(3,2,1)
    plt.imshow(imgFire,cmap='gray_r')
    c = plt.colorbar(ticks=[48,36,24,12,0])
    plt.title('Fire spread')
    plt.xlabel('km')
    plt.ylabel('km')
    c.ax.set_label('Hours')
    
    #plt.subplot(3,3,2)
    #plt.imshow(img24,cmap='jet')
    #plt.colorbar()
    #plt.title('Fire at 24 hours')
    
    plt.subplot(3,2,3)
    plt.imshow(elevImg,cmap='jet')
    plt.colorbar()
    plt.title('Elevation Difference [km]')
    
    plt.subplot(3,2,4)
    plt.bar(windXs,windYs,align='center');
    plt.xticks(windXs,windNames);
    plt.ylabel('WindSpeed (mph)');
    plt.ylim(windLimits)

    plt.subplot(3,2,5)
    plt.bar(moistXs,moistYs,align='center');
    plt.xticks(moistXs,moistNames);
    plt.ylabel('Moisture (%)');
    plt.ylim(moistLimits)
    
    plt.subplot(3,2,6)
    plt.bar(canopyXs,canopyYs,align='center');
    plt.xticks(canopyXs,canopyNames);
    plt.ylabel('Canopy (%)');
    plt.ylim(canopyLimits)
    
    plt.subplot(3,2,2)
    plt.bar(modelXs,modelYs,align='center');
    plt.xticks(modelXs,modelNames);
    plt.ylabel('Model Rank');
    plt.ylim(modelLimits)
    plt.xlim([-0.95,0.95])
    #plt.plot(x,y)
    #plt.plot(0,0,'ok')
    #xRange = x.max()-x.min()
    #yRange = y.max()-y.min()
    #plt.xlim([x.min()-xRange/2,x.max()+xRange/2])
    #plt.ylim([y.min()-yRange/2,y.max()+yRange/2])
    #plt.title('Rate of Spread')
    return imgFire

def paramListTodict(paramsRaw):
    params = dict()
    params['model'] = paramsRaw[0]
    params['canopyCover'] = float(paramsRaw[1])/100
    params['canopyHeight'] = float(paramsRaw[2])
    params['crownRatio'] = float(paramsRaw[3])
    params['m1h'] = float(paramsRaw[4])
    params['m10h'] = float(paramsRaw[5])
    params['m100h'] = float(paramsRaw[6])
    params['lhm'] = float(paramsRaw[7])
    params['lwm'] = float(paramsRaw[8])
    params['windSpeed'] = float(paramsRaw[9])
    params['windDir'] = float(paramsRaw[10])
    params['slope'] = float(paramsRaw[11])/100
    params['aspect'] = float(paramsRaw[12])
    return params

def getStandardParams():
    # model,canopyCover/100,Height,Ratio,m1h,m10h,m100h,lhm,l2m,windSpeed,windDir,slope,aspect
    #paramList = ['FM1',0,0,0.5,8,6,4,60,60,10,0,0.5,0]
    paramList = ['FM1',0,0,0.5,8,9,10,60,60,10,0,0.5,0]
    params = paramListTodict(paramList)
    return params

def determineFastestModel(params=None,toPrint=False):
    if params is None:
        params = getStandardParams()
    fuelModels = buildFuelModels(allowDynamicModels=True,allowNonBurningModels=True)
    
    updatedModels = []
    Rs = []
    
    for fuelModel in list(fuelModels.keys()):
        params['model'] = fuelModel
        direction, R = getROSfromParams(params,maxOnly=True)
        updatedModels.append(fuelModel)
        Rs.append(R)
    
    numZero = len(np.where(np.array(Rs) <= 0.01)[0])
    
    inds = np.argsort(Rs)
    updatedModelsSort = np.array(updatedModels)[inds]
    RsSort = np.sort(Rs)
    
    modelIndexDict = dict()
    
    for i in range(0,len(inds)):
        value = max(0,i-numZero+1)
        modelIndexDict[updatedModelsSort[i]] = value
        if toPrint:
            print("Model = %s,\tR = %.2f"%(updatedModelsSort[i],RsSort[i]))
    
    return modelIndexDict

def rearrangeDatas(datas):
    sz = datas[0].shape
    szrs = sz[0]*sz[1]
    datasNew = np.zeros((szrs*len(datas),))
    
    for i in range(0,len(datas)):
        datasNew[i*szrs:(i+1)*szrs] = np.reshape(datas[i],(szrs,))
    return datasNew
            
def getStandardParamsInput():
    paramsInput = dict()
    paramsInput['model']        = [None,True,False] # string
    paramsInput['canopyCover']  = [None,0.0,1.0]    # percent (0-1)
    paramsInput['canopyHeight'] = [None,1.0,20.0]   # ft (1-20)
    paramsInput['crownRatio']   = [None,0.1,1.0]    # fraction (0.1-1)
    paramsInput['m1h']          = [None,1.0,40.0]  # percent (1-60)
    paramsInput['m10h']         = [None,1.0,40.0]     # percent (1-60)
    paramsInput['m100h']        = [None,1.0,40.0]     # percent (1-60)
    paramsInput['lhm']          = [None,30.0,100.0] # percent (30-300)
    paramsInput['lwm']          = [None,30.0,100.0] # percent (30-300)
    paramsInput['windSpeed']    = [None,0.0,30.0]   # mph (0-30)
    paramsInput['windDir']      = [None,0.0,360.0]  # degrees (0-360)
    paramsInput['slope']        = [None,0.0,1.0]    # fraction (0-1)
    paramsInput['aspect']       = [None,0.0,360.0]  # degrees (0-360)
    paramsInput['Mth']          = [None,5,9] # integer
    paramsInput['Day']          = [None,0,31] # integer
    paramsInput['Pcp']          = [None,0.3,10.9] # mm
    paramsInput['mTH']          = [None,400,600] # 24-Hour
    paramsInput['xTH']          = [None,1200,1500] # 24-Hour
    paramsInput['mT']           = [None,2.0,16.6] # degrees C
    paramsInput['xT']           = [None,28.9,37.2] # degrees C
    paramsInput['mH']           = [None,39.2,50.0] # Percent
    paramsInput['xH']           = [None,39.2,50.0] # Percent
    paramsInput['PST']          = [None,0,2400] # Precipitation Start Time
    paramsInput['PET']          = [None,0,2400] # Precipitation End Time
    paramsInput['startTime']    = [None,0,24] # Fire start hour
    
    return paramsInput

def manyFiresInputFigure(modelInputs):

    fig, ax = plt.subplots(figsize=(8,8))
    a = []
    lims = [[-30,30],[-20,20],[-20,20],[30,150],[30,150],[0,30],[0,30],[0,30],[0,1],[0,20],[0,1],[0,53]]
    names = ['Elevation','East Wind','North Wind','Live Herbaceous Moisture','Live Woody Moisture','1-Hour Moisture','10-Hour Moisture','100-Hour Moisture','Canopy Cover','Canopy Height','Crown Ratio','Fuel Model']
    textOffset = [0]
    #modelInputs = [elevImg,windX,windY,lhmImg,lwmImg,m1hImg,m10hImg,m100hImg,canopyCoverImg,canopyHeightImg,crownRatioImg,modelImg]
    for i in range(len(modelInputs)-1,-1,-1):
        img = modelInputs[i].copy()
        img[-1,-1] = lims[i][0]
        img[-1,-2] = lims[i][1]
        oi = OffsetImage(img, zoom = 2.0, cmap='jet')
        box = AnnotationBbox(oi, (-0.5*i,1*i), frameon=True)
        a.append(ax.add_artist(box))
        ax.annotate(names[i],xy=(-0.5*i-1.1,1*i-0.9),xycoords='data',textcoords='data',xytext=(-0.5*i-4-(len(names[i])-10)*0.1,1*i-0.85),arrowprops=dict(facecolor='black',shrink=0.05))
    i = -1
    oi = OffsetImage(imgFire, zoom = 2.0, cmap='jet')
    box = AnnotationBbox(oi, (-0.5*i,1*i), frameon=True)
    ax.annotate('Fire Perimiter',xy=(-0.5*i-1.1,1*i-0.9),xycoords='data',textcoords='data',xytext=(-0.5*i-4,1*i-0.85),arrowprops=dict(facecolor='black',shrink=0.05))
    a.append(ax.add_artist(box))
    plt.xlim(-2,6.15)
    plt.ylim(-1.9,12.2)
    plt.xlim(-9.0,1.4)
    #plt.ylim(-50,50)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('inputsExampleManyFires.png',dpi=300)

def makeFirePerimetersFigure(imgFire):
    import skimage.transform as sktf
    import skimage.filters as skfi
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    oi = skfi.gaussian(imgFire,sigma=1.0,preserve_range=True)
    imgFire = visualizeInputValues(directions,rosVectors,params,resX=250,resY=250)
    imgFire[125:126,125:126] = 0
    imgFire = imgFire[25:175,100:]
    imgFire = imgFire[::-1,:]
    #oi = OffsetImage(imgFire, zoom = 2.0, cmap='jet')
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    fs=32
    im = ax.imshow(imgFire,cmap='hot_r')
    plt.gca().invert_yaxis()
    plt.xlabel('km',fontsize=fs)
    plt.ylabel('km',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.xticks([0,20,40,60,80,100,120,140])
    plt.yticks([0,20,40,60,80,100,120,140])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%", pad=0.05)
    c = plt.colorbar(im,ticks=[48,36,24,12,0],cax=cax)
    #plt.title('Fire spread')
    plt.tick_params(labelsize=fs)
    plt.ylabel('Hours',fontsize=fs)
    #c.ax.set_label(fontsize=fs)
    plt.tight_layout()
    
    plt.savefig('exampleFirePerimiter.eps')

if __name__ == "__main__":
    ''' case0: Generate 1 set of random inputs and visualize the results.
        case1: Generate set of 100 random inputs and save inputs for validation
               with behavePlus.
        case2: Re-generate prediction with same random inputs.
        case3: Re-generate single validation output.
        case4: Generate validation plots.
        case5: Generate neural network dataset
    '''
    
    case = 5
    
    paramsInput = dict()
    paramsInput['model']        = [None,True,False] # string
    paramsInput['canopyCover']  = [None,0.0,1.0]    # percent (0-1)
    paramsInput['canopyHeight'] = [None,1.0,20.0]   # ft (1-20)
    paramsInput['crownRatio']   = [None,0.1,1.0]    # fraction (0.1-1)
    paramsInput['m1h']          = [None,1.0,40.0]  # percent (1-60)
    paramsInput['m10h']         = [None,1.0,40.0]     # percent (1-60)
    paramsInput['m100h']        = [None,1.0,40.0]     # percent (1-60)
    paramsInput['lhm']          = [None,30.0,100.0] # percent (30-300)
    paramsInput['lwm']          = [None,30.0,100.0] # percent (30-300)
    paramsInput['windSpeed']    = [None,0.0,30.0]   # mph (0-30)
    paramsInput['windDir']      = [None,0.0,360.0]  # degrees (0-360)
    paramsInput['slope']        = [None,0.0,1.0]    # fraction (0-1)
    paramsInput['aspect']       = [None,0.0,360.0]  # degrees (0-360)
    paramsInput['Tmin']         = [None,2.0,16.6] # degrees C
    paramsInput['Tmax']         = [None,28.9,37.2] # degrees C
    
    resX = 50
    resY = 50
    
    """

    """
    #params['m1h'] = 40.0
    #params['windSpeed'] = 0.0 # mph (0-30)
    #params['windDir'] = -135.0 # degrees (0-360)
    #params['slope'] = 0.0 # fraction (0-1)
    #params['aspect'] = 135
    
    if case == 0:
        params = getRandomConditions(paramsInput,allowDynamicModels=True)
        params['model'] = 'TU2'
        params['canopyCover'] = 0.0607 # percent (0-1)
        params['canopyHeight'] = 17.46 # ft (1-20)
        params['crownRatio'] = 0.99 # fraction (0-1)
        params['m1h'] = 8.4 # percent (1-100)
        params['m10h'] = 6 # percent (1-100)
        params['m100h'] = 4 # percent (1-100)
        params['lhm'] = 82.75 # percent (30-300)
        params['lwm'] = 75.98 # percent (30-300)
        params['windSpeed'] = 12.08 # mph (0-30)
        params['windDir'] = 223.57 # degrees (0-360)
        params['slope'] = 0.9942 # fraction (0-1)
        params['aspect'] = 248.29 # degrees (0-360)
        directions, rosVectors = getROSfromParams(params,toPrint=True)
        visualizeInputImgs(directions,rosVectors,params,resX=resX,resY=resY)
        visualizeInputValues(directions,rosVectors,params,resX=resX,resY=resY)
    elif case == 1:
        allParams = []
        allDirections = []
        allRosVectors = []
        for i in range(0,1000):
            params = getRandomConditions(paramsInput,allowDynamicModels=True)
            directions, rosVectors = getROSfromParams(params)
            allParams.append(orderParams(params))
            allDirections.append(directions)
            allRosVectors.append(rosVectors)
        allParams = np.array(allParams).T
        #pd.DataFrame(allParams[1:,:],columns=allParams[0,:]).astype(float).round(2).to_csv('../rothermelData/validationInputs.csv')
        #pd.DataFrame(allDirections).T.to_csv('../rothermelData/validationDirections.csv')
        #pd.DataFrame(allRosVectors).T.to_csv('../rothermelData/validationRosVectors.csv')
    elif case == 2:
        allParams = pd.read_csv('../rothermelData/validationInputs.csv')
        allDirections = []
        allRosVectors = []
        for i in range(1,allParams.values.shape[1]):
            paramsRaw = allParams.values[:,i]
            params = paramListTodict(paramsRaw)
            directions, rosVectors = getROSfromParams(params)
            allParams.append(orderParams(params))
            allDirections.append(directions)
            allRosVectors.append(rosVectors)
        allParams = np.array(allParams).T
        pd.DataFrame(allDirections).T.to_csv('../rothermelData/validationDirections.csv')
        pd.DataFrame(allRosVectors).T.to_csv('../rothermelData/validationRosVectors.csv')
    elif case == 3:
        numToRepeat = 5
        allParams = pd.read_csv('../rothermelData/validationInputs.csv')
        paramsRaw = allParams.values[:,numToRepeat]
        params = paramListTodict(paramsRaw)
        directions, rosVectors = getROSfromParams(params,toPrint=True)
        
        behaveResults = pd.read_csv('../rothermelData/validationBehaveOutputs.csv')
        behaveDirections = behaveResults.values[:,0]
        behaveRos = behaveResults.values[:,numToRepeat]
        
        rosVectorsResample = np.interp(behaveDirections,directions,rosVectors)
        rmse = np.mean((rosVectorsResample-behaveRos)**2)**0.5
        
        plt.figure(figsize=(4,4))
        plt.plot(directions,rosVectors,label='prediction')
        plt.plot(behaveDirections,behaveRos,label='behavePlus')
        plt.legend()
        ylim = [min([np.min(rosVectors),np.min(behaveRos),0]),max([np.max(rosVectors),np.max(behaveRos),1.0])]
        plt.ylim(ylim)
        plt.title(str(i)+': '+str(np.round(rmse,2)))
        
    elif case == 4:
        behaveResults = pd.read_csv('../rothermelData/validationBehaveOutputs.csv')
        behaveDirections = behaveResults.values[:,0]
        allDirections = pd.read_csv('../rothermelData/validationDirections.csv')
        allRosVectors = pd.read_csv('../rothermelData/validationRosVectors.csv')
        
        rmses = []
        toPlot = True
        
        for i in range(1,51):
            behaveRos = behaveResults.values[:,i]
            directions = allDirections.values[:,i]
            rosVectors = allRosVectors.values[:,i]
            rosVectorsResample = np.interp(behaveDirections,directions,rosVectors)
            rmse = np.mean((rosVectorsResample-behaveRos)**2)**0.5
            rmses.append(rmse)
            if toPlot:
                plt.figure(figsize=(4,4))
                plt.plot(directions,rosVectors,label='prediction')
                plt.plot(behaveDirections,behaveRos,label='behavePlus')
                plt.legend()
                ylim = [min([np.min(rosVectors),np.min(behaveRos),0]),max([np.max(rosVectors),np.max(behaveRos),1.0])]
                plt.ylim(ylim)
                plt.title(str(i)+': '+str(np.round(rmse,2)))
    elif case == 5:
        outdir = '../rothermelData/'
        nsbase = outdir+'data'
        modelIndexDict = determineFastestModel()
        datasIn = []
        datasOut = []
        i = 0
        k = 0
        t1 = uc.tic()
        while i <= 0:
        #for i in range(0,4000):
            params = getRandomConditions(paramsInput,allowDynamicModels=True)
            if i == 0:
                params['aspect'] = 160
                params['model'] = 'FM1'
                params['slope'] = 0.805
                params['m1h'] = 5.26
                params['m10h'] = 6.26
                params['m100h'] = 7.26
                params['lhm'] = 69
                params['lwm'] = 49
                params['canopyCover'] = 0.7
                params['canopyHeight'] = 14
                params['crownRatio'] = 0.2
                params['windDir'] = 34
                params['windSpeed'] = 13.5

            directions, rosVectors = getROSfromParams(params,toPrint=False)
            params['modelInd'] = modelIndexDict[params['model']]
            if True:
                fireImages, modelInputs = visualizeInputImgs(directions,rosVectors,params,resX=resX,resY=resY,toPlot=False)
                for j in range(0,len(fireImages)-1):
                    data = [fireImages[j]]
                    data.extend(modelInputs)
                    datasIn.append(rearrangeDatas(data))
                    datasOut.append(rearrangeDatas([fireImages[j+1]]))
            
            if i % 1000 == 0 and False:
                datasIn = np.squeeze(datasIn)
                datasOut = np.squeeze(datasOut)
                uc.dumpPickle([datasIn,datasOut],outdir+'dataRemakeTest'+str(len(datasOut))+'_'+str(k)+'.pkl')
                datasIn = []
                datasOut = []
                k = k + 1
            i = i + 1
        print(uc.toc(t1))
        #assert False, "Stopped"
        #uc.dumpPickle([datasIn,datasOut],outdir+'dataBehaveMoist'+str(len(datasOut))+'_'+str(k)+'.pkl')
        imgFire = visualizeInputValues(directions,rosVectors,params,resX=resX,resY=resY)
        
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        
        
        networkRaw = np.loadtxt('exampleNetworkRaw0.csv',delimiter=',')
        networkProcessed = np.loadtxt('exampleNetworkProcessed0.csv',delimiter=',')
        
        fs = 16
        
        fig, ax = plt.subplots(figsize=(16,8))
        a = []
        lims = [[-30,30],[-20,20],[-20,20],[30,150],[30,150],[0,30],[0,30],[0,30],[0,1],[0,20],[0,1],[0,53]]
        names = ['Elevation','East Wind','North Wind','Live Herbaceous Moisture','Live Woody Moisture','1-Hour Moisture','10-Hour Moisture','100-Hour Moisture','Canopy Cover','Canopy Height','Crown Ratio','Fuel Model']
        textOffset = [0]
        #modelInputs = [elevImg,windX,windY,lhmImg,lwmImg,m1hImg,m10hImg,m100hImg,canopyCoverImg,canopyHeightImg,crownRatioImg,modelImg]
        for i in range(len(modelInputs)-1,-1,-1):
            img = modelInputs[i].copy()
            img[-1,-1] = lims[i][0]
            img[-1,-2] = lims[i][1]
            oi = OffsetImage(img, zoom = 2.0, cmap='hot_r')
            box = AnnotationBbox(oi, (-0.5*i,1*i), frameon=True)
            a.append(ax.add_artist(box))
            ax.annotate(names[i],xy=(-0.5*i-1.5,1*i-1.6),xycoords='data',textcoords='data',xytext=(-0.5*i-5-(len(names[i])-10)*0.1,1*i-1.525),arrowprops=dict(facecolor='black',shrink=0.05),fontsize=fs)
        i = -1
        fireImages[1][fireImages[1] == 255] = 1
        #fireImages[1][-1,-1] = 2
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
        oi = OffsetImage(fireImages[1], zoom = 2.0, cmap='hot_r',norm=norm)
        box = AnnotationBbox(oi, (-0.5*i,1*i), frameon=True)
        ax.annotate('Initial Burn Map',xy=(-0.5*i-1.5,1*i-1.6),xycoords='data',textcoords='data',xytext=(-0.5*i-5,1*i-1.525),arrowprops=dict(facecolor='black',shrink=0.05),fontsize=fs)
        a.append(ax.add_artist(box))
        
        '''
        oi = OffsetImage([[0,0],[0,0]], zoom = 2.0, cmap='hot_r')
        box = AnnotationBbox(oi, (-0.5*i,1*i), frameon=True)
        ax.annotate('Initial Burn Map',xy=(-0.5*i-1.5,1*i-1.6),xycoords='data',textcoords='data',xytext=(-0.5*i-5,1*i-1.525),arrowprops=dict(facecolor='black',shrink=0.05),fontsize=fs)
        a.append(ax.add_artist(box))
        '''
        
        
        
        i = 6# 6
        imgX = 3.50
        ax.annotate('',xy=(imgX,1*i),xycoords='data',textcoords='data',xytext=(imgX-2,1*i),arrowprops=dict(facecolor='black',shrink=0.01,width=50,headwidth=100,headlength=30))
        ax.annotate('Convolutional\nNeural Network',xy=(imgX-0.9,1*i-0.9),xycoords='data',textcoords='data',xytext=(imgX-2.5,1*i-2.85),fontsize=fs)
        imgX = 5.5
        i = 3
        oi = OffsetImage(networkRaw, zoom = 2.0, cmap='hot_r')
        box = AnnotationBbox(oi, (imgX,1*i), frameon=True)
        ax.annotate('Probability of\nFire',xy=(imgX-0.8,1*i-0.9),xycoords='data',textcoords='data',xytext=(imgX-1.1,1*i-2.85),fontsize=fs)
        a.append(ax.add_artist(box))
        imgX = 5.5
        i = 10
        oi = OffsetImage(1-networkRaw, zoom = 2.0, cmap='hot_r')
        box = AnnotationBbox(oi, (imgX,1*i), frameon=True)
        ax.annotate('Probability of\nNot Fire',xy=(imgX-0.8,1*i-0.9),xycoords='data',textcoords='data',xytext=(imgX-1.1,1*i-2.85),fontsize=fs)
        a.append(ax.add_artist(box))
        
        i = 6# 6
        imgX = 9.5
        ax.annotate('',xy=(imgX,1*i),xycoords='data',textcoords='data',xytext=(imgX-2,1*i),arrowprops=dict(facecolor='black',shrink=0.01,width=50,headwidth=100,headlength=30))
        ax.annotate('Post Processing',xy=(imgX-1.1,1*i-1.4),xycoords='data',textcoords='data',xytext=(imgX-2.25,1*i-2.85),fontsize=fs)
        
        imgX = 11.5
        i = 6
        oi = OffsetImage(networkProcessed, zoom = 2.0, cmap='hot_r')
        box = AnnotationBbox(oi, (imgX,1*i), frameon=True)
        ax.annotate('Burn Map\nAfter 6 Hours',xy=(imgX-1.5,1*i-0.9),xycoords='data',textcoords='data',xytext=(imgX-1.15,1*i-2.85),fontsize=fs)
        a.append(ax.add_artist(box))
        
        
        plt.ylim(-2.6,12.5)
        plt.xlim(-10,12.5)
        #plt.ylim(-50,50)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('inputsExampleSingleFire.eps')
        
        
        
        
        
        
        #manyFiresInputFigure(modelInputs)
        makeFirePerimetersFigure(imgFire)


        
    elif case == 6:
        import glob
        outdir = '../rothermelData/'
        dataFile = outdir+'dataBehaveMoist3000'
        files = glob.glob(dataFile+'*.pkl')
        ns = outdir+'behaveMoistData'
        allIn = []
        allOut = []
        for i in range(0,len(files)):
            [inData,outData] = uc.readPickle(files[i])
            allIn.extend(inData)
            allOut.extend(outData)
        datas = (inData,outData)
    
    #weightedMoistureDead, weightedMoistureLive = getWeightedMoistures(fuelModel,intermediates)
    #weightedSilicaDead, weightedSilicaLive = getWeightedSilicas(fuelModel,intermediates)
    #weightedFuelLoadDead, weightedFuelLoadLive = getWeightedFuelLoads(fuelModel,intermediates)
    #weightedHeatDead, weightedHeatLive = getWeightedHeats(fuelModel,intermediates)
    
    #plt.plot(directions,rosVectors)
    #plt.ylim([0,40])
    #plt.xlim([0,400])


    #truthDir = np.linspace(0,360,13)    
    # Wind 15 at 0 deg FM1
    # Slope 0.5 at 0 FM1
    #truthR = [4.5,5.1,7.9,19.2,46.9,72.7,82.5,72.7,46.9,19.2,7.9,5.1,4.5]
    
    # Wind 25 at 30 deg FM1
    # Slope 0.25 at 0 FM1
    #truthR = [5.9,5.3,6.2,10.1,28.8,76.4,117.8,131.8,113.4,69.3,24.2,9.2,5.9]
    
    # Wind 25 at 30 deg FM4
    # Slope 0.25 at 0 FM4
    #truthR = [9.3,8.3,9.6,15.7,45.3,121.1,187.1,209.5,180.3,110.2,38.3,14.3,9.3]
    
    # Wind 25 at 30 deg FM4
    # Slope 0.50 at 270 FM4
    #truthR = [8.4,8.6,11.5,23.6,69.8,133.6,172.1,169.3,126.2,61.5,20.8,10.9,8.4]
    
    # Wind 25 at 30 deg GS1
    # Slope 0.50 at 270 GS1
    #truthR = [1.9,1.9,2.6,5.2,15.5,30.0,39.0,38.5,28.9,14.3,4.8,2.5,1.9]
    
    #plt.plot(truthDir,truthR)
    
    """
    
    

    print(heatSink,heatFlux)
    """
    