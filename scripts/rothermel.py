# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 07:53:06 2018

@author: JHodges
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import network_analysis as na
import util_common as uc
from scipy.ndimage.interpolation import zoom
from scipy.interpolate import griddata

def windFactor(model,u,modelLine=0,rhoP=1.0):
    sigma = getSigma(model,modelLine)
    beta = getBeta(model,rhoP)
    C = 7.47*np.exp(-0.133*(sigma**0.55))
    B = 0.02526*(sigma**0.54)
    E = 0.715*np.exp(-3.59*(10**-4)*sigma)
    betaOP = getBetaOP(sigma)
    if u > 0:
        wF = C*(u**B)*((beta/betaOP)**(-E))
    else:
        wF = 0.0
    return wF

def slopeFactor(model,phi,rhoP=1.0):
    beta = getBeta(model,rhoP)
    if phi > 0:
        sF = 5.275*(beta**(-0.3))*(np.tan(phi)**2)
    else:
        sF = -1.5*(np.tan(abs(phi)))
    return sF

def getBeta(model,rhoP=1.0,modelLine=0):
    fuelDepth = getDelta(model)
    w0 = getW0(model,modelLine)
    rhoB = w0/fuelDepth
    rhoB = 32
    beta = rhoP/rhoB # I reversed this?
    #print("Beta=",beta)
    return beta

def getBetaOP(sigma):
    return 3.348*(sigma**(-0.8189))

def getA(sigma,revised=True):
    if revised:
        return 133*(sigma**(-0.7913))
    else:
        return 1/(4.774*(sigma**0.1)-7.27) # Rothermel 1972

def getWn(model,St=0.0555,modelLine=0,revised=True):
    w0 = getW0(model,modelLine=0)
    if revised:
        return w0*(1-St)
    else:
        return w0/(1+St) # Rothermel 1972

def getNs(Se=0.010):
    return 0.174*(Se**(-0.19))

def getNm(model,m1h,m10h,m100h):
    Mf = getMf(model,m1h,m10h=m10h,m100h=m100h)
    Mx = getMx(model)
    return 1-2.59*(Mf/Mx)+5.11*((Mf/Mx)**2)-3.52*((Mf/Mx)**3)

def getSigma(model,modelLine=0):
    sigmas = [[0,0,0,0],
              [3500,0,0,0],
              [3000,109,30,1500],
              [1500,0,0,0],
              [2000,109,30,1500],
              [2000,109,0,1500],
              [1750,109,30,0],
              [1750,109,30,1550],
              [2000,109,30,0],
              [2500,109,301,0],
              [2000,109,30,1500],
              [1500,109,30,0],
              [1500,109,30,0],
              [1500,109,30,0]]
    return sigmas[model][modelLine]
  
def getW0(model,modelLine=0):
    w0s = [[0,0,0,0],
           [0.034,0,0,0],
           [0.092,0.046,0.023,0.023],
           [0.138,0,0,0],
           [0.230,0.184,0.092,0.230],
           [0.046,0.023,0,0.092],
           [0.069,0.115,0.092,0],
           [0.052,0.086,0.069,0.017],
           [0.069,0.046,0.115,0],
           [0.134,0.019,0.007,0],
           [0.138,0.092,0.230,0.092],
           [0.069,0.207,0.253,0],
           [0.184,0.644,0.759,0],
           [0.322,1.058,1.288,0]]
    return w0s[model][modelLine]

def getDelta(model):
    deltas = [0.0,1.0,1.0,2.5,6.0,2.0,2.5,2.5,0.2,0.2,1.0,1.0,2.3,3.0]
    return deltas[model]

def getMx(model):
    Mxs = [0,12,15,25,20,20,25,40,30,25,25,15,20,25]
    return Mxs[model]

def getReactionIntensity(model,m1h,m10h,m100h,rhoP,modelLine=0,h=8000):   
    gammaPrime = getGammaPrime(model,rhoP=rhoP,modelLine=modelLine)
    Wn = getWn(model)
    nS = getNs()
    nM = getNm(model,m1h,m10h,m100h)
    #nM = 1.0
    #nS = 1.0
    
    reactionIntensity = gammaPrime*Wn*h*nM*nS
    return reactionIntensity
    

def getGammaPrimeMax(sigma):
    return (sigma**1.5)/(495+0.0594*(sigma**1.5))

def getGammaPrime(model,rhoP=1.0,modelLine=0):
    sigma = getSigma(model,modelLine)
    #print("RhoP=",rhoP)
    beta = getBeta(model,rhoP)
    betaOP = getBetaOP(sigma)
    A = getA(sigma)
    gammaPrimeMax = getGammaPrimeMax(sigma)
    return gammaPrimeMax*((beta/betaOP)**A)*np.exp(A*(1-(beta/betaOP)))
  
def getMxLiving(alpha,mfDead,revised=True):
    if revised:
        #wPrime = 
        return max([2.9*((1-alpha)/alpha)*(1-10*mfDead/3)-0.226,0.3]) # Need to implement
    else:
        return max([2.9*((1-alpha)/alpha)*(1-10*mfDead/3)-0.226,0.3]) # Rothermel 1972

def getMf(model,m1h,m10h=0,m100h=0):
    if model == 11 or model == 12 or model == 13:
        return 0.76*m1h+0.18*m10h+0.06*m100h
    elif model == 6 or model == 7:
        return 0.89*m1h+0.09*m10h+0.02*m100h
    else:
        return m1h

def getUpsilon(model,rhoP=1.0,modelLine=0):
    sigma = getSigma(model,modelLine)
    
    beta = getBeta(model,rhoP)
    #print(sigma,beta,model,rhoP)
    return ((192+0.2595*sigma)**(-1))*np.exp((0.792+0.681*(sigma**0.5))*(beta+0.1))

def getEpsilon(model,modelLine=0):
    sigma = getSigma(model,modelLine)
    return np.exp(-138/sigma)
  
def getQig(model,m1h,m10h=0,m100h=0):
    Mf = getMf(model,m1h,m10h=m10h,m100h=m100h)
    return 250 + 1116 * (Mf/100)

def getRhoP(model):
    rhoPs = [0,32.75127,20.51292,32.27313,14.736442,15.1791398]
      
    return rhoPs[model]

def getRmodel(model,m1h=8,lhm=30,lwm=30):
    # Note, these are with m1h, m10h, m100h = 8, lhm, lwm = 30
    Rs_m1h1 = [0,7.7,4.7,7.9,11.7,3.5,3.1,2.8,0.4,1.5,2.0,1.1,2.5,3.3]
    Rs_m1h8 = [0,3.6,2.7,3.9,8.0,2.5,1.5,1.8,0.2,0.7,1.3,0.5,1.2,1.6]
    Rs_m1h16 = [0,7.7,4.7,7.9,11.7,0.5,3.1,2.8,0.4,1.5,2.0,1.1,2.5,3.3]
    Rs = Rs_m1h8
    return Rs[model]

def getRmoist(model,m1h=8,lhm=30,lwm=30):
    # Note, these are for model 5
    Rs5_lwm30_m1 = [3.5,2.9,2.5,2.1,1.9,1.7,1.6,1.5,1.4,1.2,1.1,1.0,0.9,0.8,0.6,0.5,0.4,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    Rs5_lwm30_m2 = [3.3,2.7,2.3,2.0,1.8,1.6,1.5,1.4,1.3,1.2,1.0,0.9,0.8,0.6,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2]
    Rs5_lwm30_m3 = [3.1,2.5,2.2,1.9,1.7,1.6,1.4,1.3,1.2,1.1,0.9,0.8,0.6,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m4 = [2.9,2.4,2.1,1.8,1.6,1.5,1.4,1.2,1.1,0.9,0.8,0.6,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m5 = [2.8,2.3,2.0,1.8,1.6,1.4,1.3,1.2,1.0,0.8,0.6,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m6 = [2.7,2.2,1.9,1.7,1.5,1.4,1.2,1.1,0.9,0.6,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m7 = [2.6,2.2,1.9,1.7,1.5,1.3,1.1,0.9,0.6,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m8 = [2.5,2.1,1.8,1.6,1.4,1.2,1.0,0.7,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m9 = [2.4,2.1,1.8,1.6,1.4,1.1,0.7,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m10= [2.3,2.0,1.8,1.5,1.2,0.8,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m11= [2.3,2.0,1.7,1.3,0.8,0.4,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m12= [2.2,1.9,1.5,0.9,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m13= [2.1,1.7,0.9,0.5,0.4,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    Rs5_lwm30_m14= [1.9,1.0,0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.1,0.1]
    Rs5_lwm30_m15= [1.0,0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1,0.1]
    Rs5_lwm30_m16= [0.5,0.5,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    Rs5_lwm30_m17= [0.4,0.4,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    Rs5_lwm30_m18= [0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    Rs5_lwm30_m19= [0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    Rs5_lwm30_m20= np.zeros(np.shape(Rs5_lwm30_m1))
    m1_vals = np.linspace(1,30,30)
    lwm_vals = np.linspace(30,300,28)
    m1Grid, lwmGrid = np.meshgrid(m1_vals,lwm_vals)
    m1GridRs = np.reshape(m1Grid,(m1Grid.shape[0]*m1Grid.shape[1]))
    lwmGridRs = np.reshape(lwmGrid,(lwmGrid.shape[0]*lwmGrid.shape[1]))
    points = np.array([m1GridRs,lwmGridRs]).T
    Rs = np.zeros(m1Grid.shape)
    Rs[:,0] = Rs5_lwm30_m1
    Rs[:,1] = Rs5_lwm30_m2
    Rs[:,2] = Rs5_lwm30_m3
    Rs[:,3] = Rs5_lwm30_m4
    Rs[:,4] = Rs5_lwm30_m5
    Rs[:,5] = Rs5_lwm30_m6
    Rs[:,6] = Rs5_lwm30_m7
    Rs[:,7] = Rs5_lwm30_m8
    Rs[:,8] = Rs5_lwm30_m9
    Rs[:,9] = Rs5_lwm30_m10
    Rs[:,10]= Rs5_lwm30_m11
    Rs[:,11]= Rs5_lwm30_m12
    Rs[:,12]= Rs5_lwm30_m13
    Rs[:,13]= Rs5_lwm30_m14
    Rs[:,14]= Rs5_lwm30_m15
    Rs[:,15]= Rs5_lwm30_m16
    Rs[:,16]= Rs5_lwm30_m17
    Rs[:,17]= Rs5_lwm30_m18
    Rs[:,18]= Rs5_lwm30_m19
    values = np.reshape(Rs,(Rs.shape[0]*Rs.shape[1],))
    R = griddata(points,values,[m1h,lwm])
    
    return R[0]

def getRateOfSpread(model,u,phi,m1h,m10h,m100h,rhoP,modelLine,rhoB=32.0,lhm=30,lwm=30):
    rI = getReactionIntensity(model,m1h,m10h,m100h,rhoP,modelLine=modelLine,h=8000)
    
    upsilon = getUpsilon(model,rhoP=rhoP,modelLine=modelLine)
    epsilon = getEpsilon(model,modelLine=modelLine)
    Qig = getQig(model,m1h,m10h,m100h)
    #Qig = 11.1
    #upsilon = 2424/11.1
    #rhoP = getRhoP(model)
    
    phiW = windFactor(model,u,modelLine,rhoP)
    phiS = slopeFactor(model,phi,rhoP)
    #print(rI,upsilon,rhoB,epsilon,Qig,phiW,phiS)
    #print(phiW)
    R = (rI*upsilon/(rhoP*epsilon*Qig))
    #phiW = 53.96
    #R = 3.6 #*1.1
    #R = getRmodel(model)
    R = getRmoist(model,m1h=m1h,lhm=lhm,lwm=lwm)
    return R*(1+phiW+phiS)

def cartCoords(theta,R):
    coords = np.zeros((len(R),2))
    theta = (theta-1.57) #/180*3.1415926535
    polar2z = lambda r,t: r * np.exp( 1j * t )
    z = polar2z(R,theta)
    coords[:,0] = np.real(z)
    coords[:,1] = np.imag(z)
    #Rmat = np.array([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
    #for i in range(0,len(R)):
    #    coords[i,:] = [np.cos(theta[i])*R[i], np.sin(theta[i])*R[i]]
    return coords

def rothermelOuputToImg(theta,R,resX=50,resY=50):
    coords = cartCoords(theta,R)
       
    (coords[:,0],coords[:,1]) = (coords[:,0]+resX/2,coords[:,1]+resY/2)
    coords = np.array(coords,dtype=np.int32)
    
    coordsTuple = []
    for c in coords:
        coordsTuple.append((c[0],c[1]))
    
    img = Image.new('LA',(resX,resY))
    draw = ImageDraw.Draw(img)
    draw.polygon(coordsTuple,fill='black',outline=None)
    img = np.asarray(img)[:,:,1]
    
    return img

def slopeToElevImg(phi,phiDir,resX=50,resY=50):
    slopeX = phi*np.sin(phiDir)
    slopeY = phi*np.cos(phiDir)
    img = np.zeros((2,2))
    #img[img == 0] = np.nan
    img[0,0] = -resX/2*slopeX+resY/2*slopeY
    img[0,-1]= resX/2*slopeX+resY/2*slopeY
    img[-1,0] = -resX/2*slopeX-resY/2*slopeY
    img[-1,-1] = resX/2*slopeX-resY/2*slopeY
    img = zoom(img,resX/2,order=1)
    return img

def convertInputs(u,uDir,phi,phiDir):
    phi = np.arctan(phi)
    u = u*88 # convert to ft/min
    phiDir = phiDir*3.1415926535/180
    uDir = uDir*3.1415926535/180
    return u, uDir, phi, phiDir

def convertR(R):
    R = R*1.1 # Convert to ft/min
    R = R*60.0/5280.0 # Convert to mi/hour
    R = R*1.60934 # Convert to km/hour
    return R

def randomInput(bounds):
    mx = np.max(bounds)
    mn = np.min(bounds)
    value = np.random.random()*(mx-mn)+mn
    return value

def rearrangeDatas(datas):
    sz = datas[0].shape
    szrs = sz[0]*sz[1]
    datasNew = np.zeros((szrs*len(datas),))
    
    for i in range(0,len(datas)):
        datasNew[i*szrs:(i+1)*szrs] = np.reshape(datas[i],(szrs,))
    return datasNew

if __name__ == "__main__":
    
    """ Unchanging inputs
    """
    outdir = '../rothermelData/'
    nsbase = outdir+'data'
    debugPrint = False
    
    # Output parameters
    resX = 50
    resY = 50
    totalSteps = 4
    timeStep = 6    

    # Rothermel parameters    
    model = 1
    modelLine = 0
    rhoP = 0.0340 
    
    """ Changing inputs
    """
    u = 0 #7.8 # specify in mph
    uDir = 30 # specify in degrees from north (CW positive)
    phi = 0.5 # specify in percent
    phiDir = 45 # specify in degrees from north

    u_bounds = [0,6]
    uDir_bounds = [0,360]
    phi_bounds = [0,0.50]
    phiDir_bounds = [0,360]
    lwm_bounds = [30,210]
    m1h_bounds = [1,20]
    datasIn = []
    datasOut = []
    for j in range(0,1):
        u = randomInput(u_bounds)
        uDir = randomInput(uDir_bounds)
        phi = randomInput(phi_bounds)
        phiDir = randomInput(phiDir_bounds)
        lwm = randomInput(lwm_bounds)
        m1h = randomInput(m1h_bounds)
        m10h = m1h
        m100h = m1h
        u, uDir, phi, phiDir = convertInputs(u,uDir,phi,phiDir)
        
        theta = np.linspace(0,3.1415926535*2,181)
        R = np.zeros((len(theta),))
        
        for i in range(0,len(theta)):
            u0 = np.max([u*np.cos((uDir-theta[i])),0])
            phi0 = phi*np.cos(((phiDir-theta[i])))
            R[i] = getRateOfSpread(model,u0,phi0,m1h,m10h,m100h,rhoP,modelLine,lwm=lwm)
            if debugPrint:
                print('theta=%.0f \tu0=%.2f \tphi0=%.2f\tR=%.2f'%(theta[i],u0/88,phi0/3.1415926535*180,R[i]))
        
        R = convertR(R)
        
        windX = np.zeros((resX,resY))+u*np.sin(uDir)
        windY = np.zeros((resX,resY))+u*np.cos(uDir)
        elev = slopeToElevImg(phi,phiDir,resX=resX,resY=resY)
        
        ns = nsbase+'_u_'+str(np.round(u,decimals=2))
        ns = ns+'_uDir_'+str(np.round(uDir,decimals=2))
        ns = ns+'_p_'+str(np.round(phi,decimals=2))
        ns = ns+'_pDir_'+str(np.round(phiDir,decimals=2))
        
        for i in range(0,totalSteps):
            startImg = rothermelOuputToImg(theta,R*timeStep*(i+1),resX=resX,resY=resY)
            finalImg = rothermelOuputToImg(theta,R*timeStep*(i+2),resX=resX,resY=resY)
            data = [startImg,elev,windX,windY,finalImg]
            names = ['input','Elevation','WindX','WindY','Correct']
            clims = [[0,1],[-20,20],[-500,500],[-500,500],[0,1]]
            if i == 0 and False:
                na.plotWildfireTest(data,names,clims=clims,saveFig=False,saveName=ns+'Trial.png')
                print(ns)
            datasIn.append(rearrangeDatas(data[0:4]))
            datasOut.append(rearrangeDatas([data[-1]]))
            
    datasIn = np.squeeze(datasIn)
    datasOut = np.squeeze(datasOut)
    uc.dumpPickle([datasIn,datasOut],outdir+'data1.pkl')
    """
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.figure(figsize=(4,4))
    plt.imshow(img2)
    """
    
    
    # Brush fire Dead Fuel Fine
    # sigma = 2000
    # w0 = 0.046
    # Brush fire Dead Fuel Medium
    # sigma = 109
    # w0 = 0.023
    # Brush fire Living Fuel
    # sigma = 1500
    # w0 = 0.092