/*
 * NOTICE OF RELEASE TO THE PUBLIC DOMAIN
 *
 * This work was created using public funds by employees of the 
 * USDA Forest Service's Fire Science Lab and Systems for Environmental 
 * Management.  It is therefore ineligible for copyright under title 17, 
 * section 105 of the United States Code.  You may treat it as you would 
 * treat any public domain work: it may be used, changed, copied, or 
 * redistributed, with or without permission of the authors, for free or 
 * for compensation.  You may not claim exclusive ownership of this code 
 * because it is already owned by everyone.  Use this software entirely 
 * at your own risk.  No warranty of any kind is given.
 * 
 * FARSITE is a trademark owned by Mark Finney.  You may not call derived 
 * works by the name FARSITE without explicit written permission.
 * 
 * A copy of 17-USC-105 should have accompanied this distribution in the file 
 * 17USC105.html.  If not, you may access the law via the US Government's 
 * public websites: 
 *   - http://www.copyright.gov/title17/92chap1.html#105
 *   - http://www.gpoaccess.gov/uscode/  (enter "17USC105" in the search box.)
 */

//------------------------------------------------------------------------------
// 	FSXW.H 		Header File contains functions for accessing global data
// 				associated with the fire growth model
//
//
//
//  				Copyright 1994, 1995
//   			Mark A. Finney, Systems for Environmental Management
//------------------------------------------------------------------------------


#ifndef GlobalFarsiteModelFunctions
#define GlobalFarsiteModelFunctions

#include <stdlib.h>
//#include <owl\window.h>

#define GETVAL -1		// definitions used for getting and storing data
#define XCOORD 0		// in fire perimeter1 arrays:
#define YCOORD 1
#define ROSVAL 2
#define FLIVAL 3
#define RCXVAL 4

#define NUMDATA 5

#define MAXNUM_STOPLOCATIONS 10

#define DISTCHECK_SIMLEVEL  	0
#define DISTCHECK_FIRELEVEL  1

//typedef unsigned BYTE;

//----------------- Model Parameter Access Functions --------------------------

/*double GetDistRes();
void SetDistRes(double input);
void SetDynamicDistRes(double input);
double GetDynamicDistRes();
double GetPerimRes();
void SetPerimRes(double input);
bool AccelerationON();
void SetAccelerationON(bool State);
double GetRosRed(int fuel);
void SetRosRed(int fuel, double rosred);
void InitializeRosRed();
double GetActualTimeStep();
double GetVisibleTimeStep();
double GetTemporaryTimeStep();
void SetActualTimeStep(double input);
void SetVisibleTimeStep(double input);
void SetTemporaryTimeStep(double input);
void SetSecondaryVisibleTimeStep(double input);
double GetSecondaryVisibleTimeStep();
double EventMinimumTimeStep(double time);   		// event driven time step
bool CheckExpansions(long YesNo);   		   	// check illogical perimeter expansions
bool CheckPostFrontal(long YesNo);
long DistanceCheckMethod(long Method);

//----------------- Fire Data Access Functions --------------------------------

long GetInout(long FireNumber);
long GetNumPoints(long FireNumber);
void SetInout(long FireNumber, int Inout);
void SetNumPoints(long FireNumber, long NumPoints);
void GetPerimeter2(long coord, double* xpt, double* ypt, double* ros,
	double* fli, double* rct);
double GetPerimeter2Value(long coord, long value);
void SetPerimeter2(long coord, double xpt, double ypt, double ros, double fli,
	double rct);
double* AllocPerimeter2(long NumPoints);
void FreePerimeter2();
double* AllocPerimeter1(long NumFire, long NumPoints);
void FreePerimeter1(long NumFire);
double GetPerimeter1Value(long NumFire, long NumPoint, int coord);
void SetPerimeter1(long NumFire, long NumPoint, double xpt, double ypt);
void SetFireChx(long NumFire, long NumPoint, double ros, double fli);
void SetReact(long NumFire, long NumPoint, double ReactionIntensity);
double* GetPerimeter1Address(long NumFire, long NumPoint);
long SwapFirePerims(long NumFire1, long NumFire2);
long GetNumPerimAlloc();
bool AllocFirePerims(long num);
bool ReAllocFirePerims();
void FreeAllFirePerims();


//----------------- Fire Accounting Functions ---------------------------------

bool CreateSpotSemaphore();
void CloseSpotSemaphore();
void GetNumSpots(long* NumSpots, bool inc);
void SetNumSpots(long input);
void IncNumSpots(long increment);
long GetNumFires();
void SetNumFires(long input);
void IncNumFires(long increment);
long GetNewFires();
void SetNewFires(long input);
void IncNewFires(long increment);
long GetSkipFires();
void SetSkipFires(long newvalue);
void IncSkipFires(long increment);
double PercentIgnition(double);
double IgnitionDelay(double delay);
bool EnableCrowning(long);
bool EnableSpotting(long);
bool EnableSpotFireGrowth(long);
bool ConstantBackingSpreadRate(long);

// -----------------Elevation Functions ---------------------------------------

void AllocElev(long CurrentFire);
void SetElev(long Num, long elev);
long GetElev(long Num);
long* GetElevAddress(long Num);
void FreeElev();

// -----------------Stop-location Functions ---------------------------------------

long SetStopLocation(double xcoord, double ycoord);
bool GetStopLocation(long StopNum, double* xcoord, double* ycoord);
void ResetStopLocation(long StopNum);
bool EnableStopLocation(long StopNum, long Action);
void ResetAllStopLocations();
long GetNumStopLocations();*/

#endif    //GlobalFarsiteModelFunctions

