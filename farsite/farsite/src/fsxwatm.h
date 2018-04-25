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
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <mem.h>
#include <iostream>
//#include <boost/filesystem/path.hpp>
//#include <boost\filesystem\path.hpp>
class Farsite5;

#define ATMTEMP  0
#define ATMHUMID 1
#define ATMRAIN  2
#define ATMWSPD  3
#define ATMWDIR  4
#define ATMCLOUD 5

class Atmosphere
{
private:
	double North;
	double South;
	double East;
	double West;
	double ResolutionX;
	double ResolutionY;
	long CellX;
	long CellY;
	short* Value;

public:
	long XNumber;
	long YNumber;
	long NumCellsPerGrid;
	Farsite5 *pFarsite;
	Atmosphere();
	~Atmosphere();
	void SetHeaderInfo(double north, double south, double east, double west,
		double resolutionx, double resolutiony);
	bool GetHeaderInfo(double* north, double* south, double* east,
		double* west, double* resolutionx, double* resolutiony);
	void GetResolution(double* resolutionx, double* resolutiony);
	bool CompareHeader(double north, double south, double east, double west,
		double resolutionx, double resolutiony);
	bool AllocAtmGrid(long timeintervals);
	bool FreeAtmGrid();
	bool GetAtmValue(double xpt, double ypt, long time, short* value);
	bool SetAtmValue(long number, long time, short value);
};


class AtmosphereGrid
{
	// base class for access and loading of atmospheric variables
	FILE* InputTable;
	FILE* ThisFile;
	long month, day, hour;
	long* Month;		// array of pointers to longs that store dates
	long* Day;
	long* Hour;
	long NumGrids;
	long StartGrid;
	long TimeIntervals;
	long Metric;
	Atmosphere atmosphere[6];
	Farsite5 *pFarsite;

public:
	bool AtmGridWTR;
	bool AtmGridWND;
	char ErrMsg[256];

	AtmosphereGrid(long numgrids, Farsite5 *_pFarsite);  	// will default to 6 if all themes included,
	~AtmosphereGrid();  				// and 3 if only wind spd dir & cloud %
	bool ReadInputTable(char *InputFileName);
	bool ReadHeaders(long FileNumber);
	bool CompareHeader(long FileNumber);
	bool SetAtmosphereValues(long timeinterval, long filenumber);
	bool GetAtmosphereValue(long FileNumber, double xpt, double ypt,
		long time, short* value);
	void GetResolution(long FileNumber, double* resolutionx,
		double* resolutiony);
	long GetAtmMonth(long count);
	long GetAtmDay(long count);
	long GetAtmHour(long count);
	long GetTimeIntervals();
	void FreeAtmData();
};

