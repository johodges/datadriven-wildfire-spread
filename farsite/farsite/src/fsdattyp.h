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
#include <stdlib.h>

#ifndef FARSITE_DATATYPES
#define FARSITE_DATATYPES

typedef struct
{
	long Month;
	long Day;
	long Hour;
	long Min;
}  TimeStruct;

struct TimeData
{
	long Number;
	long numarrays;
	TimeStruct* Time1, * Time2;

	TimeData();
	~TimeData();
	long GetDay(long num);
	long GetHour(long num);
	long GetMin(long num);
	long GetMonth(long num);
	void SetData(long day, long hour, long min, long month);
	void ReAllocData();
	void ReSet();
};


typedef struct
{
	double Horiz;
	double Slope;
} GraphStruct;


struct GraphData
// contains horizontal and topologic area and perimeter data
// stores data and functions for area and perimeter data
{
	double DataMax, DataMin;
	long number;			// number of observations
	long numarrays;		// number of 50 unit arrays
	GraphStruct* Graph1;
	GraphStruct* Graph2;

	GraphData();
	~GraphData();
	void ReSet();
	void ReAllocData();
	double GetHoriz(long Num);
	double GetSlope(long Num);
	void SetData(double Horiz, double Slope);
};


double GetTopologicalData(long Type, long Num);
double GetHorizontalData(long Type, long Num);

typedef struct
{
	long Fires;
	long Spots;
	long Enclaves;
} FireStruct;


struct FireData
{
	long number, count, numarrays;
	//long *Fires, *fires;
	//long *Spots, *spots;
	//long *Enclaves, *enclaves;
	FireStruct* Fire1, * Fire2;

	FireData();
	~FireData();
	long GetFires(long num);
	long GetSpots(long num);
	long GetEnclaves(long num);
	void SetData(long fires, long spots, long enclaves);
	void ReAllocData();
	void ReSet();
};

#endif // FARSITE_DATATYPES
