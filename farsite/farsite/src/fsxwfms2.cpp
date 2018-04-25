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
//
//   fsxwfms2.cpp
//   FARSITE 4.0
//   10/20/2000
//   Mark A. Finney (USDA For. Serv., Fire Sciences Laboratory)
//
//   Contains functions for calculating dead fuel moisture maps for an entire
//  	  landscape file (declarations in fsx4.hpp)
//
//------------------------------------------------------------------------------

#include <math.h>
#include <string.h>
//#include "newfms.h"//fms-0.4.0.h"
#include "Farsite5.h"
//#ifdef WIN32
#include "fsxsync.h"
//#endif
#include "fsx4.hpp"
#include "cdtlib.h"
//#include "portablestrings.h"
//#include <windows.h>
//#include <process.h>
//extern const double PI;
static const double PI = acos(-1.0);

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//   Global Access Functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


FELocalSite::FELocalSite(Farsite5 *_pFarsite) : LandscapeData(_pFarsite)
{
	pFarsite = _pFarsite;
	StationNumber = -1;
	pFarsite->WindLoc = 0;
    pFarsite->WindLoc_New = 0;

	//	SetCanopyChx(15.0, 4.0, 0.2, 30.0, 100, 2, 1); // for each thread
}



long FELocalSite::GetLandscapeData(double xpt, double ypt)
{
	if (xpt<pFarsite->GetLoEast() ||
		xpt>pFarsite->GetHiEast() ||
		ypt<pFarsite->GetLoNorth() ||
		ypt>pFarsite->GetHiNorth())
		return -1;

	long posit;

	pFarsite->CellData(xpt, ypt, cell, crown, ground, &posit);
	ElevConvert(cell.e);				// need to convert units and codes
	SlopeConvert(cell.s);
	AspectConvert(cell.a);
	FuelConvert(cell.f);
	CoverConvert(cell.c);
	HeightConvert(crown.h);
	BaseConvert(crown.b);
	DensityConvert(crown.p);
	DuffConvert(ground.d);
	WoodyConvert(ground.w);
	StationNumber = pFarsite->GetStationNumber(xpt, ypt) - 1;
	XLocation = xpt;
	YLocation = ypt;

	return posit;
}


long FELocalSite::GetLandscapeData(double xpt, double ypt, LandscapeStruct& ls)
{
	if (xpt<pFarsite->GetLoEast() ||
		xpt>pFarsite->GetHiEast() ||
		ypt<pFarsite->GetLoNorth() ||
		ypt>pFarsite->GetHiNorth())
	{
		ld.elev = ld.slope = ld.fuel = ld.cover = -9999;
		ld.aspectf = ld.height = ld.base = ld.density = ld.duff = -9999;
		ld.woody = NODATA_VAL;//-9999;
		return 0;
	}
	long posit;
	celldata cell;
	crowndata crown;
	grounddata ground;

	pFarsite->CellData(xpt, ypt, cell, crown, ground, &posit);
	ElevConvert(cell.e);				// need to convert units and codes
	SlopeConvert(cell.s);
	AspectConvert(cell.a);
	FuelConvert(cell.f);
	CoverConvert(cell.c);
	HeightConvert(crown.h);
	BaseConvert(crown.b);
	DensityConvert(crown.p);
	DuffConvert(ground.d);
	WoodyConvert(ground.w);
	StationNumber = pFarsite->GetStationNumber(xpt, ypt) - 1;
	XLocation = xpt;
	YLocation = ypt;
	ls = ld;

	return posit;
}

void FELocalSite::GetEnvironmentData(EnvironmentData* mw)
{
	(*mw).ones = onehour;
	(*mw).tens = tenhour;
	(*mw).hundreds = hundhour;
	(*mw).liveh = (double) pFarsite->GetInitialFuelMoisture(ld.fuel, 3) / 100.0;
	(*mw).livew = (double) pFarsite->GetInitialFuelMoisture(ld.fuel, 4) / 100.0;
	(*mw).windspd = mwindspd;
	(*mw).tws = twindspd;
	if (wwinddir >= 0.0)
		(*mw).winddir = wwinddir / 180.0 * PI;
	else
		(*mw).winddir = wwinddir;
	(*mw).thousands = thousands;
	//(*mw).tenthous = tenthous;
	(*mw).temp = airtemp;
	(*mw).humid = relhumid;
	(*mw).solrad = solrad;
}

/********************************************************************************************************/
void FELocalSite::GetFireEnvironment( double SimTime, bool All)
{
long date, cloud, hour, l_Elev;
double hours;
double Radiate = 0, EquilMx;

	 tenhour = hundhour = onehour = thousands = solrad =mwindspd= -1.0;
	 if (ld.elev == -9999)
		  return;

	 date = pFarsite->Chrono(SimTime, &hour, &hours, false);				// get hours days etc.
	 if (pFarsite->AtmosphereGridExists())
		  AtmWindAdjustments(date, hours, &cloud);
 	else
	  	windadj(date, hours, &cloud);

	 if (ld.fuel < 1)
		  tenhour = hundhour = onehour = thousands = 0.0;
 	else {
		  if (All)
			   Radiate = 1;

//	  	tenhour = env->GetMx (SimTime, ld.fuel, ld.elev, ld.slope, ld.aspectf,	ld.cover, &EquilMx, &Radiate, SIZECLASS_10HR);	// Radiate is not returned if set to 0
     tenhour = this->pFarsite->cfmc.GetMx (SimTime, ld.fuel, ld.elev, ld.slope, ld.aspectf,	ld.cover, &EquilMx, &Radiate, SIZECLASS_10HR);	// Radiate is not returned if set to 0

	  	if (All) {                          // env->elref always in feet.
//			   env->fmsthread[0].SiteSpecific ( (long)(env->elevref - ld.elev * 3.2808), env->tempref, &airtemp, env->humref, &relhumid);
      l_Elev = ( (double)ld.elev * 3.2808 ) + 0.5;
      this->pFarsite->cfmc.SiteSpecific (l_Elev, &airtemp, &relhumid);
    }  /* if (All) */

		  onehour = (4.0 * EquilMx + tenhour) / 5.0;
		  solrad = Radiate;
		  Radiate = 0;

//	 hundhour = env->GetMx (SimTime, ld.fuel, ld.elev, ld.slope, ld.aspectf,	ld.cover, &EquilMx, &Radiate, SIZECLASS_100HR);	// Radiate is not returned if set to 0
 	  hundhour = this->pFarsite->cfmc.GetMx (SimTime, ld.fuel, ld.elev, ld.slope, ld.aspectf, ld.cover, &EquilMx, &Radiate, SIZECLASS_100HR);	// Radiate is not returned if set to 0

//  thousands = env->GetMx (SimTime, ld.woody, ld.elev, ld.slope,	ld.aspectf, ld.cover, &EquilMx, &Radiate,	SIZECLASS_1000HR);	// Radiate is not returned if set to 0
	  	thousands = this->pFarsite->cfmc.GetMx  (SimTime, ld.woody, ld.elev, ld.slope,  ld.aspectf, ld.cover, &EquilMx, &Radiate,	SIZECLASS_1000HR);	// Radiate is not returned if set to 0

		  tenthous = 0.30;
	 } /* else */

// test-larry
#ifdef xxxxxx
int _LogFile (char cr[]);
char cr[300];
    sprintf (cr, "%6.0f %3d,  %2d %4d %2d %3.0f %2d  --> %5.3f %5.3f %5.3f\n", SimTime,date,
            ld.woody, ld.elev, ld.slope, ld.aspectf, ld.cover,
            onehour, tenhour, hundhour);
   _LogFile (cr);
#endif

}

/***************************************************************************
*
****************************************************************************/
void FELocalSite::windadj(long date, double hours, long* cloud)
{

  //windadj_Old(date,hours, cloud);

  windadj_New(date,hours, cloud);

 //
// printf ( "\n");

}

/***************************************************************
* Name: windadj_New
* Desc: Fi
*       This function relies on the fact that there is as least
*       one wind record for each day.
* Note-1: This function will always be ask for the same or next
*          date, so we can alway start searching the wind table
*          from where we left off on the previous call.
*
****************************************************************/
void FELocalSite::windadj_New(long date, double hours, long* cloud)
{
int i, j, Last, iX, mth, day;
//short s;
int count;
//char cr[20];

  GetMthDay (date, &mth, &day);
  Last = (int)pFarsite->MaxWindObs[0] - 2;   /* -1 don't want last 13 Month Record */

  j = pFarsite->WindLoc_New - 1;    /* See Note-1 above */
  if ( j < 0 )                      /* don't let array in go below */
   j = 0;

/* Find the Month and Day */
  for ( i = j; i <= Last; i++ ) {
    if ( pFarsite->wddt[0][i].mo == mth && pFarsite->wddt[0][i].dy == day )
       break;  }

/* Find the hourly record within the day */
  j = i;
  for ( i = j; i <= Last; i++ ) {
    if ( pFarsite->wddt[0][i].dy != day || pFarsite->wddt[0][i].mo != mth )
      break;
    if ( hours < pFarsite->wddt[0][i].hr  )
      break;  }

  if ( i != Last )
    iX = i - 1;
  else
    iX = i;

  count = iX;

// test
//  printf ("New: %d %6.0f --> %3d  %2d %2d %4d  \n", date, hours, count, pFarsite->wddt[0][count].mo,pFarsite->wddt[0][count].dy,pFarsite->wddt[0][count].hr );
// test

  pFarsite->WindLoc_New = count;

  /*if ( pFarsite->wddt[StationNumber][count].a_FWN != NULL ) {

    s = pFarsite->GetWindGridDirByCoord (this->XLocation, this->YLocation, pFarsite->wddt[StationNumber][count].a_FWN);
    wwinddir = (double) s;

    s = pFarsite->GetWindGridSpeedByCoord (this->XLocation, this->YLocation, pFarsite->wddt[StationNumber][count].a_FWN);
    twindspd = (double) s;
   *cloud = pFarsite->wddt[StationNumber][count].cl;
    windreduct();			// GET MIDFLAME WINDSPEED
    return;  }*/


  twindspd = pFarsite->GetWindSpeed (StationNumber, count);
  wwinddir = pFarsite->GetWindDir   (StationNumber, count);
  *cloud   = pFarsite->GetWindCloud (StationNumber, count);

	//twindspd=(double) wss;
  windreduct();			// GET MIDFLAME WINDSPEED

  }

/******************************************************
* Name: GetMthDay
*
******************************************************/
int FELocalSite::GetMthDay (int date, int *ai_Mth, int *ai_Day)
{
int i;
/*         First julian day of each month */
int ir[] = { 1,32,60,91,121,152,182,213,244,274,305,335,366,999};

  if ( date > 365 ) {   /* Shouldn't happen, caller screwed up */
      *ai_Mth = 12;
      *ai_Day = 31;
      return 0; }

  for ( i = 0; i < 50; i++ ) {
    if ( date < ir[i] ) {
      *ai_Mth = i;
      break; }
  }
   i--;
  *ai_Day = (date - ir[i]) + 1;
  return 1;

}

/************************************************************************************************************/
void FELocalSite::windadj_Old(long date, double hours, long* cloud)
{
	// FINDS MOST CURRENT WINDSPD AND CALCULATES MIDFLAME WINDSPDS FROM 20-FT WS (MPH)

	int count, month, xmonth, xday, day, hhour = 0, xdate;
	double MinTimeStep;

	count = pFarsite-> WindLoc - 1;
	do
	{
		count++;
		month = pFarsite->GetWindMonth(StationNumber, count);
		day = pFarsite->GetWindDay(StationNumber, count);
		if (month < 13)
			xmonth = pFarsite->GetJulianDays(month);
		else
		{
			day = 0;
			xmonth = date;
		}
		xdate = xmonth + day;
	}
	while (xdate < date);

	if (month != 13)						// if hit end of windspeed data
	{
		xday = day;
		hhour = pFarsite->GetWindHour(StationNumber, count);
		while (hours >= (double) hhour)
		{
			count++;
			day = pFarsite->GetWindDay(StationNumber, count);
			xmonth = pFarsite->GetWindMonth(StationNumber, count);
			hhour = pFarsite->GetWindHour(StationNumber, count);
			if (day > xday || xmonth > month)
				break;
		}
	}

	if (hours <= hhour)	{
		 MinTimeStep = (double) hhour - hours;
	 	pFarsite->EventMinimumTimeStep(MinTimeStep);  	   // load into global variable
	}

	count--;

// test
//  printf ("Old: %d %6.0f --> %3d  %2d %2d %4d  \n", date, hours, count, pFarsite->wddt[0][count].mo,pFarsite->wddt[0][count].dy,pFarsite->wddt[0][count].hr );
// test


// WN-Test.........................
/* If WindNinja was run */
 /*if ( pFarsite->wddt[StationNumber][count].a_FWN != NULL ) {
    short s;
    s = pFarsite->GetWindGridDirByCoord (this->XLocation, this->YLocation, pFarsite->wddt[StationNumber][count].a_FWN);
    wwinddir = (double) s;

    s = pFarsite->GetWindGridSpeedByCoord (this->XLocation, this->YLocation, pFarsite->wddt[StationNumber][count].a_FWN);
    twindspd = (double) s;

   *cloud = pFarsite->wddt[StationNumber][count].cl;
   	pFarsite->WindLoc = count;
    windreduct();			// GET MIDFLAME WINDSPEED
    return;  }*/


    twindspd = pFarsite->GetWindSpeed(StationNumber, count);
	wwinddir = pFarsite->GetWindDir(StationNumber, count);
	*cloud = pFarsite->GetWindCloud(StationNumber, count);
	pFarsite->WindLoc = count;
	//twindspd=(double) wss;
	windreduct();			// GET MIDFLAME WINDSPEED
}



/*************************************************************************/
void FELocalSite::AtmWindAdjustments(long curdate, double hours, long* cloud)
{
	long count = -1, ddate, month, day, hhour, xmonth, xday;
	short twspd, wndir, cloud_s;
	double MinTimeStep;

	do
	{
		count++;
		month = pFarsite->GetAtmosphereGrid()->GetAtmMonth(count);
		if (month == -1)					// hit end of ATMDATA
			break;
		day = pFarsite->GetAtmosphereGrid()->GetAtmDay(count);
		ddate = day + pFarsite->GetJulianDays(month);
	}
	while (ddate != curdate);
	if (month != -1)						// if hit end of ATMDATA data
	{
		hhour = pFarsite->GetAtmosphereGrid()->GetAtmHour(count);
		xday = day;
		while (hours >= hhour)
		{
			count++;
			xmonth = pFarsite->GetAtmosphereGrid()->GetAtmMonth(count);
			xday = pFarsite->GetAtmosphereGrid()->GetAtmDay(count);
			hhour = pFarsite->GetAtmosphereGrid()->GetAtmHour(count);
			if (xday > day || xmonth > month)
				break;
		}
		count--;
		pFarsite->GetAtmosphereGrid()->GetAtmosphereValue(ATMWSPD, XLocation, YLocation,
								count, &twspd);
		pFarsite->GetAtmosphereGrid()->GetAtmosphereValue(ATMWDIR, XLocation, YLocation,
								count, &wndir);
		pFarsite->GetAtmosphereGrid()->GetAtmosphereValue(ATMCLOUD, XLocation,
								YLocation, count, &cloud_s);
		*cloud = cloud_s;
		twindspd = (double) twspd;
		wwinddir = (double) wndir;
		windreduct();
	}
	if (hours <= hhour)
	{
		MinTimeStep = (double) hhour - hours;
		pFarsite->EventMinimumTimeStep(MinTimeStep);  	   // load into global variable
	}
}


void FELocalSite::windreduct()
{
	// FUNCTION TO REDUCE WINDSPEED (MI/HR) TO MIDFLAME OR VEGETATION HEIGHT

	double ffactor, htfuel, htflame, m1, m2;
	double canopyht = ld.height*3.2808;//GetDefaultCrownHeight()*3.28;     // convert to feet

	if (ld.cover <= 5 || canopyht == 0)	// ld.cover==0
	{
		if (ld.fuel > 0)
		{
			switch (ld.fuel)
			{
			case 1:
				htfuel = 1.0; break;
			case 2:
				htfuel = 1.0; break;
			case 3:
				htfuel = 2.5; break;
			case 4:
				htfuel = 6.0; break;
			case 5:
				htfuel = 2.0; break;
			case 6:
				htfuel = 2.5; break;
			case 7:
				htfuel = 2.5; break;
			case 8:
				htfuel = 0.2; break;
			case 9:
				htfuel = 0.2; break;
			case 10:
				htfuel = 1.0; break;
			case 11:
				htfuel = 1.0; break;
			case 12:
				htfuel = 2.3; break;
			case 13:
				htfuel = 3.0; break;
			default:
				htfuel = pFarsite->GetFuelDepth(ld.fuel);  // retrieve from cust models
				if (htfuel == 0.0)
					htfuel = 0.01;
				break;
			}
		}
		else
			htfuel = 0.01;		// no fuel so height is essentially zero
		htflame = htfuel;		// from Baughman and Albini 6th conf. FFM 1980
		m1 = (1.0 + 0.36 * (htfuel / htflame)) /
			(log((20.0 + 0.36 * htfuel) / (0.13 * htfuel)));
		m2 = log(((htflame / htfuel + 0.36) / .13)) - 1.0;
		mwindspd = m1 * m2 * twindspd;
	}
	else
	{
		ffactor = ((double) ld.cover / 100.0) * 0.33333; // volume ratio of cone to cylinder
		ffactor *= PI / 4.0;						// area ratio of circle to square

		m1 = 0.555 /
			(sqrt(canopyht * ffactor) * log((20.0 + 0.36 * canopyht) /
											(0.13 * canopyht)));
		if (m1 > 1.0)
			m1 = 1.0;
		mwindspd = m1 * twindspd;
	}
}


/******************************************************************************
*
*
******************************************************************************/
FILE *fh = NULL;
int  iS_Log = 0;
int _LogFile (char cr[])
{
char  cr_FN[100];
   strcpy (cr_FN, "C:\\larryf\\StuTest\\LogFil.txt");

  if ( iS_Log == 0  )
    remove (cr_FN);

   iS_Log++;
  fh = fopen (cr_FN, "a+");
  fprintf (fh,"%6d: %s",iS_Log, cr);
  fclose (fh);
  return 0;
}


