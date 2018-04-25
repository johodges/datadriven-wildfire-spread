/*********************************************************************************
* Name: fmc_ci3.cpp - FlamMap Input Class
* Desc: This class holds and makes available all needed inputs.
*
*
*
**********************************************************************************/
#define  _CRT_SECURE_NO_WARNINGS   /* Get rid of sprintf Warnings */

#ifdef WIN32
#include <windows.h>
#include <conio.h>
#else
#include <stdio.h>
#include <string.h>
#endif

#include "cdtlib.h"
#include "newfms.h"

#include "semtime.h"
#include "deadfuelmoisture.h"

#include "FMC_CI.h"    /* FlamMap Input Class */
#include "FMC_FE2.h"    /* FireEnvironment2 Class */

//.....................................................
// Need to include icf_def.h for the error return #defines
//  and the error message table, BUT there is a d_RAWS struct
//  defined in it which conflicts with the one in fmc_ci.h
//  this will cause it to not use it.......
#define SKIP_RAWS_TYPEDEF
#include "icf_def.h"



/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
void Chk (long val, long low, long high,  int *aj);
int  ChkWeather (WeatherData *wtrdt, char cr_ErrMes[]);
int  CheckNextDayTime (int m, int d, int t, int mm, int dd, int tt );
int  CheckMonthDay (int m, int  d);
int  CheckNextDay (int m, int d, int mm, int dd );
int RAWS_Seq (int y, int m, int d, int t, int ny,int nm,int nd, int nt );


float _Min (long A, long B);
float _Max (long A, long B);

/*******************************************************************/
void CI::ResetThreads()
{
// I put this call in CFMC::ResetThreads becasue env-> isn't in CI anymore
//    burn->env->ResetAllThreads();

  FreeFarsiteEvents(EVENT_MOIST);
  FreeFarsiteEvents(EVENT_MOIST_THREAD);

	 char eventInstanceStr[64];

	 sprintf(eventInstanceStr, "FlamEventMoist_%ld", instanceID);
  AllocFarsiteEvents(EVENT_MOIST, GetMaxThreads(), eventInstanceStr, false, false);

	 sprintf(eventInstanceStr, "FlamEventMoistThread_%ld", instanceID);
  AllocFarsiteEvents(EVENT_MOIST_THREAD, GetMaxThreads(), eventInstanceStr, true, false);

	 for (int i = 0; i < 64; i++)
	 	 ThreadProgress[i] = 0.0;


// May have to implement this later
//	 conditioningProgress = 0.0;

}
/*********************************************************/
bool	CI::FreeFarsiteEvents(long EventNum)
{
bool ret = false;

#ifdef WIN32
   switch (EventNum) {
//     	case 1: ret=hBurnEvent.FreeEvents(); break;

     	case 2:
         ret = hMoistEvent.FreeEvents();
         break;

//     	case 3: ret=hBurnupEvent.FreeEvents(); break;
//     	case 4: ret=hIntegEvent.FreeEvents(); break;
//     	case 5: ret=hBurnThreadEvent.FreeEvents(); break;
      	case 6:
          ret=hMoistThreadEvent.FreeEvents();
          break;
//     	case 7: ret=hBurnupThreadEvent.FreeEvents(); break;
//     	case 8: ret=hIntegThreadEvent.FreeEvents(); break;
//     	case 9: ret=hCrossEvent.FreeEvents(); break;
//     	case 10: ret=hCrossThreadEvent.FreeEvents(); break;
     default:
        this->LogicError(" CI::FreeFarsiteEvents() ERROR");
        break;
     }
#endif
     return ret;
}

bool CI::AllocFarsiteEvents(long EventNum, long numevents, char *basename, bool ManReset, bool InitState)
{
	bool ret = false;
#ifdef WIN32
     switch(EventNum) {
//	     case 1: ret=hBurnEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
     	case 2:
         ret = hMoistEvent.AllocEvents(numevents, basename, ManReset, InitState);
         break;
//     	case 3: ret=hBurnupEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
//     	case 4: ret=hIntegEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
//     	case 5: ret=hBurnThreadEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
     	case 6:
         ret = hMoistThreadEvent.AllocEvents(numevents, basename, ManReset, InitState);
         break;
//     	case 7: ret=hBurnupThreadEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
//     	case 8: ret=hIntegThreadEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
//     	case 9: ret=hCrossEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
//     	case 10: ret=hCrossThreadEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
     default:
        this->LogicError(" CI::AllocFarsiteEvents() ERROR");
        break;
     }
#endif
     return ret;
}


/***********************************************************/
void CI::SetInstanceID(long id)
{
	instanceID = id;
}

long CI::GetStartProcessor ()
{
  return StartProcessor;
}

/*******************************************************************
* Name: CheckInputs
* Desc: Check and make sure all the inputs needed to run are within
*        limits and valid - see NOTE below
* NOTE: a function to check for unset or unload data should have
*       been done before coming here.
*  Ret: 1 ok
*       < 0  ICF error message code
*            use the ICF error message function to get the message text
********************************************************************/
int CI::CheckInputs ()
{
int i;//,j,n;

  strcpy (cr_ErrMes,"");

  if ( !this->InitCheck(cr_ErrMes) )  /* simple check to make sure */
    return e_EMS_cMiss;               /*  were at least loaded */

  if ( NumWeatherStations < 1 || NumWeatherStations > eC_Sta ) {
    sprintf (cr_ErrMes,"Bad # w stations %ld",NumWeatherStations );
    return e_EMS_cWtSt;  }

  if ( NumWindStations < 1 ) {
    sprintf (cr_ErrMes,"Bad # wind obs %ld",NumWindStations );
    return e_EMS_cWdOb;  }

 if ( //l_ElevHigh < 0 || l_ElevHigh > 20000 ||
       //l_ElevLow  < 0 || l_ElevLow  > 20000 ||
       l_ElevLow > l_ElevHigh ||
       s_EUnits < 0 || s_EUnits > 1 )  {
      strcpy (cr_ErrMes,"Bad Elev or Unit");
      return e_EMS_cElv;  }

  if ( (l_SlopeHigh < l_SlopeLow) ||
       s_SUnits < 0 || s_SUnits > 1 ) {
      strcpy (cr_ErrMes,"Bad Slope or Unit");
      return e_EMS_cSlp; }

  if ( l_CoverHigh < 0 || l_CoverLow < 0 ||
       l_CoverHigh > 100 || l_CoverLow > 100 ||
       (l_CoverLow > l_CoverHigh)) {
      strcpy (cr_ErrMes, "Bad Cover value");
      return e_EMS_cCov; }

/*...............................................................................*/
  i = 0;
  if ( !CheckMonthDay (startmonth,startday ) )i++;
  if ( starthour < 0 || startday > 2400 ) i++;
  if ( i > 0 ) {
    strcpy (cr_ErrMes, "Bad start date/time");
    return e_EMS_cSDt; }

/*...............................................................................*/
  if ( StartProcessor < 0 || MaxThreads < 0 ) {
     strcpy (cr_ErrMes, "Bad # processors");
     return e_EMS_cPro; }

/*------------------------------------------------------------------------------*/
/* Check weather/wind or weather stream data table */
 if ( a_RAWS[0] != 0 ) {                   /* if we have  Weather Stream Data */
    if ( !CheckWeatherStream())
      return e_EMS_cSqWt;
    return 1; }

/* If there is no weather stream data than we need weather & wind data */
 if ( wtrdt[0] == 0 ) {
    strcpy (this->cr_ErrMes,"No Weather Data");
    return e_EMS_cNoWt; }

 if ( this->wddt[0] == 0 ) {
    strcpy (this->cr_ErrMes,"No Wind Data");
    return e_EMS_cNoWn; }

 i = CheckWeatherWind ();
 if ( i < 0 )
   return i;

  return 1;
}

/***********************************************************************************/
int CI::CheckWeatherStream()
{
int i;
d_RAWS *a,*b;

  for ( i = 1; i < this->irN_RAWS[0]; i++ ) {
      a = &this->a_RAWS[0][i-1];
      b = &this->a_RAWS[0][i];

      if ( !RAWS_Seq (a->i_Yr,a->i_Mth,a->i_Day,a->i_Time,
                     b->i_Yr,b->i_Mth,b->i_Day,b->i_Time) ) {
        sprintf (this->cr_ErrMes,"Bad Wth Rec order %d %d %d",b->i_Mth,b->i_Day,b->i_Time);
        return 0; }

  }
  return 1;
}

/***********************************************************************
* Name: RAWS_Seq
* Desc: Check the sequence year, month, day, time (military)
*   In...'n' means next (following) date time
*  Ret: 1 OK, 0 Error
***********************************************************************/
int RAWS_Seq (int y, int m, int d, int t, int ny,int nm,int nd, int nt )
{
  if ( ny > y )
    return 1;
  if ( ny < y )
    return 0;

  if ( nm > m )
    return 1;
  if ( nm < m )
    return 0;

  if ( nd > d )
    return 1;
  if ( nd < d )
    return 0;

  if ( nt > t )
    return 1;

  return 0;
}

/**********************************************************************************
* Name: CheckWeatherWind
* Desc: validate data in weather & wind tables.
* Note-1 When I loaded the Wind Data I stuck an extra record on the end
*         and put 13 in it for a month, because GetCloud() looks for a 13
*         which I might rework at some point. But for now we don't want to
*         try and check that last 13 month record so.... MaxWindObs[0] - 1;
* NOTE: Make sure there is data loaded be coming here
*  Ret: 1 OK,
*       < 0  ICF Error code,
* Note: an error message is put into this->cr_ErrMess
*       BUT - error messages should be retreived the ICF
*       using the ICF error code
***********************************************************************************/
int CI::CheckWeatherWind()
{
int i,j,k, n;
   strcpy (this->cr_ErrMes,"");

/* basic error check, only do for station 0, setup to only report the  */
  j = 0;
  n = MaxWindObs[0] -1;   /* See Note-1 above */
  for ( i = 0; i < n ; i++ ) {
    Chk (wddt[0][i].mo, 1, 12, &j);
		  Chk (wddt[0][i].dy, 1, 31, &j);
		  Chk (wddt[0][i].hr, 0, 2359, &j);
		  Chk (wddt[0][i].cl, 0, 100, &j);
    if ( j > 0 ) {  /* if anything was found in error */
      sprintf (this->cr_ErrMes,"Invalid wind date, see line %d", i+1);
      return e_EMS_cInWn; } }

/*......................................................................*/
/* Check weather stream data */
  for ( i = 0; i < this->MaxWeatherObs[0]; i++ ) {
    k = ChkWeather (&wtrdt[0][i],cr_ErrMes);
     if ( k < 0 )
       return k; }

/* Check that all weather month-days are consecutive */
   for ( i = 0; i < (this-> MaxWeatherObs[0] - 1); i++ ) {
     if ( !CheckNextDay (wtrdt[0][i].mo,    wtrdt[0][i].dy,
                         wtrdt[0][i+1].mo,  wtrdt[0][i+1].dy )){
        sprintf (this->cr_ErrMes,"Bad wthr date M%ld D%ld ", wtrdt[0][i+1].mo, wtrdt[0][i+1].dy);
       return e_EMS_cInWt;}}

  return 1;
}


/*********************************************************************/
void Chk (long val, long low, long high, int *aj)
{
  if ( val < low || val > high )
    *aj = 1;
}

/**********************************************************
* Name: GetErrMes
* Desc: get what's ever in the CI error message string
*        will be a Null if none is there.
*  Ret: 1 if NO error message found
*       0 Error message present and returned
**********************************************************/
int  CI::GetErrMes(char cr_EMess[])
{
  if ( !strcmp ( this->cr_ErrMes,"") )
    return 1;
  strcpy (cr_EMess, this->cr_ErrMes);
  return 0;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkWeather   Check Weather inputs
* Desc: Check the fields in the array of entries
*       switch = WEATHER_STREAM or  WEATHER_DATA:
*       Both switches are optional but you can't have both
*
*   i_Mth   = month,
*   i_Day   = day,
*   f_Per   = precip in 100th of inch (ex 10 = 0.1 inches
*   i_mTH   = min_temp_hour 0-2359,
*   i_xTH   = max_temp_hour 0 - 2359,
*   f_mT    = min_temp,
*   f_xT    = max_temp,
*   i_xH    = max_humidity,
*   i_mH    = mix_humidity,
*   i_Elv   = elevation,
*   i_PST   = precip_start_time 0-2359, if f_Per > 0
*   i_PET   = precip_end_time 0-2359    if f_Per > 0
*
*  Out: ai...error number
*
*  Ret: 1 ok, < 0  = error ICF error code ,
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ChkWeather (WeatherData *w, char cr_ErrMes[])
{
#define e_HrMax 2359

   strcpy (cr_ErrMes,"");
   if ( w->mo < 1 || w->mo > 12 ) {
     sprintf (cr_ErrMes,"Bad wtr month %ld",w->mo);
     return e_EMS_cbMt;  }
   if ( w->dy < 1 || w->dy > 31 ) {
     sprintf (cr_ErrMes,"Bad wth day %ld",w->dy);
     return e_EMS_cbDy;  }

   if ( w->rn < 0 || w->rn > 10000 ) {
       sprintf (cr_ErrMes,"Bad Precip amt %f",w->rn);
       return e_EMS_cbPr; }

   if ( w->t1 < 0 || w->t1 > e_HrMax ) {
     sprintf (cr_ErrMes,"Bad Min temp time %ld", w->t1);
     return e_EMS_cbMtt; }
   if ( w->t2 < 0 || w->t2 > e_HrMax ) {
     sprintf (cr_ErrMes,"Bad Max temp time %ld", w->t2);
     return e_EMS_cbXt; }

   if ( w->H1 > 100 || w->H1 < 1 ){
     sprintf (cr_ErrMes,"Bad Max Humidity %ld", w->H1);
     return e_EMS_cbXh; }

   if ( w->H2 < 1 || w->H2 > 100 ) {
     sprintf (cr_ErrMes,"Bad Min Humidity %ld", w->H2);
     return e_EMS_cbMh; }

/* Take this out, this may not always be the case */
//   if ( w->H2 > w->H1 ) {
//     sprintf (cr_ErrMes,"Humidity Min %d is > Max %d",w->H2, w->H1);
//    return 0; }

   if ( w->el < 0 || w->el > 14000 ) {
     sprintf (cr_ErrMes,"Bad Elev %4.0f", w->el);
     return e_EMS_cbEl; }



     if ( w->H2 < 0 || w->H2 > e_HrMax ) {
      sprintf (cr_ErrMes,"Bad Precip start time %ld", w->H2);
      return e_EMS_cbPs; }

     if ( w->H1 < 0 || w->H1 > e_HrMax ) {
      sprintf (cr_ErrMes,"Bad Precip end time %ld", w->H1);
      return  e_EMS_cbPe; }


	   if ( w->rn > 0 ) {
	      if ( w->tr2 == 0 ){
	         sprintf (cr_ErrMes, "Precip %6.1f no str/end time", w->rn);
	       	 return e_EMS_cbPrr; }

	      if ( w->tr1 >= w->tr2 ) {
		       sprintf (cr_ErrMes, "Bad Precip str/end time %ld/%ld", w->tr1, w->tr2);
	        return e_EMS_cbPB; } }


    if ( w->rn == 0 ) {
	     	if ( w->tr2 > 0 || w->tr1 > 0 ){
	      	 sprintf (cr_ErrMes, "0 Precip bad str/end time %ld/%ld", w->t2, w->t1);
	        return e_EMS_cbP; } }
  return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckNextDaytIME
* Desc: See if days are consecutive, one day apart
*       If days are the same then check time of day
* NOTE: THIS doesn't do a Feb 29, we're not worry about that though
*       This will do end of year wrap around though
*   In: month day combinations, mm dd is the day after combination
*  Ret: 1 ok,
*       0 days are not consectutive or bad month or day
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  CheckNextDayTime (int m, int d, int t, int mm, int dd, int tt )
{
int i,j,k;

   if ( m == 12 && d == 31 && mm == 1 && dd == 1 )
     return 1;                      /* Special Case - Year end wrap around */

   i = CheckMonthDay (mm, dd);      /* Get julian dates  */
   j = CheckMonthDay (m, d);
   if ( i == 0 || j == 0 )          /* First check for valid month day  */
     return 0;
   k = i - j;                       /* number of days apart */
   if ( k > 0)//== 1 )                    /* If they are consectutive days   */
     return 1;                      /*  OK */
   //if ( k > 1 || k < 0 )            /* if days are more than one day apart */
   //  return 0;                      /*    error  */

   if ( t >= tt )                   /* Days are same, is time later */
     return 0;                      /*  time is not in order - error */

   return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckNextDay
* Desc: See if days are consecutive, one day apart
* NOTE: THIS doesn't do a Feb 29, we're not worry about that though
*       This will do end of year wrap around though
*   In: month day combinations, mm dd is the day after combination
*  Ret: 1 ok,
*       0 days are not consectutive or bad month or day
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  CheckNextDay (int m, int d, int mm, int dd )
{
int i;

   if ( m == 12 && d == 31 && mm == 1 && dd == 1 )
     return 1;                      /* Special Case - Year end wrap around */

   i = CheckMonthDay (mm, dd) - CheckMonthDay (m, d);
   if ( i != 1 )
     return 0;
   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckMthDay
* Desc: Return the day number - julian date
* NOTE: THIS doesn't do a Feb 29
*   In: month 1->12, day 1->N
*  Ret: julian date or 0 for invalid date
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
/*int CheckMonthDay (int m, int  d)
{
int rd[] = {0, 31, 28, 31, 30, 31,30, 31, 31, 30, 31, 30, 31};
int i,n;

    if ( m < 1 || m > 12 )              // Chk for invalid month
      return 0;
    if ( d < 1 )                        // Chk day
      return 0;

    n = 0;
    for ( i = 0; i < m; i++ )           //total up days in each month
      n += rd[i];                       // thru the desired mth

    if ( d > rd[i] )                    // Chk for valid day
      return 0;

    n = n + d;                          //Add day to total
    return n;
}
*/
/****************************************************************************
* Name: Set_MoistCalcHourIntermal
* Desc: Set the fuel size hour that will be used to determine what
*        the Conditioning simulation will use for a time interval
* NOTE: FlamMap originally just used the SIZECLASS_10HR, SO in here
*       and in the CI::Init() the default is 10 hr.
*   In: Send in a 0 to set the default
*       1,10,100,1000 Fuel Size Class,
*       Any other number will cause the Default to get set.
****************************************************************************/
void CI::Set_MoistCalcHourInterval (int i)
{
  if ( i == 1 )
    this->i_MCHI = SIZECLASS_1HR;
  else if ( i == 10 )
    this->i_MCHI = SIZECLASS_10HR;
  else if ( i == 100 )
    this->i_MCHI = SIZECLASS_100HR;
  else if ( i == 1000 )
    this->i_MCHI = SIZECLASS_1000HR;
  else                                 /* Use the default */
    this->i_MCHI = SIZECLASS_10HR;
}
/*-----------------------------------------------------------------------*/
/* See the above function */
long CI::Get_MoistCalcHourInterval ()
{
  return (long) this->i_MCHI;
}

/**********************************************************************************
* Name: RAWS_Allocate
* Desc: Allocate memory for the Weather Stream Data
*   In: iN......number records to allocate in table
*        0 Error Can't Alloc mem
********************************************************************************/
int CI::RAWS_Allocate (int iN)
{
int i,i_Station;
   i_Station = 0;   /* For now all were doing is station 1 */
   if ( this->a_RAWS[i_Station] )
     delete[] this->a_RAWS[i_Station];

   this->a_RAWS[i_Station] = new d_RAWS[iN + 1]; /* NOTE do 1 extra to mark end of table */
   if (this->a_RAWS[i_Station] == NULL )
     return 0;

   this->irN_RAWS[i_Station] = iN;

   for ( i = 0; i <= iN; i++ )
     this->a_RAWS[i_Station][i].i_Yr = e_RAWSend;  /* might use this to detect end of table */

   return 1;
}

/*****************************************************************************
* Name: RAWS_LoadDate
* Dese: Load RASW (Remote Automated Weather Station) Date/Time data.
*       SEE CI::RAWS_LoadInfo()
* NOTE----------> Don't load any FEB 29 leap year days <-------- NOTE
* NOTE: Limit is 365 days of data,
* NOTE: Station 0 is always used.
*   In: iX....index
*       i_Yr..year.....No more than 2 years
*       i_Mth..Month,
*       i_Time....Military Time 0000->2359
*****************************************************************************/
int CI::RAWS_LoadDate (int iX, int i_Yr, int i_Mth, int i_Day, int i_Time)
{
  if ( iX < 0 || iX >= this->irN_RAWS[0] )
     return 0;

  this->a_RAWS[0][iX].i_Yr = i_Yr;             /* Year, not sure if/how we'll use this */
  this->a_RAWS[0][iX].i_Mth = i_Mth;
  this->a_RAWS[0][iX].i_Day = i_Day;
  this->a_RAWS[0][iX].i_Time = i_Time;         /* Military 0000->2359 */
  return 1;
}

/******************************************************************************
* Name: RASW_LoadInfo()
* Desc: Load RAWS  weather data
*       See CI::LoadDate()
*   In: iX index into table
*******************************************************************************/
int CI::RAWS_LoadInfo ( int iX, float f_Temp, float f_Humidity, float f_PerHou,
                        float f_CloCov)
{

  if ( iX < 0 || iX >= this->irN_RAWS[0] )
     return 0;

  this->a_RAWS[0][iX].f_Temp = f_Temp;         /* Temperature */
  this->a_RAWS[0][iX].f_Humidity = f_Humidity;
  this->a_RAWS[0][iX].f_PerHou = f_PerHou;        /* Perceipitation Hourly */
  this->a_RAWS[0][iX].f_CloCov = f_CloCov;       /* Cloud Cover percent 0 -> 100  */

   return 1;
}

/*****************************************************************
* Name:
* Desc:
*****************************************************************/
int CI::RAWS_Chrono (double d_SimTim, long *al_MilTim, double *ad_MilTim)
{
int date;
long l_MinTime;
d_RAWS  s_RAWS;

/* Get the Minute Time */
  l_MinTime = this->l_StartMinDate + (int) d_SimTim;
  this->RAWS_GetRecord(l_MinTime, &s_RAWS);

  date = GetJulian (s_RAWS.i_Mth,s_RAWS.i_Day);
  *al_MilTim = (long) s_RAWS.i_Time;
  *ad_MilTim = (double) s_RAWS.i_Time;
  return date;

}

/****************************************************************
* Name: RAWS_CalcMinuteDate
* Desc: Get the total minutes for the given date - time
*       This is based on starting date of Jan 1 or e_SimYearOne
*        and going thru Dec 31 e_SimYearTwo
*       Starting at:
*       Jan 1 year 1 at 12 AM = 0, same day at 8 am = 4800 minutes
*   In: Yr .... e_SimYearOne, e_SimYearTwo - Remember a max sim
*               time is 356, across a max of two years
*       Mth, Day....1->12, days per month
*       MilTim......0->2359
****************************************************************/
long  CI::RAWS_CalcMinuteDate (int Yr, int Mth, int Day, int MilTim)
{
long min,JulDays;
   min = MilTim_To_Min (MilTim);
   JulDays = (long) GetJulian (Mth,Day);
   JulDays = JulDays - 1;           /* See Note-1 above */
   min = min + JulDays * ( 24 * 60 );

   if ( Yr == e_SimYearTwo ) {
      min = min + ( 365 * 24 * 60 ); }

   return min;

}

/******************************************************************
* Name: MilTim_To_Min
* Desc: Convert a military time to total minutes
*       Ex:  1030 =  (10 * 60) + 30 = 630 minute
*  Ret: total minutes
********************************************************************/
int CI::MilTim_To_Min (int i_MilTim)
{
int  i,h,m;
  h = i_MilTim / 100;
  m = i_MilTim % 100;
  i = ( h * 60 ) + m;
  return i;
}


/*****************************************************************
* Name: RAWS_GetCloud
* Desc:
*****************************************************************/
int CI::RAWS_GetCloud (int i_Date, int i_Time)
{
//int i, iX, i_CloCov;
//d_RAWS *a;
#ifdef wowo
  i_CloCov = -1;

  for ( i = 0; i < irN_WDS[0]; i++ ) {
       a = &a_RAWS[0][i];
       i_JulFnd = CI::GetJulian(a->i_Mth, a->i_Day);
       if ( i_JulFnd != i_Date )
         continue;

       for ( j = i; j < irN_RAWS[0]; j++ )
         a = &a_RAWS[0][j];
         f_CloCov = a->f_CloCov;
         i_Jul = CI::GetJulian(a->i_Mth, a->i_Day);
         if ( i_Jul != i_JulFnd )
            return f_CloCov;
         if ( i_Time > a->i_Time )
            return f_CloCov;


         if ( i_Time >

       if ( i_JulDat > i_Date )

         break;
       if ( a->i_Time < i_Time )
         continue;
       if ( a->i_Time > i_Time )
         break;          }

#endif
   return 1;
}

/**************************************************************************
* Name: RAWS_SetMinuteDate
* Desc: Set a Minute Date into the RAWS Table.
*       Minute Date gets used to located records during the simulation
*       Minute Date is the total number of minutes based on a starting
*        time of Jan 1 first year and going to Dec 31 of second year
*       No more than two separte years are allowed in tbl and now more
*        that 365 days.
* Note-1 RAWS.i_SimYear doesn't even need to be in the RAWS tbl, but
*         it might help during development or debug
**************************************************************************/
void CI::RAWS_SetMinuteDate()
{
int i_StrYr, iX;
d_RAWS *a;

   if ( this->a_RAWS[0] == NULL )   /* only do if we have RAWS input data  */
     return;

   i_StrYr =  this->a_RAWS[0][0].i_Yr;   /* start year, first entry in tbl */
   iX = 0;
   while (1) {                           /* Go thru RAWS tbl */
     a = &this->a_RAWS[0][iX++];
     if ( a->i_Yr == e_RAWSend )         /* end of tbl */
      break;
     if ( a->i_Yr == i_StrYr )           /* if equal to starting year */
        a->i_SimYear = e_SimYearOne;     /*  mark as Year one */
     else
        a->i_SimYear = e_SimYearTwo;     /* See Note-1 above */

     a->l_MinDate = this->RAWS_CalcMinuteDate(a->i_SimYear,a->i_Mth,a->i_Day,a->i_Time);
  }  /* while */

  if ( this->i_StartYear == i_StrYr )
    this->i_StartSimYear = e_SimYearOne;
  else
    this->i_StartSimYear = e_SimYearTwo;

/* Start Simulation Time, expressed as a Minute Date, see declaration notes */
  this->l_StartMinDate = this->RAWS_CalcMinuteDate(this->i_StartSimYear,
                                                 this->startmonth,
                                                 this->startday,
                                                 this->starthour);

 // RAWS_Display ();

}

/*****************************************************************
* Name: RAWS_GetRecord
* Desc: Get the record for the specified Minute Date.
*   In: l_MinDate....Minute Date, see notes e_RAWS definition
*  Out: a_RAWS........copy of record
*  Ret: index of found record
*****************************************************************/
int CI::RAWS_GetRecord (long l_MinDate, d_RAWS *a_RAWS)
{
int i,iX;
d_RAWS *a;

  iX = 0;

  while (1) {
    i = this->RAWS_isEnd(iX);   /* Check for end of tbl */
    if ( i != 0 ) {            /* if at end */
      iX = i;                  /* point to last valid rec in tble */
      break; }

    a = &this->a_RAWS[0][iX];

    if ( l_MinDate == a->l_MinDate )
       break;
    if ( l_MinDate < a->l_MinDate ) {
       iX--;
       break; }
    iX++;
  }

  if ( iX < 0 )               /* This shouldn't happen */
    iX = 0;

  memcpy (a_RAWS,&this->a_RAWS[0][iX],sizeof(d_RAWS));
  return iX;
}

/***********************************************************************
* Name: RAWS_isEnd
* Desc: See if the table index is at or past the end of the table
*  Ret: if at or past end of table than return the index of the
*        last valid record in table
*       0.... Not at end of table
***********************************************************************/
int CI::RAWS_isEnd (int iX)
{
  if ( iX >= this->irN_RAWS[0] )     /* is iX > the # of recs in tbl */
    return this->irN_RAWS[0] - 1;    /* if yes return idx to lst tbl rec */

  if ( this->a_RAWS[0][iX].i_Yr == e_RAWSend ) /* if rec is marked and end tbl */
      return this->irN_RAWS[0] - 1;  /* idx of lst rec in tbl */

  return 0;
}

/************************************************************************
* Name: RAWS_GetData
* Desc:  Get weather data from RAWS table. A time interval is use to
*        determine how many hourly records need to be used.
*       Temp, Humd, Cloud Cover are average when more than one record is
*       is used for time interval.
* Note: There isn't necessarily always a RAWS record for each hour in the
*       table. And we deal with that below in the code.
* Note; There should have been enough extra RAWS records loaded on end of
*       table past the end condition date to not worry about end of
*       table issues, or calculations made on the last record.
*   In: l_StrMinDate....Minute Date to start gathering data
*       l_Interval..number of minutes average across and collect rain
*  Out: a_RAWS......this will contain the date/time info from the staring
*                   record used
*                   Average Temp,Humid,CloudCover are returned in here
*                   Rain(precip) is accumulated and put in here
************************************************************************/
void CI::RAWS_GetData (long l_StrMinDate, long l_Interval, d_RAWS *a_RAWS )
{
int iX;
long l;
float f_TotMin, f_MinInRec;
float f_TotMinAvg = 0;

  f_TotMin = l_Interval;        /* Total Minutes to accumulate rain for */

/* Get index of RAWS rec that has the starting Minute Date */
  iX = RAWS_GetRecord (l_StrMinDate, a_RAWS);

  f_MinInRec = RAWS_MinInRec (iX);     /* # of minutes this rec represents */

  if ( l_StrMinDate > this->a_RAWS[0][iX].l_MinDate ){ /* if str time is in middl of rec */
    l = l_StrMinDate - this->a_RAWS[0][iX].l_MinDate;
    f_MinInRec = f_MinInRec - l; }

  if ( f_MinInRec > f_TotMin )           /* if accum time is < than in rec */
     f_MinInRec = f_TotMin;

  f_TotMin -= f_MinInRec;               /* minutes left to accum info */

  a_RAWS->f_Humidity = 0;
  a_RAWS->f_Temp = 0;
  a_RAWS->f_CloCov = 0;
  f_TotMinAvg = 0;
  RAWS_Sum (a_RAWS, &this->a_RAWS[0][iX], f_MinInRec, &f_TotMinAvg);
  iX++;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-*/
/* Go thru any prceeding records foward in time and accum info */
  while (1) {
    if ( f_TotMin <= 0 )
       break;

    if ( this->a_RAWS[0][iX+1].i_Yr == e_RAWSend )        /* End of Table */
      break;

    f_MinInRec = RAWS_MinInRec (iX);            /* minutes in rec */
    if ( f_TotMin < f_MinInRec )               /* don't use all minutes */
       f_MinInRec = f_TotMin;
    RAWS_Sum (a_RAWS, &this->a_RAWS[0][iX], f_MinInRec, &f_TotMinAvg);
    f_TotMin -= f_MinInRec;                    /* reduce minutes to accum */
    iX++;
 }

  a_RAWS->f_Temp     = a_RAWS->f_Temp / f_TotMinAvg;
  a_RAWS->f_Humidity = a_RAWS->f_Humidity / f_TotMinAvg;
  a_RAWS->f_CloCov   = a_RAWS->f_CloCov / f_TotMinAvg;
}


/**********************************************************************
* Name: RAWS_GetAccumRan
* Desc: Sum up all the rain between the two specified dates.
*       This function will handle Start & End Minutes dates that
*        are not on hourly boundries should they occur.
*       This function will handle RAWS records the represent any
*        amount of time, 1 hour, 2 hours, 15,30,37 minutes whatever...
*       If MISSING Records occur in the table than the amount of rain in
*        the first record found is consider to have fallen for the whole
*        duration of the missing records.
*        So if a record with a Minute date of 120 has 1 inch of rain
*        and the following record's Minute date is 240, than the 1 inch
*        is consider to be for 120 minutes. So if the caller is using
*        a Fuel Size 10 Hr 60 minute interval, they get 0.5 inches
*        of rain returned.
*   In: l_StrMinDate...start accumating rain on this Minute Date
*       l_EndMinDate...accumulate up to but not including this Minute Date
*  Ret: total rain
**********************************************************************/
 double CI::RAWS_GetAccumRain (long l_StrMinDate, long l_EndMinDate )
{
int iX;
float A,B,Dif, MIR,pc;
double d_Tot;
d_RAWS  s_RAWS;

  d_Tot = 0;
  iX = RAWS_GetRecord (l_StrMinDate, &s_RAWS); /* index to rec with start minute date */

/* Deal with the 1st rec - See Notes above */
  A = _Max ( a_RAWS[0][iX].l_MinDate, l_StrMinDate);
  B = _Min ( a_RAWS[0][iX+1].l_MinDate, l_EndMinDate );
  Dif = (float) B - A;   /* Minutes in record */
  MIR = (float) RAWS_MinInRec(iX);
  pc = ( Dif / MIR ) ;
  d_Tot = pc * a_RAWS[0][iX].f_PerHou;
  iX++;

  while (1) {
    if ( l_EndMinDate <= a_RAWS[0][iX].l_MinDate )
       break;
    if ( l_EndMinDate >= a_RAWS[0][iX+1].l_MinDate )
       pc = 1.0;
    else {
       Dif = l_EndMinDate - a_RAWS[0][iX].l_MinDate;
       MIR = (float) RAWS_MinInRec(iX);
       pc = Dif / MIR; }

    d_Tot += pc * a_RAWS[0][iX].f_PerHou;
    iX++;
  }

  return d_Tot;
}


/*****************************************************************************************/
float _Min (long A, long B)
{
  if ( A < B )
    return (float) A;
  return (float) B;
}
float _Max (long A, long B)
{
  if ( A > B )
    return (float) A;
  return (float) B;
}




/******************************************************************************/
void CI::RAWS_Sum (d_RAWS *a_To, d_RAWS *a_Frm, float f_Min, float *af_TotMinAvg)
{
   a_To->f_Temp     += a_Frm->f_Temp * f_Min;          /* Temperature */
   a_To->f_Humidity += a_Frm->f_Humidity * f_Min;
   a_To->f_CloCov   += a_Frm->f_CloCov * f_Min;       /* Cloud Cover percent 0 -> 100  */
   *af_TotMinAvg += f_Min;
}

/*********************************************************************
* Name: RAWS_MinInRec
* Desc: Get the number of minutes that a record represents.
*       This based on what time the following record has.
* Note-1: The last record should get called because we checked at
*         the begin of the program to make sure the Cond end time
*         has a days weather in after it.
*   In:
*  Ret:
**********************************************************************/
long CI::RAWS_MinInRec (int iX)
{
long l;

  if ( this->a_RAWS[0][iX].i_Yr == e_RAWSend )   /* Should be getting called */
    return 0;                          /* Just in case */

  if ( this->a_RAWS[0][iX+1].i_Yr == e_RAWSend )  /* See Note-1 */
    return 60;                         /* if happens assume 60 minutes */

  l = this->a_RAWS[0][iX+1].l_MinDate - this->a_RAWS[0][iX].l_MinDate;

  return l;

}


/*****************************************************************
* Name: RAWS_Display
* Desc: Display - used for TESTING
*****************************************************************/
void CI::RAWS_Display ()
{
int iX;
d_RAWS *a;

  printf ("        MinTime  Yr  Mth Day Time Temp  Humid PerHou WinSpd WinDir CloCov\n");

  iX = 0;
  while (1) {
    a = &this->a_RAWS[0][iX++];

    if ( a->i_Yr == e_RAWSend )
      break;

    printf ("%3d, %6ld %4d(%1d) %2d %2d %4d %6.2f %6.2f %6.2f %6.2f\n",
            iX,
            a->l_MinDate,
            a->i_Yr,a->i_SimYear,a->i_Mth,a->i_Day,a->i_Time,
            a->f_Temp,a->f_Humidity,a->f_PerHou,
             a->f_CloCov);
  }
}

