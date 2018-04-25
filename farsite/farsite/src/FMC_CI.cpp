/*********************************************************************************
* Name: FMC_CI.cpp - Fuel Moisture Conditioning Input Class
* Desc: This class holds and makes available all the inputs needed to run
*        the dead fuel mositure conditioning.
* NOTE: This class substitutes what the FlamMap class use to do
*       when the conditioning code was part of the FlamMap project/program.
* Date: 10-20-09
**********************************************************************************/

#define _CRT_SECURE_NO_WARNINGS   /* Stop compiler 'depriciated' Warnings */

#ifdef WIN32
#include <windows.h>
#include <conio.h>
#endif

#include <string.h>

#include "cdtlib.h"
#include "newfms.h"

#include "semtime.h"
#include "deadfuelmoisture.h"

#include "FMC_CI.h"    /* Fuel Moisture Conditioning Input Class */
#include "FMC_FE2.h"    /* FireEnvironment2 Class */
#include <stdio.h>

//.....................................................
// Need to include icf_def.h for the error return #defines
//  and the error message table, BUT there is a d_RAWS struct
//  defined in it which conflicts with the one in fmc_ci.h
//  this will cause it to not use it.......
#define SKIP_RAWS_TYPEDEF
#include "icf_def.h"



void Get_MthDay (long JulDay, long *Mth, long *Day);

/*********************************************************************
* Name: Set_FuelModel
* Desc: Set the Fuel Model categories
*   In: array of values and count
*********************************************************************/
int CI::Set_FuelModel(int32 lr_Data[], int32 lN, char cr_ErrMes[])
{
int i;
   i = CI::SetFuelCats (lr_Data, lN,  e_FuelModel, cr_ErrMes);
   return i;
}

/*********************************************************************
* Name: Set_Woody
* Desc: Set the woody categories
* NOTE: The Woody categories don't appear to get used in the
*         Conditioning even though there is code there for them.
*********************************************************************/
int  CI::Set_Woody (int32 lr_Data[], int32 lN, char cr_ErrMes[])
{
int i;
   i = CI::SetFuelCats (lr_Data, lN, e_Woody, cr_ErrMes);
   return i;
}

/*********************************************************************
* Name: SetFuelCats
* Desc: Set Fuel Category values, Fuel Moisture Models.
*  Ret: 1 ok,
*       < 0 is an ICF error code - which is used to get the error
*         message from the ICF error message table
*  NOTE NOTE -> and error message is also return by this function,
*               this is how error were originally setup - so I left
*               it in place, and the caller might want to use it ??
*********************************************************************/
int CI::SetFuelCats (int32 lr_Data[],int32 lN, int i_Type, char cr_ErrMess[])
{
long i,n;
long *lr;

   strcpy (cr_ErrMess,"");
   strcpy (this->cr_ErrMes,"");

   if ( lN < 0 || lN > (long) eC_Hdr ) {  /* Watch out - this is a fixed size */
     sprintf (cr_ErrMess, "to many Fuel Moist Models: %ld",lN);
     strcpy (this->cr_ErrMes, cr_ErrMess);  /* also save in Class, in case */
    return e_EMS_cFMM; }

   if ( i_Type == e_Woody ) {                /* Woody */
      n = this->l_NumWoodie = lN;
      lr = this->lr_Woodie; }
   else if ( i_Type == e_FuelModel ) {           /* Fuel */
      n = this->l_NumFuel = lN;
      lr = this->lr_Fuel; }
   else {
     sprintf (cr_ErrMess, "Bad Fuel Moist Model: %d", i_Type);
     strcpy  (this->cr_ErrMes, cr_ErrMess);   /* also save in class, in case */
     return e_EMS_cBFM; }

   for ( i = 0; i <= n; i++ ) {
     lr[i] = lr_Data[i]; }

  return 1;
}

/**********************************************************************
* Name: GetFuelCats
* Desc: Get the array of Fuels data load by DLL caller,
* Note: for example if the FlamMap program is using this DLL then the
*       data being returned is the data that FlamMap got from it's .lcp file
*       and loaded into this DLL as input.
*  In: i_Type..see #defines e_FuelModel, e_Woody
* Out: lr_Data...requested data
* Ret: number of values in array
**********************************************************************/
int CI::GetFuelCats (int i_Type, long lr_Data[])
{
long i, n;
long *lr;
   if ( i_Type == e_Woody ) {           /* Woody  */
      n = this->l_NumWoodie;
      lr = this->lr_Woodie; }
   if ( i_Type == e_FuelModel ) {           /* Fuel */
      n = this->l_NumFuel;
      lr = this->lr_Fuel; }
   else {
     LogicError ((char *)"GetFuelCats(), invlaid request");
     return 0; }
   for ( i = 0; i <= n; i++ )
     lr_Data[i] = lr[i];
   return n;
}

/*********************************************************************************
* Name: ConvertActualTimeToSimTine
* Desc: Get the difference in minutes between the date sent in and the
*       conditioning starting time.
* Note-1: If the function's inputed 'mo' is a previous month to the starting date
*          it's assumed to be in the following year.
*
* NOTE: This function was revamped by removing code that was never used
*        because the condtions were never meet.
*
*********************************************************************************/
double CI::ConvertActualTimeToSimtime(long mo, long dy, long hr, long mn)
{
double SimTime, RefStart;

/* minutes - Year to date of conditioning starting date */
  RefStart = ( GetJulianDays (this->startmonth) + this->startday) * 1440.0;

  if ( mo >= this->startmonth )
   	SimTime = (GetJulianDays(mo) + dy) * 1440.0 + hr / 100.0 * 60.0 + mn;
  else
    SimTime = (GetJulianDays(mo) + dy + 365.0) * 1440.0 + hr / 100.0 * 60.0 + mn;

  SimTime -= RefStart;
  return SimTime;
}

/***********************************************************************/
int CI::GetJulian(int i_Mth, int i_Days)
{
int i;
   i = (int) CI::GetJulianDays((long)i_Mth);
   i = i + i_Days;
   return i;
}

/**************************************************************************/
long CI::GetJulianDays (long Month)
{
long days = 0;
	 switch (Month) {
	   case 1: days=0; break;		    	// cumulative days to begin of month
		  case 2: days=31; break;           // except ignores leapyear, but who cares anyway,
		  case 3: days=59; break;
		  case 4: days=90; break;
		  case 5: days=120; break;
		  case 6: days=151; break;
		  case 7: days=181; break;
		  case 8: days=212; break;
		  case 9: days=243; break;
	  	case 10: days=273; break;
		  case 11: days=304; break;
		  case 12: days=334; break;}
  return days;
}

/*****************************************************************************************/
bool CI::EnvironmentChanged(long YesNo, long StationNumber, long FuelSize)
{
	 if ( FuelSize > 3 )
    	FuelSize = 3;
	 if ( YesNo > -1 )
     EnvtChanged[FuelSize][StationNumber] = (bool) YesNo;

  return EnvtChanged[FuelSize][StationNumber];
}

/************************************************************************/
void CI::SetLatitude (long lat)
{
  l_Latitude = lat;
}
long CI::GetLatitude()
{
  return l_Latitude;
}

/*************************************************************************/
void CI::SetNumStations (long WeaSta, long WidSta)
{
 NumWeatherStations = WeaSta;
 NumWindStations = WidSta;
}

/************************************************************/
long CI::GetNumStations()
{
	  if ( NumWeatherStations > NumWindStations)
	   	return NumWindStations;
	  return NumWeatherStations;
}

/*************************************************************
* Name: Delete
* Desc: NEED to do this when when done with object to clean
*        up
* NOTE: There is no destructor for the class SO THIS needs
*         must be called
**************************************************************/
void CI::Delete()
{
int i;

/* inputed wind data, if allocated than free it up  */
  for ( i = 0; i < eC_Sta; i++) {   /* each Station */
    if ( wddt[i] )	{                /* Wind Data */
      delete[] wddt[i];
      wddt[i] = 0;
      MaxWindObs[i] = 0; }
    if ( wtrdt[i] ) {               /* Weather Data */
      delete[] wtrdt[i];
      wtrdt[i] = 0;
      MaxWeatherObs[i] = 0; }
    if ( a_RAWS[i] ) {               /* Weather Stream Data */
      delete[] a_RAWS[i];
      a_RAWS[i] = 0;
      irN_RAWS[i] = 0; } }

  NumWindStations = 0;
  NumWeatherStations = 0;
}

/*********************************************************************
* Name: Init
* Desc: Init and set default values for the Fuel Moisture Conditioning
*        Inputs class
* Note-1: The InitNLC() initalizies most relevant inputs items with
*         an 'init' value and then later after defaults and the
*         user loads inputs the InitCheck(), if it finds any of
*         the 'init' values that means something hasn't been set.
**********************************************************************/
void CI::Init()
{
int i,s, fs;

   this->InitNLC();     /* see Note-1 above */

   fuelconversion.Init();  /* loads with the conversion values */

/*................................................................*/
/* Not sure how these should really be set */

  for ( s = 0; s < eC_Sta; s++ ) {    /* each station */
    for ( fs = 0; fs < 4; fs++ ) {    /* each fuel size */
       EnvironmentChanged (false, s, fs);}}

/* This is how FlamMap did it in SetDates() */
  s = 0;    /* Station 0, each fuel size  */
  EnvironmentChanged (true, s, 0);
  EnvironmentChanged (true, s, 1);
  EnvironmentChanged (true, s, 2);
  EnvironmentChanged (true, s, 3);

/*....................................................................*/
   strcpy (cr_ErrMes,"");  /* return error message */

   for ( i = 0; i < eC_Hdr; i++ ) {
     lr_Fuel[i] = 0;
     lr_Woodie[i] = 0; }

   for ( i = 0; i < eC_fm; i++ )  /* Just init on field so we can check later */
     this->fm[i].TL1 = -1;        /* to make sure array was loaded by user */

/*---------------------------------------------------------------------------*/
   for (int i = 0; i < eC_Sta; i++)     {
     wtrdt[i] = NULL;
     MaxWeatherObs[i] = 0;
     wddt[i] = NULL;
     MaxWindObs[i] = 0;
     a_RAWS[i] = NULL;
     irN_RAWS[i] = 0;}

  strcpy (cr_FuelStickModel,e_FSM_Default); /* Conditioning Nelson Stick Model */

//  ResetNewFuels();

  for (int i = 0; i < 64; i++)
    ThreadProgress[i] = 0.0;


/* Default Moisture Conditioning Time Simulation Interval - based on Fuel Siz Cls */
   i_MCHI = SIZECLASS_10HR;

   this->iS_1k = 1;      /* Do 1k moisture values */

/*..............................................................................*/
/* 1 Hour  */
 	MoistCalcInterval[0][0] = 60;    /* simulation time */
 	MoistCalcInterval[0][1] = 200;   /* Elev */
	 MoistCalcInterval[0][2] = 10;    /* Slope */
	 MoistCalcInterval[0][3] = 45;    /* Aspect */
	 MoistCalcInterval[0][4] = 15;    /* Cover */

 	MoistCalcInterval[1][0] = 60;  /* 10 Hr */
 	MoistCalcInterval[1][1] = 200;
 	MoistCalcInterval[1][2] = 10;
 	MoistCalcInterval[1][3] = 45;
 	MoistCalcInterval[1][4] = 15;

 	MoistCalcInterval[2][0] = 120;   /* 100 Hr */
 	MoistCalcInterval[2][1] = 400;
	 MoistCalcInterval[2][2] = 10;
	 MoistCalcInterval[2][3] = 45;
	 MoistCalcInterval[2][4] = 20;

	 MoistCalcInterval[3][0] = 240;   /* 1000 Hr */
	 MoistCalcInterval[3][1] = 400;
	 MoistCalcInterval[3][2] = 15;
	 MoistCalcInterval[3][3] = 45;
 	MoistCalcInterval[3][4] = 30;
}

/******************************************************************
* Name: InitNCL
* Desc: This initialize is used so that later we can perform a
*        not loaded check, meaning if we find any of these
*        init values then the input wasn't loaded.
* Note: See InitCheck()
******************************************************************/
void CI::InitNLC()
{
    cr_FuelStickModel[0] = 0;
    d_RunTime =  eF_init;
    instanceID = eI_init;
    StartProcessor = 0;
    l_Latitude = eI_init;
    l_NumWoodie = eI_init;
    lr_Woodie[0] = eI_init;
    l_NumFuel = eI_init;
    lr_Fuel[0] = eI_init;
    NumWeatherStations = eI_init;
    NumWindStations = eI_init;

    startmonth = eI_init;
    startday = eI_init;
    starthour = eI_init;

    MaxThreads  = 1;

    wtrdt[0] =   0;
    MaxWeatherObs[0] = eI_init;

    wddt[0] =    0;                /* WindData */
    MaxWindObs[0]  = eI_init;

    a_RAWS[0] = 0;                  /* Weather Stream Data */
    irN_RAWS[0] = eI_init;


    fuelconversion.Type[0] = eI_init;  /* FuelConversions.type[]  init */

   // newfuels[0].number  = eI_init;     /* NewFuel */

//    HAVE_CUST_MODELS = eI_init;

    fm[0].TL1 = eI_init;               /* InitialFuelMoisture */

    l_ElevHigh = eI_init;
    l_ElevLow = eI_init;
    s_EUnits = eI_init;

    l_SlopeHigh = eI_init;
    l_SlopeLow = eI_init;
    s_SUnits = eI_init;

    l_CoverHigh = eI_init;
    l_CoverLow = eI_init;

    MoistCalcInterval[0][0] = eI_init;

    i_MCHI = eI_init;

 //   hMoistEvent.NumEvents = eI_init;          /* FarsiteEvent */
 //   hMoistThreadEvent.NumEvents = eI_init;    /* FarsiteEvent */
}

/******************************************************************
* Name: InitCheck
* Desc: If any of the input variables stil have an init value in
*       them it means they haven't been load, which is not a
*       good thing.
*       See InitNLC()
******************************************************************/
int CI::InitCheck (char cr[])
{
  strcpy (cr,"");
  if ( cr_FuelStickModel[0]    == 0       ) { strcpy (cr, "cr_FuelStickModel[0]"); return 0; }
  if ( d_RunTime               == eF_init ) { strcpy (cr, "d_RunTime           "); return 0; }
  if ( instanceID              == eI_init ) { strcpy (cr, "instanceID          "); return 0; }
  if ( StartProcessor          == eI_init ) { strcpy (cr, "StartProcessor      "); return 0; }
  if ( l_Latitude              == eI_init ) { strcpy (cr, "l_Latitude          "); return 0; }
  if ( l_NumWoodie             == eI_init ) { strcpy (cr, "l_NumWoodie         "); return 0; }
//  if ( lr_Woodie[0]            == eI_init ) { strcpy (cr, "lr_Woodie[0]        "); return 0; }
  if ( l_NumFuel               == eI_init ) { strcpy (cr, "l_NumFuel           "); return 0; }
//  if ( lr_Fuel[0]              == eI_init ) { strcpy (cr, "lr_Fuel[0]          "); return 0; }
  if ( NumWeatherStations      == eI_init ) { strcpy (cr, "NumWeatherStations  "); return 0; }
  if ( NumWindStations         == eI_init ) { strcpy (cr, "NumWindStations     "); return 0; }
  if ( startmonth              == eI_init ) { strcpy (cr, "startmonth          "); return 0; }
  if ( startday                == eI_init ) { strcpy (cr, "startday            "); return 0; }
  if ( starthour               == eI_init ) { strcpy (cr, "starthour           "); return 0; }
//  if ( MaxThreads              == eI_init ) { strcpy (cr, "MaxThreads          "); return 0; }
  if ( fuelconversion.Type[0]  == eI_init ) { strcpy (cr, "fuelconversion.Type[0]"); return 0; };
 // if ( newfuels[0].number      == eI_init ) { strcpy (cr, "newfuels[0].number");   return 0; }
//  if ( HAVE_CUST_MODELS        == eI_init ) { strcpy (cr, "HAVE_CUST_MODELS  ");   return 0; }
  if ( fm[0].TL1               == eI_init ) { strcpy (cr, "fm[0].TL1         ");   return 0; }
  if ( l_ElevHigh              == eI_init ) { strcpy (cr, "l_ElevHigh        ");   return 0; }
  if ( l_ElevLow               == eI_init ) { strcpy (cr, "l_ElevLow         ");   return 0; }
  if ( s_EUnits                == eI_init ) { strcpy (cr, "s_EUnits          ");   return 0; }
  if ( l_SlopeHigh             == eI_init ) { strcpy (cr, "l_SlopeHigh       ");   return 0; }
  if ( l_SlopeLow              == eI_init ) { strcpy (cr, "l_SlopeLow        ");   return 0; }
  if ( s_SUnits                == eI_init ) { strcpy (cr, " s_SUnits         ");   return 0; }
  if ( l_CoverHigh             == eI_init ) { strcpy (cr, "l_CoverHigh       ");   return 0; }
  if ( l_CoverLow              == eI_init ) { strcpy (cr, "l_CoverLow        ");   return 0; }
  if ( MoistCalcInterval[0][0] == eI_init ) { strcpy (cr, "MoistCalcInterval ");   return 0; }

#ifdef WIN32
  if ( hMoistEvent.NumEvents       == eI_init ) { strcpy (cr, "hMoistEvent.NumEvents"); return 0; }
  if ( hMoistThreadEvent.NumEvents == eI_init)  { strcpy (cr, "hMoistThreadEvent.NumEvents"); return 0; }
#endif

/* A defualt should have been set in the Init even if the user didn't set it */
  if ( this->i_MCHI == eI_init )  { strcpy (cr, "i_MCHI isn't set"); return 0; }

  if ( wtrdt[0] == 0 && wddt[0] == 0 && a_RAWS[0] == 0 ) {
     strcpy (cr, "Weather/Wind or Weather Stream Data");
     return 0; }
  if ( MaxWeatherObs[0] == eI_init && MaxWindObs[0] == eI_init && irN_RAWS[0] == eI_init ) {
     strcpy (cr, "Weather/Wind/Cloud Data array count");
     return 0; }

  return 1;
}

/*********************************************************************
* Name: Chrono
* Desc: Given a time in minutes add it to the inputed starting
*        simulation month, day, hour (0->2359) and return it as the
*        Julian date and hour
*   In: SIMTIME = minutes
*  Out: *hour....0 -> 2359 - Military Time
*       *hours...
*  Ret; julian date
*
* NOTE: This has been modified from the orginal FlamMap::Chrono()
*       the following "if else" code was removed because RelCond was always
*       True and UseCondtioningPeriod() was always False
*      if (RelCondit && UseConditioningPeriod(GETVAL))
*
*
*     This function isn't used anymore, it was only used to get
*        the date/hour for getting cloud,

*********************************************************************/
long CI::Chrono(double SIMTIME, long *hour, double *hours)
{
long date, month, day, min;
double mins, days;

/*----------------------------------------------------------*/

/* Get totaal sim time hours */
 	*hours = SIMTIME / 60.0;
 	*hour = (long) *hours;                       // truncates time to nearest hour

/* Get sim time remaining minutes */
	 min = (long) ((*hours - *hour) * 60);
	 mins = (*hours - *hour) * 60.0;

/* Get hours to Military Time the 0->2359 form */
	*hours = ((double) *hour) * 100.0 + (double) this->starthour;	// elapsed time in hours

/* The number of days to simulate */
 	days = *hours / 2400.0;
 	day = (long) days;

/* Get Julian starting date that the sim will run to  */
	 date = day + this->startday;
	 month = GetJulianDays (this->startmonth);
  date = date + month;

/* if it goes into the the next year, bring date back to approbriate date */
	 if ( date > 365 )
    date -= 365;

/* Number of total days from start to run the simulation */
	 days = day;

/* Here's the hour of the last day to stop sim at */
	*hour = (long) (*hours - days * 2400) + min;         // integer hour
	*hours = (*hours - days * 2400.0) + mins;            // double precision hours

	return date;
}



/***************************************************************************************/
void CI::GetWeatherRainTimes(long StationNumber, long NumObs, long *tr1, long *tr2)
{
   if ( NumObs > MaxWeatherObs [StationNumber] - 1)
     	NumObs = MaxWeatherObs [StationNumber] - 1;

  *tr1 = wtrdt [StationNumber][NumObs].tr1;
  *tr2 = wtrdt [StationNumber][NumObs].tr2;
}

/***************************************************************************
* NOTE:
*  This is a dummy function - It says that there is no AtmosphereGrid
*   in use.
*  The AtmosphereGrid wasn't implemented in FlamMap even though there was
*   code in the FlamMap program, but it never got called, So I didn't
*   implement it in here either, I also did bring in FlamMap.AtmGrid
*  SEE FlamMap::AtmosphereGridExist() for the orginal code that was
*   in this function.
****************************************************************************/
long CI::AtmosphereGridExists()
{
   return 0;
}


/*************************************************************************
* Name: wtfdt_idx
* Desc: return the table index of the weather record for the specified
*        Julian Date
*   In: julian date
*       sn....station
*  Ret: index into table
*       if date isn't in table return either index of 1st or last rec
*************************************************************************/
int CI::wtrdt_idx ( int i_JulDatFind, int sn)
{
int i, i_JulDat;

   for ( i = 0; i < this->MaxWeatherObs[sn]; i++ ) {

     i_JulDat = this->GetJulianDays (wtrdt[sn][i].mo) + wtrdt[sn][i].dy;

     if ( i == 0 ) {                    /* just incase requested date is before */
       if ( i_JulDatFind < i_JulDat )   /*  wtr data, shouldn't  really happen */
         return i; }

     if ( i_JulDatFind == i_JulDat )    /* this is the one we want */
         return i;
   }

   i = this->MaxWeatherObs[sn] - 1;     /* found so return last rec's idx */
   return i;
}


/**************************************************************************/
#ifdef OLDOLD
long CI::GetWeatherMonth(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].mo;
}

long CI::GetWeatherDay(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].dy;
}
#endif

/******************************************************************/
double CI::GetWeatherTemp1(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].T1;
}

double CI::GetWeatherTemp2(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].T2;
}

long CI::GetWeatherTime1(long StationNumber, long NumObs)
{
   if ( NumObs > MaxWeatherObs[StationNumber] - 1)
     	NumObs = MaxWeatherObs[StationNumber] - 1;
  	return wtrdt[StationNumber][NumObs].t1;
}

long CI::GetWeatherTime2(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].t2;
}

/***********************************************************************/
long CI::GetWeatherHumid1(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].H1;
}

long CI::GetWeatherHumid2(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].H2;
}

double CI::GetWeatherElev(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].el;
}

double CI::GetWeatherRain(long StationNumber, long NumObs)
{
     if(NumObs>MaxWeatherObs[StationNumber]-1)
     	NumObs=MaxWeatherObs[StationNumber]-1;

	return wtrdt[StationNumber][NumObs].rn;
}

/*****************************************************************
* Name: GetCloud
* Desc: Get cloud cover for the simulation time sent in,
*       The simulation time is added to the starting conditioning
*       time then that entry is found in the table
* NOTE: This function replaces the orginal one FE2:GetCloud() in
*        order to handle hitting then end of table better and to
*        get rid of the Chron() time function
*   In: l_SimTim.....simulation time (minutes)
*****************************************************************/
long CI::GetCloud (long l_SimTim, long sn)
{
int i;
long l_TblTim, l_SimMinTim;

/* NOTE ----> */
/* We can use the RAWS total minute type date to find the record we want */

/* Get the total minutes plus the simtime */
  l_SimMinTim = CI::RAWS_CalcMinuteDate  (this->i_StartYear, this->startmonth, this->startday, this->starthour);
  l_SimMinTim += l_SimTim;                /* add on the simtime */

  for ( i = 0; i < MaxWindObs[sn]; i++ ) {  /* look thru wind records */
     l_TblTim = CI::RAWS_CalcMinuteDate (wddt[sn][i].yr, wddt[sn][i].mo, wddt[sn][i].dy,wddt[sn][i].hr);

     if ( i == 0 ) {                   /* just in case time is before start time */
       if ( l_SimMinTim < l_TblTim )
         return wddt[sn][i].cl; }

     if ( l_SimMinTim < l_TblTim )   /* time we're looking for is ahead */
       return wddt[sn][i-1].cl;
   }

   i = MaxWindObs[sn] - 1;           /* just in case requested time is past */
   return wddt[sn][i].cl;            /*  what table has, return last in tbl */
 }


/**********************************************************************/
long CI::GetWindCloud(long StationNumber, long NumObs)
{
   if ( NumObs > MaxWindObs[StationNumber] - 1 )
     	NumObs = MaxWindObs[StationNumber] - 1;
	  return wddt[StationNumber][NumObs].cl;
}

long CI::GetWindMonth(long StationNumber, long NumObs)
{
   if ( NumObs > MaxWindObs[StationNumber] - 1 )
     	NumObs = MaxWindObs[StationNumber] - 1;
  	return wddt[StationNumber][NumObs].mo;
}

long CI::GetWindDay(long StationNumber, long NumObs)
{
   if ( NumObs > MaxWindObs[StationNumber]-1)
     	NumObs=MaxWindObs[StationNumber]-1;
	  return wddt[StationNumber][NumObs].dy;
}

long CI::GetWindHour(long StationNumber, long NumObs)
{
   if ( NumObs > MaxWindObs[StationNumber]-1)
     NumObs=MaxWindObs[StationNumber]-1;
	  return wddt[StationNumber][NumObs].hr;
}

/***********************************************************************/
long CI::GetMaxThreads()
{
	return MaxThreads;
}

/********************************************************************/
int CI::GetFuelConversion(int fuel)
{// retrieve fuel model conversions
	int cnv = -1;

	if (fuel >= 0 && fuel < eC_FCTyp )
	{        // check fuel for valid array range
		cnv = fuelconversion.Type[fuel];    // get new fuel
		if ( cnv < 1 )
		{
			if ( cnv < -9)
				cnv=-1;
		}
		/*else if ( !IsNewFuelReserved (cnv) )
		{
			if ( newfuels[cnv].number == 0 )
				cnv = -1;
		}*/
		else if ( cnv > 256 )
			cnv = -1;
	}

	return cnv;
}
/**********************************************************************/
/*bool CI::IsNewFuelReserved (long number)
{
	 if (number < 0)
    return false;

	 if (newfuels[number].number < 0 )
    return true;

  return false;
}*/

/********************************************************************/
/*void CI::ResetNewFuels()
{
  memset(newfuels, 0x0, 257*sizeof(NewFuel));
  InitializeNewFuel();
  long i;
  for ( i=0; i < 257; i++ ) {
    if ( newfuels[i].number )
       newfuels[i].number *= -1;	  // indicate permanence
  }

  HAVE_CUST_MODELS = false;
}*/
/********************************************************************/
long CI::GetInitialFuelMoisture(long Model, long FuelClass)
{
  if (Model > 256 || Model < 1 )
    return 2;

long mx;

	 switch(FuelClass) {
     case 0: mx=fm[Model-1].TL1; break;
     case 1: mx=fm[Model-1].TL10; break;
     case 2: mx=fm[Model-1].TL100; break;
     case 3: mx=fm[Model-1].TLLH; break;
     case 4: mx=fm[Model-1].TLLW; break;
		 // case 5: mx = fm[Model-1].
    }
   return mx;
}


/***********************************************************************
* Name: LogicError
* Desc: All place to send logic errors, things that shouldn't be
*       happening. Mostly used during development.
***********************************************************************/
void CI::LogicError (char cr[])
{
/* For now just print out */
/* later comment this can be commented out. */
  printf ("CI LOGIC ERROR: %s\n",cr);
  //_getch ();
}


/**********************************************************************
* Name: FuelConversions
* Desc: This is declared inside the CI Class
*       This function goes off as a constructor when the CI is
*        instanciated.
**********************************************************************/
void FuelConversions::Init()
{
long i;

 	for ( i = 0; i < eC_FCTyp; i++ )    // could also read default file here
		  Type[i]=i;

	Type[99]=-1;
	Type[98]=-2;
	Type[97]=-3;
	Type[96]=-4;
	Type[95]=-5;
	Type[94]=-6;
	Type[93]=-7;
	Type[92]=-8;
	Type[91]=-9;
}


/*****************************************************************
*  Looks like any code in the Conditioning that calls this function
* never even get used, so this function never gets called.
*  Evidently this is for using Coarse Woody, which FlamMap never
* used to run Conditioning..
*  BUT I've kind of left things in place, in case we need it, I
* not sure yet if FarSite will want to use Coarse Woody, if not
* maybe this and the other code can get removed.
*******************************************************************/
double CI::GetWoodyFuelMoisture(long ModelNumber, long SizeClass)
{
  if ( ModelNumber > MAXNUM_COARSEWOODY_MODELS)
    return 0.0;

  return 1; /* for testing now */

/* need to implement code below */

//	 if ( coarsewoody[ModelNumber-1].NumClasses < SizeClass)
//   	return 0.0;

//  return coarsewoody[ModelNumber-1].wd[SizeClass].FuelMoisture;
}


/***********************************************************
* Name: Set_RunTime
* Desc: Set the time in minutes that you want to run the s
*       Conditioning simulation
*   In: d = time in minutes.
***********************************************************/
void CI::Set_RunTime (double d)
{
  d_RunTime = d;
}

/***********************************************************
* Name: FuelStickModel_***
* Desc: Set the Nelson Fuel Stick Moisture Model that is
*        to be used
*       See comments in CI Class definition for cr_FuelStickModel
***********************************************************/
void CI::FuelStickModel_Nelson_070 ()
{
   strcpy (cr_FuelStickModel, e_FSM_Nelson_070);
}

void CI::FuelStickModel_Nelson_100 ()
{
   strcpy (cr_FuelStickModel, e_FSM_Nelson_100);
}

/* True = the model is set to the orginal Nelson 0.7.0 */
bool CI::FuelStickModel_IsNelson_070()
{
  if ( !strcmp (cr_FuelStickModel,e_FSM_Nelson_070) )
    return true;
  return false;
}


/************************************************************************************
* Name: ConvertSimtimeToActualTime
* Desc: This functions takes the start condition mth day, adds the simtime to it
*        and converts that to a mth day.
* NOTE: this function had a bug in it, it didn't always get the right mth/day and
*        minutes would come out as 59 sometimes instead 0. BUT it kind of seemed
*        like it didn't matter much to the caller. BUT I rewrote it because it was
*        cause the wronge mth/day to go into the moisture history records, which are
*        mostly used for reference, but maybe needed later for actual use.
*
************************************************************************************/
void CI::ConvertSimtimeToActualTime (double SimTime, long *mo, long *dy, long *hr, long *mn)
{

/* This is the fixed function */
 ConvertSimtimeToActualTime_Fix (SimTime, mo, dy, hr, mn);
 return;

//**************************************************************************
// This is the orginal code
//
#ifdef OLD_CODE

long months, days, hours, mins;
double fday, fhour;
long i;

 fday = (SimTime / 1440.0);	// days from minutes
  days = (long) fday;
  fhour = (fday-days)*24.0;  // hours from days
  hours = (long) fhour;
  mins = (fhour-hours)*60;	 // minutes from hours

  if (mins>60) {
     mins-=60;
     hours++; }

  if(hours>=24) {
    	hours-=24;
     days++; }

  hours *= 100;

  long ConDate = GetJulianDays(startmonth) + startday;
  if ( ConDate + days > 365) {
    days =- 365;
    for (i=1; i<12; i++) {
      if ( days < GetJulianDays(i))
         break; }
    days -= GetJulianDays(i);
    months = i;}
  else {
    for ( i = 1; i < 12; i++) {
      if (days + ConDate < GetJulianDays ( startmonth + i ))
         break; }

    days -= ( GetJulianDays (startmonth + i - 1 ) - ConDate);
    months = startmonth + (i-1);}

  *mo = months;
  *dy = days;
  *hr = hours;
  *mn = mins;
#endif
}

/**********************************************************************
* See Calling function for comments.
*
************************************************************************/
void CI::ConvertSimtimeToActualTime_Fix (double SimTime, long *mo, long *dy, long *hr, long *mn)
{
long A,l, days, hours, mins, l_Time;
long ConDays;

 l_Time = (long) SimTime;
 days = l_Time / 1440;
 l = l_Time % 1440;
 hours = l / 60;
 mins  = l % 60;

  hours = hours * 100;

  ConDays = GetJulianDays(this->startmonth) + this->startday;

  A  = ConDays + days;

  Get_MthDay (A, mo, dy );

  *hr = hours;
  *mn = mins;
}

/**************************************************************************/
void Get_MthDay (long JulDay, long *Mth, long *Day)
{
int i;
int dd[13] = { 0, 31, 59, 90,120,151, 181, 212, 243, 273, 304, 334, 365 };

  if ( JulDay > 365 )
    JulDay = JulDay - 365;

  for ( i = 1; i <= 12; i++ ) {
    if ( JulDay <= dd[i] ) {
      *Mth = i;
      *Day = JulDay - dd[i-1];
      break; } }
}
