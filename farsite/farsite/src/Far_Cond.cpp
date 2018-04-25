/********************************************************************************

  Stuff for hooking Cond Dll to FarSite

  > Farsite5.h    put in #include  fmc_cfmc.h
                  put CFMC cfmc in the Farsite5 class

                 call from Farsite5::FarsiteSimulationLoop()

                  get moistures back in FELocalSite::GetFireEnvironment()
                    see the tenhour = etc.
  >



  >

**********************************************************************************/

#include "Farsite5.h"
#include <memory.h>
#include <cstring>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
// #include "rand2.h"
#include "barriers.hpp"

#define e_MinPerDay 1440
#define e_DaysPerYear  365
#define e_StrYear 2001


//FILE *fTest = NULL;

// int MilToMin (int i_MilTim);
//  int GetMCDate(int mth, int day, int year);
int CheckMonthDay (int m, int  d);


/**********************************************************************
* Name: Run_CondDLL
* Desc: Init, load & check inputs and run the moisture conditioning DLL
*  Ret: 1 - conditioning ran ok
*       < 0  negative error code, the error message can be obtained using
*            the icf.ErrorMessage()
**********************************************************************/
int Farsite5::Run_CondDLL()
{
long i, i_AdjTim;
char  cr_ErrMes[1000];

  this->cfmc.Init();

  i_AdjTim = FMC_LoadInputs (&this->icf,       /* input command file data */
                      &this->Header,    /* Landscape file header data */
                      &this->cfmc,
                      cr_ErrMes);
  if ( i_AdjTim < 0 ){                      /* Error occured */
    strcpy (this->icf.cr_ErrExt,cr_ErrMes);
    cfmc.Delete();
    return i_AdjTim;}

  i = cfmc.CheckInputs();
  if ( i < 0 ) {
     cfmc.GetErrMes(cr_ErrMes);                /* Error occured */
     strcpy (this->icf.cr_ErrExt,cr_ErrMes);
     cfmc.Delete();                       /* cleanup alloc'd memory */
     return i; }

  i = cfmc.Run();  /* use the MaxTime in FlamMap obj */
  if ( i == 0 ) {   /* run was aborted by user */
    strcpy (this->icf.cr_ErrExt,"User Aborted");
    cfmc.Delete();                /* cleanup alloc'd memory */
    return e_EMS_Abort; }

/* Adjust the SimTimes in Cond Dll History records, so that the records before the */
/*  burning start time are negative and the record at the starting burn simulation */
/*  time is at 0 */
    cfmc.Set_FarsiteDef(i_AdjTim);

 //   i = cfmc.ExportMoistureDataText("C:/larryf/stutest/Time-Bug-FIX.txt", "");

 return 1;

}

/*********************************************************************
* Name: FMC_LoadInputs
* Desc: Fuel Moisture Conditioning - Load inputs into the CFMC class
*        object.
*
*   In: fm....FlamMap object - various input data is in here see below
*       icf...Input Command File object - data from FlamMap inputs file
*       lcp...HeadData - data from input LCP file
*  Out: cfmc..Class Fuel Moisture Conditioning - class for DLL
*  Ret: Minute difference between start condition date and starting
*        burn simulation date
*       0 error
**********************************************************************/
int Farsite5::FMC_LoadInputs ( ICF *icf, headdata *lcp, CFMC *cfmc, char cr_ErrMes[])
{
int  i,j, i_StrYr, obs;
int  i_MthStart, i_DayStart, i_HourStart;
int  date, min, StrMin, EndMin, MaxMin;
double d;
d_RAWS *ar;
 d_Wtr *a;

   strcpy (cr_ErrMes,"");

   i_StrYr = -1;

/* Currently Farsite only uses 1 processor */
   this->cfmc.SetNumStations (this->MaxThreads, 1); /* processors - weather stations */

// NOTE - at this point the Cond DLL doesn't even handle this large woody but....
//  for now we set it so we don't get an error,
   i = cfmc->Set_Woody (lcp->woodies, lcp->numwoody, cr_ErrMes);
   if ( i < 0 )
     return i;

/* FuelModel categories */
   i = cfmc->Set_FuelModel (lcp->fuels, lcp->numfuel, cr_ErrMes);
   if ( i < 0 )
     return i;

/*........................................................................*/
/* Load Fuel Moisture Models from the input command file settings struct */
   for ( i = 0; i < icf->iN_FMD; i++ ) {  /* Find defualt Fuel Model */
     if ( icf->a_FMD[i].i_Model == e_DefFulMod )
       break; }

/* Set the default fuel moist model */
   cfmc->Set_DefFuelMoistModel (icf->a_FMD[i].i_TL1,   icf->a_FMD[i].i_TL10,
                                icf->a_FMD[i].i_TL100, icf->a_FMD[i].i_TLLH,
                                icf->a_FMD[i].i_TLLW);
/* Now set all the fuel moist model */
   for ( i = 0; i < icf->iN_FMD; i++ ) {          /* for each Model             */
     if ( icf->a_FMD[i].i_Model == e_DefFulMod )  /* Skip the Default Model     */
       continue;
     cfmc->Set_FuelMoistModel (icf->a_FMD[i].i_Model, icf->a_FMD[i].i_TL1,
                               icf->a_FMD[i].i_TL10, icf->a_FMD[i].i_TL100,
                               icf->a_FMD[i].i_TLLH, icf->a_FMD[i].i_TLLW); }
/*..................................................................*/

   cfmc->Set_Elev(lcp->hielev, lcp->loelev, lcp->EUnits);    /* Unit 1 == feet, 0 = meter */
   cfmc->Set_Slope(lcp->hislope, lcp->loslope, lcp->SUnits); /* Unit 0 = percent, 1 = degrees */
   cfmc->Set_Cover(lcp->hicover, lcp->locover, lcp->CUnits);

	//SYSTEM_INFO sysinf;
	// GetSystemInfo(&sysinf);
	 cfmc->Set_ThreadsProcessor (1,0);  /* # of threads, and starting processor # */
  // cfmc->Set_ThreadsProcessor (this->MaxThreads,0);  /* # of threads, and starting processor # */

	 	cfmc->ResetThreads();
   //cfmc->SetInstanceID(1);  /* make unique for each use of DLL */
   cfmc->SetLatitude(lcp->latitude);

    int N = 0;
/*----------------------------------------------------------------------*/
/* RAWS - load it if we have it */
   if ( this->icf.a_RAWS != NULL ) {   /* Do we have RAWS hourly Weather Stream Data */
     i = this->cfmc.RAWS_Allocate (this->icf.iN_RAWS);
     if ( i == 0 ){
       strcpy (cr_ErrMes,"Can't Allocating RAWS Memory");
       return e_EMS_cMem; }

     for ( i = 0; i < this->icf.iN_RAWS; i++ ) {
       ar = &this->icf.a_RAWS[i];
       this->cfmc.RAWS_LoadDate(i,ar->i_Yr,ar->i_Mth,ar->i_Day,ar->i_Time);
       this->cfmc.RAWS_LoadInfo(i,ar->f_Temp,ar->f_Humidity,ar->f_PerHou,ar->f_CloCov);  }

//  this->cfmc.WSD_Display ();
/* Conditioning for FarSite starts with the first date/time */
    i_StrYr = this->icf.a_RAWS[0].i_Yr;
    i_MthStart = this->icf.a_RAWS[0].i_Mth;
    i_DayStart = this->icf.a_RAWS[0].i_Day;
    i_HourStart = this->icf.a_RAWS[0].i_Time;

    this->cfmc.RAWS_SetElev(this->icf.i_RAWSElev);
    goto CondDate;
  }                 /* hav Wthr Strm so no Wind/Wthr  */


/*------------------------------------------------------------------------*/
/* Load the Daily Wind Data */
  N = icf->iN_Wnd;
  if ( N > 0 ) {
    if ( !cfmc->AllocWindData_Sta0 (N) ) {
      strcpy (cr_ErrMes, "Can't Allocate Wind Memory");
      return e_EMS_cMem; } }

  d_Wnd *w;
  for ( j = 0; j < N; j++ ) {
    w = &icf->a_Wnd[j];
    cfmc->SetWindData_Sta0 (j, w->i_Year, w->i_Mth, w->i_Day, w->i_Hr, w->i_CloCov);
//    printf ("%d, %d, %d, %d, %d \n", j, w->i_Mth, w->i_Day, w->i_Hr, w->i_CloCov);
 }


/*......................................................................./
 Load Daily Weather Data                                                 */
  obs = icf->iN_Wtr;
  if ( obs > 0 ) {
    if ( !cfmc->AllocWeatherData_Sta0 (obs) ) {  /* allocate space in DLL */
      strcpy (cr_ErrMes, "Can't Alloc Weather Memory");
      return e_EMS_cMem; } }

  for ( int j = 0; j < obs; j++ ) {
     a = &icf->a_Wtr[j];
    cfmc->SetWeatherData_Sta0(
//   printf ("%3d, %2d %2d %6.2f %4d %3d %6.2f %6.2f %3d %3d %6.2f %3d %3d\n",
                               j,a->i_Mth,a->i_Day,
                               a->f_Per,
                               a->i_mTH,  /* Min Time  hour */
                               a->i_xTH,  /* Max Time  hour */
                               a->f_mT,   /* min Temp */
                               a->f_xT,   /* max Temp */
                               a->i_xH,   /* max Humid */
                               a->i_mH,   /* min Humid */
                      (float)  a->i_Elv,  /* Elev */
                               a->i_PST,  /* Percip Start Time */
                               a->i_PET); /* Percip End time */
   }

    i_StrYr =  icf->a_Wtr[1].i_Year;     /* Start Conditioning For Daily Weather */
    i_MthStart = icf->a_Wtr[1].i_Mth;    /*  on the second day */
    i_DayStart = icf->a_Wtr[1].i_Day;
    i_HourStart = 0;

/*--------------------------------------------------------------------*/
CondDate:

   cfmc->Set_DateStart (i_StrYr, i_MthStart, i_DayStart, i_HourStart);

   date = GetMCDate (i_MthStart, i_DayStart, i_StrYr);
   min = MilToMin (i_HourStart);              /* Military time to minutes */
   StrMin = (date * e_MinPerDay) + min;       /* to total minutes */

   date = GetMCDate (icf->i_FarsiteEndMth, icf->i_FarsiteEndDay, icf->i_FarsiteEndYear);
   min = MilToMin (icf->i_FarsiteEndHour);
   EndMin = (date * e_MinPerDay) + min;      /* to total minutes */

   MaxMin = EndMin - StrMin;
   d = (double) MaxMin;

   d += this->actual;

   cfmc->Set_RunTime(d);

//-----------------------------------------------------------------------------
/* Simulation time interval is based on a Fuel Size Class, 1, 10, 100, 1000 hr */
/* the 0 used below will cause the default to get set   */
   cfmc->Set_MoistCalcHourInterval (0);

/*-------------------------------------------------------------------*/
/* Set the conditioning underlying Nelson Fuel Moisture Stick Model */
/* If none set than the default model is used - see FMC DLL code */
     cfmc->FuelStickModel_Nelson_070();  /* This is the Original one used */

// Still have to put an switch in the inputs for the newer Nelson in the inputs file
    //cfmc->FuelStickModel_Nelson_100();  /* Nelson 1.0.0 added in Oct 09 */


/* Tell Cond DLL not to do 1000 hr moisture calcs */
  cfmc->Set_1kOff();

/* Get difference in minutes between the starting condition date and the starting burn date */
   date = GetMCDate (icf->i_FarsiteStartMth, icf->i_FarsiteStartDay, icf->i_FarsiteStartYear);
   min = MilToMin (icf->i_FarsiteStartHour);
   EndMin = (date * e_MinPerDay) + min;      /* to total minutes */
   MaxMin = EndMin - StrMin;

  return (int) MaxMin;
}

/*********************************************************************
* Name: GetMCDate
* Date: Get a Mositure Conditioning Date
*       This a Julian date with another 365 added to it if
*        the year is the e_StrYear+1 year.
*       FlamMap Cond only operates with 2 psuedo years and a
*        maximum number of one year worth of data
*   In: mth,day, year = use only e_StrYear or e_StrYear+1
*  Ret: 1 -> e_DaysPerYear * 2
*       0 = invalid date
**********************************************************************/
int GetMCDate(int mth, int day, int year)
{
int i;
   i = CheckMonthDay(mth, day);
   if ( i == 0 )
     return 0;

  if ( year != e_StrYear )
      i += e_DaysPerYear;
   return i;
}


/*********************************************************************
* Name: MilToMin
* Desc: Convert Military time to total minutes.
*       Ex: 1030 = ( 10 * 60 ) + 30 = 630 minutes
*******************************************************************/
int MilToMin (int i_MilTim)
{
int hour,min,Tot;
    hour = i_MilTim / 100;
    min = i_MilTim % 100;
    Tot = hour * 60;
    Tot = Tot + min;
    return Tot;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckMthDay
* Desc: Return the day number - julian date
* NOTE: THIS doesn't do a Feb 29
*   In: month 1->12, day 1->N
*  Ret: julian date or 0 for invalid date
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int CheckMonthDay (int m, int  d)
{
int rd[] = {0, 31, 28, 31, 30, 31,30, 31, 31, 30, 31, 30, 31};
int i,n;

    if ( m < 1 || m > 12 )              /* Chk for invalid month             */
      return 0;
    if ( d < 1 )                        /* Chk day                           */
      return 0;

    n = 0;
    for ( i = 0; i < m; i++ )           /* total up days in each month       */
      n += rd[i];                       /*  thru the desired mth             */

    if ( d > rd[i] )                    /* Chk for valid day */
      return 0;

    n = n + d;                          /* Add day to total                  */
    return n;
}


#ifdef wowowow

/***********************************************************************
* Name: DeadMoistureHistory_Display
* Desc: Test function to display out the list of records.
**********************************************************************/
int FireEnvironment2::DeadMoistureHistory_Display (int FuelSize, int sn)
{
DeadMoistureHistory *a;
int i,j,k, iN, Cnt;
float f_Tot;
char  CR[1000];

    Cnt = 1;
    iN = MxDesc.NumAlloc[sn][FuelSize];  /* actual # of Moist vales in the DeadMoistureHistory.Moistue[] */

    a = this->FirstHist[sn][FuelSize];

    while (1) {
      if ( a == NULL )
        break;

      sprintf (CR, "%2d - %6.0f %6.0f e%4d t%2.0f h%3.0f c%3d r%4.2f :", Cnt,
              a->LastTime, a->SimTime, a->Elevation,
              a->AirTemperature,a->RelHumidity, a->CloudCover, a->Rainfall / 2.54);
      Cnt++;
      TestFile (CR);

      f_Tot = 0;                                 /* Get Average Moist */
      for ( i = 0; i < iN; i++ )
        f_Tot += a->Moisture[i];
      f_Tot = f_Tot / (float) iN ;
      j = (int) (f_Tot * 100.0) ;
      sprintf (CR, "m %2d ",j);
      TestFile (CR);

      for ( i = 0; i < 10; i++ ) {           /* Show a few individul moists */
        j = (int) (a->Moisture[i] * 100.0);
        sprintf (CR, " %2d", j);
        TestFile (CR); }

      TestFile  ("\n");

      a = (DeadMoistureHistory *)  a->next;
    }

  return 1;
}



/***************************************************************************/
int FireEnvironment2::TestFile (char cr[])
{

   if ( !_stricmp (cr,"open") ) {
     fTest = fopen ("c:\\LarryF\\Test-Far\\FarSiteMoistHist.txt","w");
     return 1; }

   if ( !_stricmp (cr,"close") ) {
     fclose (fTest);
     return 1; }

   fprintf (fTest, "%s", cr);

   return 1;
}

#endif
