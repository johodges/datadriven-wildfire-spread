
/******************************************************************
* Change  10-29-09
*   Update code to use the new DeadFuelMoistuer class, replace the
*   Fms class in FMS_Cover structure, see flm4.cpp
*
*   See UpdateMapMoistuer_NewDFM() & AllocCover() below.
*   Search file for "Change" or "10-29-09"
*
* NOTE NOTE:
*   To switch running between the old and new class see
*   FmsThread::UpdateMoistures() below, also so AllocCover()
*
*
********************************************************************/
//------------------------------------------------------------------------------
//
//   fsxwfms2.cpp
//   FARSITE 4.0
//   10/20/2000
//   Mark A. Finney (USDA For. Serv., Fire Sciences Laboratory)
//
//   Contains functions for calculating dead fuel moisture maps for an entire
//        landscape file (declarations in fsx4.hpp)
//
//------------------------------------------------------------------------------
#ifdef WIN32
#include <windows.h>
#include <process.h>
#include <conio.h>
#else
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <sys/stat.h>
#endif


#include "cdtlib.h"
#include "newfms.h"
#include "semtime.h"
#include "deadfuelmoisture.h"
#include "FMC_CI.h"
#include "FMC_FE2.h"
#include <math.h>

using namespace std;

static const double PI=acos(-1.0);

static long WindLoc;

// Test-lar
void TestFile (char cr[]);
char gcr[1000];
// test-lar

/**********************************************************************
* Name: CondMngr - Conditioning Manager
* Desc: Run the conditioning thru each time step.
*       All inputs need to be set before coming here.
* NOTE: Function is setup to return integers in case we need to pass
*
* Note-1: the orginal code in FlamMap just returned a 0 with the GetActualTimeStep()
*  MoistSimTime = GetActualTimeStep(); see FlamMap:MoistThread()
*
*        back other errors at some point.
*  Ret: 1 = completed successfully,
*       0 = terminated by user
***********************************************************************/
int  FE2::CondMngr ()
{
double MoistSimTime, Interval, MaxTime;
long  lX;

   b_Terminate = false ;
   MoistSimTime = 0;              /* See Note-1 above */
   lX = a_CI->Get_MoistCalcHourInterval();
   Interval = a_CI->GetMoistCalcInterval (lX, FM_INTERVAL_TIME);
   MaxTime = a_CI->d_RunTime;                /* time of simulation - minutes */
	  d_Progress = 0.0;

/* Set Minute-Date in RAWS table if we have RAWS data */
   this->a_CI->RAWS_SetMinuteDate();

// test-larry
//    TestFile("open");

/* Run simuation for each time increment */
	 do {
     if (b_Terminate == true)              /* User terminated - hit a key */
        break;
		   MoistSimTime += Interval;
     CalcMapFuelMoistures(MoistSimTime);
		   d_Progress = MoistSimTime / MaxTime;  /* Percent completed */
     if ( d_Progress > 1.0 )               /* might go a little over */
       d_Progress = 1.0;
  } while ( MoistSimTime < MaxTime);      // MaxTime is class variable



// test-larry
//   TestFile ("Size 1 Hr\n");
//    this->DeadMoistureHistory_Display(0,0);

//    TestFile ("Size 10 Hr\n");
//    this->DeadMoistureHistory_Display(1,0);

//   TestFile ("Size 100 Hr\n");
//    this->DeadMoistureHistory_Display(2,0);

//   TestFile ("Size 1000 Hr\n");
//    this->DeadMoistureHistory_Display(3,0);

//   TestFile ("close");
// test-larry


  if ( b_Terminate == true )  /* If conditiong run had a termination request */
    return 0;

  return 1;
}

/**********************************************************************
* Name: CalcMapFuelMoistures
* Desc: Calculate the moistures for the specified simulation time
* Note-1: Time to run simulation too.
*         For example, succesive calls to this function would be
*         60, 120,  180, . . . 600
* Note-2: the NormalizeMoistureStartTime() was removed because it
*          would always return without doing anything because of it's
*          check for constant moisture.
*   In: SimTime...in minutes, see Note-1 above
*
***********************************************************************/
bool FE2::CalcMapFuelMoistures(double SimTime)
{
long i, j, k, PrevNumAlloc;
bool UpDated=false, NeedToRecalculate=true;

  PrevNumAlloc = MxDesc.NumAlloc[0][SIZECLASS_10HR];

/* Build Moisture Map on the first time thru here */
  if ( NumStations < a_CI->GetNumStations() ) {
    AllocStations ( a_CI->GetNumStations() );  /* Build the Fuel Moisture Map */
    ResetData(0); }

/* See Note-2 above  NormalizeMoistureStartTime() */

/*----------------------------------------------------------------------------------*/
  for ( i = 0; i < NumStations; i++) {
    StationNumber = i;
      for ( k = 1; k < NUM_FUEL_SIZES; k++ ) {
          if ( a_CI->iS_1k == 0 && k == SIZECLASS_1000HR ) {  /* user requested don't do 1k hour */
             continue; }

          if ( Stations[i].ReAllocFuels(k) || a_CI->EnvironmentChanged (GETVAL, i, k) ) {
             FreeHistory (i, k);
             if ( k == 1 ) {
	               FreeHistory(i, 0);
                PrevNumAlloc=0; }
     	     	 RefreshHistoryDescription(i, k);
          	  ResetData(k);
		           for ( j = 0; j < Stations[i].NumFuels[k]; j++) {
	               Stations[i].FMS[k][j]->FirstTime = 0.0;    /* Start time of moisture */
     	    			   Stations[i].FMS[k][j]->LastTime = 0.0; }   /* Ending time of moisture */
     	    }
 // will only happen if moistures not calculated in .FMM file
          else if (Stations[i].NumFuels[k] > 0 && MxDesc.NumAlloc[i][k] == 0) {
             FreeHistory(i, k);
             if ( k == 1 )
               FreeHistory(i, 0);
     	       RefreshHistoryDescription (i, k);
          	  ResetData(k);
		           for ( j = 0; j < Stations[i].NumFuels[k]; j++) {
	               Stations[i].FMS[k][j]->FirstTime = 0.0;
     	    			   Stations[i].FMS[k][j]->LastTime = 0.0;	} }

/* Do Daily Weather Data or Hourly RAWS data, depending on what was in the inputs */
          if ( !CheckMoistureHistory (i, k, SimTime))	{	// if the moistures have not already been calculated
	           for ( j = 0; j < Stations[i].NumFuels[k]; j++) {  /* for each Fule Type (Model) */
              if ( this->a_CI->a_RAWS[i] != NULL )
                RunFmsThreads_RAWS(SimTime,i,j,k);      /* Hourly RAWS data */
     	     		 else
                RunFmsThreads_Stu (SimTime, i, j, k);} }  /* i = Station, j = Fuel Type, k = Fuel Size */

	         else if ( k == 1 ) {
			          CloudCount=0;
			          WindLoc = 0;
		  	        CheckMoistureHistory (i, SIZECLASS_1HR, SimTime); }
		           a_CI->EnvironmentChanged(false, i, k);
          if ( k == 1 )
            a_CI->EnvironmentChanged(false, i, 0);
     } /* for k */
   }  /* for i */

   if ( PrevNumAlloc > 0 ) {
     if ( UpDated == true )
     	  NeedToRecalculate = true;	   // must recalc if data have been changed;
     	else
        NeedToRecalculate = false; }   // don't have to recalc if data already exist

   return NeedToRecalculate;
}

/******************************************************************
* Name: AllocStations
* Desc: Build Fuel Moisture Maps
*       Allocates FuelMoistureMap class and its classes. elev,
*        aspect, etc
******************************************************************/
bool FE2::AllocStations(long Num)
{
  long i, j;
  FreeStations();

   if (Num > 0) {
	     Stations = new FuelMoistureMap[Num];
      if ( Stations == NULL)
         return false;
      for ( i = 0; i < Num; i++) {
			     Stations[i].a_CI = a_CI;
			     Stations[i].AllocFuels(SIZECLASS_10HR);
        Stations[i].AllocFuels(SIZECLASS_100HR);
        if ( a_CI->iS_1k == 1 )
           Stations[i].AllocFuels(SIZECLASS_1000HR);
     }

      NumStations=Num; }
    else
     	return false;

  	for ( i = 0; i < eC_Sta; i++) {
     for ( j = 1; j < NUM_FUEL_SIZES; j++)
	 	    RefreshHistoryDescription(i, j);	  // first call to this
   }

	  return true;
}
// ***************************************************************
void FE2::FreeStations()
{
	  if ( Stations == NULL)
     	return;

   long i, j;
   for ( i = 0; i < NumStations; i++)	{
     for ( j = 0; j < NUM_FUEL_SIZES; j++)
      		Stations[i].FreeFuels(j);}

   delete[] Stations;
   Stations=0;
   NumStations=0;

   for ( i = 0; i < eC_Sta; i++) {
     for ( j = 0; j < NUM_FUEL_SIZES; j++)
     		FreeHistory(i, j); }

   CloseFmsThreads();
   SimStart = -1.0;
}

/******************************************************************
* I'm not sure why this function has that parameter or why it
*  gets called in a loop, but I'm leaving it as is for now
*******************************************************************/
void FE2::ResetData(long FuelSize)
{
   CloudCount=0;
   WindLoc=0;
   SimStart=-1.0;
}

/*********************************************************************************/
void FE2::CloseFmsThreads()
{
#ifdef WIN32
long m;

  if ( NumFmsThreads == 0 )
     	return;

  for ( m = 0; m < NumFmsThreads; m++)
		   fmsthread[m].SetRange (0.0, 0, 0, 0, 0, 0, 0, 0, -1);

  for ( m = 0; m < NumFmsThreads; m++)
     fmsthread[m].StartFmsThread (m, 0, &MxDesc, false);

  Sleep(50);
  a_CI->WaitForFarsiteEvents(EVENT_MOIST, NumFmsThreads, true, INFINITE);
#endif
 	FreeFmsThreads();
}

//------------------------------------------------------------------------------
static long FM_TOLERANCE=1;

long GetFmTolerance()
{
	return FM_TOLERANCE;
}

//******************************************************************
DeadMoistureDescription::DeadMoistureDescription()
{
long i, j;

// change 10-29-09 - see comments in struct DeadMoistureDescription
//   fms = 0;

   for ( i = 0; i < eC_Sta; i++) {
     for ( j=0; j<NUM_FUEL_SIZES; j++)
		    	ResetDescription(i, j); }
}

void DeadMoistureDescription::ResetDescription(long Station, long FuelSize)
{
long i;

   i = FuelSize;
   NumStations = 0;
   memset ( &(FuelKey[Station][i]), 0x0, MAXNUM_FUEL_MODELS*sizeof(long));
  	NumAlloc[Station][i] = 0;
  	NumHist[Station][i] = 0;
   NumFuels[Station][i]=0;
   EndTime[Station][i]=0.0;
	  NumElevs[Station][i]=NumSlopes[Station][i]=NumAspects[Station][i]=NumCovers[Station][i]=0;

// change 10-29-09 - see comments in struct DeadMoistureDescription
// if (fms)
//    delete[] fms;
//	 fms=0;
}


DeadMoistureDescription::~DeadMoistureDescription()
{
	long i, j;

     for(i=0; i<eC_Sta; i++)
     {	for(j=0; j<NUM_FUEL_SIZES; j++)
			ResetDescription(i, j);
     }
}

/***************************************************************
* Name: Init
* Desc: This replaces the FlamMap FireEnvironment2 Constructor,
*       One difference here is that we 'new' the a_CI instead
*        of passing it in.
*       a_CI is new'd with a CI object, the CI object will do what
*        the FlamMap object did in FlamMap, and that is to get
*        inputs when called upon to do so.
*************************************************************/
int FE2::Init ()
{
long i, j;

  a_CI = new CI;     /* FlamMap Input Class - will hold inputs */
  if ( a_CI == NULL )
    return 0;
  a_CI->Init();       /* Init the CI class */

	 b_Terminate = false;
  d_Progress = -1.0;      /* % completed, -1 indicated not started */
  Stations = 0;
  NumStations = 0;
  CloudCount = 0;
  fmsthread = 0;
  NumFmsThreads = 0;
  SimStart = -1.0;
  this->HistoryCount = 0;
  humidmn = humidmx = humref = 0;
  rain = 0;
  StationNumber = 0;
  tempmn = tempmx = tempref = 0;

  elevref = 0;
  for ( i = 0; i < eC_Sta; i++ ){
    for ( j = 0; j < NUM_FUEL_SIZES; j++){
	      CurHist[i][j] = 0;
		     NextHist[i][j] = 0;
     	 FirstHist[i][j] = 0; } }

  return 1;
}

/******************************************************************/
bool FE2::AllocFmsThreads()
{
	 if ( NumFmsThreads == a_CI->GetMaxThreads())
    return true;

  CloseFmsThreads();
  fmsthread = new FmsThread[a_CI->GetMaxThreads()];

  if ( !fmsthread )	{
   	NumFmsThreads=0;
    return false; }

  NumFmsThreads=a_CI->GetMaxThreads();
		for ( int i = 0; i < NumFmsThreads; i++)
		  	fmsthread[i].a_CI = a_CI;

  return true;
}


void FE2::ResetAllThreads()
{
	CloseFmsThreads();
}


void FE2::FreeFmsThreads()
{
 	if(fmsthread)
    delete[] fmsthread;

  fmsthread=0;
  NumFmsThreads=0;
}

/*******************************************************************************
* Name: GetClouds
* Desc: Find the percent of cloud cover in the wind data table for
*        the specified date and time.
*   In: date......Julian Date
*       hour......military 0000->2300

      This function is NOT used anymore, see it's replacement
       CI::GetCloud()
**
******************************************************************************/
long FE2::GetClouds(long date, double hour)
{
long cloud;
long count, month, xmonth, xday, day, hhour, xdate;

  count = 0;

  do	{
    count++;
		  month = a_CI->GetWindMonth(StationNumber, count);
		  day = a_CI->GetWindDay(StationNumber, count);
    if ( month < 13 )
      xmonth = a_CI->GetJulianDays(month);
    else {
     	day = 0;
      xmonth = date; }

		  xdate = xmonth + day;

	 }	while (xdate != date);


 	if ( month != 13)	{					// if hit end of windspeed data
	  	xday = day;
	  	hhour = a_CI->GetWindHour(StationNumber, count);
	  	while ( hour >= hhour ) {
      count++;
			   day = a_CI->GetWindDay(StationNumber, count);
			   xmonth = a_CI->GetWindMonth(StationNumber, count);
		   	hhour = a_CI->GetWindHour(StationNumber, count);
			   if ( day > xday || xmonth > month)
			    	break;
		  }
  }

  	count--;
	  cloud = a_CI->GetWindCloud (StationNumber, count);
	  cloud = std::min(cloud, (long)99);
   CloudCount=count;




   return cloud;
}

/*************************************************************************
* Name: DMH_FindRec
* Desc: Find a DeadMoistureHistory record in the linked list
*   In: FuelSize...tells which list to look in
*       d_Time...ending time (SimTime) in record
*  Ret: address of record,
* NOTE: this should never return a NULL it would mean a logic
*       error in the calling function.
*************************************************************************/
DeadMoistureHistory *FE2::DMH_FindRec (long FuelSize, double d_Time)
{
DeadMoistureHistory *adr;
long sn = 0;

  adr = FirstHist[sn][FuelSize];

 if ( adr == NULL )
    return NULL;

  while (1) {                  /* Go thru list */
    if ( d_Time == adr->SimTime )
       return adr;
    adr = (DeadMoistureHistory *) adr->next;
    if ( adr == NULL )
      return  NULL;  }

  return NULL;
}

/**************************************************************************************
* Name: AllocHistory
* Desc: Put a DeadMoistureHistory Struct into the specified list
*       There are lists for each each Station Fuel Size combination.
*
**************************************************************************************/
//int static alN = 0;
bool FE2::AllocHistory (long Station, long FuelSize)
{

/* First one in list */
  if ( FirstHist [Station][FuelSize] == 0 ) {
    FirstHist [Station][FuelSize] = new DeadMoistureHistory;
    CurHist   [Station][FuelSize] = FirstHist[Station][FuelSize];}

/* Go thru list and add to end */
  else {
    while ( CurHist[Station][FuelSize]->next != NULL) {
       NextHist[Station][FuelSize] = (DeadMoistureHistory *) CurHist[Station][FuelSize]->next;
       CurHist[Station][FuelSize] = NextHist[Station][FuelSize]; };

    CurHist[Station][FuelSize]->next = (DeadMoistureHistory *) new DeadMoistureHistory;
    NextHist[Station][FuelSize]      = (DeadMoistureHistory *) CurHist[Station][FuelSize]->next;
    CurHist[Station][FuelSize]       = NextHist[Station][FuelSize];
  }

  CurHist[Station][FuelSize]->Moisture = new double[MxDesc.NumAlloc[Station][FuelSize]];
  if ( CurHist[Station][FuelSize]->Moisture == NULL)
     	return false;

  memset (CurHist[Station][FuelSize]->Moisture, 0x0, MxDesc.NumAlloc[Station][FuelSize] * sizeof(double));
  for(int i = 0; i < MxDesc.NumAlloc[Station][FuelSize]; i++)
		CurHist[Station][FuelSize]->Moisture[i] = -1.0;
	 CurHist[Station][FuelSize]->next = 0;

 	MxDesc.NumHist[Station][FuelSize]++;

// printf (" FuelSize: %d, Cnt: %d \n", FuelSize,	MxDesc.NumHist[Station][FuelSize]);

  return true;
}

/*************************************************************************
* Name:
* Desc:
*      FE2.MxDesc is a struct DeadMoistureDescription
*************************************************************************/
void FE2::RefreshHistoryDescription(long Station, long FuelSize)
{
   if ( NumStations == 0 )
     	return;

   MxDesc.NumStations=NumStations;
	  if ( Stations[0].NumFuels[FuelSize] == 0 )
     	return;

   MxDesc.NumElevs[Station][FuelSize]  = Stations[0].FMS[FuelSize][0]->NumElevs;
	  MxDesc.NumSlopes[Station][FuelSize] = Stations[0].FMS[FuelSize][0]->Fms_Elev[0]->NumSlopes;
	  MxDesc.NumAspects[Station][FuelSize]= Stations[0].FMS[FuelSize][0]->Fms_Elev[0]->Fms_Slope[0]->NumAspects;
	  MxDesc.NumCovers[Station][FuelSize] = Stations[0].FMS[FuelSize][0]->Fms_Elev[0]->Fms_Slope[0]->Fms_Aspect[0]->NumFms;
	  MxDesc.NumFuels[Station][FuelSize]  = Stations[0].NumFuels[FuelSize];

   //CopyMemory (MxDesc.FuelKey[Station][FuelSize], Stations[0].FuelKey[FuelSize], MAXNUM_FUEL_MODELS*sizeof(long));
   memcpy (MxDesc.FuelKey[Station][FuelSize], Stations[0].FuelKey[FuelSize], MAXNUM_FUEL_MODELS*sizeof(long));

 	MxDesc.NumAlloc[Station][FuelSize] = MxDesc.NumFuels[Station][FuelSize] * MxDesc.NumElevs[Station][FuelSize] *
                                       MxDesc.NumSlopes[Station][FuelSize] * MxDesc.NumAspects[Station][FuelSize] *
                                       MxDesc.NumCovers[Station][FuelSize];
   MxDesc.EndTime[Station][FuelSize] = 0.0;

   if ( FuelSize == 1 ) {    /* if 10 Hour,   [0] = 1 Hr,  [1] = 10 Hr  */
      MxDesc.NumElevs[Station][0]   = MxDesc.NumElevs[Station][1];
	     MxDesc.NumSlopes[Station][0]  = MxDesc.NumSlopes[Station][1];
	     MxDesc.NumAspects[Station][0] = MxDesc.NumAspects[Station][1];
	     MxDesc.NumCovers[Station][0]  = MxDesc.NumCovers[Station][1];
	     MxDesc.NumFuels[Station][SIZECLASS_1HR] = MxDesc.NumFuels[Station][1];
	     MxDesc.NumAlloc[Station][0] = MxDesc.NumAlloc[Station][SIZECLASS_10HR]; }

// change 10-29-09 - see comments in struct DeadMoistureDescription
//   if ( MxDesc.fms )
//     	delete[] MxDesc.fms;
//   MxDesc.fms=0;
}

/*****************************************************************/
bool FE2::CheckMoistureTimes(double SimTime)
{
long i, j;

   for (i=0; i<NumStations; i++) {
	    for (j=1; j<NUM_FUEL_SIZES; j++) {
	      CheckMoistureHistory(i, j, SimTime);
       if (j==1)
          CheckMoistureHistory(i, 0, SimTime); }}

   return true;
}

/****************************************************************************
* Name:CheckMoistureHistory
* Desc: Check and move the moisture values in the Moist Hist record for
*        the specified time into the Map
* Note-1 Modified so that if the requested SimTime isn't in the Moist Hist
*         list the last one in the list gets used. SO THAT 'return false'
*         should never happen, I had a problem with Farsite doing this because
*         didn't have Farsite running the Conditioning long enough, which
*         I fixed, BUT I still want to make sure this doesn't return false
*         BECAUSE it would cause a crash with an NULL CurrHist[]
* NOTE: This function is weird see notes in code.
*        AND I'm not sure that we even want or need this function getting
*         call while we run the Conditioning, it gets call to retrieve the
*         moistures which is relevant.
*
*****************************************************************************/
bool FE2::CheckMoistureHistory(long Station, long FuelSize, double SimTime)
{
long i;

/* All this EnvironmentChanged business, I think could probably get yanked */
	  if ( a_CI->EnvironmentChanged(GETVAL, Station, FuelSize))
     	return false;

/* I'm not sure this is even relevant or we want it to return a false */
  if ( SimTime > MxDesc.EndTime[Station][FuelSize])
     	return false;

   if ( MxDesc.NumFuels[Station][FuelSize] == 0 )
     	return true;

   if ( MxDesc.NumHist[Station][FuelSize] == 0 )
     	return false;

   if (SimTime >= CurHist[Station][FuelSize]->LastTime && SimTime <= CurHist[Station][FuelSize]->SimTime)	// if within current time span
     	return true;

   DeadMoistureHistory *lasthist, *thishist;

/* NOTE - both of the following 'for' loops start at the beginning of the Hist */
/*  list, so i'm not sure what the point is */


// test-larry
// printf ("-->%6.1f  ", SimTime);
// test-larry

   if ( SimTime > CurHist[Station][FuelSize]->SimTime ) {
    	CurHist[Station][FuelSize] = lasthist = FirstHist[Station][FuelSize];

     for ( i = 0; i < MxDesc.NumHist[Station][FuelSize]; i++ )	{

      if ( (SimTime >= CurHist[Station][FuelSize]->LastTime &&
            SimTime <= CurHist[Station][FuelSize]->SimTime)  ||
            CurHist[Station][FuelSize]->next == NULL ) {   /* See Note-1 above */
         thishist = CurHist[Station][FuelSize];
	        CurHist[Station][FuelSize] = lasthist;
	        CopyMoistureHistory (Station, FuelSize);
     	   CurHist[Station][FuelSize] = thishist;
         CopyMoistureHistory(Station, FuelSize);
         break;  }

      lasthist = CurHist[Station][FuelSize];
      NextHist[Station][FuelSize] = (DeadMoistureHistory *) CurHist[Station][FuelSize]->next;
      CurHist[Station][FuelSize] = NextHist[Station][FuelSize];
     }  /* for i */

/* See Note-1 above */
     if ( i == MxDesc.NumHist[Station][FuelSize])	// this will indicate if SimTime is greater than the
       return false;				// data in the History
   }

// if SimTime is earlier than current, start from 0 and work forward until current
   else if ( SimTime < CurHist[Station][FuelSize]->LastTime) {
     CurHist[Station][FuelSize] = lasthist = FirstHist[Station][FuelSize];

     for ( i = 0; i < MxDesc.NumHist[Station][FuelSize]; i++) {
       if ( SimTime >= CurHist[Station][FuelSize]->LastTime &&
            SimTime <= CurHist[Station][FuelSize]->SimTime )  {
         thishist = CurHist[Station][FuelSize];
	        CurHist[Station][FuelSize] = lasthist;
	        CopyMoistureHistory(Station, FuelSize);
     	   CurHist[Station][FuelSize] = thishist;
         CopyMoistureHistory(Station, FuelSize);
         break;  }

       lasthist = CurHist[Station][FuelSize];
       NextHist[Station][FuelSize] = (DeadMoistureHistory *) CurHist[Station][FuelSize]->next;
       CurHist[Station][FuelSize] = NextHist[Station][FuelSize];
     }
   }

	return true;
}


/*******************************************************************/
void FE2::CopyMoistureHistory(long Station, long FuelSize)
{
long i, j, k, m, n, p, loc;
long NFuel;
double Moist;

  NFuel=MxDesc.NumFuels[Station][FuelSize];
  i=Station;
  switch (FuelSize)  {
    case SIZECLASS_1HR:		// 1hour
      for ( j=0; j<NFuel; j++) {
        for ( k=0; k < MxDesc.NumElevs[i][FuelSize]; k++)	{
          for ( m=0; m < MxDesc.NumSlopes[i][FuelSize]; m++)	{
				        for ( n=0; n < MxDesc.NumAspects[i][FuelSize]; n++)	{
				          for ( p=0; p < MxDesc.NumCovers[i][FuelSize]; p++)	{
                if ( CurHist[i][FuelSize]->LastTime == 0.0  )
                  Stations[i].FMS[SIZECLASS_10HR][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->LastEq[p] =
	                      ((double) a_CI->GetInitialFuelMoisture (Stations[i].FMS[SIZECLASS_10HR][j]->Fuel, FuelSize)) / 100.0;
	               else
                  Stations[i].FMS[SIZECLASS_10HR][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->LastEq[p]
                    = Stations[i].FMS[SIZECLASS_10HR][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->NextEq[p];
                loc =	j * MxDesc.NumElevs[i][FuelSize] * MxDesc.NumSlopes[i][FuelSize] * MxDesc.NumAspects[i][FuelSize] * MxDesc.NumCovers[i][FuelSize]
                    + k * MxDesc.NumSlopes[i][FuelSize] * MxDesc.NumAspects[i][FuelSize] * MxDesc.NumCovers[i][FuelSize]
                    + m * MxDesc.NumAspects[i][FuelSize] * MxDesc.NumCovers[i][FuelSize]
                    +	n * MxDesc.NumCovers[i][FuelSize]
                    + p;
               	Stations[i].FMS[SIZECLASS_10HR][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->NextEq[p]
                 = CurHist[i][FuelSize]->Moisture[loc];                                               }}}}}
     break; /* SIZECLASS_1Hr */

     default: 	//10,,100, 1000 hr
       for (j=0; j < NFuel; j++) {
        	for (k=0; k < MxDesc.NumElevs[i][FuelSize]; k++){
	          for (m=0; m < MxDesc.NumSlopes[i][FuelSize]; m++)	{
			       		for (n=0; n < MxDesc.NumAspects[i][FuelSize]; n++)	{
					         for (p=0; p < MxDesc.NumCovers[i][FuelSize]; p++)	{
               	if ( CurHist[i][FuelSize]->LastTime == 0.0 )	{
                     if ( FuelSize < SIZECLASS_1000HR)
	                   Stations[i].FMS[FuelSize][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->LastMx[p]
                      = ((double) a_CI->GetInitialFuelMoisture(Stations[i].FMS[FuelSize][j]->Fuel, FuelSize))/100.0;
							          	else if ( FuelSize == SIZECLASS_1000HR)
 	                  Stations[i].FMS[FuelSize][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->LastMx[p]
                     = ((double) a_CI->GetInitialFuelMoisture(Stations[i].FMS[FuelSize][j]->Fuel, SIZECLASS_100HR))/100.0;
                  else {
									           Moist = a_CI->GetWoodyFuelMoisture(Stations[i].FMS[FuelSize][j]->Fuel, FuelSize) / 100.0;
									          	if ( Moist <= 0.10)
                       Moist = 0.10;
	                   Stations[i].FMS[FuelSize][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->LastMx[p]=Moist;
                  } /* else */
                } /* if */
	               else
                  Stations[i].FMS[FuelSize][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->LastMx[p]
                   = Stations[i].FMS[FuelSize][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->NextMx[p];
                loc = j * MxDesc.NumElevs[i][FuelSize] * MxDesc.NumSlopes[i][FuelSize] * MxDesc.NumAspects[i][FuelSize] * MxDesc.NumCovers[i][FuelSize]
                    +	k * MxDesc.NumSlopes[i][FuelSize] * MxDesc.NumAspects[i][FuelSize] * MxDesc.NumCovers[i][FuelSize]
                    + m * MxDesc.NumAspects[i][FuelSize] * MxDesc.NumCovers[i][FuelSize]
                    + n * MxDesc.NumCovers[i][FuelSize]
                    + p;

                Stations[i].FMS[FuelSize][j]->Fms_Elev[k]->Fms_Slope[m]->Fms_Aspect[n]->NextMx[p]
                 = CurHist[i][FuelSize]->Moisture[loc];
                Stations[i].FMS[FuelSize][j]->FirstTime = CurHist[i][FuelSize]->LastTime;
                Stations[i].FMS[FuelSize][j]->LastTime = CurHist[i][FuelSize]->SimTime;
                if ( FuelSize == SIZECLASS_10HR) {
                  elevref = CurHist[i][FuelSize]->Elevation;
	                 tempref = CurHist[i][FuelSize]->AirTemperature;
     	            humref = CurHist[i][FuelSize]->RelHumidity;}
            }}}}} /* For */
     	 break;
     } /* switch (fuelsize) */
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//   FuelMoistureMap functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
FuelMoistureMap::FuelMoistureMap(CI *_a_CI)
{
 	a_CI = _a_CI;
  long i;

/* Fixit - get rid of ZeroMemory that just be in a loop */
	 for ( i = 0; i < 4; i++)
	 {
        NumFuels[i]=0;
        //ZeroMemory (FuelKey[i], MAXNUM_FUEL_MODELS * sizeof(long));
        memset(FuelKey[i], 0x0, MAXNUM_FUEL_MODELS *sizeof(long));
        FMS[i]=0;
    }

  FmTolerance[0][0]=1;
  FmTolerance[0][1]=2;
  FmTolerance[0][2]=4;
  FmTolerance[0][3]=4;
  FmTolerance[1][0]=3;
  FmTolerance[1][1]=4;
  FmTolerance[1][2]=8;
  FmTolerance[1][3]=8;
  FmTolerance[2][0]=5;
  FmTolerance[2][1]=6;
  FmTolerance[2][2]=10;
  FmTolerance[2][3]=10;
  FmTolerance[3][0]=8;
  FmTolerance[3][1]=10;
  FmTolerance[3][2]=15;
  FmTolerance[3][3]=15;
  for ( i = 0; i < NUM_FUEL_SIZES; i++ )
     CuumRain[i] = 0.0;
}


FuelMoistureMap::~FuelMoistureMap()
{
	long i;

     for(i=0; i<4; i++)
		FreeFuels(i);
}

/***********************************************************************/
void FuelMoistureMap::SearchCondenseFuels(long FuelSize)
{
// condense the number of fuel types used based on tolerance for the
// initial moisture content of each size class
//-> Can probably make the criteria sensitive to the timespan of the simulation
//-> where differences in 1hr fuels are ignored after ~6 hours and diff in 10hr fuels
//-> are ignored after 3 days or so.

long i, j, num, FuelOrder;
long t10;
long t10a;
long FmTol[4];

// don't need to worry about initial conditions if using a conditioning period
//if(UseConditioningPeriod(GETVAL))
//	CopyMemory(FmTol, FmTolerance[3], 3*sizeof(long));
// else	 // if no conditioning period, then tolerances are needed
// {

  switch ( GetFmTolerance()) {
     case FM_HARDASS:
		     //CopyMemory(FmTol, FmTolerance[0], 4*sizeof(long));
 		     memcpy(FmTol, FmTolerance[0], 4*sizeof(long));
      	break;
	    case FM_MODERATE:
		    	memcpy(FmTol, FmTolerance[1], 4*sizeof(long));
       break;
	    case FM_LIBERAL:
		 	   memcpy(FmTol, FmTolerance[2], 4*sizeof(long));
      	break;
	    case FM_SLOPPY:
		    	memcpy(FmTol, FmTolerance[3], 4*sizeof(long));
       	break;
   }

	FuelOrder=1;

	if ( FuelSize <= SIZECLASS_1000HR )	{
	  	for ( i=0; i < MAXNUM_FUEL_MODELS; i++)	{
			   if ( FuelKey[FuelSize][i] < FuelOrder)
			    	continue;
		   	FuelKey[FuelSize][i]=FuelOrder;
			   num = i+1;      // Fuel Model Number is not zero based
			   if ( FuelSize < SIZECLASS_1000HR)
				    t10 = a_CI->GetInitialFuelMoisture(num, FuelSize);
			   else if (FuelSize == SIZECLASS_1000HR)
			    	t10 = a_CI->GetInitialFuelMoisture(num, SIZECLASS_100HR);

			   for (j=0; j<MAXNUM_FUEL_MODELS; j++)	{
				    if (j==i)
				     	continue;
				    if ( FuelKey[FuelSize][j] < FuelOrder)
			     		continue;
				    num = j+1;
				    if ( FuelSize < SIZECLASS_1000HR)
		     			t10a=a_CI->GetInitialFuelMoisture(num, FuelSize);
			    	else
		     			t10a = a_CI->GetInitialFuelMoisture (num, SIZECLASS_100HR);
					   if ( t10 > t10a - FmTol[FuelSize] && t10 < t10a+FmTol[FuelSize])	{
					    	FuelKey[FuelSize][j]=FuelKey[FuelSize][i];     // set to same
						    NumFuels[FuelSize]--;	}
		  	}  /* for j */

     FuelOrder++;
			  if ( FuelOrder > NumFuels[FuelSize])
			   	break;
		} /* for i */
	}  /* if FuelSize <= SIZECLASS_1000HR */

  else {
	   for ( i=0; i<MAXNUM_COARSEWOODY_MODELS; i++)	{
	     if ( FuelKey[FuelSize][i] < FuelOrder)
		    		continue;
	     FuelKey[FuelSize][i]=FuelOrder;
	     num=i+1;      // Fuel Model Number is not zero based
			   t10 = a_CI->GetWoodyFuelMoisture(num, FuelSize);
      for ( j=0; j < MAXNUM_COARSEWOODY_MODELS; j++)	{
         if (j==i)
           continue;
	        if ( FuelKey[FuelSize][j] < FuelOrder)
     	     continue;
	     	  num = j+1;
			     	t10a = a_CI->GetWoodyFuelMoisture (num, FuelSize);
	        if (t10>t10a-FmTol[SIZECLASS_100HR] && t10<t10a+FmTol[SIZECLASS_100HR]) {
            FuelKey[FuelSize][j]=FuelKey[FuelSize][i];     // set to same
        		  NumFuels[FuelSize]--; }
    		 }  /* for j */
       FuelOrder++;
	      if(FuelOrder>NumFuels[FuelSize])
  		     	break;
	    }
   }
}



/****************************************************************
* Name: FuelMoistureMap
* Desc: For the specified Fuel Size load the convert condensed
*       fuel types into the FuelKey FuelMoistureMap.FuelKey
*       A Moisture Map can also get build, depending - see code.
* Note-1; This code never gets used, it would blowup if it did
*         because there is no  FuelKey[FuelSize] for coarse wood
*         only 0 -> 3 for each fuel size. I'm leaving this
*         code in place in case it's needed for doing coarse someday.
*****************************************************************/
bool FuelMoistureMap::ReAllocFuels(long FuelSize)
{
bool UPDATE;
long i, j, k, fueltype, oldnumfuels, numfuels;
long NewKey[MAXNUM_FUEL_MODELS];
long FuelCats[256];

//   printf ("Siz %d - ReAllocFuels()\n", FuelSize);

   memset(FuelCats, 0x0, 256 * sizeof(long));

// reindex the new fuel types based on fuel moisture
   if ( FuelSize <= SIZECLASS_1000HR)
      numfuels = a_CI->GetFuelCats(e_FuelModel, FuelCats);

   else {
      numfuels = a_CI->GetFuelCats (e_Woody, FuelCats);
     	if ( numfuels == 0)
        return false; }

/* Fixit - CopyMemory and ZeroMemory should be in for loops */
   //CopyMemory(NewKey, FuelKey[FuelSize], MAXNUM_FUEL_MODELS*sizeof(long));
   memcpy(NewKey, FuelKey[FuelSize], MAXNUM_FUEL_MODELS*sizeof(long));
   //ZeroMemory(FuelKey[FuelSize], MAXNUM_FUEL_MODELS*sizeof(long));
   memset(FuelKey[FuelSize], 0x0, MAXNUM_FUEL_MODELS*sizeof(long));
   k=1;

   for ( i = 0; i <= numfuels; i++) {
     if ( FuelSize <= SIZECLASS_1000HR )	{
       fueltype = FuelCats[i];    //GetLandscapeTheme()->AllCats[3][i];
		    	fueltype = a_CI->GetFuelConversion(fueltype);   // see if there are conversions

	     	if ( fueltype < MAXNUM_FUEL_MODELS+1 && fueltype > 0 )	{
	        if ( FuelKey[FuelSize][fueltype-1] == 0 )
            FuelKey[FuelSize][fueltype-1] = k++; }}

/*  NOTE: else for the if ( FuelSize <= SIZECLASS_1000HR ) SEE NOTE-1 above */
     else	{
   	   fueltype = FuelCats[i];        //GetLandscapeTheme()->AllCats[9][i];
			    if ( fueltype < MAXNUM_COARSEWOODY_MODELS && fueltype > 0) {
	        if ( FuelKey[FuelSize][fueltype-1] == 0)
            FuelKey[FuelSize][fueltype-1]=k++; } }

   } /* for i */

   numfuels = k - 1;				             	// new potential number, not condensed
   oldnumfuels = NumFuels[FuelSize]; 	// save the previous number
   NumFuels[FuelSize] = numfuels;    	// set new number for condensation scan
   SearchCondenseFuels (FuelSize);

   UPDATE = false;
   if ( oldnumfuels != NumFuels[FuelSize]) // if original number is not the new number
     	UPDATE = true;
   else {
    	for (i = 0; i < MAXNUM_FUEL_MODELS; i++) {
       if ( FuelKey[FuelSize][i] != NewKey[i] ) {
         UPDATE = true;
         break;}}}

   if ( !UPDATE )
     	return false;

/* Looks like if this code gets used it will build a Map */
   else {
     numfuels = NumFuels[FuelSize];		      // save new number
     NumFuels[FuelSize] = oldnumfuels;	    // replace old one for freeing
     memcpy (NewKey, FuelKey[FuelSize], MAXNUM_FUEL_MODELS * sizeof(long));
     FreeFuels (FuelSize);         	     	 // Free Fuels erases all fuel types
     memcpy (FuelKey[FuelSize], NewKey, MAXNUM_FUEL_MODELS*sizeof(long));
     NumFuels[FuelSize] = numfuels;        // restore new number of fuel types
	   	FMS[FuelSize] = new FMS_Elevations*[numfuels];
     long next=1;

    	for ( i=0; i < NumFuels[FuelSize]; i++) {
        FMS[FuelSize][i] = new FMS_Elevations (FuelSize, a_CI);
        for ( j = 0; j < MAXNUM_FUEL_MODELS; j++) {
	          if ( FuelKey[FuelSize][j]==next) {
	             FMS[FuelSize][i]->SetFuel(j+1);
	     		     	FMS[FuelSize][i]->AllocElevations();
              next++;
              break; }
           else if ( next > NumFuels[FuelSize] )
              break;  }  } /* for i j */

   } /* else */

  return true;
}

/*************************************************************
* Name: AllocFuels
* Desc: Build a Fuel Mositure Map for the specified FuelSize
*       FuelMoistureMap are the Stations in FE2 class,
*       FuelMoistureMap contains the map of, elev, aspect,
*         slope, cover, stick
*   In: FuelSize - see define SIZECLASS_* in fmc_fe2.h
**************************************************************/
bool FuelMoistureMap::AllocFuels(long FuelSize)
{
long i, j, k, next, fueltype;
long FuelCats[256];

  memset (FuelCats, 0x0, 256 * sizeof(long));
  FreeFuels(FuelSize);
  if ( FuelSize <= SIZECLASS_1000HR) {
    NumFuels[FuelSize] = a_CI->GetFuelCats(e_FuelModel, FuelCats); }
  else {
		  NumFuels[FuelSize] = a_CI->GetFuelCats(e_Woody, FuelCats);
    if (NumFuels[FuelSize]==0)
      return false; }
 	k=1;

	 for ( i=0; i <= NumFuels[FuelSize]; i++) {
    if (FuelSize <= SIZECLASS_1000HR) {
     	fueltype = FuelCats[i];            //GetLandscapeTheme()->AllCats[3][i];

			   fueltype = a_CI->GetFuelConversion(fueltype);   // see if there are conversions

			   if ( fueltype < 257 && fueltype > 0)	{
        if ( FuelKey[FuelSize][fueltype-1] == 0 )
          FuelKey[FuelSize][fueltype-1]=k++; } }
        else 	{
          fueltype = FuelCats[i];           //GetLandscapeTheme()->AllCats[9][i];
		        if ( fueltype < MAXNUM_COARSEWOODY_MODELS && fueltype > 0 )	{
 	           if ( FuelKey[FuelSize][fueltype-1] == 0 )
             FuelKey[FuelSize][fueltype-1]=k++;} }
  } /* for i */

	 NumFuels[FuelSize] = k - 1;
  SearchCondenseFuels (FuelSize);
  if ( NumFuels[FuelSize] == 0 )
    return true;

  next = 1;

  FMS[FuelSize] = new FMS_Elevations * [NumFuels[FuelSize]];
  for ( i = 0; i < NumFuels[FuelSize]; i++)	{
    FMS[FuelSize][i] = new FMS_Elevations(FuelSize, a_CI);
    for ( j = 0; j < MAXNUM_FUEL_MODELS; j++) {
      if ( FuelKey[FuelSize][j] == next ) {
	       FMS[FuelSize][i]->SetFuel (j+1);
     			FMS[FuelSize][i]->AllocElevations();
        next++;
        break; }
    }
  }

 return true;
}


/**************************************************************************/
void FuelMoistureMap::FreeFuels(long FuelSize)
{
long i;

  for ( i = 0; i < NumFuels[FuelSize]; i++) {
	   	FMS[FuelSize][i]->FreeElevations();
	    delete FMS[FuelSize][i]; }

  if ( NumFuels[FuelSize]>0) {
    if ( FMS[FuelSize] )
		     delete[] FMS[FuelSize];
	   FMS[FuelSize]=0; }

  NumFuels[FuelSize]=0;
/* Fixit ZeroMemory should be using a loop instead */
  //ZeroMemory(FuelKey[FuelSize], MAXNUM_FUEL_MODELS * sizeof(long));
  memset(FuelKey[FuelSize], 0x0, MAXNUM_FUEL_MODELS * sizeof(long));
  CuumRain[FuelSize] = 0.0;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//   FMS_Elevation functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
FMS_Elevations::FMS_Elevations(long fuelsize, CI *_a_CI)
{
	 Fms_Elev = 0;
  NumElevs = 0;
  Fuel = 0;
  FirstTime = 0.0;
  LastTime = 0.0;
  FuelSize = fuelsize;
	 a_CI = _a_CI;
}

/**************************************************************************/
bool FMS_Elevations::AllocElevations()
{
	  FreeElevations();

long i;
double conv=1.0;

   if (a_CI->GetTheme_Units(E_DATA)==1)      // if feet, convert to meters
     conv = 0.3048;

    NumElevs = ((a_CI->GetHiElev() - a_CI->GetLoElev()) * conv) / a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_ELEV) + 1;
    Fms_Elev = new FMS_Slopes * [NumElevs];   //(FMS_Slopes **) GlobalAlloc(GMEM_FIXED, NumElevs*sizeof(FMS_Slopes *));
    if (Fms_Elev==NULL)
    	 	return false;
    for ( i = 0; i < NumElevs; i++) {
       Fms_Elev[i] = new FMS_Slopes (FuelSize, a_CI);
      	if ( Fms_Elev[i] == NULL)
         	return false;
       else {
          Fms_Elev[i]->Elev = i * a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_ELEV)
                                +  a_CI->GetLoElev() * conv;
          Fms_Elev[i]->Fuel = Fuel; }
	      if ( ( Fms_Elev[i]->AllocSlopes() ) == false)
     			  return false;
     }

   return true;
}

void FMS_Elevations::SetFuel(long fuel)
{
	Fuel=fuel;
}

void FMS_Elevations::FreeElevations()
{
	 if(Fms_Elev==0)
     	return;

     long i;

     for(i=0; i<NumElevs; i++)
     {    if(Fms_Elev[i]==0)
     		continue;
     	Fms_Elev[i]->FreeSlopes();
   		delete Fms_Elev[i];
          Fms_Elev[i]=0;
     }
	delete[] Fms_Elev;//GlobalFree(Fms_Elev);
     Fms_Elev=0;
     NumElevs=0;
     Fuel=0;
     FirstTime=0.0;
     LastTime=0.0;
}



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//   FMS_Slope functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

FMS_Slopes::FMS_Slopes(long fuelsize, CI *_a_CI)
{
	 NumSlopes=0;
  Fms_Slope=0;
  FuelSize=fuelsize;
	 a_CI = _a_CI;
}


bool FMS_Slopes::AllocSlopes()
{
long i;

	 LoVal=a_CI->GetTheme_LoValue(S_DATA);
  HiVal=a_CI->GetTheme_HiValue(S_DATA);

  double fraction, ipart;
  double slopef;

  if (a_CI->GetTheme_Units(S_DATA)==1) {
     LoVal = slopef = atan((double) LoVal/100.0)/PI*180.0;
     fraction=modf(slopef, &ipart);
     if(fraction>=0.5)
     	 	LoVal++;
	   	HiVal = slopef = atan((double) HiVal/100.0)/PI*180.0;
     fraction=modf(slopef, &ipart);
     if(fraction>=0.5)
     		HiVal++; }

   if ( LoVal > 50)
     	LoVal=50;
   if ( HiVal > 50)
     	HiVal=50;
   NumSlopes = (HiVal-LoVal) / a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE) + 1;

   if (NumSlopes==0)
     	return false;

   Fms_Slope = new FMS_Aspects * [NumSlopes];
   if (Fms_Slope==0)
     	return false;

   for ( i=0; i<NumSlopes; i++) {
     Fms_Slope[i] = new FMS_Aspects(FuelSize, a_CI);
     if (Fms_Slope[i]!=0) {
	       Fms_Slope[i]->Slope = i * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE)+LoVal;
        Fms_Slope[i]->Elev=Elev;
        Fms_Slope[i]->Fuel=Fuel; }
     else
        return false;
     if ((Fms_Slope[i]->AllocAspects())==false)
        return false;
    }

  return true;
}

void FMS_Slopes::FreeSlopes()
{
	long i;

	for(i=0; i<NumSlopes; i++)

     {	Fms_Slope[i]->FreeAspects();

     	delete Fms_Slope[i];

     }

     delete[] Fms_Slope;
}



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//   FMS_Aspect functions
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


FMS_Aspects::FMS_Aspects(long fuelsize, CI *_a_CI)
{
	NumAspects=0;
     Fms_Aspect=0;
     FuelSize=fuelsize;
	 a_CI = _a_CI;
}

/*************************************************************************/
bool FMS_Aspects::AllocAspects()
{
long i;

	  LoVal = 0;              // GetTheme_LoValue(A_DATA);
   HiVal = 360;            // GetTheme_HiValue(A_DATA);
   NumAspects = (HiVal-LoVal) / a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_ASP)+1;

   if (NumAspects==0)
     	return false;

   Fms_Aspect = new FMS_Cover * [NumAspects];
   if (Fms_Aspect==0)
     	return false;

   for ( i=0; i < NumAspects; i++) {
	     Fms_Aspect[i]=new FMS_Cover(FuelSize, a_CI);
      if (Fms_Aspect[i] != 0) {
         Fms_Aspect[i]->Aspect = i * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP)+LoVal;
         Fms_Aspect[i]->Slope = Slope;
         Fms_Aspect[i]->Elev = Elev;
         Fms_Aspect[i]->Fuel = Fuel; }
       else
         return false;
       if (( Fms_Aspect[i]->AllocCover()) == false)
         return false;
     }

     return true;
}

// **********************************************************************
void FMS_Aspects::FreeAspects()
{
long i;

 	for(i=0; i<NumAspects; i++)  {
    Fms_Aspect[i]->FreeCover();
    delete Fms_Aspect[i];
  }
  delete[] Fms_Aspect;
}


// ******************************************************************************
FMS_Cover::FMS_Cover(long fuelsize, CI *_a_CI)
{
 	 NumFms = 0;
   fms = 0;    /* old 0.7.0 Nelson Fuel Stick Model */

/* Change 10-29-09 */
   dfm = 0;    /* new 1.0.0 Nelson Fuel Stick Model */

   LastMx=0;
   NextMx=0;
   LastEq=0;
   NextEq=0;
   Elev=-1;
   FuelSize=fuelsize;
	  a_CI = _a_CI;
}

// ********************************************************************************
bool FMS_Cover::AllocCover()
{
long i;

	  LoVal = a_CI->GetTheme_LoValue(C_DATA);
   HiVal = a_CI->GetTheme_HiValue(C_DATA);
   if(a_CI->GetTheme_Units(C_DATA) == 0)//classes
   {
	   switch(LoVal)
	   {
	   case 99:
		   LoVal = 0;
		   break;
	   case 1:
		   LoVal = 10;
		   break;
	   case 2:
		   LoVal = 30;
		   break;
	   case 3:
		   LoVal = 60;
		   break;
	   case 4:
		   LoVal = 75;
		   break;
	   default:
		   LoVal = 0;
		   break;
	   }
	   switch(HiVal)
	   {
	   case 99:
		   HiVal = 0;
		   break;
	   case 1:
		   HiVal = 10;
		   break;
	   case 2:
		   HiVal = 30;
		   break;
	   case 3:
		   HiVal = 60;
		   break;
	   case 4:
		   HiVal = 75;
		   break;
	   default:
		   HiVal = 0;
		   break;
	   }
   }
   if ( HiVal > 80 )
     HiVal = 80;                       //percent
   NumFms = ( HiVal - LoVal ) / a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV) + 1;

/* Depending on which Nelson Fuelstick Model we are running */
   if ( a_CI->FuelStickModel_IsNelson_070() ) {
     fms = new Fms * [NumFms];
	    if ( fms == NULL ) {
       NumFms=0;
       return false; }
     for ( i = 0; i < NumFms; i++){
       fms[i]=0;}
   }

/* Change 10-29-09 add this code for new Nelson Fuelstick Model 1.0.0  */
   else {
     dfm = new DeadFuelMoisture * [NumFms];
     if ( dfm == NULL ) {
       NumFms=0;
       return false;}
     for ( i = 0; i < NumFms; i++) {
       dfm[i] = 0; }
   } /* else */
/* ....................................................... */

   LastMx = new double[NumFms];    //(double *) GlobalAlloc(GMEM_FIXED, NumFms*sizeof(double));
	  if ( LastMx == NULL ) {
     NumFms=0;
     return false;}
   for ( i = 0; i < NumFms; i++){
     LastMx[i]=0.0;}

   NextMx = new double[NumFms];    //(double *) GlobalAlloc(GMEM_FIXED, NumFms*sizeof(double));
	  if ( NextMx == NULL ){
	    NumFms = 0;
     return false; }
   for (i=0; i<NumFms; i++)
     NextMx[i]=0.0;

   LastEq=new double[NumFms];   //(double *) GlobalAlloc(GMEM_FIXED, NumFms*sizeof(double));
	  if ( LastEq==NULL)	{
     NumFms = 0;
     return false;}
   for ( i = 0; i < NumFms; i++)
     LastEq[i]=0.0;
     NextEq = new double[NumFms];  //(double *) GlobalAlloc(GMEM_FIXED, NumFms*sizeof(double));

  	if ( NextEq == NULL ) {
    	NumFms=0;
     return false;}
   for ( i = 0; i < NumFms; i++)
     NextEq[i]=0.0;

  	SolRad = new double[NumFms];
  	if ( SolRad == NULL ){
     NumFms = 0;
     return false;}
   for ( i = 0; i < NumFms; i++)
     SolRad[i]=0.0;

   return true;
}

void  FMS_Cover::FreeCover()
{
long i;
	 if ( fms ) {
    for ( i = 0; i < NumFms; i++) {
       if ( fms[i] )
	        Fms_Destroy ( fms[i] ); }
    delete[] fms;  }

/* Change 10-29-09 newer Deadfuelmoisutre class */
 if ( dfm ) {
   for ( i = 0; i < NumFms; i++) {
      if ( dfm[i] )
       delete dfm[i];}
   delete[] dfm;
   dfm = 0; }

	if (LastMx)
     	delete[] LastMx;//GlobalFree(LastMx);
	if (NextMx)
     	delete[] NextMx;//GlobalFree(NextMx);
	if (LastEq)
     	delete[] LastEq;//GlobalFree(LastEq);
	if (NextEq)
     	delete[] NextEq;//GlobalFree(NextEq);
	if (SolRad)
	 	delete[] SolRad;

     fms=0;
     LastMx=0;
     NextMx=0;
     LastEq=0;
     NextEq=0;
     NumFms=0;
}


//------------------------------------------------------------------------------
//
//   FmsThread Functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
FmsThread::FmsThread(CI *_a_CI)
{
	 Begin=End = 0;
  CurHist = 0;
  ThreadOrder = -1;
	 a_CI = _a_CI;
}

FmsThread::~FmsThread()
{
}

// ****************************************************************************
void FmsThread::StartFmsThread(long ID, long sizeclass, DeadMoistureDescription *mxdesc, bool firsttime)
{
X_HANDLE hFmsThread;

    FuelSize = sizeclass;
    FirstTime = firsttime;
    MxDesc = mxdesc;
#ifdef WIN32
    if ( ThreadOrder < 0 ){
      ThreadOrder = ID;
      hFmsEvent = a_CI->GetFarsiteEvent(EVENT_MOIST_THREAD, ThreadOrder);
      SYSTEM_INFO sysinf;
      GetSystemInfo(&sysinf);
      long NumTimes = (long) (ThreadOrder / sysinf.dwNumberOfProcessors);
		    long ProcNum = a_CI->GetStartProcessor() + ThreadOrder;

		    while ( ProcNum >= sysinf.dwNumberOfProcessors )
			      ProcNum -= sysinf.dwNumberOfProcessors;

/* ---> starts FmsThread::UpdateMoistures() */
      hFmsThread=(HANDLE) ::_beginthreadex(NULL, 0, &FmsThread::RunFmsThread, this, CREATE_SUSPENDED, &ThreadID);

		    //unsigned long Affinity =pow(2.0, (int)ProcNum);
		    //SetThreadAffinityMask (hFmsThread, Affinity);
	  SetThreadIdealProcessor(hFmsThread, ProcNum);
			//printf("FmsThread affinity to processor %ld\n", ProcNum);
      ResumeThread(hFmsThread);
      CloseHandle(hFmsThread);
    }
     else
      	SetEvent(hFmsEvent);
#else
    RunFmsThread(this);
#endif
}

// *********************************************************************
unsigned FmsThread::RunFmsThread(void *fmsthread)
{
	  static_cast <FmsThread *> (fmsthread)->UpdateMoistures();
   return 1;
}

// ********************************************************************************
void FmsThread::UpdateMoistures()
{

  do {
    if ( End < 0 )
     	break;

/* Change 10-31-09 */
/* Depending on which Nelson Fuel Stick Model - old or new */

   if ( a_CI->FuelStickModel_IsNelson_070()) /* What Model to use */
		     UpdateMapMoisture();       /* uses Old 0.7.0 Nelson Model */
    else
       UpdateMapMoisture_NewDFM (); /* uses newer 1.0.0 Nelson Model */

#ifdef WIN32
    a_CI->SetFarsiteEvent(EVENT_MOIST, ThreadOrder);
     DWORD ret = WaitForSingleObject(hFmsEvent, INFINITE);
     ResetEvent(hFmsEvent);
#else
    break;
#endif

  }  while ( End > -1 );

#ifdef WIN32
  a_CI->SetFarsiteEvent(EVENT_MOIST, ThreadOrder);   //    SetEvent(hMoistureEvent[ThreadOrder]);
#endif
}

/*****************************************************************************
* Name: UpdateMapMoisture
* Desc: THIS FUNCTION actually calls Fms_Upate() to do the conditioning
*       Runs on each entry in the Fuel Moisture Map (Elev-Slope-Aspect-Cov)
*       Runs Only for the specied Fuel Type and Fuel Size as specified in the
*        FmsThread.FuelType, FmsTread.FuelSize
*       Runs for the specified time interval - NOTE see the actual call below
*        Fms_Update() for the time parameter.
*       Output from Conditioning SEE the code below --->
*         curhist->Moisture[loc]
*         Fms_MeanWtdMoisture(cov->fms[n]);
*
******************************************************************************/
void FmsThread::UpdateMapMoisture()
{
long 	i, j, k, m, n, loc;    //, FuelSize=1;
long		sn = StationNumber;
long 	ElevInc, ElevDiff;
long		Slope, Aspect, Cover;
double 	Radiate, temp, Ctemp, humid, mx, emult=1.0; // 3.2808
double 	ElevRef, TempRef, HumidRef, Rain, d_FueSizInt;
long 	Cloud;                  //, NumNodes, BlockLen;
FMS_Cover *cov;
DeadMoistureHistory *curhist, *nexthist, *onehist;

   	if ( a_CI->GetTheme_Units(E_DATA) == 0 )
	     	emult = 3.2808;        //convert to feet for SiteSpecific

    i = FuelType;  /* This is cute - watch out i gets used */

	   curhist = CurHist[FuelSize];
    if ( FuelSize == SIZECLASS_10HR)
	     onehist = CurHist[0];

/*-----------------------------------------------------------------------------------*/
/* UpdateMapMoistures()  first time thru */
    if (FirstTime) {
       ElevRef = curhist->Elevation;
       TempRef = curhist->AirTemperature;
       HumidRef = curhist->RelHumidity;
       Rain = curhist->Rainfall;
       Cloud = curhist->CloudCover;
      	if (a_CI->GetTheme_Units(E_DATA)==1)
		       	ElevInc = a_CI->GetLoElev()-a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult;	// must be less, so that increment will start at LoElev
       else
          ElevInc = (a_CI->GetLoElev()-a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV))*emult;
       for ( j = 0; j < Begin; j++ )
	          ElevInc += ( a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult);

       if (FuelSize < SIZECLASS_1000HR)
	          mx = ((double) a_CI->GetInitialFuelMoisture (Stations[sn].FMS[FuelSize][i]->Fuel, FuelSize)) / 100.0;
       else if (FuelSize == SIZECLASS_1000HR)
	          mx = ((double) a_CI->GetInitialFuelMoisture(Stations[sn].FMS[SIZECLASS_1000HR][i]->Fuel, SIZECLASS_100HR))/100.0;
		     else	{
           mx = a_CI->GetWoodyFuelMoisture(Stations[sn].FMS[FuelSize][i]->Fuel, FuelSize) / 100.0;
           if (mx <= 0.10)
              mx = 0.10; }

  	    for ( j = Begin; j < End; j++) {     // for each elevation interval
        	ElevInc += (a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult);
		      	ElevDiff = ElevRef - ElevInc;
		       SiteSpecific (ElevDiff, TempRef, &temp, HumidRef, &humid); // adjusts to site specific temp and relhum
		       Ctemp = temp - 32.0;                                   // convert to C
		       Ctemp /= 1.8;
         humid /= 100.0;
         for ( k = 0; k < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->NumSlopes; k++) {
           Slope = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->LoVal + k * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE);
           for ( m = 0; m < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->NumAspects; m++) {
             Aspect =Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->LoVal + m * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP);
    		       for ( n=0; n < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->NumFms; n++) {
               Cover = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->LoVal + n * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV);
               cov = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m];
		             cov->LastMx[n] = mx;
			            cov->LastEq[n] = mx;
               if (cov->fms[n] == NULL) {
                 switch ( FuelSize )	{
                   case SIZECLASS_10HR:
                     cov->fms[n] = Fms_Create10Hour((char *)"10hr"); break;
                   case SIZECLASS_100HR:
                     cov->fms[n] = Fms_Create100Hour((char *)"100hr"); break;
                   case SIZECLASS_1000HR:
                     cov->fms[n] = Fms_Create1000Hour((char *)"1000hr"); break;	} }

               Radiate = SimpleRadiation (Date, Hour, Cloud, ElevInc, Slope, Aspect, Cover);
               cov->SolRad[n] = Radiate;
               Fms_Initialize (cov->fms[n], Ctemp, humid, Radiate, Rain, Ctemp, humid, mx);   	// go directly to Fms_Update, not Fms_UpdateAt

               d_FueSizInt = (double) a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME);

               Fms_Update ( cov->fms[n], (d_FueSizInt / 60.0), Ctemp, humid, Radiate, Rain);	        // go directly to Fms_Update, not Fms_UpdateAt

               loc = GetLoc (FuelSize,i,j,k,m,n,sn);
				           cov->NextMx[n] = Fms_MeanWtdMoisture (cov->fms[n]);
               curhist->Moisture[loc] = cov->NextMx[n];
               cov->NextEq[n] = cov->fms[n]->sem;
               if ( FuelSize == SIZECLASS_10HR )
	                onehist->Moisture[loc] = cov->NextEq[n];
	       } } } }  /* for n, m, k, j */

	       LastTime = curhist->LastTime = Stations[sn].FMS[FuelSize][i]->FirstTime;
        if ( FuelSize == SIZECLASS_10HR)
	          onehist->LastTime = LastTime;

        LastTime += a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME);			// force constant time interval between calcs.
	       Stations[sn].FMS[FuelSize][i]->LastTime = curhist->SimTime = MxDesc->EndTime[sn][FuelSize] = LastTime;

        if ( FuelSize == SIZECLASS_10HR)
          onehist->SimTime = MxDesc->EndTime[sn][0] = LastTime;

	       return;
    }    /* if ( FirstTime ) */

/*------------------------------------------------------------------------------------------*/
/* UpdateMapMoisture() succesive passes                                                    */
     LastTime = Stations[sn].FMS[FuelSize][i]->LastTime;

     while ( LastTime < SimTime ) {

        ElevRef = curhist->Elevation;
        TempRef = curhist->AirTemperature;
        HumidRef = curhist->RelHumidity;
        Rain = curhist->Rainfall;
        Cloud = curhist->CloudCover;

       	if ( a_CI->GetTheme_Units(E_DATA) == 1 )
		       	ElevInc = a_CI->GetLoElev() - a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult;	// must be less, so that increment will start at LoElev
	      	else
     	   	ElevInc = (a_CI->GetLoElev() - a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV))*emult;
		      for ( j = 0; j < Begin; j++)
	          ElevInc += (a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult);

	      	for ( j = Begin; j < End; j++) 	{ /* for each Elev */
	         ElevInc += a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult;
			       ElevDiff = ElevRef - ElevInc;
     	    SiteSpecific(ElevDiff, TempRef, &temp, HumidRef, &humid); // adjusts to site specific temp and relhum
          Ctemp = temp - 32.0;                                   // convert to C
			       Ctemp /= 1.8;
     	    humid /= 100.0;

          for ( k = 0; k < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->NumSlopes; k++) {
            Slope = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->LoVal + k * a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_SLOPE);
            for ( m = 0; m < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->NumAspects; m++)  {
	             Aspect = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->LoVal + m * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP);
    		        for ( n = 0; n < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->NumFms; n++) {
	               Cover = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->LoVal +
               		       n * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV);
                cov = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m];
				            cov->LastMx[n] = cov->NextMx[n];
				            cov->LastEq[n] = cov->NextEq[n];
    			         Radiate = SimpleRadiation(Date, Hour, Cloud, ElevInc, Slope, Aspect, Cover);
     			        cov->SolRad[n] = Radiate;

                d_FueSizInt = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME);
                Fms_Update (cov->fms[n], (d_FueSizInt / 60.0), Ctemp, humid, Radiate, Rain);	// go directly to Fms_Update, not Fms_UpdateAt

                loc = GetLoc (FuelSize,i,j,k,m,n,sn);
                curhist->Moisture[loc] = cov->NextMx[n] = Fms_MeanWtdMoisture(cov->fms[n]);

// test-larry
// if ( loc == 0 && FuelSize == 1 )
//  printf ("%f  %d  %d %d  %d %d %d %d - %f  %f  %f  %f \n",
//	curhist->Moisture[loc], Date, Hour, Cloud, ElevInc,		Slope, Aspect, Cover,
//  	Ctemp, humid, Radiate, Rain);
// test-larry



                if ( FuelSize == SIZECLASS_10HR) {
                   onehist->Moisture[loc] = cov->NextEq[n] = cov->fms[n]->sem;  }

       	} } } }  /* for j,k,m,n */

        curhist->LastTime = LastTime;
        Stations[sn].FMS[FuelSize][i]->FirstTime = LastTime;

        if ( FuelSize == SIZECLASS_10HR)
	         onehist->LastTime = LastTime;

        LastTime += a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME);
        curhist->SimTime = MxDesc->EndTime[sn][FuelSize] = LastTime;
        //Stations[sn].FMS[FuelSize][i]->LastTime = LastTime;

        if ( FuelSize == SIZECLASS_10HR ) {
           onehist->SimTime = MxDesc->EndTime[sn][0] = LastTime;
           nexthist = (DeadMoistureHistory *) onehist->next;
           onehist = nexthist; }

        nexthist = (DeadMoistureHistory *) curhist->next;
        curhist = nexthist;

     }   /* while  LastTime < SimTime */
        Stations[sn].FMS[FuelSize][i]->LastTime = LastTime;

  return;     /* just a place to put a debug break */
}

/******************************************************************************
* Name: GetLoc
* Desc: Get the location (index) into the Moisture Array - see caller
*   In: see caller
******************************************************************************/
long FmsThread::GetLoc (long FuelSize, long i, long j, long k, long m, long n, long sn)
{
long loc;
  loc=  i * MxDesc->NumElevs  [sn][FuelSize] *
            MxDesc->NumSlopes [sn][FuelSize] *
            MxDesc->NumAspects[sn][FuelSize] *
            MxDesc->NumCovers [sn][FuelSize] +
        j * MxDesc->NumSlopes [sn][FuelSize] *
            MxDesc->NumAspects[sn][FuelSize] *
            MxDesc->NumCovers [sn][FuelSize] +
        k * MxDesc->NumAspects[sn][FuelSize] *
            MxDesc->NumCovers [sn][FuelSize] +
        m * MxDesc->NumCovers [sn][FuelSize] + n;
  return loc;
}

/****************************************************************************************/
double FmsThread::SimpleRadiation(long date, double hour, long cloud, long elev, long slope, long aspect, long cover)
{

// calculates solar radiation (W/m2) using Collin Bevins model
double cloudTransmittance=1.0-(double) cloud/100.0;
double Rad, jdate, latitude=(double) a_CI->GetLatitude();
double atmTransparency=0.7;
double canopyTransmittance=1.0-(double) cover/100.0;
long i, month, pdays, days;

     for ( i = 1; i <= 13; i++) {
	      switch (i) {
         case 1: days=0; break;			// cumulative days to begin of month
	     	  case 2: days=31; pdays=0; break;           // except ignores leapyear, but who cares anyway,
		       case 3: days=59; pdays=31; break;
     	  	case 4: days=90; pdays=59; break;
	       	case 5: days=120; pdays=90; break;
		       case 6: days=151; pdays=120; break;
       		case 7: days=181; pdays=151; break;
	       	case 8: days=212; pdays=181; break;
		       case 9: days=243; pdays=212; break;
       		case 10: days=273; pdays=243; break;
	       	case 11: days=304; pdays=273; break;
		       case 12: days=334; pdays=304; break;
         default: days=367; pdays=334; break; }
      if ( date < days ) {
        month = i - 1;
        days = date - pdays;
        break; }
   }

   jdate = CDT_JulianDate(2000, month, days, hour/100, 0, 0, 0);
   Rad = CDT_SolarRadiation (jdate, 0.0, latitude, 0.0, (double) slope, (double) aspect, (double) elev / 3.2808,
                            atmTransparency, cloudTransmittance, canopyTransmittance);

   return Rad *= 1370.0;       //W/m2
}


/*************************************************************************
* Change 10-29-09 - this function added
* Name: UpdateMapMoistuer_NewDFM
* Desc: This function replaces the UpdateMapMoisture() and uses the
*       new Deadfuelmoisture class
*        see deadfuelmoisture.h
* NOTE:
**************************************************************************/
void FmsThread::UpdateMapMoisture_NewDFM()
{
long 	i, j, k, m, n, loc;
long		sn = StationNumber;
long 	ElevInc, ElevDiff,	Slope, Aspect, Cover;
double d,	Radiate, temp, Ctemp, humid, mx, emult=1.0;     // 3.2808
double 	ElevRef, TempRef, HumidRef, Rain;
long 	Cloud;       //, NumNodes, BlockLen;
FMS_Cover *cov;
DeadMoistureHistory *curhist, *nexthist, *onehist;


  if ( a_CI->GetTheme_Units(E_DATA)==0)
    emult = 3.2808;                           //convert to feet for SiteSpecific
  i = FuelType;
  curhist = CurHist[FuelSize];
  if ( FuelSize == SIZECLASS_10HR)
     onehist=CurHist[0];

/*-----------------------------------------------------------------------------------*/
  if ( FirstTime ) {
     ElevRef = curhist->Elevation;
     TempRef = curhist->AirTemperature;
     HumidRef = curhist->RelHumidity;
     Rain = curhist->Rainfall;
     Cloud = curhist->CloudCover;

     if ( a_CI->GetTheme_Units(E_DATA)==1)
        ElevInc = a_CI->GetLoElev()-a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult;	// must be less, so that increment will start at LoElev
     else
        ElevInc = (a_CI->GetLoElev()-a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV))*emult;
     for ( j = 0; j < Begin; j++)
        ElevInc += (a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult);

     // get starting fuel moisture
     if ( FuelSize < SIZECLASS_1000HR)
       mx = ((double) a_CI->GetInitialFuelMoisture(Stations[sn].FMS[FuelSize][i]->Fuel, FuelSize))/100.0;
     else if ( FuelSize == SIZECLASS_1000HR)
       mx = ((double) a_CI->GetInitialFuelMoisture(Stations[sn].FMS[SIZECLASS_1000HR][i]->Fuel, SIZECLASS_100HR))/100.0;
     else {
       mx = a_CI->GetWoodyFuelMoisture(Stations[sn].FMS[FuelSize][i]->Fuel, FuelSize)/100.0;
       if ( mx <= 0.10)
         mx = 0.10; }

     for ( j = Begin; j < End; j++)  { // for each elevation
       ElevInc += (a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult);
       ElevDiff = ElevRef-ElevInc;
       SiteSpecific(ElevDiff, TempRef, &temp, HumidRef, &humid); // adjusts to site specific temp and relhum
       Ctemp = temp-32.0;                                   // convert to C
       Ctemp /= 1.8;
       humid /= 100.0;

       for ( k = 0; k < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->NumSlopes; k++)  {
         Slope = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->LoVal +
                 k * a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_SLOPE);

         for ( m = 0; m < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->NumAspects; m++){
            Aspect= Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->LoVal +
                    m * a_CI->GetMoistCalcInterval ( FuelSize, FM_INTERVAL_ASP);

           for ( n = 0; n < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->NumFms; n++)  {
              Cover = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->LoVal+
                     n * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV);
            cov = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m];
            cov->LastMx[n] = mx;
            cov->LastEq[n] = mx;

            if ( cov->dfm[n] == NULL ) {
              switch(FuelSize) {
              	case SIZECLASS_10HR: {
                   cov->dfm[n] = new DeadFuelMoisture (0.64, "10Hr");    /* New-DFM ..................*/
                   break; }

               case SIZECLASS_100HR: {
                   cov->dfm[n] = new DeadFuelMoisture (2.0, "100Hr");     /* New-DFM ..................*/
                   break; }

               case SIZECLASS_1000HR: {
                   cov->dfm[n] = new DeadFuelMoisture (6.4, "1000Hr");  /* New-DFM ..................*/
                   break;}}
            }

            Radiate = SimpleRadiation (Date, Hour, Cloud, ElevInc, Slope, Aspect, Cover);
            cov->SolRad[n] = Radiate;

/* New-DFM  */
#define e_BaroPres  0.0218    /* default used to init Fms class - Barometric pressure (cal/cm3) */

            cov->dfm[n]->initializeEnvironment (
                       Ctemp,                 /*  ta Initial ambient air temperature (oC).         double */
                       humid,                 /*  ha Initial ambient air relative humidity (g/g).  double */
                       Radiate,               /*  sr Initial solar radiation (W/m2).               double */
                       Rain,                  /*  rc Initial cumulative rainfall amount (cm).      double */
                       Ctemp,                 /*  ti Initial stick temperature (oC).               double */
                       humid,                 /*  hi Initial stick surface relative humidty (g/g). double */
                       mx,                    /*  wi Initial stick fuel moisture fraction (g/g).   double */
                       e_BaroPres);           /*  bp Initial stick barometric pressure (cal/cm3).  double */

//  this is part of the CummRain bug
//           Stations[sn].CuumRain += (double) a_CI->GetMoistCalcInterval ( FuelSize, FM_INTERVAL_TIME) / 24.0 * Rain;

/* New-DFM   */
            d = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME)/60.0;
            cov->dfm[n]->update(
                    d,          /* double  et,   Elapsed time since the previous observation (h).         */
                    Ctemp,      /* double  at,   Current observation's ambient air temperature (oC).         */
                    humid,      /* double  rh,   Current observation's ambient air relative humidity (g/g).   */
                    Radiate,    /* double  sW,   Current observation's solar radiation (W/m2).                */
                    Rain,       /* double  rcum, Current observation's total cumulative rainfall amount (cm). */
                    e_BaroPres);/* double  bpr   Current observation's stick barometric pressure (cal/cm3). */

            loc =
               i * MxDesc->NumElevs[sn][FuelSize] * MxDesc->NumSlopes[sn][FuelSize] *
                   MxDesc->NumAspects[sn][FuelSize]*MxDesc->NumCovers[sn][FuelSize] +
               j * MxDesc->NumSlopes[sn][FuelSize] * MxDesc->NumAspects[sn][FuelSize] *
                   MxDesc->NumCovers[sn][FuelSize] +
               k * MxDesc->NumAspects[sn][FuelSize] * MxDesc->NumCovers[sn][FuelSize] +
               m * MxDesc->NumCovers[sn][FuelSize] + n;
/* New-DFM */
           cov->NextMx[n] = cov->dfm[n]->meanWtdMoisture ();
           curhist->Moisture[loc] = cov->NextMx[n];

           cov->NextEq[n] = cov->dfm[n]->m_Sem();
           if ( FuelSize == SIZECLASS_10HR )
             onehist->Moisture[loc] = cov->NextEq[n];

      }}}}  /* for n m k j  */

   LastTime = curhist->LastTime = Stations[sn].FMS[FuelSize][i]->FirstTime;
   if ( FuelSize == SIZECLASS_10HR )
     onehist->LastTime = LastTime;

   LastTime += a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME);			// force constant time interval between calcs.
   Stations[sn].FMS[FuelSize][i]->LastTime = curhist->SimTime=
   MxDesc->EndTime[sn][FuelSize] = LastTime;

   if ( FuelSize == SIZECLASS_10HR )
     onehist->SimTime = MxDesc->EndTime[sn][0] = LastTime;

   return;
 } /* if firsttime */


/***********************************************************************************************/
/* Successive loops  (non-first)                                                               */

  LastTime = Stations[sn].FMS[FuelSize][i]->LastTime;
  while ( LastTime < SimTime ) {
     ElevRef = curhist->Elevation;
     TempRef = curhist->AirTemperature;
     HumidRef = curhist->RelHumidity;
     Rain = curhist->Rainfall;
     Cloud = curhist->CloudCover;
     if ( a_CI->GetTheme_Units(E_DATA) == 1 )
       ElevInc = a_CI->GetLoElev() - a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult;	// must be less, so that increment will start at LoElev
     else
       ElevInc = (a_CI->GetLoElev() - a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_ELEV)) * emult;
     for ( j = 0; j < Begin; j++)
       ElevInc += (a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult);

     for ( j = Begin; j < End; j++) {                // for each elevation
      	ElevInc += a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult;
       ElevDiff = ElevRef-ElevInc;
       SiteSpecific (ElevDiff, TempRef, &temp, HumidRef, &humid); // adjusts to site specific temp and relhum
       Ctemp = temp-32.0;                                   // convert to C
       Ctemp /= 1.8;
       humid /= 100.0;

       for ( k = 0; k < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->NumSlopes; k++) {
        	Slope = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->LoVal +
                 k * a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_SLOPE);
         for ( m = 0; m < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->NumAspects; m++){
          	Aspect = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->LoVal +
                    m * a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_ASP);
           // cov = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m];
           for ( n = 0; n < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->NumFms; n++) {
             Cover = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->LoVal +
                     n * a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_COV);
             cov = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m];
             cov->LastMx[n] = cov->NextMx[n];
             cov->LastEq[n] = cov->NextEq[n];

             Radiate = SimpleRadiation (Date, Hour, Cloud, ElevInc, Slope, Aspect, Cover);

             cov->SolRad[n] = Radiate;

/* New-DFM */
             cov->dfm[n]->update (a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME) / 60.0, Ctemp, humid, Radiate, Rain);

             loc=
               i * MxDesc->NumElevs[sn][FuelSize] * MxDesc->NumSlopes[sn][FuelSize] * MxDesc->NumAspects[sn][FuelSize] * MxDesc->NumCovers[sn][FuelSize] +
               j * MxDesc->NumSlopes[sn][FuelSize] * MxDesc->NumAspects[sn][FuelSize] * MxDesc->NumCovers[sn][FuelSize] +
               k * MxDesc->NumAspects[sn][FuelSize] * MxDesc->NumCovers[sn][FuelSize] +
               m * MxDesc->NumCovers[sn][FuelSize] + n;

/* New-DFM */
             cov->NextMx[n] = cov->dfm[n]->meanWtdMoisture();
             curhist->Moisture[loc] = cov->NextMx[n];

             if ( FuelSize == SIZECLASS_10HR ) {
                cov->NextEq[n] = cov->dfm[n]->m_Sem();
                onehist->Moisture[loc] = cov->NextEq[n]; }
        }}}}   /* for n,m,k,j */

    curhist->LastTime = Stations[sn].FMS[FuelSize][i]->FirstTime = LastTime;
    if ( FuelSize == SIZECLASS_10HR )
       onehist->LastTime = LastTime;
    LastTime += a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME);
    curhist->SimTime=MxDesc->EndTime[sn][FuelSize] = Stations[sn].FMS[FuelSize][i]->LastTime = LastTime;
    if ( FuelSize == SIZECLASS_10HR) {
      onehist->SimTime=MxDesc->EndTime[sn][0] = LastTime;
      nexthist = (DeadMoistureHistory *) onehist->next;
      onehist = nexthist; }

   nexthist = (DeadMoistureHistory *) curhist->next;
   curhist = nexthist;
 } /* while */

  return;

}

/****************************************************************************/
void FE2::FreeHistory(long Station, long FuelSize)
{
long i, j;

  i=FuelSize;
	 CurHist[Station][i] = FirstHist[Station][i];
	 for ( j = 0; j < MxDesc.NumHist[Station][i]; j++)	{
     delete[] CurHist[Station][i]->Moisture;
     NextHist[Station][i] = (DeadMoistureHistory *) CurHist[Station][i]->next;
     delete CurHist[Station][i];
// printf ("Delete CurHist FuelSize %d \n",j);
    	CurHist[Station][i] = NextHist[Station][i]; }

  FirstHist[Station][i] = 0;
  CurHist[Station][i] = 0;
  NextHist[Station][i] = 0;
  MxDesc.ResetDescription (Station, FuelSize);
}

/*********************************************************************
* Name: ElevTempHum
* Desc: Get the requested data, which was calc'd and set when the
*        conditioning was run
**********************************************************************/
void FE2::ElevTempHum (long *al_elev, double *ad_temp, double *ad_hum)
{
   *al_elev = this->elevref;
   *ad_temp = this->tempref;
   *ad_hum  = this->humref;
}

/***************************************************************
* Name: HaveFuelMoist
* Desc: See if a fuel moist has been calculation for the
*       specified station & Fuel Size Class
*  Ret: True, yes it's been calc'd - False - nope
****************************************************************/
bool FE2::HaveFuelMoist(long Station, long FuelSize)
{
  if ( FuelSize < SIZECLASS_1HR )
    return false;
  if ( FuelSize > SIZECLASS_1000HR )
    return false;

  if ( this->MxDesc.NumFuels[Station][FuelSize] == 0 )
    return false;
  return 1;
}

/*************************************************************************
* Name: Get_Progress
* Desc: Get the percent of completion of the conditioning run
*  ret: 0 -> 1.0
*        1.0 == Completed
*       -1 == Conditioning not run yes,
*************************************************************************/
double FE2::Get_Progress()
{
 return this->d_Progress;
}


/**********************************************************************/
void FE2::SiteSpecific (long elev, double *ad_airtemp, double *ad_relhumd)
{
long l_ElevDiff;
     l_ElevDiff = this->elevref - elev;
    this->fmsthread[0].SiteSpecific(l_ElevDiff, this->tempref, ad_airtemp, this->humref, ad_relhumd);
}

/*********************************************************************
* Name: Terminate
* Desc: Set the termination flag. Setting this flag will cause the
*       simulation to abort.
*********************************************************************/
void FE2::Terminate()
{
  this->b_Terminate = true;
}

// **********************************************************************
void FmsThread::SetRange ( double simtime, long date, long hour,
                          long stationnumber, long fueltype,
		     		                 	FuelMoistureMap *map, DeadMoistureHistory **hist,
                           long begin, long end)
{
  SimTime=simtime;
	 Date=date;
     Hour=hour;
     StationNumber=stationnumber;
     FuelType=fueltype;
     Stations=map;
     CurHist=hist;
     Begin=begin;
     End=end;
}
/*******************************************************************************/
void FmsThread::SiteSpecific(long ElevDiff, double tempref, double *temp, double humref, double *humid)
{// FROM ROTHERMEL, WILSON, MORRIS, & SACKETT  1986
double dewptref, dewpt;
     // ElevDiff is always in feet, temp is in Fahrenheit, humid is %

	 dewptref = -398.0-7469.0 / (log(humref/100.0)-7469.0/(tempref+398.0));
 	*temp=tempref+(double) ElevDiff/1000.0*5.5;       	// Stephenson 1988 found summer adiabat at 3.07 F/1000ft
 	dewpt=dewptref+(double) ElevDiff/1000.0*1.1; 		    // new humidity, new dewpt, and new humidity
 	*humid=7469.0*(1.0/(*temp+398.0)-1.0/(dewpt+398.0));
	 *humid=exp(*humid)*100.0;		                      		// convert from ln units
	 if ( *humid > 99.0 )
		  *humid=99.0;
}

/*******************************************************************
*
*
*
*******************************************************************/
FILE *fTest = NULL;

void TestFile (char cr[])
{

//if ( strstr (cr, "AllocHistory") )
//   return;

 /*  if ( !_stricmp (cr,"open") ) {
     fTest = fopen ("c:\\LarryF\\Test-Far\\test-new.txt","w");
     return; }

   if ( !_stricmp (cr,"close") ) {
     fclose (fTest);
     return; }

   fprintf (fTest, "%s", cr);*/

 //  printf ("%s",cr);

}
// ************************************************************************
// ************************************************************************
// ************************************************************************


/**************************************************************************
* Name:
* Desc:
*
* NOTE NOTE:  Newer updeate function by Stu
*             Inserted 3-18-10
***************************************************************************/
void FE2::RunFmsThreads_Stu (double SimTime, long sn, long FuelType, long FuelSize)
{
bool      StartAtBeginning=false;
long 	i, j, l_Intv;                     //, FuelSize=1;
long 	date, hour, tr1, tr2;
long		Cloud, MoistLoc;
long 	begin, end, threadct, range;
long		mo, dy, hr, mn, min1, min2;
double	fract, interval, ipart, RainRate, Interval, r1, r2;
double 	lasttime, checktime, hours;

   DeadMoistureHistory **hist = 0;

   if ( MxDesc.NumFuels[sn][FuelSize] == 0 )
     	return;

   AllocFmsThreads();     /* allocate FmsThread class object(s) */
   i = FuelType;

   interval = ((double) Stations[sn].FMS[FuelSize][i]->NumElevs) / ((double) NumFmsThreads);
   fract = modf (interval, &ipart);
   range = (long) interval;
   if ( fract > 0.0)
      range++;

/*--------------------------------------------------------------------------------------------*/
/* First time loop                                                                           */
   if ( Stations[sn].FMS[FuelSize][FuelType]->LastTime == 0.0 ||
        MxDesc.NumHist[sn][FuelSize] == 0) {   	// will allocate all stations, so only do for 1st station
     hist=new DeadMoistureHistory*[NUM_FUEL_SIZES];
     if ( FuelType == 0 ) {
	      switch (FuelSize)	{
         case SIZECLASS_100HR:
	           AllocHistory(sn, SIZECLASS_100HR);          		// 100hr fuels
            break;
         case SIZECLASS_1000HR:
			         AllocHistory(sn, SIZECLASS_1000HR);          	// 1000hr fuels
            break;
         default:
            AllocHistory(sn, SIZECLASS_1HR);          		// 1hr fuels
		     	    AllocHistory(sn, SIZECLASS_10HR);          		// 10hr fuels
            break; } }
     else {
       CurHist[sn][FuelSize] = FirstHist[sn][FuelSize];
       if ( FuelSize == SIZECLASS_10HR)
          CurHist[sn][SIZECLASS_1HR] = FirstHist[sn][SIZECLASS_1HR];
       StartAtBeginning = true; }

     hist[FuelSize]=CurHist[sn][FuelSize];
     if ( FuelSize == SIZECLASS_10HR)
	       hist[0]=CurHist[sn][0];

     l_Intv = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME); /* In minutes */
     if ( a_CI->AtmosphereGridExists () )
       Cloud = 0;
     else
       Cloud = a_CI->GetCloud(l_Intv,sn); /* 10-18-10 new GetCloud() */

     date = a_CI->Chrono(l_Intv,&hour, &hours);
	   	HumTemp_Stu (date, hour, &tempref, &humref, &elevref, &rain, &humidmx,
                  &humidmn, &tempmx, &tempmn, &tr1, &tr2);
		   if ( rain > 0.0 ) 	{
			    if ( tr2 == 0 ) 	{		// use actual rain duration
			      tr2 = 2400;
				     tr1 = 0; }
		  	  if ( tr2 > 0) 	{		// use actual rain duration
         l_Intv = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME);
				     a_CI->ConvertSimtimeToActualTime(l_Intv, &mo, &dy, &hr, &mn);
				     min1 = ((long) (tr1/100.0))*100;
				     mn = tr1 - min1;
			     	r1 = a_CI->ConvertActualTimeToSimtime(mo, dy, min1, mn);
				     min2 = ((long) (tr2/100.0))*100;
			     	mn = tr2-min2;
				     r2 = a_CI->ConvertActualTimeToSimtime(mo, dy, min2, mn);
				     RainRate=rain / (r2-r1);
				     lasttime=0.0;
				     Interval = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME);
				     if ( r1 >= lasttime || r2<lasttime-Interval) 		// if interval starts before r1 or after r2
					      rain=0.0;
				     else if ( r1 > lasttime-Interval && r1<lasttime) {	// if r1 starts inside interval
					      if ( r2 > lasttime)					// if interval ends before r2
						       rain = RainRate * (lasttime-r1);
					      else                                    // if r1 && r2 inside interval
						       rain = RainRate * (r2-r1);}
			 	    else if ( lasttime - Interval > r1 && r2 >= lasttime) //if interval between r1 && r2
				     	 rain = RainRate * Interval;
				     else if ( r2 >= lasttime - Interval && lasttime > r2 ) //if interval brackets r2
					      rain = RainRate * ( r2 - (lasttime-Interval));
			  } /* if ( tr2 > 0 ) */

			  if ( rain > 0.0 )	{
				   humref=max(humidmx, 90.0);
				   Cloud = 100.0;	}
		}  /* if (rain > 0.0) */


    rain *= 2.54;			               // rain in cm
		  Stations[sn].CuumRain[FuelSize] = rain;
		  CurHist[sn][FuelSize]->AirTemperature=tempref;
		  CurHist[sn][FuelSize]->RelHumidity=humref;
	  	CurHist[sn][FuelSize]->Elevation=elevref;
    CurHist[sn][FuelSize]->CloudCover=Cloud;
    CurHist[sn][FuelSize]->Rainfall = Stations[sn].CuumRain[FuelSize];

    if ( FuelSize==SIZECLASS_10HR) {
      CurHist[sn][0]->AirTemperature = tempref;
			   CurHist[sn][0]->RelHumidity = humref;
			   CurHist[sn][0]->Elevation = elevref;
      CurHist[sn][0]->CloudCover = Cloud;
	     CurHist[sn][0]->Rainfall = Stations[sn].CuumRain[FuelSize]; }

    begin=threadct=0;
    for ( j = 0; j < NumFmsThreads; j++) {
      end = begin + range;
      if ( begin >= Stations[sn].FMS[FuelSize][i]->NumElevs)
        continue;
      if ( end > Stations[sn].FMS[FuelSize][i]->NumElevs)
        end = Stations[sn].FMS[FuelSize][i]->NumElevs;
      fmsthread[j].SetRange (SimTime, date, hour, StationNumber, FuelType, Stations,		hist, begin, end);
// Test
//  printf ("First: %6.0f \n", SimTime);
      threadct++;
      begin = end; }

    for (j=0; j<threadct; j++)
      fmsthread[j].StartFmsThread(j, FuelSize, &MxDesc, true);
#ifdef WIN32
    a_CI->WaitForFarsiteEvents(EVENT_MOIST, threadct, true, INFINITE);
#endif
  //WaitForMultipleObjects(threadct, &(a_CI->hMoistEvent.GetEvent(0)), TRUE, INFINITE);
  } /* if station[sn] */



/*-----------------------------------------------------------------------------------*/
/* Successive loop (non-first)                                                      */

  if (hist)
    delete[] hist;
  hist=0;
/*.................................................................................*/
  if (Stations[sn].FMS[FuelSize][i]->LastTime < SimTime) {
    hist=new DeadMoistureHistory*[NUM_FUEL_SIZES];
			 j=0;
    lasttime=Stations[sn].FMS[FuelSize][i]->LastTime;
    do	{
      if (i==0) {                   // FuelType is the first one
						  switch(FuelSize) 	{
						    case SIZECLASS_100HR:
	     			   AllocHistory(sn, SIZECLASS_100HR);          		// 100hr fuels
            break;
          case SIZECLASS_1000HR:
			         AllocHistory(sn, SIZECLASS_1000HR);              	// 1000hr fuels
            break;
     	    default:
	     		    AllocHistory(sn, SIZECLASS_10HR);          		// 10hr fuels
		     	    AllocHistory(sn, SIZECLASS_1HR);          		// 1hr fuels
            break;
         } /* switch */
      }
      else
	  {
		  if (StartAtBeginning)
		  {
			  NextHist[sn][FuelSize]=(DeadMoistureHistory *) CurHist[sn][FuelSize]->next;
			  CurHist[sn][FuelSize]=NextHist[sn][FuelSize];
			  if (FuelSize == SIZECLASS_10HR)
			  {
				  NextHist[sn][0]=(DeadMoistureHistory *) CurHist[sn][0]->next;
				  CurHist[sn][0]=NextHist[sn][0];
			  }
		  }
		  else if (j==0)
		  { // first time through
			  MoistLoc=MxDesc.NumAlloc[sn][FuelSize]/MxDesc.NumFuels[sn][FuelSize]*FuelType;
			  CurHist[sn][FuelSize]=FirstHist[sn][FuelSize];
			  while (CurHist[sn][FuelSize]->Moisture[MoistLoc]>0.0)
			  {
				  NextHist[sn][FuelSize]=(DeadMoistureHistory *) CurHist[sn][FuelSize]->next;
				  CurHist[sn][FuelSize]=NextHist[sn][FuelSize];
			  }
			  if ( FuelSize==SIZECLASS_10HR)
			  {
				  MoistLoc=MxDesc.NumAlloc[sn][0]/MxDesc.NumFuels[sn][0]*FuelType;
				  CurHist[sn][0]=FirstHist[sn][0];
				  while(CurHist[sn][0]->Moisture[MoistLoc]>0.0)
				  {
					  NextHist[sn][0]=(DeadMoistureHistory *) CurHist[sn][0]->next;
					  CurHist[sn][0]=NextHist[sn][0];
				  }
			  }
		  } /* else if */
		  else
		  {
			  NextHist[sn][FuelSize]=(DeadMoistureHistory *) CurHist[sn][FuelSize]->next;
			  CurHist[sn][FuelSize]=NextHist[sn][FuelSize];
			  MoistLoc=MxDesc.NumAlloc[sn][FuelSize]/MxDesc.NumFuels[sn][FuelSize]*FuelType;
			  while(CurHist[sn][FuelSize]->Moisture[MoistLoc]>0.0)
			  {
				  NextHist[sn][FuelSize]=(DeadMoistureHistory *) CurHist[sn][FuelSize]->next;
				  CurHist[sn][FuelSize]=NextHist[sn][FuelSize];
			  }
			  if (FuelSize==SIZECLASS_10HR)
			  {
				  NextHist[sn][0]=(DeadMoistureHistory *) CurHist[sn][0]->next;
				  CurHist[sn][0]=NextHist[sn][0];
				  MoistLoc=MxDesc.NumAlloc[sn][0]/MxDesc.NumFuels[sn][0]*FuelType;
				  while (CurHist[sn][0]->Moisture[MoistLoc]>0.0)
				  {
					  NextHist[sn][0] =(DeadMoistureHistory *) CurHist[sn][0]->next;
					  CurHist[sn][0]=NextHist[sn][0];
				  }
			  }
		  } /* else */
      } /* else */

      if ( j == 0 )  {
         hist[FuelSize] = CurHist[sn][FuelSize];
         if (FuelSize==SIZECLASS_10HR)
           hist[0]=CurHist[sn][0];}
      j++;
		    CurHist[sn][FuelSize]->LastTime=lasttime;
      if (FuelSize==SIZECLASS_10HR)
       	CurHist[sn][0]->LastTime=lasttime;
      lasttime+=a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME);
      CurHist[sn][FuelSize]->SimTime=lasttime;
      if(FuelSize==SIZECLASS_10HR)
        CurHist[sn][0]->SimTime = lasttime;
      checktime = lasttime;

		    if (a_CI->AtmosphereGridExists())
        Cloud = 0;               //GetAtmClouds(date, hours);
		    else
			     Cloud = a_CI->GetCloud(checktime,sn); /* 10-18-10 new GetCloud() */

      date = a_CI->Chrono(checktime,&hour,&hours);
      HumTemp_Stu (date,hour, &tempref, &humref, &elevref, &rain, &humidmx,
                  &humidmn, &tempmx, &tempmn, &tr1, &tr2);
		    if ( rain > 0.0 ) {
         if ( tr2 > 0 ) {		// use actual rain duration
			     		a_CI->ConvertSimtimeToActualTime(lasttime, &mo, &dy, &hr, &mn);

	          min1 = ((long) (tr1 / 100.0)) * 100;
		         mn = tr1 - min1;
     		    r1 = a_CI->ConvertActualTimeToSimtime(mo, dy, min1, mn);
		       		min2 = ((long) (tr2/100.0))*100;
           mn = tr2 - min2;
          	r2 = a_CI->ConvertActualTimeToSimtime(mo, dy, min2, mn);
		         RainRate = rain / ( r2 - r1) ;
     		    Interval = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME);

           if (r1 >= lasttime || r2 < lasttime - Interval) 		// if interval starts before r1 or after r2
             rain = 0.0;
           else if (r1 > lasttime - Interval && r1 < lasttime) {	// if r1 starts inside interval
								     if ( r2 > lasttime )					// if interval ends before r2
               rain = RainRate * ( lasttime - r1 );
             else                                    // if r1 && r2 inside interval
               rain = RainRate * ( r2 - r1 );
           }
           else if ( lasttime-Interval>=r1 && r2>=lasttime) //if interval between r1 && r2
             rain=RainRate*Interval;
           else if(r2>=lasttime-Interval && lasttime>r2) //if interval brackets r2
            	rain=RainRate*(r2-(lasttime-Interval));
	        } /* if ( tr2 > 0 */
 					   if ( rain>0.0)	{
				     		humref=max(humidmx, 90.0);
           Cloud=100;	}
     	} /* if rain > 0 ) */


			   rain *= 2.54;		// rain in cm
		    Stations[sn].CuumRain[FuelSize] += rain;
			   CurHist[sn][FuelSize]->AirTemperature = tempref;
			   CurHist[sn][FuelSize]->RelHumidity = humref;
				  CurHist[sn][FuelSize]->Elevation = elevref;
      CurHist[sn][FuelSize]->CloudCover = Cloud;
      CurHist[sn][FuelSize]->Rainfall = Stations[sn].CuumRain[FuelSize];
		    if(FuelSize==SIZECLASS_10HR)		{
				   	CurHist[sn][0]->AirTemperature = tempref;
				   	CurHist[sn][0]->RelHumidity = humref;
				    CurHist[sn][0]->Elevation = elevref;
		      CurHist[sn][0]->CloudCover = Cloud;
			     CurHist[sn][0]->Rainfall = Stations[sn].CuumRain[FuelSize]; }
    } while (lasttime < SimTime );


	  	begin=threadct=0;
    for (j=0; j<NumFmsThreads; j++) {
			   end=begin+range;
      if(begin>=Stations[sn].FMS[FuelSize][i]->NumElevs)
         continue;
      if(end>Stations[sn].FMS[FuelSize][i]->NumElevs)
       	end=Stations[sn].FMS[FuelSize][i]->NumElevs;
      fmsthread[j].SetRange(SimTime, date, hour, StationNumber, FuelType, Stations,	hist, begin, end);
// Test
//  printf ("Next: %6.0f, FuelSize: %d \n", SimTime,FuelSize);
      threadct++;
      begin=end; }

    for (j=0; j<threadct; j++)
      fmsthread[j].StartFmsThread(j, FuelSize, &MxDesc, false);

#ifdef WIN32
    a_CI->WaitForFarsiteEvents(EVENT_MOIST, threadct, true, INFINITE);
#endif
          //WaitForMultipleObjects(threadct, hMoistureEvent, TRUE, INFINITE);
          //Sleep(5);
    MxDesc.EndTime[sn][FuelSize]=Stations[sn].FMS[FuelSize][i]->LastTime=lasttime;

    if ( FuelSize == SIZECLASS_10HR ) {
       MxDesc.EndTime[sn][0] = MxDesc.EndTime[sn][FuelSize];}

   } /* if station[sn] */
/*...........................................................................*/


/*--------------------------------------------------------------------*/
/* Put the date and time in the Moist Hist Rec, for now this is for  */
/*  exporting testing of data - but it may be used more later  */
/* Note the LastTime + 0.001, keeps things even, else could get small rounding issues */
			a_CI->ConvertSimtimeToActualTime(CurHist[sn][FuelSize]->LastTime, &mo, &dy, &hr, &mn);
// if ( mo > 6 )
//  printf ("Here \n");

   CurHist[sn][FuelSize]->i_Mth = mo;
   CurHist[sn][FuelSize]->i_Day = dy;
   CurHist[sn][FuelSize]->i_Time = hr  + mn;
   if ( FuelSize == SIZECLASS_10HR ) {
     CurHist[sn][0]->i_Mth = mo;
     CurHist[sn][0]->i_Day = dy;
     CurHist[sn][0]->i_Time = hr + mn; }
/*-------------------------------------------------------------------*/



   if (hist)
    	delete[] hist;
}


/*************************************************************************************
* Name: HumTemp
* Desc: Find the weather info in the weather stream table
*       Finds and interpolates Humidity and temperature for Current Hour
*   In; date....Julian date
*       hour....hour of day 0000->2300
* NOTE NOTE: Newer updated Stu version, insert here 3-18-10
*
*************************************************************************************/
void FE2::HumTemp_Stu (long date, long hour, double *tempref, double *humref,
	                     long *elevref, double *rain, double *humidmx, double *humidmn,
                      double *tempmx, double *tempmn, long *tr1, long *tr2)
{

long count = -1, hmorn, haft, Tmin, Tmax, Hmax, Hmin;
long elref, hx, garbage, ppt;
double h1, h2, dtprime, dtmxmn, humid, temp, tempf, humf, sign;
double Thalf, Hhalf;
int  LastCount;

   LastCount =   a_CI->wtrdt_idx(date,StationNumber);

//	while ( LastDate != date) 	{
//     count++;
//		   month = a_CI->GetWeatherMonth(StationNumber, count);
//		   day = a_CI->GetWeatherDay(StationNumber, count);
//     LastDate = day + a_CI->GetJulianDays(month);
//     LastCount = count;
//     if ( LastDate > 365 )
//       count=-1;	}


  hmorn = a_CI->GetWeatherTime1(StationNumber, LastCount);
	 haft  = a_CI->GetWeatherTime2(StationNumber, LastCount);
	 ppt   = a_CI->GetWeatherRain(StationNumber, LastCount);
	 Tmin  = a_CI->GetWeatherTemp1(StationNumber, LastCount);
	 Tmax  = a_CI->GetWeatherTemp2(StationNumber, LastCount);
	 Hmax  = a_CI->GetWeatherHumid1(StationNumber, LastCount);
	 Hmin  = a_CI->GetWeatherHumid2(StationNumber, LastCount);
	 elref = a_CI->GetWeatherElev(StationNumber, LastCount);

		a_CI->GetWeatherRainTimes(StationNumber, LastCount, tr1, tr2);


	 count = LastCount;	            // only meaningful if LastDate==date;

  if (hour > haft) {
    count++;
		  garbage = a_CI->GetWeatherElev(StationNumber, count);
		  hmorn = a_CI->GetWeatherTime1(StationNumber, count);
		  Tmin = a_CI->GetWeatherTemp1(StationNumber, count);
	  	Hmax = a_CI->GetWeatherHumid1(StationNumber, count);
		  if (garbage != elref)	{
      tempf = (double) Tmin;
		   	humf = (double) Hmax;
		   	fmsthread[0].SiteSpecific(elref-garbage, tempf, &temp, humf, &humid);	/* adjusts to site specific temp and relhum at a given pixel */
			   Tmin=(int) temp;
		   	Hmax=(int) humid;}
	  }
  	else	if ( hour < hmorn) {
      count--;
     	if ( count  <0 )
			     count=0;
	    	garbage=a_CI->GetWeatherElev(StationNumber, count);
		    //ppt=a_CI->GetWeatherRain(StationNumber, count);
	    	haft=a_CI->GetWeatherTime2(StationNumber, count);
	    	Tmax=a_CI->GetWeatherTemp2(StationNumber, count);
	    	Hmin=a_CI->GetWeatherHumid2(StationNumber, count);
	   	if ( garbage != elref ) {
       tempf=Tmax;
		    	humf=Hmin;
	    		fmsthread[0].SiteSpecific(elref-garbage, tempf, &temp, humf, &humid);	/* adjusts to site specific temp and relhum at a given pixel */
		    	Tmax=(int) temp;
		    	Hmin=(int) humid; }
  	}

	  hx = hmorn / 100;
	  hx *= 100;
  	if ( hx < hmorn )
	   	h1 = (double) hx + (10.0 * (hmorn-hx) / 6.0);
  	else
		   h1 = (double) hmorn;
	  hx = haft / 100;
	  hx *= 100;
	  if ( hx < haft )
		   h2 = (double) hx + (10.0 * (haft-hx) / 6.0);
	  else
	   	h2=(double) haft;

	  dtmxmn = ( 2400 - h2 ) + h1;		      /* this section interpolates temperature */

	  if ( hour >= h1 && hour <= h2)	{   	/* and humidity from high and low obs */
		    dtprime = (double) hour-h1;	     /* and their time of observation */
		    dtmxmn = 2400 - dtmxmn;
	    	sign = 1.0;	}
	  else	{
	    if (hour > h2)
		    	dtprime = double (hour) - h2;   // time since the maximum
		   else
		    	dtprime = ( 2400 - h2) + hour;    // time since the maximum
		   sign = -1.0;	}

	  Thalf = ((double) Tmax - (double) Tmin) / 2.0 * sign;
	  Hhalf = ((double) Hmin - (double) Hmax) / 2.0 * sign;

  	if ( dtmxmn != 0)	{
	   	*tempref = ((double) (Tmax+Tmin))/2.0+(Thalf*sin(PI*(dtprime/dtmxmn-0.5)));
	   	*humref = ((double) (Hmax+Hmin))/2.0+(Hhalf*sin(PI*(dtprime/dtmxmn-0.5)));	}
	  else	{
		   *tempref = Tmax;
	   	*humref = Hmin;}

	  *elevref = elref;
	  *rain = ((double) ppt) / 100.0;
	  *tempmx = (double) Tmax;
  	*tempmn = (double) Tmin;
  	*humidmx = (double) Hmax;
  	*humidmn = (double) Hmin;
}




