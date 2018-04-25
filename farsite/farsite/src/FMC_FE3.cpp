
/******************************************************************
* Name: FMC_FE3.cpp
* Desc: Contains the new FE2::RunFmsThreads_RAWS() (and related functions)
*        which is a copy of FE2::RunFmsThreads_Stu() and modified
*        to work with the newer Weather Stream Data.
* Date: 4-14-10
********************************************************************/

#define _CRT_SECURE_NO_WARNINGS

#ifdef WIN32
#include <windows.h>
#include <process.h>
#include <conio.h>
#else
#include <stdio.h>
#endif

#include "cdtlib.h"
#include "newfms.h"
#include "semtime.h"
#include "deadfuelmoisture.h"
#include "FMC_CI.h"
#include "FMC_FE2.h"
static const double PI=acos(-1.0);

// Test-lar
void TestFile (char cr[]);

/***************************************************************************
* Name: _MinStickMoist
* Desc: check and set the minimum moisture that we want to see coming
*        back from the Nelson moisture stick calcs.
****************************************************************************/
void _MinStickMoist (double *ad)
{
  if ( *ad < 0.02 )
     *ad = 0.02;
}

/*********************************************************************
* Name: RunFmsThreads_RAWS
* Desc: This function is only used with RAWS weather data.
*       Run the conditioning threads for the specified input
*       parameters - Simulation Time, Station, Fuel Type and Fuel Size.
*
**********************************************************************/
void FE2::RunFmsThreads_RAWS (double SimTime, long sn, long FuelType, long FuelSize)
{
long 	j,l, l_Intv,date,hour,begin, end, threadct, range,l_MinDate, l_LastTime;
double	fract, interval, ipart, d_RaiAcm;
d_RAWS s_RAWS;

  if ( MxDesc.NumFuels[sn][FuelSize] == 0 )  /* if no Fuel Types of this Fuel Size */
     	return;

  l_Intv = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME); /* In minutes */
  l_LastTime = Stations[sn].FMS[FuelSize][FuelType]->LastTime;
  l_LastTime = l_LastTime + l_Intv;
  if ( l_LastTime != SimTime )
     return;

   AllocFmsThreads();             /* Alloc Thread objects if no done already */

   if ( FuelType != 0 ) {
     CurHist[sn][FuelSize] = DMH_FindRec (FuelSize, SimTime);
     if ( FuelSize == SIZECLASS_10HR)
        CurHist[sn][0] = DMH_FindRec (0, SimTime);}

   if ( FuelType == 0 ) {
	    switch (FuelSize)	{
       case SIZECLASS_100HR:
	        AllocHistory (sn, SIZECLASS_100HR);
         break;
       case SIZECLASS_1000HR:
			      AllocHistory (sn, SIZECLASS_1000HR);
         break;
       default:
         AllocHistory (sn, SIZECLASS_1HR);
		     	 AllocHistory (sn, SIZECLASS_10HR);
         break; }

     l_Intv = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME); /* In minutes */
     l_MinDate = (this->a_CI->l_StartMinDate + (long)SimTime) - l_Intv;
     a_CI->RAWS_GetData(l_MinDate,l_Intv,&s_RAWS);
     l = this->a_CI->l_StartMinDate + (long) SimTime;
     d_RaiAcm = a_CI->RAWS_GetAccumRain (this->a_CI->l_StartMinDate, l );
     d_RaiAcm = d_RaiAcm * 2.54;   /* To Centimeters */
     Load_CurHist (CurHist[sn][FuelSize], &s_RAWS, a_CI->i_RAWS_Elev,d_RaiAcm, SimTime, l_Intv);
     if ( FuelSize == SIZECLASS_10HR )
         Load_CurHist (CurHist[sn][0], &s_RAWS, a_CI->i_RAWS_Elev,d_RaiAcm, SimTime, l_Intv);
    }  /* if FuelType == 0 */


   Stations[sn].FMS[FuelSize][FuelType]->FirstTime = SimTime - l_Intv;
   Stations[sn].FMS[FuelSize][FuelType]->LastTime = SimTime;
   MxDesc.EndTime[sn][FuelSize] = SimTime;
   if ( FuelSize == SIZECLASS_10HR )
      MxDesc.EndTime[sn][0] = SimTime;

   date = this->a_CI->GetJulian (CurHist[sn][FuelSize]->i_Mth, CurHist[sn][FuelSize]->i_Day);
   hour = CurHist[sn][FuelSize]->i_Time;            /* military hourly/min time */

/*______________________________________________________________________________*/
/* Figure out how many elev will be in the thread */
   interval = ((double) Stations[sn].FMS[FuelSize][FuelType]->NumElevs) / ((double) NumFmsThreads);
   fract = modf(interval, &ipart);
   range = (long) interval;
   if ( fract > 0.0)
      range++;

   begin = threadct = 0;
   for ( j = 0; j < NumFmsThreads; j++) {
     end = begin + range;
     if ( begin >= Stations[sn].FMS[FuelSize][FuelType]->NumElevs)
       continue;
     if ( end > Stations[sn].FMS[FuelSize][FuelType]->NumElevs)
       end = Stations[sn].FMS[FuelSize][FuelType]->NumElevs;
     fmsthread[j].SetRange (SimTime, date, hour, sn, FuelType, Stations,	CurHist[sn], begin, end);
     threadct++;
     begin = end; }

   for ( j = 0; j < threadct; j++)
     fmsthread[j].StartFmsThread_RAWS (j, FuelSize, &MxDesc, false);

#ifdef WIN32
   a_CI->WaitForFarsiteEvents(EVENT_MOIST, threadct, true, INFINITE);
#endif

//   this->Disp_CurHistMoist(FuelSize, FuelType);

}


/*****************************************************************************
* Name: UpdateMapMoisture_RAWS
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
void FmsThread::UpdateMapMoisture_RAWS()
{
long 	i, j, k, m, n, loc;
long		sn = StationNumber;
long 	ElevInc, ElevDiff;
long		Slope, Aspect, Cover;
double 	Radiate, temp, Ctemp, humid, emult = 1.0;
double 	ElevRef, TempRef, HumidRef, Rain;
long 	Cloud;
FMS_Cover *cov;

  	if ( a_CI->GetTheme_Units(E_DATA) == 0 )
	     	emult = 3.2808;        //convert to feet for SiteSpecific

    i = FuelType;  /* This is cute - watch out i gets used */

/*-----------------------------------------------------------------------------------*/
    ElevRef  = CurHist[FuelSize]->Elevation;
    TempRef  = CurHist[FuelSize]->AirTemperature;
    HumidRef = CurHist[FuelSize]->RelHumidity;
    Rain     = CurHist[FuelSize]->Rainfall;
    Cloud    = CurHist[FuelSize]->CloudCover;

    if ( a_CI->GetTheme_Units(E_DATA) == 1 )
		     ElevInc = a_CI->GetLoElev()-a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV)*emult;	// must be less, so that increment will start at LoElev
    else
       ElevInc = (a_CI->GetLoElev()-a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV))*emult;

    for ( j = 0; j < Begin; j++ )
	      ElevInc += ( a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult);

  	 for ( j = Begin; j < End; j++) {     // for each elevation interval
       ElevInc += (a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV) * emult);
		     ElevDiff = ElevRef - ElevInc;
		     SiteSpecific (ElevDiff, TempRef, &temp, HumidRef, &humid); // adjusts to site specific temp and relhum
		     Ctemp = (temp - 32.0) / 1.8;                                   // convert to C
       humid /= 100.0;

       for ( k = 0; k < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->NumSlopes; k++) {
         Slope = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->LoVal + k * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE);
         for ( m = 0; m < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->NumAspects; m++) {
           Aspect =Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->LoVal + m * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP);
    		     for ( n = 0; n < Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->NumFms; n++) {
             Cover = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m]->LoVal + n * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV);

             cov = Stations[sn].FMS[FuelSize][i]->Fms_Elev[j]->Fms_Slope[k]->Fms_Aspect[m];

             loc = GetLoc (FuelSize,i,j,k,m,n,sn);
    			      Radiate = SimpleRadiation (Date, Hour, Cloud, ElevInc, Slope, Aspect, Cover);

             if ( a_CI->FuelStickModel_IsNelson_070())
               Stick07_Mngr (n, cov, Radiate, Ctemp, humid,  Rain,  loc);
             else
               Stick10_Mngr (n, cov, Radiate, Ctemp, humid,  Rain,  loc);

	   } } } }  /* for n, m, k, j */

   return;
}

/*****************************************************************************
* Name: Stk07_Mngr - Fuel Moistuure Content Stick Nelson 0.07 (older version)
* Desc: Run a moisture condition on a Nelson stick,
*   In: See caller
*  Out: see called functions below
******************************************************************************/
void FmsThread::Stick07_Mngr (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc)
{
/* If stick not allocated yet, get one */
  if (cov->fms[n] == NULL) {
    if ( FuelSize ==	SIZECLASS_10HR )
      cov->fms[n] = Fms_Create10Hour ((char *)"10hr");
    else if ( FuelSize == SIZECLASS_100HR )
      cov->fms[n] = Fms_Create100Hour ((char *)"100hr");
    else
      cov->fms[n] = Fms_Create1000Hour ((char *)"1000hr"); }

/* Check and do the 'first time' to init stick, etc */
  if ( CurHist[FuelSize]->LastTime == 0 )
    this->Stick07_First (n,cov,Radiate,Ctemp,humid,Rain,loc);
  else
    this->Stick07_Next (n,cov,Radiate,Ctemp,humid,Rain,loc);

}

/*****************************************************************************
* Name: Stick07_First
* Desc: Fuel Moisture Content Stick Nelson 0.07 (older version)
*       This is done when the stick a new stick is created,
******************************************************************************/
void FmsThread::Stick07_First(long n, FMS_Cover *cov, double Radiate,
                             double Ctemp, double humid, double Rain, long loc)
{
double mx, d_FueSizInt;

   if ( FuelSize < SIZECLASS_1000HR)
	    mx = ((double) a_CI->GetInitialFuelMoisture (Stations[StationNumber].FMS[FuelSize][FuelType]->Fuel, FuelSize)) / 100.0;
   else
	    mx = ((double) a_CI->GetInitialFuelMoisture(Stations[StationNumber].FMS[SIZECLASS_1000HR][FuelType]->Fuel, SIZECLASS_100HR)) / 100.0;

   cov->LastMx[n] = mx;
			cov->LastEq[n] = mx;

   cov->SolRad[n] = Radiate;

   Fms_Initialize (cov->fms[n], Ctemp, humid, Radiate, Rain, Ctemp, humid, mx);   	// go directly to Fms_Update, not Fms_UpdateAt

   d_FueSizInt = (double) a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME);
   Fms_Update ( cov->fms[n], (d_FueSizInt / 60.0), Ctemp, humid, Radiate, Rain);	        // go directly to Fms_Update, not Fms_UpdateAt

			cov->NextMx[n] = Fms_MeanWtdMoisture (cov->fms[n]);
  _MinStickMoist (&cov->NextMx[n]);
   CurHist[FuelSize]->Moisture[loc] = cov->NextMx[n];

   cov->NextEq[n] = cov->fms[n]->sem;
  _MinStickMoist (&cov->NextEq[n]);
   if ( FuelSize == SIZECLASS_10HR )
     CurHist[0]->Moisture[loc] = cov->NextEq[n];

// Test-larry ................................................
//if ( cov->NextMx[n] <= 0.02 )
//  printf ("mx %f \n",  cov->NextMx[n]) ;
//if ( cov->NextEq[n] <= 0.02)
//  printf ("eq1 %f - %f  \n",  cov->NextEq[n], cov->NextMx[n] );

}

/*****************************************************************************
* Name:
* Desc:
*
******************************************************************************/
void FmsThread::Stick07_Next (long n, FMS_Cover *cov, double Radiate,
                             double Ctemp, double humid, double Rain, long loc)
{
double d_FueSizInt;

   cov->LastMx[n] = cov->NextMx[n];
			cov->LastEq[n] = cov->NextEq[n];
   cov->SolRad[n] = Radiate;

   d_FueSizInt = a_CI->GetMoistCalcInterval (FuelSize, FM_INTERVAL_TIME);
   Fms_Update (cov->fms[n], (d_FueSizInt / 60.0), Ctemp, humid, Radiate, Rain);	// go directly to Fms_Update, not Fms_UpdateAt

   cov->NextMx[n] = Fms_MeanWtdMoisture (cov->fms[n]);
  _MinStickMoist (&cov->NextMx[n]);         /* Chk - Set minimum */
   CurHist[FuelSize]->Moisture[loc] = cov->NextMx[n] ;

   if ( FuelSize == SIZECLASS_10HR) {
     cov->NextEq[n] = cov->fms[n]->sem;
    _MinStickMoist (&cov->NextEq[n]);       /* chk - set minimum */
     CurHist[0]->Moisture[loc] = cov->NextEq[n];   }

// Test-larry ................................................
//if ( cov->NextMx[n] <= 0.02)
//  printf ("mx %f \n",  cov->NextMx[n] );
//if ( cov->NextEq[n] <= 0.02)
//  printf ("eq2 %f - %f  \n",  cov->NextEq[n], cov->NextMx[n] );

}

/*****************************************************************************
* Name: Stk10_Mngr - Fuel Moistuure Content Stick Nelson 0.10 (newer version)
* Desc: Run a moisture conditioning on a Nelson stick,
*   In: See caller
*  Out: see called functions below
******************************************************************************/
void FmsThread::Stick10_Mngr (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc)
{
  if ( CurHist[FuelSize]->LastTime == 0 )
    Stick10_First (n, cov, Radiate, Ctemp, humid, Rain, loc);
  else
    Stick10_Next (n, cov, Radiate, Ctemp, humid, Rain, loc);
}

/*****************************************************************************
* Name: Stick10_First
* Desc: Fuel Moisture Content Stick Nelson 0.10 (newer version)
*
******************************************************************************/
void FmsThread::Stick10_First (long n, FMS_Cover *cov, double Radiate,
                            double Ctemp, double humid, double Rain, long loc)
{
double d,mx;

   if ( FuelSize < SIZECLASS_1000HR)
	    mx = ((double) a_CI->GetInitialFuelMoisture (Stations[StationNumber].FMS[FuelSize][FuelType]->Fuel, FuelSize)) / 100.0;
   else
	    mx = ((double) a_CI->GetInitialFuelMoisture(Stations[StationNumber].FMS[SIZECLASS_1000HR][FuelType]->Fuel, SIZECLASS_100HR)) / 100.0;

   cov->LastMx[n] = mx;
   cov->LastEq[n] = mx;

   if ( cov->dfm[n] == NULL ) {
     switch ( FuelSize ) {
       case SIZECLASS_10HR: {
         cov->dfm[n] = new DeadFuelMoisture (0.64, "10Hr");    /* New-DFM ..................*/
         break; }
       case SIZECLASS_100HR: {
         cov->dfm[n] = new DeadFuelMoisture (2.0, "100Hr");     /* New-DFM ..................*/
         break; }
       case SIZECLASS_1000HR: {
         cov->dfm[n] = new DeadFuelMoisture (6.4, "1000Hr");  /* New-DFM ..................*/
         break;}}}

   cov->SolRad[n] = Radiate;

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

   d = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME) / 60.0;
   cov->dfm[n]->update(
              d,          /* double  et,   Elapsed time since the previous observation (h).         */
              Ctemp,      /* double  at,   Current observation's ambient air temperature (oC).         */
              humid,      /* double  rh,   Current observation's ambient air relative humidity (g/g).   */
              Radiate,    /* double  sW,   Current observation's solar radiation (W/m2).                */
              Rain,       /* double  rcum, Current observation's total cumulative rainfall amount (cm). */
              e_BaroPres  /* double  bpr   Current observation's stick barometric pressure (cal/cm3). */
              );

   cov->NextMx[n] = cov->dfm[n]->meanWtdMoisture ();
  _MinStickMoist (&cov->NextMx[n]);
   CurHist[FuelSize]->Moisture[loc] = cov->NextMx[n];

   cov->NextEq[n] = cov->dfm[n]->m_Sem();
  _MinStickMoist (&cov->NextEq[n]);
   if ( FuelSize == SIZECLASS_10HR )
      CurHist[0]->Moisture[loc] = cov->NextEq[n];
}

/*****************************************************************************
* Name: Stick10_Next
* Desc: Fuel Moisture Content Stick Nelson 0.10 (newer version)
*
******************************************************************************/
void FmsThread::Stick10_Next (long n, FMS_Cover *cov, double Radiate,
                            double Ctemp, double humid, double Rain, long loc)
{
double d;
   cov->LastMx[n] = cov->NextMx[n];
   cov->LastEq[n] = cov->NextEq[n];
   cov->SolRad[n] = Radiate;

   d = a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_TIME) / 60.0;
   cov->dfm[n]->update (d, Ctemp, humid, Radiate, Rain);

   cov->NextMx[n] = cov->dfm[n]->meanWtdMoisture();
  _MinStickMoist (&cov->NextMx[n]);
   CurHist[FuelSize]->Moisture[loc] = cov->NextMx[n];

   if ( FuelSize == SIZECLASS_10HR ) {
     cov->NextEq[n] = cov->dfm[n]->m_Sem();
    _MinStickMoist (&cov->NextEq[n]);
     CurHist[0]->Moisture[loc] = cov->NextEq[n]; }
}

/*****************************************************************************
* Name: Load_CurHist
* Desc: Load a DeadMoistureHistory struct with weather, date, time info
*
******************************************************************************/
void FE2::Load_CurHist (DeadMoistureHistory *CurHist, d_RAWS *a_RAWS, int i_Elev,
                        double d_RaiAcm, long SimTime, long l_Intv)
{
		 CurHist->AirTemperature = a_RAWS->f_Temp;
		 CurHist->RelHumidity    = a_RAWS->f_Humidity;
   CurHist->Elevation      = i_Elev;
   CurHist->CloudCover     = (long)a_RAWS->f_CloCov;
   CurHist->Rainfall       = d_RaiAcm;
   CurHist->i_Mth          = a_RAWS->i_Mth;
   CurHist->i_Day          = a_RAWS->i_Day;
   CurHist->i_Time         = a_RAWS->i_Time;
   CurHist->LastTime       = SimTime - l_Intv;
   CurHist->SimTime        = SimTime;
}

// ****************************************************************************
void FmsThread::StartFmsThread_RAWS(long ID, long sizeclass, DeadMoistureDescription *mxdesc, bool firsttime)
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

      hFmsThread=(HANDLE) ::_beginthreadex(NULL, 0, &FmsThread::RunFmsThread_RAWS, this, CREATE_SUSPENDED, &ThreadID);

		    //unsigned long Affinity =pow(2.0, (int)ProcNum);
		   // SetThreadAffinityMask (hFmsThread, Affinity);
	  SetThreadIdealProcessor(hFmsThread, ProcNum);
			//printf("FmsThread affinity to processor %ld\n", ProcNum);
      ResumeThread(hFmsThread);
      CloseHandle(hFmsThread);
    }
     else
      	SetEvent(hFmsEvent);
#else
    RunFmsThread_RAWS(this);
#endif
}

// *********************************************************************
unsigned  FmsThread::RunFmsThread_RAWS(void *fmsthread)
{
	  static_cast <FmsThread *> (fmsthread)->UpdateMoistures_RAWS();
   return 1;
}

// ********************************************************************************
void FmsThread::UpdateMoistures_RAWS()
{
  do {
    if ( End < 0 )
     	break;
    UpdateMapMoisture_RAWS();
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

/********************************************************************************
*
*  Used for testing
*
*******************************************************************************/
/*void FE2::Disp_CurHistMoist (long FuelSize, long FuelType)
{
int i,j,sn,Sta, FUEL_SIZES;
double d;
char  CR[500];
  FUEL_SIZES = FuelSize;
  Sta = 0;
  sn = 0;  //station 0 //
  j = MxDesc.NumAlloc[sn][FuelSize];

  printf( "------------------------------------------------------------------\n");
//  TestFile (CR);

  printf ( "FSiz: %d, FTyp: %d    EndTime    %4.0f\n", FuelSize, FuelType,
           MxDesc.EndTime [Sta] [FUEL_SIZES]	);
//  TestFile (CR);

//  printf ("NumAlloc   %d\n", MxDesc.NumAlloc  [Sta] [FUEL_SIZES]	);
//  printf ("NumElevs   %d\n", MxDesc.NumElevs  [Sta] [FUEL_SIZES]);
//  printf ("NumSlopes  %d\n", MxDesc.NumSlopes [Sta] [FUEL_SIZES]);
//  printf ("NumAspects %d\n", MxDesc.NumAspects[Sta] [FUEL_SIZES]);
//  printf ("NumCovers  %d\n", MxDesc.NumCovers [Sta] [FUEL_SIZES]);
//  printf ("NumFuels   %d\n", MxDesc.NumFuels  [Sta] [FUEL_SIZES]	);
//  printf ("NumHist    %d\n", MxDesc.NumHist   [Sta] [FUEL_SIZES]);


  printf ( "LastTime: %4.0f   SimTime: %4.0f \n", CurHist[sn][FuelSize]->LastTime,CurHist[sn][FuelSize]->SimTime);
//  TestFile (CR);

  for ( i = 0; i < j; i++ ) {
    d = CurHist[sn][FuelSize]->Moisture[i];
    if ( d > 0 )
      printf ("%4.2f ", d);
  }
  printf ("\n");

}*/
/***********************************************************************
* Name: DeadMoistureHistory_Display
* Desc: Test function to display out the list of records.
**********************************************************************/
int FE2::DeadMoistureHistory_Display (int FuelSize, int sn)
{
DeadMoistureHistory *a;
int i,j,iN, Cnt;
float f_Tot;
char  CR[1000];
    Cnt = 1;
    iN = MxDesc.NumAlloc[sn][FuelSize];  /* actual # of Moist vales in the DeadMoistureHistory.Moistue[] */

    a = this->FirstHist[sn][FuelSize];

    while (1) {
      if ( a == NULL )
        break;

      sprintf (CR, "%2d - %6.0f %6.0f e%4ld t%2.0f h%3.0f c%3ld r%4.2f :", Cnt,
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

      TestFile  ((char *)"\n");

      a = (DeadMoistureHistory *)  a->next;
    }


#ifdef wowo
    double 	LastTime;
    double 	SimTime;
    long		  Elevation;			       // feet
	   double 	AirTemperature;	 	  // fahrenheit
    double 	RelHumidity;        // percentage
    long		  CloudCover;		       // percentage
    double 	Rainfall;           // cm of rain
   	double 	*Moisture;	        	// array of 10hr moisture contents at SimTime
	   void   	*next;
#endif

//if ( First


//  DeadMoistureHistory		*NextHist [eC_Sta] [4];

  return 1;
}


/************************************************************************
* Name: ExportMoistureDataText
* Desc: Export the moisture condition history records into
*        a text file.
*   In: FileName....path file name
*       LcpName.....name of landscape file, This name is just place
*                    into text file, not used for anything else
*  Ret: true = ok,  false error opening file
************************************************************************/
bool FE2::ExportMoistureDataText(char* FileName, char* LcpName)
{
DeadMoistureHistory *a;
long i, j, NumHist,NumAlloc;


	FILE* fout = fopen(FileName, "wt");
	if (fout == NULL)
		return false;

	fprintf(fout, "Version: 1.0\n");

	fprintf(fout, "LcpName: %s\n", LcpName);

	for (i = 0; i < NumStations; i++){
		for (j = 0; j < NUM_FUEL_SIZES; j++)	{
			switch(j)	{
			  case 0:
				   fprintf(fout, "One Hour\n");
			  	 break;
			  case 1:
			   	fprintf(fout, "Ten Hour\n");
				   break;
			  case 2:
			   	fprintf(fout, "Hundred Hour\n");
			   	break;
			  case 3:
				   fprintf(fout, "Thousand Hour\n");
			   	break;
			  default:
			   	fprintf(fout, "WTF!\n");
			 	break;}

			fprintf(fout, "\tNumAlloc:   %ld\n", MxDesc.NumAlloc[i][j]);
			fprintf(fout, "\tNumElevs:   %ld\n", MxDesc.NumElevs[i][j]);
			fprintf(fout, "\tNumSlopes:  %ld\n", MxDesc.NumSlopes[i][j]);
			fprintf(fout, "\tNumAspects: %ld\n", MxDesc.NumAspects[i][j]);
			fprintf(fout, "\tNumCovers:  %ld\n", MxDesc.NumCovers[i][j]);
			fprintf(fout, "\tNumFuels:   %ld\n", MxDesc.NumFuels[i][j]);
			fprintf(fout, "\tFuelKey:    %ld\n", MxDesc.FuelKey[i][j]);
			fprintf(fout, "\tNumHist:    %ld\n", MxDesc.NumHist[i][j]);
			fprintf(fout, "\tEndTime:    %f\n", MxDesc.EndTime[i][j]);		}	}



	for (i = 0; i < NumStations; i++)	{
		 for (j = 0; j < NUM_FUEL_SIZES; j++)	{
			  switch(j)	{
	  		  case 0:
			  	   fprintf(fout, "One Hour\n");
				     break;
			    case 1:
				     fprintf(fout, "Ten Hour\n");
				     break;
			    case 2:
				     fprintf(fout, "Hundred Hour\n");
				     break;
			    case 3:
				     fprintf(fout, "Thousand Hour\n");
				     break;
			    default:
				     fprintf(fout, "WTF!\n");
				     break;	}

			   NumHist  = MxDesc.NumHist[i][j];
			   NumAlloc = MxDesc.NumAlloc[i][j];

			   fprintf(fout, "NumHist: %ld\n", NumHist);

     	fprintf(fout, "Time  Mth Day    Time     Elev    Temp    RH    CC   Rain\n");

			   a = FirstHist[i][j];
			   while (a != NULL )		{
				    fprintf(fout, "%6.0f %6.0f  %2d-%2d %4d ", a->LastTime, a->SimTime, a->i_Mth,a->i_Day,a->i_Time);
	       fprintf(fout, "  %5ld %7.2f %6.2f %3ld %6.3f",  a->Elevation, a->AirTemperature, a->RelHumidity,a->CloudCover,a->Rainfall);

			    	for (int m = 0; m < NumAlloc; m++)
					      fprintf(fout, " %6.4f", a->Moisture[m]);

				    fprintf(fout, "\n");
				    a = (DeadMoistureHistory *) a->next;
			   }
		}
	}

 	fclose(fout);
 	return true;
}


/****************************************************************************************
* Name:
* Desc:
*
*
***************************************************************************************/
int FE2::Set_FarsiteTime (long l_Adj)
{
DeadMoistureHistory *a;
int i,j;
long l;


	 for (i = 0; i < NumStations; i++)	{
		  for (j = 0; j < NUM_FUEL_SIZES; j++)	{
      a = FirstHist[i][j];
	     if ( a == NULL )
        continue;

			   while (a != NULL )		{
        l = (long) a->LastTime;
        l = l - l_Adj;
        a->LastTime = (double) l;

        l = (long) a->SimTime;
        l = l - l_Adj;
        a->SimTime = (double) l;

				    a = (DeadMoistureHistory *) a->next;
			   }
     }
    }
	return 0;
}



