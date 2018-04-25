/**********************************************************************************
* Name: fmc_cfmc.cpp
* Desc: Dead Fuel Moisture Conditioning DLL Class interface functions.
* Date: 11-01-09
***********************************************************************************/
//#include <windows.h>
#include "cdtlib.h"
#include "newfms.h"

#include "semtime.h"
#include "deadfuelmoisture.h"
#include "FMC_CI.h"
#include "FMC_FE2.h"
#include "FMC_CFMC.h"

long GUniqueInstance = 1;
int critsec = 0;

#ifdef WIN32
CRITICAL_SECTION FMC_InstanceCS;

//*********************************************************
BOOL APIENTRY DllMain ( HMODULE hModule, DWORD ul_reason_for_call,LPVOID lpReserved)
{
   // Perform actions based on the reason for calling.
    switch( ul_reason_for_call )  {
        case DLL_PROCESS_ATTACH:  // Init once for each new process. Return FALSE to fail DLL load.
		        	if ( !critsec )	{
			 	        InitializeCriticalSection(&FMC_InstanceCS);
			         	critsec = 1;	}
           break;

        case DLL_THREAD_ATTACH:  // Do thread-specific initialization.
            break;

        case DLL_THREAD_DETACH:  // Do thread-specific cleanup.
            break;

        case DLL_PROCESS_DETACH:   // Perform any necessary cleanup.
			       if ( critsec )	{
		         		DeleteCriticalSection(&FMC_InstanceCS);
			         	critsec = 0;	}
          break;}
    return TRUE;
}
#endif
/**************************************************************************/
void CFMC::Set_RunTime ( double d)
{
   this->a_FE2->a_CI->Set_RunTime (d);
}

/****************************************************************
* Name: GetMx
* Desc: Get fuel conditioning calculated outputs
*   In:
*  Out:
*  Ret:
****************************************************************/
double CFMC::GetMx(double d_Time, long l_fuel, long l_elev, long l_slope,	double d_aspectf,
                  long l_cover, double *ad_equil, double *ad_solrad, long l_FuelSize)
{
double d;
    d = this->a_FE2->GetMx( d_Time,  l_fuel, l_elev, l_slope,	d_aspectf,
                  l_cover, ad_equil, ad_solrad, l_FuelSize)	;
    return d;
}

/****************************************************************
* Name: AllocWindData_Sta0
* Desc: See comments in a_CI->AllocWindData_Sta0()
*   In: number of wind records
*  Ret: 1 ok, 0 error
****************************************************************/
int CFMC::AllocWindData_Sta0 ( int iN )
{
int i;
   i = this->a_FE2->a_CI->AllocWindData_Sta0(iN);
   return i;
}

/****************************************************************
* Name:
* Desc:
*   In:
*  Ret:
****************************************************************/
int CFMC::SetWindData_Sta0 (long NumObs, long year, long month, long day, long hour, long cloudy)
{
int i;
   i = this->a_FE2->a_CI->SetWindData_Sta0 (NumObs, year, month,day,hour,cloudy);
   return i;
}

/****************************************************************
* Name:
* Desc:
*   In:
*  Ret:
****************************************************************/
int CFMC::AllocWeatherData_Sta0 (long NumObs)
{
int  i;
  i = this->a_FE2->a_CI->AllocWeatherData_Sta0 (NumObs);
  return i;
}

/***************************************************************************
* Name: SetWeatherData_Sta0
* Desc: Use english units
***********************************************************************/
int CFMC::SetWeatherData_Sta0 (long NumObs, long month, long day,
                         double rain, long time1, long time2,	double temp1,
                         double temp2, long humid1, long humid2, double elevation,
                         long tr1, long tr2)
{
int i;
    i = this->a_FE2->a_CI->SetWeatherData_Sta0 (NumObs, month, day,
             rain, time1, time2, temp1,temp2, humid1, humid2,  elevation, tr1,  tr2);
    return i;
}

/****************************************************************
* Name: Set_Elev
* Desc: Set Elevation
*   In: ThemeUnit..1 = feet, 0 meter.
*  Ret: 1 - OK
****************************************************************/
int CFMC::Set_Elev (long High, long Low, short ThemeUnit)
{
int i;
   i = this->a_FE2->a_CI->Set_Elev (High, Low, ThemeUnit);
   return i;
}

int CFMC::Set_Slope (long High, long Low, short ThemeUnit)
{
int i;
   i = this->a_FE2->a_CI->Set_Slope (High, Low, ThemeUnit);
   return i;
}

int CFMC::Set_Cover (long High, long Low, short ThemeUnit)
{
int i;
   i = this->a_FE2->a_CI->Set_Cover (High, Low, ThemeUnit);
   return i;
}

/**************************************************************************
* Name: Set_DateStart
* Desc: Date to start condtioning on
*   In: date...julian date
**************************************************************************/
int  CFMC::Set_DateStart (int i_StartYear, long month, long day, long hour)
{
   this->a_FE2->a_CI->i_StartYear = i_StartYear;
   this->a_FE2->a_CI->startmonth = month;
   this->a_FE2->a_CI->startday = day;
   this->a_FE2->a_CI->starthour = hour;
   return 1;
}

/***********************************************************************/

#ifdef  NOT_SURE_NEEDED
int  CFMC::Set_DateEnd (long date, long month, long day, long hour, long min)
{
   this->a_FE2->a_CI->enddate = date;
   this->a_FE2->a_CI->endmonth = month;
   this->a_FE2->a_CI->endday = day;
   this->a_FE2->a_CI->endhour = hour;
   this->a_FE2->a_CI->endmin = min;
   return 1;
}
#endif

/********************************************************
* Name: Run
* Desc: Run the Conditioning Simulation
*  Ret: 1 = completed ok, 0 terminiated by user
**********************************************************/
int CFMC::Run ()
{
int i;
   i = this->a_FE2->CondMngr();
   return i;
}

/*************************************************************
* Name: Init
* Desc: Allocate a FireEnvironment2 class and initialize it
* Note: The FireEnvironment2 class was taken from FlamMap, because
*        it handles doing the dead fuel moisture conditioning.
*  Ret: 1 ok
**************************************************************/
int CFMC::Init()
{
   this->a_FE2 = new FE2;
   if ( !this->a_FE2->Init() )
      return 0;
#ifdef WIN32
	EnterCriticalSection(&FMC_InstanceCS);
#endif
	GUniqueInstance++;
	SetInstanceID(GUniqueInstance);
#ifdef WIN32
	LeaveCriticalSection(&FMC_InstanceCS);
#endif
   return 1;
}

/*************************************************************
* Name: Delete
* Desc: NEED to do this when when done with object to clean
*        up
* NOTE: There is no destructor for the class SO THIS needs
*         must be called
* Note-1: This pointer will only have something in it if the
*         CFMC::Init() was run, otherwise it will be NULL from
*          the consructor function....So it's safe to call this
*         function even if the DLL wasn't used.
**************************************************************/
void CFMC::Delete()
{
  if ( this->a_FE2 == NULL ) /* See Note-1 Above */
    return;

  this->a_FE2->a_CI->Delete();
  this->a_FE2->FreeStations();
  this->a_FE2->FreeFmsThreads();

  if ( this->a_FE2->a_CI) {
    delete this->a_FE2->a_CI;
    this->a_FE2->a_CI = NULL; }

  if ( this->a_FE2) {
    delete this->a_FE2;
    this->a_FE2 = NULL; }
}
/***********************************************************
* Name: Terminate
* Desc: If the condtioning simulation is in progress this
*       will cause it to terminate
***********************************************************/
void CFMC::Terminate()
{
 if ( this->a_FE2 == NULL )
   return;

  this->a_FE2->Terminate();
}

/*******************************************************
* Name: Set_ThreadsProcessor
* Desc: Set the number of threads to be used
*       Set the starting processor number
*******************************************************/
void CFMC::Set_ThreadsProcessor (long threads, long StartProcessor)
{
#ifdef WIN32
SYSTEM_INFO sysinf;
	 	GetSystemInfo(&sysinf);
/* Don't set # of threads to more processors on system */
 	 if ( threads > sysinf.dwNumberOfProcessors )
     this->a_FE2->a_CI->MaxThreads = sysinf.dwNumberOfProcessors;
   else
     this->a_FE2->a_CI->MaxThreads = threads;

/* I don't think it's that important what the starting processor # is, */
/*  so just check and set to 0 if need be */
#endif
   if ( StartProcessor < 0 || StartProcessor > this->a_FE2->a_CI->MaxThreads )
     this->a_FE2->a_CI->StartProcessor = 0;
   else
     this->a_FE2->a_CI->StartProcessor = StartProcessor;
}


void CFMC::ResetThreads ()
{
   this->a_FE2->ResetAllThreads();
 	 this->a_FE2->a_CI->ResetThreads();
}

void CFMC::SetInstanceID (long id)
{
   this->a_FE2->a_CI->SetInstanceID(id);
}

/***************************************************************/
void CFMC::SetLatitude(long lat)
{
   this->a_FE2->a_CI->SetLatitude(lat);
}
/***************************************************************/
void CFMC::SetNumStations (long WeaSta, long WidSta)
{
   this->a_FE2->a_CI->NumWeatherStations = WeaSta;
   this->a_FE2->a_CI->NumWindStations = WidSta;
}

/**********************************************************
* Name: Set_FuelModel
* Desc: Set the landscape theme FuelModel categories
*   In: arrya of data and size
**********************************************************/
int  CFMC::Set_FuelModel (int32 lr_Data[], int32 lN, char cr_ErrMes[])
{
int i;
   i = this->a_FE2->a_CI->Set_FuelModel (lr_Data, lN, cr_ErrMes);
   return i;
}

/**********************************************************
* NOTE: At this point it doesn't appear that this even
*       gets used in the Conditioning code,
* Name: Set_Woody
* Desc: Set the landscape theme Woody categories
*   In: arrya of data and size
**********************************************************/
int CFMC::Set_Woody (int32 lr_Data[], int32 lN, char cr_ErrMes[])
{
int i;
   i = this->a_FE2->a_CI->Set_Woody (lr_Data, lN, cr_ErrMes);
   return i;
}

/********************************************************************
* Name:
* Desc:
*
*
*********************************************************************/
int CFMC::Set_FuelMoistModel (int fuelModel, int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody)
{
int i;
   i = this->a_FE2->a_CI->SetMoistures(fuelModel, _fm1, _fm10, _fm100, _fmHerb, _fmWoody);
   return i;
}
/*----------------------------------------------------------------------------*/
void CFMC::Set_DefFuelMoistModel (int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody)
{
   this->a_FE2->a_CI->SetAllMoistures(_fm1, _fm10, _fm100,	_fmHerb, _fmWoody);
}

/********************************************************************
* Name: CheckInputs
* Desc: Check all the inputed data,
*  Ret: 1 ok, ICF Error code
********************************************************************/
int CFMC::CheckInputs ()
{
int i;
  i = a_FE2->a_CI->CheckInputs();
  return i;
}

/**********************************************************************/
bool CFMC::CheckMoistureTimes(double SimTime)
{
bool b;
    b = a_FE2->CheckMoistureTimes(SimTime);
    return b;
}

/********************************************************************
* Name: Set_1kOff
* Desc: Tell DLL not to computer 1000 fuel moistures.
*       Farsite doesn't use these moistures
*******************************************************************/
void  CFMC::Set_1kOff ()
{
    a_FE2->a_CI->iS_1k = 0;
}

/****************************************************************
* Name: GetErrMes
* Desc: Get any error message that may have been generated
*  Ret: 1 if now, 0 if error message
***************************************************************/
int CFMC::GetErrMes (char cr_ErrMes[])
{
int i;
  i = a_FE2->a_CI->GetErrMes(cr_ErrMes);
  return i;
}

/************************************************************
* See CFMC
* NOTE: Initialization is not do here it MUST be done
*       by the user with the CFMC::Init()
************************************************************/
CFMC::CFMC()
{
  this->a_FE2 = NULL;
}

/*************************************************************
* Cleanup is NOT done here, user MUST use the
*  CFMC.Delete() when they are done with the object.
**************************************************************/
CFMC::~CFMC ()
{
  Delete ();
}

/***************************************************************
* Name: HaveFuelMoist
* Desc: See if a fuel moist has been calculation for the
*       specified station & Fuel Size Class
*  Ret: True, yes it's been calc'd - False - nope
****************************************************************/
bool CFMC::HaveFuelMoist(long Station, long FuelSize)
{
bool b;
  b = this->a_FE2->HaveFuelMoist(Station,FuelSize);
  return b;
}

/***************************************************************
* Name: ElevTempHum
* Desc: The returned values from the DLL
***************************************************************/
void CFMC::ElevTempHum (long *al_elev, double *ad_temp, double *ad_hum)
{
   this->a_FE2->ElevTempHum (al_elev,ad_temp, ad_hum);
}

/**************************************************************
* Name: FuelStickMode_***
* Desc: used to set which underlying Nelson Fuel Stick
*       Moisture Model is to be used.
* Note: Nelson 0.7.0  1997-2000 Original is the newfms.h .cpp
*       Nelson 1.0.0  2005 Updated is the deadfuelmoisture.h .cpp new stuff that
*        implemented in approx Oct 2009.
**************************************************************/
void CFMC::FuelStickModel_Nelson_070 ()
{
  this->a_FE2->a_CI->FuelStickModel_Nelson_070();
}

void CFMC::FuelStickModel_Nelson_100 ()
{
  this->a_FE2->a_CI->FuelStickModel_Nelson_100();
}

/****************************************************************
* Name: Get_Progress....see below
* Desc: Get the progress, percent completed, of conditiong run
* Note-1: Caller (FlamMap/Farsite) could have a progress thread
*          that as for progress before object gets alloced.
* NOTE: 3 functions
*  Ret: double 0 -> 1.0
*       float  0 -> 1.0
*       int    0 -> 100
*****************************************************************/
double CFMC::Get_ProgressD()
{
double d;
  if ( this->a_FE2 == NULL )   /* See Note-1 above */
     return 0;
  d = this->a_FE2->Get_Progress();
  return d;
}
/*-------------------------------------------------------------------*/
float CFMC::Get_ProgressF()
{
float f;
  if ( this->a_FE2 == NULL )   /* incase not alloced yet */
     return 0;

  f = (float) this->a_FE2->Get_Progress();
  return f;
}
/*------------------------------------------------------------------*/
/* Return as integer value 0 --> 100  */
int CFMC::Get_ProgressInt()
{
int i;
double d;
  if ( this->a_FE2 == NULL )   /* incase not alloced yet */
     return 0;

  d = this->a_FE2->Get_Progress();
  d = d * 100.0;  /* Get to whole number */
  i = (int) d;
  return i;
}

/*******************************************************************
* Name: Set_MoistCalcHourInterval
* Desc: Set the the fuel size class that is to be used to
*        determine the conditionings simulation time interval
* NOTE: Normaly this is set to 10 and if this function is not called
*        10 will be used for a default.
*   In: 1, 10, 100, 1000 Hour
* NOTE: any other value than above will cause default to get set
********************************************************************/
void CFMC::Set_MoistCalcHourInterval (int i)
{
  this->a_FE2->a_CI->Set_MoistCalcHourInterval(i);
}

/*************************************************************************/
void CFMC::SiteSpecific (long elev, double *ad_airtemp, double *ad_relhumd)
{
   this->a_FE2->SiteSpecific (elev, ad_airtemp, ad_relhumd);
}


/**************************************************************************
* Name:  See below
* Desc: Functions to load Weather Stream Data into the Cond DLL
*   In: see comments in the a_CI-> corresponding functions.
***************************************************************************/
int CFMC::RAWS_LoadInfo ( int iX, float f_Temp, float f_Humidity, float f_PerHou,
                         float f_CloCov)
{
int i;
   i = this->a_FE2->a_CI->RAWS_LoadInfo (iX, f_Temp, f_Humidity, f_PerHou, f_CloCov);
   return i;
}

/*----------------------------------------------------------------------------*/
int CFMC::RAWS_LoadDate (int iX, int i_Yr, int i_Mth, int i_Day, int i_Time)
{
int i;
   i = this->a_FE2->a_CI->RAWS_LoadDate (iX, i_Yr, i_Mth, i_Day, i_Time);
   return i;
 }


/*----------------------------------------------------------------------------
* Ret: 0....error allocating memory
---------------------------------------------------------------------------*/
int CFMC::RAWS_Allocate (int iN)
{
int i;
   i= this->a_FE2->a_CI->RAWS_Allocate (iN);
   return i;
}

/*--------------------------------------------------------------------------
* Elevation - feet
* Elevation that corresponds with WSD weather stream data
---------------------------------------------------------------------------*/
void CFMC::RAWS_SetElev(int i_Elev)
{
  this->a_FE2->a_CI->i_RAWS_Elev = i_Elev;
}

/*----------------------------------------------------------------------------*/
void CFMC::RAWS_Display()
{
   this->a_FE2->a_CI->RAWS_Display();
}


bool CFMC::ExportMoistureDataText(char *FileName, char * LCP_FilNam)
{
bool b;
   b = this->a_FE2->ExportMoistureDataText(FileName, LCP_FilNam);
   return b;
}

int CFMC::Set_FarsiteDef (long l)
{
int i;
   i = this->a_FE2->Set_FarsiteTime(l);
   return i;
}

int CFMC::SimTime_ActualTime (double SimTime, long* mo, long* dy, long* hr, long* mn)
{
   this->a_FE2->a_CI->ConvertSimtimeToActualTime (SimTime, mo, dy, hr, mn);

   return 1;

}

