/*********************************************************************************
* Name: fmc_ci.h - Conditioning Input Class
* Desc: Define the CI Class and related struct and classes
*
**********************************************************************************/
#pragma once

#define GETVAL -1

/* Used to identify Theme Categories, the numbers used could be anything but */
/*  they are carry overs from the FlamMap Condition code, incase of reference */
#define e_FuelModel 3
#define e_Woody     9

/* Used for initializing and checking for unloaded input data in CI class */
#define eI_init -9999
#define eF_init -99999.0

#define eC_Sta  5           /* Number of Stations */
//#ifdef WIN32
#include "fsxsync.h"
//#endif
#ifndef int32
#if UINT_MAX == 65535
typedef long int32;
#else
typedef int int32;
#endif
#endif

/*************************************************************************************/
/*                           Newer Weather Stream Data                               */
/* Note-1 Minute Date is total minutes starting with Jan 1 of e_YearOne   */
/*        thru Dec 31 of year e_YearTwo                                   */
/*        This gives us a way to do date/time arithmatic and find the need */
/*        records based on time                                            */
/* Data in input file can be english or metric, but as soon as it's read  */
/*  it convert to english and stored in the table as such */

typedef long X_HANDLE;

#define e_RAWSend -1      /* goes into i_Yr, to mark end of table */
typedef struct {
   long l_MinDate;       /* See Note-1 above */

   int i_Yr;             /* Year, set by user, ex; 2005, 2006, etc */

/* See CI::RAWS_SetYear() this field really does nothing but may come in handy */
/*  while I'm developing or debugin to help take confusion out of what year is what*/
#define e_SimYearOne 1
#define e_SimYearTwo 2
   int i_SimYear;          /* Simulation year - */

   int i_Mth;
   int i_Day;
   int i_Time;           /* Military 0000->2359 */
   float f_Temp;         /* Temperature Fahrenheit */
   float f_Humidity;     /* whole number 0->100 */
   float f_PerHou;       /* Perceipitation Hourly, Inches  1.00 = one inch */
   float f_CloCov;       /* Cloud Cover percent 0 -> 100  */
} d_RAWS;


/*************************************************************************************/
typedef struct {
	 int    i_Year;
  long  	mo;
	 long  	dy;
	 double 	rn;
	 long 	t1;
	 long 	t2;
	 double 	T1;
	 double 	T2;
	 long 	H1;
	 long 	H2;
	 double 	el;
  long 	tr1;
  long		tr2;
} WeatherData;

typedef struct {  // wind data structure
  long  yr;
	 long 	mo;
	 long 	dy;
	 long 	hr;
	 long 	cl;		// cloudiness
} WindData;


/*************************************************************************/
struct FuelConversions
{
#define eC_FCTyp 257
	int Type[eC_FCTyp];            // each fuel type contains a fuel model corresponding
	void Init();		 // load defaults
};

/*typedef struct
{
     long number;
     char code[8];
     double h1;
     double h10;
     double h100;
     double lh;
     double lw;
     long dynamic;
     long sav1;
     long savlh;
     long savlw;
     double depth;
     double xmext;
     double heatd;
     double heatl;
     char desc[256];
} NewFuel;*/

/******************************************************************/
typedef struct
{   // initial fuel moistures by fuel type
	bool FuelMoistureIsHere;
	long TL1;
	long TL10;
	long TL100;
	long TLLH;
	long TLLW;
} InitialFuelMoisture;


/******************************************************************/
/*class FarsiteEvent
{
public:
	 X_HANDLE *hEvent;
   unsigned long NumEvents;

   FarsiteEvent();
   ~FarsiteEvent();
   bool AllocEvents(long numevents, char *basename, bool ManReset, bool InitState);
   bool FreeEvents();

	  X_HANDLE GetEvent(long ThreadNum);
   bool SetEvent(long ThreadNum);
   bool ResetEvent(long ThreadNum);
   bool WaitForEvents(long numevents, bool All, unsigned long Wait);
   bool WaitForOneEvent(long ThreadNum, unsigned long Wait);
};*/



/****************************************************************************/
/*                    Conditioning Inputs Struct                                  */
/* Still need to deal with this                                            */
#define eC_Hdr 100 /* same size as FlamMap.Header data arrays */
class CI {

public:

   char cr_ErrMes[2000];

/* Tells what Nelson - Fuel Moisture Stick Model to use */
/* Original 0.7.0 - see newfms.h .cpp */
/* Updated - 1.0.0 deadfuelmoistures.h .cpp */
#define e_FSM_Nelson_070  "Nelson-0.7.0"
#define e_FSM_Nelson_100  "Nelson-1.0.0"
#define e_FSM_Default    e_FSM_Nelson_070
    char cr_FuelStickModel[50];

    double d_RunTime; /* Time in minutes for simulation to run */

    long instanceID;

	   double ThreadProgress[64];
    long StartProcessor;
    long l_Latitude;   /* from .lcp file */

    long l_NumWoodie;       /* 9 */
    long lr_Woodie[eC_Hdr];

    long  l_NumFuel;       /* 3 */
    long  lr_Fuel[eC_Hdr];

    long NumWeatherStations,  NumWindStations;

/* Minute Date is a time expressed as total minutes from the starting point of */
/* Jan 1 e_StartYear One,  Jan 1 12 PM = 0, Jan 1 1:00 AM = 60, Jan 1 1:01 AM = 61, . . . */
   long l_StartMinDate;

    int i_StartYear;        /* Actual starting year, Ex: 2001,2003, etc. */
    int i_StartSimYear;     /* Use e_YearOne, e_YearTwo */

    long  startmonth, startday, starthour;

    long  MaxThreads;

	   WeatherData *wtrdt[eC_Sta];
    long MaxWeatherObs[eC_Sta];

/* Wind observation data for each station */
    WindData *wddt[eC_Sta];
    long MaxWindObs[eC_Sta];

/* Weather Stream Data */
    d_RAWS *a_RAWS[eC_Sta];
    int irN_RAWS[eC_Sta];
    int i_RAWS_Elev;
    int RAWS_Allocate (int iN);
    int RAWS_LoadDate (int iX, int i_Yr, int i_Mth, int i_Day, int i_Time);
    int RAWS_LoadInfo ( int iX, float f_Temp, float f_Humidity, float f_PerHou, float f_CloCov);
    int  RAWS_GetCloud (int i_Date, int i_Time);
    int  RAWS_Chrono (double d_SimTim, long *al_MilTim, double *ad_MilTim);
    void RAWS_SetMinDate();
    void RAWS_Display ();
    int  RAWS_GetRecord (long l_SimMin, d_RAWS *a_RAWS);
    int  RAWS_isEnd (int iX);
    void RAWS_GetData (long l_StrTim, long l_Interval, d_RAWS *a_RAWS );
    double RAWS_GetAccumRain (long l_StrMinDate, long l_EndMinDate );
    void RAWS_Sum (d_RAWS *a_To, d_RAWS *a_Frm, float f_Min, float *af_TotMinAvg);
    long RAWS_MinInRec (int iX);
    void RAWS_SetMinuteDate();
    long RAWS_CalcMinuteDate  (int Yr, int Mth, int Day, int MilTim);

    int   MilTim_To_Min (int i_MilTim);

    FuelConversions fuelconversion;
//    NewFuel newfuels[257];
 //   int HAVE_CUST_MODELS;

#define eC_fm 257
   	InitialFuelMoisture fm[eC_fm];

   	bool EnvtChanged[4][eC_Sta];  //={{false, false, false, false, false},

    long l_ElevHigh;
    long l_ElevLow;
    short s_EUnits;

    long l_SlopeHigh;
    long l_SlopeLow;
    short s_SUnits;

    long l_CoverHigh;
    long l_CoverLow;
    short s_CUnits;

/* Interval......................................*/
/* [4] = Fuel Sizes, [5] Categories, see #defines FM_INTERVAL_TIME, etc */

	   long MoistCalcInterval[4][5];

 #ifdef WIN32
    FarsiteEvent hMoistEvent;
    FarsiteEvent hMoistThreadEvent;
#endif

    int i_MCHI;   /* Moisture Calc Hour Interval 1,10,100,1000 Hr */

/* Calculate 1000 Hr fuel moisture switch */
/* 1 = Yes do 1000 Hr,  0 = No don't do */
    int iS_1k;

/*.........................................................................*/
/*  Functions   */
//    CI ();
    void Init ();
    void Delete();
    void InitNLC();
    int InitCheck (char cr[]);

    void Set_MoistCalcHourInterval (int i);
    long  Get_MoistCalcHourInterval ();

    int CheckInputs ();
    int CheckWeatherWind ();
    int CheckWeatherStream ();

    int GetErrMes (char cr_ErrMes[]);

    int Set_FuelModel (int32 lr_Data[], int32 lN, char cr_ErrMes[]);
    int Set_Woody (int32 lr_Data[], int32 lN, char cr_ErrMes[]);
    int SetFuelCats (int32 lr_Data[], int32 lN, int i_Type, char cr_ErrMess[]);

    void FuelStickModel_Nelson_070 ();
    void FuelStickModel_Nelson_100 ();
    bool FuelStickModel_IsNelson_070();

    void Set_RunTime (double d);
    //void ResetNewFuels();
    //void InitializeNewFuel();

    void SetLatitude (long lat);
    long GetLatitude ();

    void  SetNumStations (long WeaSta, long WidSta);
    long  GetNumStations();
    int   GetFuelCats (int i_Type, long lr_Data[]);
    int   GetFuelConversion (int i);
    //bool  IsNewFuelReserved (long number);
    long  GetInitialFuelMoisture (long Model, long FuelClass);

    int   SetMoistures(int fuelModel, int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody);
    void  SetAllMoistures(int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody);

    int   Set_Elev (long High, long Low, short ThemeUnit);
    int   Set_Slope (long High, long Low, short ThemeUnit);
    int   Set_Cover (long High, long Low, short ThemeUnit);

    short	GetTheme_Units (short DataTheme);

    inline long  GetLoElev() { return l_ElevLow; }
    inline long  GetHiElev() { return l_ElevHigh; }

 	  long  GetMoistCalcInterval (long FM_SIZECLASS, long CATEGORY);

    long  GetTheme_LoValue (short DataTheme);
    long  GetTheme_HiValue (short DataTheme);

    long   GetMaxThreads();

    int AllocWindData_Sta0 (int iN);
    int SetWindData_Sta0 (long NumObs, long year, long month, long day, long hour, long cloudy);

    long AllocWeatherData_Sta0 (long NumObs);


    int SetWeatherData_Sta0 (long NumObs, long month, long day,
                         double rain, long time1, long time2,	double temp1,
                         double temp2, long humid1, long humid2, double elevation,
                         long tr1, long tr2);

    void  LogicError (char cr[]);
    int   wtrdt_idx ( int i_JulDatFind, int sn);

//    long   GetWeatherMonth(long StationNumber, long NumObs);
//    long   GetWeatherDay(long StationNumber, long NumObs);

    double GetWeatherTemp1(long StationNumber, long NumObs);
    double GetWeatherTemp2(long StationNumber, long NumObs);
    long   GetWeatherTime1(long StationNumber, long NumObs);
    long   GetWeatherTime2(long StationNumber, long NumObs);
    long   GetWeatherHumid1(long StationNumber, long NumObs);
    long   GetWeatherHumid2(long StationNumber, long NumObs);
    double GetWeatherElev(long StationNumber, long NumObs);
    double GetWeatherRain(long StationNumber, long NumObs);
    long   GetWindCloud(long StationNumber, long NumObs);
    long   GetWindDay(long StationNumber, long NumObs);
    long   GetWindMonth(long StationNumber, long NumObs);
    long   GetWindHour(long StationNumber, long NumObs);

    long   GetCloud (long time,  long sn);

    long   Chrono(double SIMTIME, long *hour, double *hours);
    long   GetJulianDays (long Month);
    int    GetJulian(int i_Mth, int i_Day);

    double ConvertActualTimeToSimtime(long mo, long dy, long hr, long mn);
    bool   EnvironmentChanged(long YesNo, long StationNumber, long FuelSize);

   void  ConvertSimtimeToActualTime (double SimTime, long *mo, long *dy, long *hr, long *mn);
   void  ConvertSimtimeToActualTime_Fix (double SimTime, long *mo, long *dy, long *hr, long *mn);


    void   GetWeatherRainTimes (long StationNumber, long NumObs, long *tr1, long *tr2);
    long   AtmosphereGridExists();

    bool    SetFarsiteEvent (long event, long threadorder);
    X_HANDLE  GetFarsiteEvent (long event, long threadorder);
    bool    WaitForFarsiteEvents (long EventNum, long numevents, bool All, unsigned long Wait);

    bool	FreeFarsiteEvents(long EventNum);
    bool AllocFarsiteEvents(long EventNum, long numevents, char *basename, bool ManReset, bool InitState);
    void ResetThreads();
    void SetInstanceID (long id);
    long GetStartProcessor ();

    double GetWoodyFuelMoisture (long ModelNumber, long SizeClass);

  };
