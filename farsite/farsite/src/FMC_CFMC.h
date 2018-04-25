/**********************************************************************************
* Name: fmc_cfmc.h
* Desc: Dead Fuel Moisture Conditioning DLL Class interface functions.
* Date: 11-01-09
***********************************************************************************/

#pragma once

/*#ifdef BUILD_FMC_DLL
#define FMC_DLL_EXPORT __declspec(dllexport)
#else
#define FMC_DLL_EXPORT __declspec(dllimport)
#endif */
#define FMC_DLL_EXPORT

class FE2;

#ifndef int32
#if UINT_MAX == 65535
typedef long int32;
#else
typedef int int32;
#endif
#endif
/*******************************************************************************/
class FMC_DLL_EXPORT CFMC {
public:
  	CFMC();
	 ~CFMC();

   FE2 *a_FE2;

   void FuelStickModel_Nelson_070 ();
   void FuelStickModel_Nelson_100 ();

   int Run ();
   int Init ();
   void Delete ();
   void Terminate ();
   bool HaveFuelMoist(long Station, long FuelSize);
   void ElevTempHum (long *al_elev, double *ad_temp, double *ad_hum);
   void SetLatitude (long lat);
   void SetNumStations (long WeaSta, long WidSta);
   void SetElev (long ar_Elev[], long NumElev);
   void Set_MoistCalcHourInterval (int i);
   void SiteSpecific (long elev, double *ad_airtemp, double *ad_relhumd);

    int Set_FuelModel (int32 lr_Data[], int32 lN, char cr_ErrMes[]);
    int Set_Woody (int32 lr_Data[], int32 lN, char cr_ErrMes[]);

   int  Set_FuelMoistModel(int fuelModel, int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody);
   void Set_DefFuelMoistModel (int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody);
   int  Set_Elev (long High, long Low, short ThemeUnit);
   int  Set_Slope (long High, long Low, short ThemeUnit);
   int  Set_Cover (long High, long Low, short ThemeUnit);
   int  Set_DateStart (int i_StartYear, long startmonth, long startday, long starthour);
   void Set_ThreadsProcessor (long threads, long StartProcessor);
   void ResetThreads ();
   void SetInstanceID (long id);

   int  AllocWindData_Sta0 (int iN);
   int  SetWindData_Sta0 (long NumObs, long year, long month, long day, long hour, long cloudy);


   int AllocWeatherData_Sta0 (long NumObs);

    int SetWeatherData_Sta0 (long NumObs, long month, long day,
                         double rain, long time1, long time2,	double temp1,
                         double temp2, long humid1, long humid2, double elevation,
                         long tr1, long tr2);

    double GetMx(double Time, long fuel, long elev, long slope,	double aspectf,
                  long cover, double *equil, double *solrad, long FuelSize);

   int  CheckInputs ();
   void Set_RunTime (double d);

   int  GetErrMes (char cr_ErrMes[]);

   float  Get_ProgressF ();
   double Get_ProgressD ();
   int    Get_ProgressInt ();

   bool CheckMoistureTimes(double SimTime);
   void Set_1kOff ();

/* Weather Stream Data functions */
   int RAWS_LoadInfo (int iX, float f_Temp, float f_Humidity, float f_PerHou, float f_CloCov);
   int RAWS_LoadDate (int iX, int i_Yr, int i_Mth, int i_Day, int i_Time);
   int RAWS_Allocate (int iN);
   void RAWS_SetElev (int i_Elev);
   void RAWS_Display ();


   bool ExportMoistureDataText(char* FileName, char* LCP_FilNam);
   int  Set_FarsiteDef (long l);

   int  SimTime_ActualTime (double SimTime, long* mo, long* dy, long* hr, long* mn);

};
