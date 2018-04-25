/* Farsite20 */

#pragma once
#include "fsxlandt.h"
#include "fsxwatm.h"
#include <stdio.h>
#include "icf_def.h"
#include "fsairatk.h"
#include "fsxpfront.h"
#include "fsxwattk.h"
#include "fsx4.hpp"
#include "themes.h"
#include "fsxwignt.h"
#include "fsxwshap.h"
#include "polygon.h"
#include "PerimeterData.h"
#include "rand3.h"
#include <list>
#include <algorithm>

#include "FMC_CFMC.h"

#include "Far_BSG.h"

// WN-Test
//#include "..\WindNinja2\CWindNinja2.h"
/*#ifndef max
#define max(a,b)        (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)        (((a) < (b)) ? (a) : (b))
#endif*/

int GetMCDate(int mth, int day, int year);
int MilToMin (int i_MilTim);

/*********************************************************************************/
inline double 	pow2(double in)//;               			// returns square of base
{
	return in * in;
}
const double TinyVal = 1e-9;
inline bool IsTiny(double in){return (in < TinyVal && in > -TinyVal) ? true : false;}
long const MaxNumCompoundCrews = 200;

#define NODATA_VAL -9999
const size_t headsize = 7316;
#define GETVAL -1		// definitions used for getting and storing data
#define XCOORD 0		// in fire perimeter1 arrays:
#define YCOORD 1
#define ROSVAL 2
#define FLIVAL 3
#define RCXVAL 4

#define NUMDATA 5

#define MAXNUM_STOPLOCATIONS 10

#define DISTCHECK_SIMLEVEL  	0
#define DISTCHECK_FIRELEVEL  1

#define SIZE_FIREPERIM_BLOCK 1000

#define RAST_ARRIVALTIME      20
#define RAST_FIREINTENSITY    21
#define RAST_SPREADRATE       22
#define RAST_FLAMELENGTH      23
#define RAST_HEATPERAREA      24
#define RAST_CROWNFIRE        25
#define RAST_FIREDIRECTION    26
#define RAST_REACTIONINTENSITY 27

#define MAXNUM_COARSEWOODY_MODELS 256
#define CWD_COMBINE_NEVER		1
#define CWD_COMBINE_ALWAYS		2
#define CWD_COMBINE_ABSENT		3

#define FM_INTERVAL_TIME  	0
#define FM_INTERVAL_ELEV		1
#define FM_INTERVAL_SLOPE	2
#define FM_INTERVAL_ASP		3
#define FM_INTERVAL_COV		4

#define E_DATA			0
#define S_DATA			1
#define A_DATA			2
#define F_DATA			3
#define C_DATA			4
#define H_DATA			5
#define B_DATA			6
#define P_DATA			7
#define D_DATA			8
#define W_DATA			9

#define NULLLOCATION		0
#define IGNITIONLOCATION 	1
#define BARRIERLOCATION 		2
#define FUELLOCATION     	3
#define WEATHERLOCATION  	4
#define ZOOMLOCATION     	5
#define EDITLOCATION     	6
#define DIRECTLOCATION   	7
#define INDIRECTLOCATION 	8
#define PARALLELLOCATION 	9
#define AERIALLOCATION   	10
#define RELOCATEDIRECT   	11
#define RELOCATEINDIRECT 	12
#define RELOCATEPARALLEL 	13
#define RELOCATEAIRATTACK 	14
#define ADDGROUNDATTACK		15
#define FIREINFORMATION 		16
#define MOVEPERIMPOINTS		17
#define PANLANDSCAPE		18
#define STOPLOCATION		19
#define MEASURELANDSCAPE		20

#define SIMREQ_NULL					0
#define SIMREQ_INITIATETERMINATE 	1
#define SIMREQ_STARTRESTART 		2
#define SIMREQ_RESUMESUSPEND 		3
#define SIMREQ_RESET				4

//----------------------------------------------------------
//
//   Burn Period functions
//
//----------------------------------------------------------

typedef struct
{
	long Month;
     long Day;
     long Year;      /* Add this 4-11-12, to help check against wther */
     long Start;
     long End;
} AbsoluteBurnPeriod;


typedef struct
{
	double Start;
     double End;
} RelativeBurnPeriod;


typedef struct
{
     double SurfaceAreaToVolume;
     double Load;
     double HeatContent;
     double Density;
     double FuelMoisture;
} WoodyData;

struct CoarseWoody
{
	long Units;
     double Depth;
     double TotalWeight;
	long NumClasses;
     char Description[64];
     WoodyData *wd;

     CoarseWoody();
};

typedef struct
{
     long ID;
     VectorTheme theme;
     void *next;
     void *last;
} VectorStorage;



//
//   new fuel model support
//
//------------------------------------------------------------------

typedef struct
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
}NewFuel;

// WN-Test...........
/*----------------------------------------------------------------------*/
/*                 Farsite WindNinja Struct                             */
/* used in the WindData struct tbl                                      */
typedef struct  {

 float f_Temperature;  /* weather & cloud at time of simulation */
 long  nWindRows;      /* size of grid */
 long  nWindCols;
 double windsXllCorner;
 double windsYllCorner;
 double windsResolution;

	short  **windDirGrid;
	short  **windSpeedGrid;
 } d_FWN;

/*********************************************************************/
typedef struct   {   // wind data structure
	 long 	mo;
	 long 	dy;
     long   yr;      // add 4-11-12
	 long 	hr;
	 double 	ws;		// windspeed mph
	 long 	wd;		// winddirection azimuth
	 long 	cl;		// cloudiness

// WN-Test
#define e_NA  0  /* Not Applicable, record isn't in a burn period */
#define e_BP  1  /* Record is in the burn period sim time */
#define e_WN  2  /* Record has Ninja run on it and contains speed & dir grids */
#define e_PT  3  /* Record contains Pointer to a record that has the grids */
  int iS_WN;
  d_FWN *a_FWN;
// WN-Test
  }  WindData;

/**********************************************************************/
typedef struct {
	 long 	mo;
	 long 	dy;
     long   yr;     // add 4-11-12
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

/*********************************************************************/
struct FuelConversions
{
	int Type[257];            // each fuel type contains a fuel model corresponding
	FuelConversions();		 // load defaults
};

typedef struct
{// initial fuel moistures by fuel type
	bool FuelMoistureIsHere;
	long TL1;
	long TL10;
	long TL100;
	long TLLH;
	long TLLW;
} InitialFuelMoisture;

struct StationGrid
{
	long 	XDim;                 // number of Easting cells
	long 	YDim;                 // number of Northing cells
	double 	Width;              // width of grid cell
	double 	Height;             // height of grid cell
	long 	*Grid;                // holds weather/wind station numbers
	StationGrid();
	~StationGrid();
};

class CSpotData
{
public:
	CSpotData();
	~CSpotData();
	double launchTime;
	double launchX;
	double launchY;
	double landTime;
	double landX;
	double landY;
};

#if UINT_MAX == 65535
typedef long int32;
#else
typedef int int32;
#endif
typedef struct
{// header for landscape file
	int32 CrownFuels;         // 20 if no crown fuels, 21 if crown fuels exist
     int32 GroundFuels;		// 20 if no ground fuels, 21 if ground fuels exist
	int32 latitude;
	double loeast;
	double hieast;
	double lonorth;
	double hinorth;
	int32 loelev;
	int32 hielev;
	int32 numelev; 			//-1 if more than 100 categories
     int32 elevs[100];
     int32 loslope;
     int32 hislope;
     int32 numslope;			//-1 if more than 100 categories
     int32 slopes[100];
     int32 loaspect;
     int32 hiaspect;
     int32 numaspect;		//-1 if more than 100 categories
     int32 aspects[100];
     int32 lofuel;
     int32 hifuel;
     int32 numfuel;			//-1 if more than 100 categories
     int32 fuels[100];
     int32 locover;
     int32 hicover;
     int32 numcover;			//-1 if more than 100 categories
     int32 covers[100];
     int32 loheight;
     int32 hiheight;
     int32 numheight;		//-1 if more than 100 categories
     int32 heights[100];
     int32 lobase;
     int32 hibase;
     int32 numbase;			//-1 if more than 100 categories
     int32 bases[100];
     int32 lodensity;
     int32 hidensity;
     int32 numdensity;		//-1 if more than 100 categories
     int32 densities[100];
     int32 loduff;
     int32 hiduff;
     int32 numduff;			//-1 if more than 100 categories
     int32 duffs[100];
     int32 lowoody;
     int32 hiwoody;
     int32 numwoody;			//-1 if more than 100 categories
     int32 woodies[100];
	int32 numeast;
	int32 numnorth;
	double EastUtm;
	double WestUtm;
	double NorthUtm;
	double SouthUtm;
	int32 GridUnits;        // 0 for metric, 1 for English
	double XResol;
	double YResol;
	short EUnits;
	short SUnits;
	short AUnits;
	short FOptions;
	short CUnits;
	short HUnits;
	short BUnits;
	short PUnits;
     short DUnits;
     short WOptions;
     char ElevFile[256];
     char SlopeFile[256];
     char AspectFile[256];
     char FuelFile[256];
     char CoverFile[256];
     char HeightFile[256];
     char BaseFile[256];
     char DensityFile[256];
     char DuffFile[256];
     char WoodyFile[256];
     char Description[512];
}headdata;

typedef struct
{// structure for holding weather/wind stream data max and min values
	int 	wtr;
	int 	wnd;
	int 	all;
}streamdata;
using std::list;
typedef list<CSpotData> CSpotList;

/*****************************************************************************/
/*                        Farsite Progress Class                    */
class FPC {
public:
#define e_Start   0
#define e_Ninja   1
#define e_NinjaPrep 2    /* Ninja preparing to run */
#define e_PreCond 3
#define e_Cond    4
#define e_Far     5
   int i_State;


   float f_WNToDo;       /* How Many ninja runs are needed */
   float f_WNComplete;   /* How many competed so far */


  float f_PreProg;

  float f_pcNinja;    /* percentages of time that each section */
  float f_pcCond;     /* is approximated to run */
  float f_pcFar;

   FPC();
   ~FPC();
   void Init();
   //float GetProgress(Farsite5 *a_F5, CWindNinja2 *a_WN2, CFMC *a_cfmc, char cr[]);
   float GetProgress(Farsite5 *a_F5, CFMC *a_cfmc, char cr[]);
  // float WindNinja_Progress (CWindNinja2 *a_WN2);
   float LeaCriSec (float f);


   int SetTimeSlice (int i_NumNinRun, Farsite5 *aFar);
   float GetCondTime (Farsite5 *a_F5);

   void Set_NinjaRunning (float f);
   void Set_NinjaPrep ();
   void Set_CondRunning ();
   void Set_FarsiteRunning ();

   void Set_NinjaInc();

	 // CRITICAL_SECTION CriSec;
 };


class Farsite5
{
public:
  BSG m_BSG;    /* Burn Spot Grid - tracks burnt areas and spotting hits */

  FPC m_FPC;    /* Farsite Progress Class */

  int ExportMoistureDataText(char *trgName);

//___________________________________________________________________
// WN-New....
 // CWindNinja2 WN2;
  float  fN_WNToDo;      /* Number of Ninja runs that will need to be run */
  float  fN_WNDone; /* Number of Ninja runs completed */

 // int  Load_WindNinja(char cr_ErrMes[]);
 // void Disp_WindTbl_WN();
 // int WindNinjaInit ( CWindNinja2 *a_WN2, double *ad_ElevRes, long *al_NumNorthRows, long *al_NumEastCols, char cr_ErrMes[]);
 // int WindNinjaRun ( CWindNinja2 *a_WN2, WindData *a_WD, double d_ElevRes, long l_NumNorthRows, long l_NumEastCols, char cr_ErrMes[]);
  //int WindNinja_SetExisting(int iX);
//  float WindNinja_Progress(float *afN_WNtoDo);

  float  GetLayerValueByCell (int i_Type, int l_Col, int l_Row);
  float  GetRough (int r, int c, int i_Type);
  float ConvertUnits(float val, int srcUnits, int destUnits, int themeType);
  void DeleteWindGrids(d_FWN *a_FWN, int iS_WN);
  short GetWindGridSpeed(long wndRow, long wndCol, d_FWN *a_FWN);
  short GetWindGridDir(long wndRow, long wndCol, d_FWN *a_FWN);
  short GetWindGridSpeedByCoord(double xCoord, long yCoord, d_FWN *a_FWN);
  short GetWindGridDirByCoord(double xCoord, long yCoord, d_FWN *a_FWN);
  float Temperature (long l_Mth, long l_Day, long l_Hr);
  float   Get_RAWS_Temp (long l_Mth, long l_Day, long l_Hr);

void HumTemp(long l_Mth, long l_Day, long l_Hr, double* tempref);

/* the numbers or order don't matter, they are just used to identify */
#define e_Rough_h   1
#define e_Rough_d   2
#define e_Roughness 4
#define e_Albedo    5
#define e_Bowen     6
#define e_Cg        7
#define e_Anthropog 8
#define e_Elev      9

float  computeSurfPropForCell ( int i_Type, double canopyHeight,
                                int canopyCover, int fuelModel, double fuelBedDepth);

// WN-New...
// ------------------------------------------------------------------------


	Farsite5(void);
	~Farsite5(void);
	CSpotList spotList;
	Random Rand;
	void 	ReadHeader();
	bool LoadLandscapeFile(char *FileName);
	int LoadInputsFile(char *FileName);
	char  *LoadInputError (int i_Num);
	void CloseLandFile();
	celldata 	CellData(double east, double north, celldata &cell, crowndata &cfuel, grounddata &gfuel, long *posit);
	long		GetCellPosition(double east, double north);
	void GetCellDataFromMemory(long posit, celldata& cell, crowndata& cfuel, grounddata& gfuel);
	void SetCellFuel(long posit, short tfuel);
	const char *GetLandFileName();
	void SetLandFileName(char* FileName);
	bool OpenLandFile();
	double 	ConvertEastingOffsetToUtm(double input);
	double 	ConvertNorthingOffsetToUtm(double input);
	double 	ConvertUtmToEastingOffset(double input);
	double 	ConvertUtmToNorthingOffset(double input);
	char*	GetHeaderDescription();
	long 	HaveCrownFuels();
	long 	HaveGroundFuels();
	double 	GetCellResolutionX();                      	// return landscape X cell dimension
	double 	GetCellResolutionY();                      	// return landscape Y cell dimension
	bool 	HaveCustomFuelModels();
	bool 	HaveFuelConversions();
	bool 	NeedCustFuelModels();
	bool 	NeedConvFuelModels();
	void 	SetCustFuelModelID(bool True_False);
	void 	SetConvFuelModelID(bool True_False);
        void    LoadBurnPeriods();
	void	LoadFuelMoist();
        void    LoadFoliarMoist(int i_FolMoi);
        int     LoadWeatherStream (d_ICF *a_ICF);


	int  LoadWind_Table (d_ICF *a_ICF);
	int  LoadWind_Regular (d_ICF *a_ICF);
	int  LoadWind_RAWS (d_ICF *a_ICF);



        bool  LoadFuelMoistureFile(char *FileName);
	long 	AllocWeatherData(long StatNum, long NumObs);
	void 	FreeWeatherData(long StationNumber);
	void 	FreeWindData(long StationNumber);


	bool SetInitialFuelMoistures(long model, long t1, long t10, long t100, long tlh, long tlw);
	bool GetInitialFuelMoistures(long model, long *t1, long *t10, long *t100, long *tlh, long *tlw);
	long GetInitialFuelMoisture(long Model, long FuelClass);
	bool InitialFuelMoistureIsHere(long Model);
	void SetAllMoistures(int _fm1, int _fm10, int _fm100, int _fmHerb, int _fmWoody);
	double 	GetFoliarMC();
	void SetFoliarMoistureContent(long Percent);
	void SetSpreadDirection(double North, double Max);
	void 	SetFileOutputOptions(long FileType, bool YesNo);
	bool		GetFileOutputOptions(long FileType);
	void 	SetRasterFileName(const char *FileName);
	char 	*GetRasterFileName(long Type);
	void 	SetVectorFileName(const char *FileName);
	char 	*GetVectorFileName();
	long 	AccessOutputUnits(long val);
	long		CanSetRasterResolution(long YesNo);		// flag to allow setting of rast res
	void 	SetRastRes(double XResol, double YResol);	// set raster output resolution
	void 	GetRastRes(double *XResol, double *YResol);  // get raster output resolution
	int 		CheckCellResUnits();					// returns 0 for metric 1 for English
	double 	GetWestUtm();
	double 	GetEastUtm();
	double 	GetSouthUtm();
	double 	GetNorthUtm();
	double  	GetLoEast();
	double 	GetHiEast();
	double 	GetLoNorth();
	double	GetHiNorth();
	long 	GetLoElev();
	long 	GetHiElev();
	long 	GetNumEast();
	long 	GetNumNorth();

	bool 	CalcFirstLastStreamData();		// compute 1st and last valid dates for simulation
	long 	GetNumStations();		  			// retrieve number of weather/wind stations
	long AtmosphereGridExists();
	long		GetJulianDays(long Month);
	bool SetAtmosphereGrid(long NumGrids);
	AtmosphereGrid* GetAtmosphereGrid();
	void 	SetGridNorthOffset(double offset);		// set offset for Atm Grid purposes
	void 	SetGridEastOffset(double offset);
	double 	GetGridNorthOffset();
	double 	GetGridEastOffset();
	void 	SetGridEastDimension(long XDim);
	void 	SetGridNorthDimension(long YDim);
	long 	GetGridEastDimension();
	long 	GetGridNorthDimension();
	void  _WSConver (d_Wtr *a, char cr[]);
	void  _WDSConver (d_Wnd *a, char cr[]);
	long 	SetWeatherData(long StationNumber, long NumObs, long month, long day, long year,
					double rain, long time1, long time2, double temp1, double temp2,
					long humid1, long humid2, double elevation, long tr1, long tr2);

	long 	AllocWindData(long StatNum, long NumObs);
        long    SetWindData (long StationNumber, long NumObs, long month, long day, long year,
                             long hour, double windspd, long winddir, long cloudy);
        long    GetOpenWeatherStation();
        long    GetOpenWindStation();
        double  MetricResolutionConvert();                       // 3.28.. for english, 1.0 for metric
        void    GetCurrentFuelMoistures(long fuelmod, long woodymod, double *MxIn, double *MxOut, long NumMx);
        long    WoodyCombineOptions(long Options);
        bool    PreserveInactiveEnclaves(long YesNo);

        int    Run_CondDLL();
        int     FMC_LoadInputs (ICF *icf, headdata *lcp, CFMC *cfmc, char cr_ErrMes[]);
        CFMC  cfmc;       /* Fuel Moisture Conditioning Class - Cond DLL export class */


	ICF icf;          /* input command file class  */
	short* landscape;
	bool CantAllocLCP;
	FILE* landfile;
	char LandFName[256];
	char InputsFName[256];
	CanopyCharacteristics CanopyChx;
	headdata Header;
	long NumVals;
	long OldFilePosition;
	bool NEED_CUST_MODELS;// = false;	// custom fuel models
	bool HAVE_CUST_MODELS;// = false;
	bool NEED_CONV_MODELS;// = false;     // fuel model conversions
	bool HAVE_CONV_MODELS;// = false;
	double RasterCellResolutionX;
	double RasterCellResolutionY;
	double SimulationDuration;// = 0.0;
	bool CondPeriod;// = false;
	FuelConversions fuelconversion;
	InitialFuelMoisture fm[257];
	bool EnvtChanged[4][5];
	long MoistCalcInterval[4][5];//={{60, 200, 10, 45, 15},           // 1hr
	long CanSetRastRes;// = true;

	//----------------- Model Parameter Access Functions --------------------------

	double GetDistRes();
	void SetDistRes(double input);
	void SetDynamicDistRes(double input);
	double GetDynamicDistRes();
	double GetPerimRes();
	void SetPerimRes(double input);
	bool AccelerationON();
	void SetAccelerationON(bool State);
	double GetRosRed(int fuel);
	void SetRosRed(int fuel, double rosred);
	void InitializeRosRed();
	double GetActualTimeStep();
	double GetVisibleTimeStep();
	double GetTemporaryTimeStep();
	void SetActualTimeStep(double input);
	void SetVisibleTimeStep(double input);
	void SetTemporaryTimeStep(double input);
	void SetSecondaryVisibleTimeStep(double input);
	double GetSecondaryVisibleTimeStep();
	double EventMinimumTimeStep(double time);   		// event driven time step
	bool CheckExpansions(long YesNo);   		   	// check illogical perimeter expansions
	bool CheckPostFrontal(long YesNo);
	long DistanceCheckMethod(long Method);

	//----------------- Fire Data Access Functions --------------------------------

	long GetInout(long FireNumber);
	inline long GetNumPoints(long FireNumber){	return numpts[FireNumber];}
	void SetInout(long FireNumber, int Inout);
	void SetNumPoints(long FireNumber, long NumPoints);
	void GetPerimeter2(long coord, double* xpt, double* ypt, double* ros,
		double* fli, double* rct);
	double GetPerimeter2Value(long coord, long value);
	void SetPerimeter2(long coord, double xpt, double ypt, double ros, double fli,
		double rct);
	double* AllocPerimeter2(long NumPoints);
	void FreePerimeter2();
	double* AllocPerimeter1(long NumFire, long NumPoints);
	void FreePerimeter1(long NumFire);
	inline double GetPerimeter1Value(long NumFire, long NumPoint, int coord)
	{
		return (perimeter1[NumFire]) ?
			perimeter1[NumFire][NumPoint * NUMDATA + coord] : 0.0;
	}

	void SetPerimeter1(long NumFire, long NumPoint, double xpt, double ypt);
	void SetFireChx(long NumFire, long NumPoint, double ros, double fli);
	void SetReact(long NumFire, long NumPoint, double ReactionIntensity);
	double* GetPerimeter1Address(long NumFire, long NumPoint);
	long SwapFirePerims(long NumFire1, long NumFire2);
	long GetNumPerimAlloc();
	bool AllocFirePerims(long num);
	bool ReAllocFirePerims();
	void FreeAllFirePerims();


	//----------------- Fire Accounting Functions ---------------------------------

	bool CreateSpotSemaphore();
	void CloseSpotSemaphore();
	void GetNumSpots(long* NumSpots, bool inc);
	void SetNumSpots(long input);
	void IncNumSpots(long increment);
	long GetNumFires();
	void SetNumFires(long input);
	void IncNumFires(long increment);
	long GetNewFires();
	void SetNewFires(long input);
	void IncNewFires(long increment);
	long GetSkipFires();
	void SetSkipFires(long newvalue);
	void IncSkipFires(long increment);
	double PercentIgnition(double);
	double IgnitionDelay(double delay);
	bool EnableCrowning(long);
	bool EnableSpotting(long);
	bool EnableSpotFireGrowth(long);
	bool ConstantBackingSpreadRate(long);

	// -----------------Elevation Functions ---------------------------------------

	void AllocElev(long CurrentFire);
	void SetElev(long Num, long elev);
	long GetElev(long Num);
	long* GetElevAddress(long Num);
	void FreeElev();

	// -----------------Stop-location Functions ---------------------------------------

	long SetStopLocation(double xcoord, double ycoord);
	bool GetStopLocation(long StopNum, double* xcoord, double* ycoord);
	void ResetStopLocation(long StopNum);
	bool EnableStopLocation(long StopNum, long Action);
	void ResetAllStopLocations();
	long GetNumStopLocations();

	//newfuel support functions
	void ResetNewFuels();
	bool SetNewFuel(NewFuel *newfuel);
	bool GetNewFuel(long number, NewFuel *newfuel);
	void InitializeNewFuel();
	bool IsNewFuelReserved(long number);

	//burn period support
	void AddDownTime(double time);
	double GetDownTime();
	bool AllocBurnPeriod(long num);
	long GetNumBurnPeriods();
	void FreeBurnPeriod();
	void SetBurnPeriod(long num, long mo, long dy, long yr, long start, long end);
	bool GetBurnPeriod(long num, long *mo, long *dy, long *start, long *end);
	bool InquireInBurnPeriod(double SimTime);
	void ConvertAbsoluteToRelativeBurnPeriod();
	bool CheckBurnPeriod(long mo1, long dy1, long mo2, long dy2);

	long GetMaxThreads();
	void SetMaxThreads(long numthreads);

	double PerimeterResolution;// = 0.0;			// maximum tangential perimeter resolution
	double DistanceResolution;// = 0.0; 	   	// minimum radial spread distance resolution
	double DynamicDistanceResolution;// = 0.0;	// minimum run-time radial resolution
	double actual;// = 0.0;					// actual time step
	double visible;// = 0.0;				   	// visible time step
	double TemporaryTimestep;// = 0.0;			// temp timestep for sim-level proc. control
	double secondaryvisible;// = -1.0;  	   	// seconary visible timestep
	double EventMinTimeStep;// = -1.0;			// event driven minimum timestep
	bool checkexpansions;// = false;
	bool checkpostfrontal;// = false;
	long DistanceMethod;// = 1;				// method for distance checking
	bool AccelerationState;// = true;			// Flag for using acceleration constants

	long NumPerimAlloc;// = 0;				// Current number of perimeter arrays allocated with new
	long* inout;// = 0;						// fire doesn't exist (0), burning out(2), in(2)
	long* numpts;// = 0;				   		// number of points in each fire
	double** perimeter1;// = 0; 		   		// pointers to arrays with perimeter points
	double* perimeter2;// = 0;					// swap array

	double redros[257];  					// rate of spread reduction factors
	size_t nmemb;
	double PercentageOfEmberIgnitions;// = 5.0;  	// % embers that start fires
	double SpotIgnitionDelay;// = 0.0;			// delay (minutes) for ignition of spot fires
	bool CrowningOK;// = true;  				 // enable crowning
	bool SpottingOK;// = true;  				 // enable spotting (just ember generation and flight)
	bool SpotGrowthOK;// = false;   			 // enable spot fire growth
	bool ConstBack;// = false;  				 // use constant backing spread rate (no wind no slope)

	long numfires;// = 0;					// number of fires
	long newfires;// = 0;   				   	// number of new fires
	long numspots;// = 0;   				   	// number of spot fires
	long NumLocalSpots;
	long skipfire;// = 0;   				   	// number of extinguished fires
	long p2numalloc;// = 0; 				   	// allocated space in perimeter2 array
	long* GroundElev;// = 0;					// stores elevation of points
	long numelev;// = 0;

	long NumStopLocations;// = 0;
	double StopLocation[MAXNUM_STOPLOCATIONS*2];
	bool StopEnabled[MAXNUM_STOPLOCATIONS];

	long OutputUnits;// = 0;

	char RasterArrivalTime[256];
	char RasterFireIntensity[256];
	char RasterFlameLength[256];
	char RasterSpreadRate[256];
	char RasterHeatPerArea[256];
	char RasterCrownFire[256];
	char RasterFireDirection[256];
	char RasterReactionIntensity[256];
	char VectFName[256];
	char RastFName[256];

	bool rast_arrivaltime;// = false;
	bool rast_fireintensity;// = false;
	bool rast_spreadrate;// = false;
	bool rast_flamelength;// = false;
	bool rast_heatperarea;// = false;
	bool rast_crownfire;// = false;
	bool rast_firedirection;// = false;
	bool rast_reactionintensity;// = false;

	streamdata FirstMonth[5];
	streamdata FirstDay[5];
	streamdata FirstHour[5];
	streamdata LastMonth[5];
	streamdata LastDay[5];
	streamdata LastHour[5];

	AtmosphereGrid* AtmGrid;// = 0;			// pointer to AtmGrid;
	long NumWeatherStations;// = 0;
	long NumWindStations;// = 0;

	WindData* wddt[5];
	WeatherData* wtrdt[5];
	long MaxWeatherObs[5];
	long MaxWindObs[5];

	StationGrid grid;
	double NorthGridOffset;// = 0.0;
	double EastGridOffset;// = 0.0;

	NewFuel newfuels[257];									// custom and standard fuel models
	long CombineOption;// = CWD_COMBINE_ABSENT;
	CoarseWoody coarsewoody[MAXNUM_COARSEWOODY_MODELS];
	long MaxThreads;// = 1;

	double DownTime;// = 0.0;
	long NumAbsoluteData;// = 0;
	long NumRelativeData;// = 0;
	long LastAccess;// = -1;
	AbsoluteBurnPeriod* abp;// = 0;
	RelativeBurnPeriod* rbp;// = 0;

	bool InactiveEnclaves;// = true;


	long NumAirCraft;// = 0;
	AirCraft* aircraft[500];

	long NumAirAttacks;// = 0;
	long AirAttackCounter;// = 0;
	AirAttackData* FirstAirAttack;
	AirAttackData* NextAirAttack;
	AirAttackData* CurAirAttack;
	AirAttackData* LastAirAttack;
	char AirAttackLog[256];

	long GetNumAirCraft();
	void SetNumAirCraft(long NumCraft);
	bool LoadAirCraft(char* FileName, bool AppendList);
	long SetNewAirCraft();
	AirCraft* GetAirCraft(long AirCraftNumber);
	long SetAirCraft(long AirCraftNumber);
	void FreeAirCraft(long AirCraftNumber);

	long SetupAirAttack(long AirCraftNumber, long CoverageLevel, long Duration,
		double* startpoint);
	void CancelAirAttack(long AirAttackCounter);
	void LoadAirAttack(AirAttackData airattackdata);
	void FreeAllAirAttacks();
	AirAttackData* GetAirAttack(long AirAttackCounter);
	AirAttackData* GetAirAttackByOrder(long OrdinalAttackNumber);
	void SetNewFireNumberForAirAttack(long OldFireNum, long NewFireNum);
	long GetNumAirAttacks();
	void SetNumAirAttacks(long NumAtk);
	void WriteAirAttackLog(AirAttackData* atk);

	long NumGroupAttacks;// = 0;
	long GroupAttackCounter;// = 0;
	GroupAttackData* FirstGroupAttack;
	GroupAttackData* NextGroupAttack;
	GroupAttackData* CurGroupAttack;
	GroupAttackData* LastGroupAttack;
	GroupAttackData* ReAtk;

	long SetupGroupAirAttack(long AirCraftNumber, long CoverageLevel,
		long Duration, double* line, long FireNum, char* GroupName);
	GroupAttackData* ReassignGroupAirAttack(GroupAttackData* atk);
	void CancelGroupAirAttack(long GroupCounter);
	void LoadGroupAttack(GroupAttackData groupattackdata);
	void FreeAllGroupAirAttacks();
	GroupAttackData* GetGroupAirAttack(long GroupAttackCounter);
	GroupAttackData* GetGroupAttackByOrder(long OrdinalAttackNumber);
	GroupAttackData* GetGroupAttackForFireNumber(long CurrentFire, long StartNum,
		long* LastNum);
	void SetNewFireNumberForGroupAttack(long OldFireNum, long NewFireNum);
	long GetNumGroupAttacks();
	void SetNumGroupAttacks(long NumGroups);

	long GetCanopySpecies();
	double 	GetAverageDBH();

//------------------------------------------------------------------------------
//  PostFrontal data for firering structures
//------------------------------------------------------------------------------
	long PRECISION;// = 12;		// number of sampling points
	long BURNUP_TIMESTEP;// = 15;	// seconds

	long NumRingStructs;// = 0;
	long NumRings;// = 0;
	RingStruct* FirstRing;// = 0;
	RingStruct* NextRing;// = 0;
	RingStruct* CurRing;// = 0;

//------------------------------------------------------------------------------
//
//	PostFrontal access functions for post frontal
//
//------------------------------------------------------------------------------

	FireRing* AllocFireRing(long NumPoints, double start, double end);
	void FreeAllFireRings();
	void FreeFireRing(long RingNum);
	FireRing* GetRing(long RingNum);
	FireRing* GetSpecificRing(long FireNumber, double StartTime);
	void GetLastRingStruct();
	long GetNumRings();
	void SetNumRings(long NewNumRings);
	long GetNumRingStructs();
	void CondenseRings(long RingNum);
	void SetNewFireNumber(long OldNum, long NewNum, long RefNum);
	bool AddToCurrentFireRing(FireRing* firering, long PointNum,
		long SurfFuelType, long WoodyModel, double DuffLoad, double* Moistures,
		double CrownLoadingBurned);


	bool AllocCoarseWoody(long ModelNumber, long NumClasses);
	void FreeCoarseWoody(long ModelNumber);
	void FreeAllCoarseWoody();
	bool SetWoodyData(long ModelNumber, long ClassNumber, WoodyData *wd, long Units);
	bool SetWoodyDataDepth(long ModelNumber, double depth, const char *Description);
	bool SetNFFLWoody();
	void GetWoodyData(long WoodyModelNumber, long SurfModelNumber, long *NumClasses, WoodyData *woody, double *Depth, double *TotalLoad);
	long GetWoodyDataUnits(long ModelNumber, char *Description);
	double GetWoodyFuelMoisture(long ModelNumber, long SizeClass);
	//double *GetAllCurrentMoistures(long *NumMx);
//	void GetCurrentFuelMoistures(long fuelmod, long woodymod, double *MxIn, double *MxOut, long NumMx);
	double WeightLossErrorTolerance(double value);
//	long WoodyCombineOptions(long Options);

//	CoarseWoody coarsewoody[MAXNUM_COARSEWOODY_MODELS];
	double WeightLossErrorTol;// = 1.0;	// Mg/ha
	CoarseWoody NFFLWoody[256];
	CoarseWoody tempwoody;
//	long CombineOption;// = CWD_COMBINE_ABSENT;

	void 	AllocStationGrid(long XDim, long YDim);	 // allocate memory for weather stations
	long 	FreeStationGrid();                       // free memory for weather stations
	long 	GetStationNumber(double xpt, double ypt);// find station number for coordinate

	void 	ResetFuelConversions();
	void 	ResetCustomFuelModels();
	int 		GetFuelConversion(int fuel);
	int 		SetFuelConversion(int fueltype, int fuelmodel);
	double	GetFuelDepth(int Number);
	long 	AccessFuelModelUnits(long Val);

	long FuelModelUnits;// = 0;

	short	GetTheme_Units(short DataTheme);
	double 	GetDefaultCrownHeight();
	double 	GetDefaultCrownBase();
	double 	GetDefaultCrownBD(short cover);
	bool CrownDensityLinkToCC;// = false;    // crown bulk density is a function of canopy cover
	bool		LinkDensityWithCrownCover(long TrueFalse);   // returns true or false

	long		GetConditMonth();
	long		GetConditDay();
	void 	SetConditMonth(long input);
	void 	SetConditDay(long input);
	long		GetConditMinDeficit();
	long		GetLatitude();
	long		GetStartMonth();
	long		GetStartDay();
	long		GetStartHour();
	long		GetStartMin();
	long		GetStartDate();
	void 	SetStartMonth(long input);
	void 	SetStartDay(long input);
	void 	SetStartHour(long input);
	void 	SetStartMin(long input);
	void 	SetStartDate(long input);
	long		GetEndMonth();
	long		GetEndDay();
	long		GetEndHour();
	long		GetEndDate();
	long		GetEndMin();
	void 	SetEndMonth(long input);
	void 	SetEndDay(long input);
	void 	SetEndHour(long input);
	void		SetEndMin(long input);
	void 	SetEndDate(long input);
	long		GetMaxMonth();
	long		GetMaxDay();
	long		GetMaxHour();
	long		GetMinMonth();
	long		GetMinDay();
	long		GetMinHour();
	long conditmonth, conditday;
	long startmonth, startday, starthour, startmin,	startdate;		//INITIAL CONDITIONS AT STARTPOINT
	long endmonth, endday, endhour, endmin, enddate;

	long 	GetWeatherMonth(long StationNumber, long NumObs);
	long 	GetWeatherDay(long StationNumber, long NumObs);
	double 	GetWeatherRain(long StationNumber, long NumObs);
	long 	GetWeatherTime1(long StationNumber, long NumObs);
	long 	GetWeatherTime2(long StationNumber, long NumObs);
	double 	GetWeatherTemp1(long StationNumber, long NumObs);
	double 	GetWeatherTemp2(long StationNumber, long NumObs);
	long 	GetWeatherHumid1(long StationNumber, long NumObs);
	long 	GetWeatherHumid2(long StationNumber, long NumObs);
	double 	GetWeatherElev(long StationNumber, long NumObs);
	void		GetWeatherRainTimes(long StationNumber, long NumObs, long *tr1, long *tr2);
	long 	GetMaxWeatherObs(long StationNumber);

	long 	GetWindMonth(long StationNumber, long NumObs);
	long 	GetWindDay(long StationNumber, long NumObs);
	long 	GetWindHour(long StationNumber, long NumObs);
	double 	GetWindSpeed(long StationNumber, long NumObs);
	long 	GetWindDir(long StationNumber, long NumObs);
	long 	GetWindCloud(long StationNumber, long NumObs);
	long 	GetMaxWindObs(long StationNumber);
	void		SetWindUnits(long StationNumber, long Units);
	void 	SetWeatherUnits(long StationNumber, long Units);
	long		GetWindUnits(long StationNumber);
	long 	GetWeatherUnits(long StationNumber);
	long 	GetTolerance();

	long NumAttacks;// = 0;
	long AttackCounter;// = 0;
	AttackData* FirstAttack;
	AttackData* NextAttack;
	AttackData* CurAttack;
	AttackData* LastAttack;
	AttackData* Reassignment;// = 0;
	char AttackLog[256];

	long SetupIndirectAttack(long CrewNum, double* startpt, long numpts);    // Indirect Attack Constructor
	long ResetIndirectAttack(AttackData* atk, double* coords, long numpts);
	long SetupDirectAttack(long CrewNum, long FireNum, double* coords);	   // DirectAttack Constructor
	long ResetDirectAttack(AttackData* atk, long FireNum, double* coords);
	long GetNumAttacks();
	long GetFireNumberForAttack(long AttackCounter);
	AttackData* GetAttackForFireNumber(long NumFire, long start,
		long* LastAttackNumber);
	void SetNewFireNumberForAttack(long oldnumfire, long newnumfire);
	long GetNumCrews();

	AttackData* GetReassignedAttack();
	void ReassignAttack(AttackData* atk);
	AttackData* GetAttack(long AttackCounter);
	AttackData* GetAttackByOrder(long OrdinalAttackNum, bool IndirectOnly);
	void LoadAttacks(AttackData attackdata);
	void CancelAttack(long AttackCounter);
	void FreeAllAttacks();
	void WriteAttackLog(AttackData* atk, long Type, long Var1, long Var2);
	bool LoadCrews(char* FileName, bool AppendList);
	Crew* GetCrew(long CrewNumber);
	long SetCrew(long CrewNumber);
	long SetNewCrew();
	void FreeCrew(long CrewNumber);
	void FreeAllCrews();

	long 	GetInputMode();                              // gets input mode to the landscape window
	void 	SetInputMode(long ModeCode, bool Pending);	// sets input model to the landscape window
	bool 	ChangeInputMode(long YesNo);                 // set & retrieve input mode changes

	long NumCrews;// = 0;
//static Crew*	crew[200];
	Crew* crew[200];
	long LandscapeInputMode;// = NULLLOCATION;

	void		SetShapeFileChx(const char *ShapeFileName, long VisOnly, long Polygons, long BarriersSep);
	char*	GetShapeFileChx(long *VisOnly, long *Polygons, long *BarriersSep);
	char ShapeFileName[256];// = "";
	long ShapeVisibleStepsOnly;// = 1;
	long ShapePolygonsNotLines;// = 1;
	long ShapeBarriersSeparately;// = 0;

	long Chrono(double SIMTIME, long* hour, double* hours, bool RelCondit);
	bool		UseConditioningPeriod(long YesNo);
	bool		EnvironmentChanged(long Changed, long StationNumber, long FuelSize);
	long 	GetMoistCalcInterval(long FM_SIZECLASS, long CATEGORY);
	void 	SetMoistCalcInterval(long FM_SIZECLASS, long CATEGORY, long Val);

	double 	ConvertActualTimeToSimtime(long mo, long dy, long hr, long mn, bool FromCondit);
	void 	ConvertSimtimeToActualTime(double SimTime, long *mo, long *dy, long *hr, long *mn, bool FromCondit);

	LandscapeTheme *GetLandscapeTheme();
	LandscapeTheme* lcptheme;// = 0;

	long		GetTheme_HiValue(short DataTheme);
	long		GetTheme_LoValue(short DataTheme);

	long WindLoc;
    long WindLoc_New;

	bool RastMake;
	bool VectMake;
	bool ShapeMake;
	void 	SetRastMake(bool YesNo);                     // make/don't make raster files
	bool 	GetRastMake();                               //
	void 	SetVectMake(bool YesNo);                     // make/don't make vector files
	bool 	GetVectMake();                               //
	void		SetShapeMake(bool YesNo);
	bool		GetShapeMake();
	long RastFormat;// = 1;
	long VectFormat;// = 0;
	bool VISONLY;// = false;		// visible time step only for vector file
	void 	SetRastFormat(long Type);                    // raster output information
	long 	GetRastFormat();                             //
	void 	SetVectFormat(long Type);                    // vector output information
	long 	GetVectFormat();                             //
	void 	SetVectVisOnly(bool YesNo);                  // otp visible perim only/not
	bool 	GetVectVisOnly();
	long 	ExcludeBarriersFromVectorFiles(long YesNo);	// include barriers or not
	Vectorize vect;						// object that writes vector files
	ShapeFileUtils shape;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//            Main Interface functions and variables from TFarsiteInterface
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct
	{
		double tws;
		double winddir;
	} WStat[5];



	long NextFireAfterInterrupt, OldFireAfterInterrupt;
	Burn burn;   								  // declare Burn object, contains burn,area,intersections,embers
	long PrimaryVisEqual, PrimaryVisCount;
	long SecondaryVisEqual, SecondaryVisCount;				// visible timestep counters
	long maximum, idum;					// maximum time in simulation, updated with global "numaximum"
	long OutputVectsAsGrown, VisPerimSize;
	long ShowVectsAsGrown;// = 1;
	bool SIM_SUSPENDED, SIM_COMPLETED, FARSITE_GO, TerminateWait, WaitInProgress,
		TerminateMoist, MxRecalculated, StepThrough;
	long NumSimThreads;
	long SimRequest;
	bool LEAVEPROCESS, SIMULATE_GO,   IN_BURNPERIOD;	// using input data instead of files;



 #define e_StartUp  0
 #define e_Farsite 1
 #define e_Condition 2
 #define e_WindNinja 3   // WN-Test
 int    i_RunStatus;   /* set as Farsite or Fuel Conditioning is running */

	double progress;

	unsigned long systime1, systime2, systime;
	bool CanModify;// = true;
	bool PreCalcFuelMoistures;// = true;
	double LastFMCalcTime;
	short ProcNum;
	long CountInwardFires, CountTotalFires, VectorBarriers;		// just for display purposes
	long CurrentFire;
	char MBStatus[64], MBMode[64];
	char ElTime[12], CurTime[14];

	double MoistSimTime;
	double a1, p1, p2, sarea, firesizeh, firesizes;			// for area and perimeter calculations
	Attack Atk;
	AirAttack AAtk;
	IgnitionFile Ignition;					// Ignition file inputs
	GroupAirAttack* gaat;
	double smolder, flaming;
	PFrontData pfdata;						// glogal current post frontal stuff
	bool F_ON;// = 0;
	long ignitionCols;
	long ignitionRows;
	float **ignitionGrid;

	int SetIgnitionFileName(char *_ignitionFileName);
	int SetBarrierFileName(char *_barrierFileName);
	int SetBarrier(char *_barrierFileName);

	long IgnitionReset;// = 0;
	void 	ResetVisPerimFile();
	long 	IgnitionResetAtRestart(long YesNo);
	void ElTimeConvert(); 					// convert time to Elapsed time
	void CurTimeConvert();					// convert time to Current Time
	void CheckSteps();							 // check for changes in actual or visual timesteps
	int  FarsiteSimulationLoop();				// calls FARSITE processes and CheckMessageLoop()
	void 	SetSimulationDuration(double minutes);
	double	GetSimulationDuration();
	long      ShowFiresAsGrown(long YesNo);
	bool CheckThreadNumbers();
	void ResetThreads();
	bool	CanModifyInputs(long YesNo);
	bool		PreCalcMoistures(long YesNo);
	bool PreCalculateFuelMoisturesNoUI();
	void CountFires();  						 // just counts fires for display
	bool FarsiteProcess1();					// FARSITE process split to allow message processing
	bool FarsiteProcessSpots();				// FARSITE process split to allow message processing
	bool FarsiteProcess2();					// FARSITE process split to allow message processing
	void FarsiteProcess3();					// FARSITE process split to allow message processing
	void WriteVectorOutputs();				// output fire perimeters to vector file and monitor
	void WriteOutputs(int Type);			   	// writes area and perimeter data to arrays and draws graphs
	void WriteGISLogFile(long LogType);		// write log file for GIS outputs
	bool WriteClocks();						// update the clocks if visible
	bool WriteArea();						// write the area data and graph
	bool WritePerim();						// write the perimeter data and graph
	bool WriteFireData();			  		// write fire data to window
	bool WritePFData();						// write post frontal fire data to window

	void ProcessSimRequest();

	void LoadIgnitions();
	void SaveIgnitions();					// for internal use with start/restart
	void StartMoistThread();
	unsigned RunMoistThread(void*);
	void MoistThread();
	long InsertSpotFire(double xpt, double ypt);
	void CheckStopLocations();
	void ResetFarsite();					// reset all critical farsite data structures
	void FlatSimulateInitiateTerminate();
	void Execute_InitiateTerminate();
   /**
     * \brief Restarts or initializes the FARSITE simulation.
     *
     * The simulation then begins executing.
     */
	int Execute_StartRestart();
	void Execute_ResumeSuspend();
	void Execute_Reset();
	/**
	 * \brief Configures the FARSITE simulator for initial simulation loop.
	 *
	 * This method assumes that all has been initialized correctly.
	 */
	void Initiate() ;
	/**
	 * \brief Dumps all configuration info from the FARSITE simulator.
	 *
	 * After using this method, the user is required to load another
	 * configuration into memory before simulating.
	 */
	void SetModelParams();
	void Terminate() ;
	void ResetDuration();					// resets duration dialog box for simulation
	void 	FreeAllVectorThemes();
	void FreeAllCompoundCrews();
	char CrewFileName[256];
	char IgFileName[256];
	long MemCount;// = 0;
	VectorStorage* CurVect;// = 0;
	VectorStorage* FirstVect;// = 0;
	VectorStorage* NextVect;// = 0;
	VectorStorage* LastVect;// = 0;
	long NumVectorThemes;// = 0;
	long NumCompoundCrews;// = 0;
	CompoundCrew* compoundcrew[MaxNumCompoundCrews];
	int LaunchFarsite(void);
	int WriteArrivalTimeGrid(char *trgName);
	int WriteIntensityGrid(char *trgName);
	int WriteFlameLengthGrid(char *trgName);
	int WriteSpreadRateGrid(char *trgName);
	int WriteSpreadDirectionGrid(char *trgName);
	int WriteHeatPerUnitAreaGrid(char *trgName);
	int WriteReactionIntensityGrid(char *trgName);
	int WriteCrownFireGrid(char *trgName);
	int WriteArrivalTimeGridBinary(char *trgName);
	int WriteIntensityGridBinary(char *trgName);
	int WriteFlameLengthGridBinary(char *trgName);
	int WriteSpreadRateGridBinary(char *trgName);
	int WriteSpreadDirectionGridBinary(char *trgName);
	int WriteHeatPerUnitAreaGridBinary(char *trgName);
	int WriteReactionIntensityGridBinary(char *trgName);
	int WriteCrownFireGridBinary(char *trgName);
	int CreateIgnitionGrid();
	void FillIgnitionPolygon(xPolygon *poly, float val);
	void FillBarrierPolygon(xPolygon *poly);
	int WriteIgnitionGrid(char *trgName);
	int WriteSpotGrid(char *trgName);

	CPerimeterData *perimeters;
	CPerimeterData *lastPerimeter;
	void CleanPerimeters();
	void AddCurrPerimeter();
	int WritePerimetersShapeFile(char *trgName);

//	double GetFarsiteProgress(int *ai_RunStatus);
	double GetProgress(int *ai_RunStatus);
	int CancelFarsite(void);

	double SetFarsiteProgress(double newProgress);
 float  GetFarsiteProgress ();


// void Farsite5::SetFarsiteRunStatus (int i);

	clock_t timeLaunchFarsite;
	clock_t timeFinish;
	int WriteTimingsFile(char *trgName);

// int GetFarsiteRunStatus(char cr[]);
	int LoadCustomFuelsData();
    int LoadCustomFuelFile(char *FileName);
	int CheckCustomFuelsCoverage();

    int LoadROSAdjustFile();
	void LoadCrownFireMethod (char cr_CroFirMet[]);
	long GetCrownFireCalculation();
	long SetCrownFireCalculation(long Type);
	long CrownFireCalculation;			// 0=Finney (1998), 1=Scott&Reinhardt (2001)

	//CRITICAL_SECTION progressCS;
	int CmMapEnvironment(int themeno, double mapTime, char *outName);

	int WriteSpotDataFile(char *trgName);
	int WriteSpotShapeFile(char *trgName);
	int AddIgnitionToSpotGrid();

	int GetNumIgnitionCells();
	//debugging aids
	void WritePerimeter1Shapefile(int num, long CurFire);
	void WritePerimeter2Shapefile(int num, long curFire);
	void WritePerimeter1CSV(int num, long CurFire);
	void WritePerimeter2CSV(int num, long curFire);
	int m_nCellsLit;
};
