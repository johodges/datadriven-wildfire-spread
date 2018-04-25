

using namespace Sem;

#define MAXNUM_FUEL_MODELS 256

#define NUM_FUEL_SIZES 4

/* From flm4.hpp..............*/
#define SIZECLASS_1HR			0
#define SIZECLASS_10HR			1
#define SIZECLASS_100HR			2
#define SIZECLASS_1000HR			3

#define FM_SLOPPY 	3
#define FM_LIBERAL 	2
#define FM_MODERATE 1
#define FM_HARDASS 	0

// need to Fix-This
#define EVENT_MOIST 2  /* see calls to SetFarsiteEvent */

/* in flammap3.h */
#define E_DATA  0
#define S_DATA  1
#define C_DATA  4

#define FM_INTERVAL_TIME  0
#define FM_INTERVAL_ELEV 1
#define FM_INTERVAL_SLOPE 2
#define FM_INTERVAL_ASP  3
#define FM_INTERVAL_COV  4

/* This was in fsxync.h */
#define EVENT_MOIST_THREAD 	6

/* This was orginally in flammap3.h */
#define MAXNUM_COARSEWOODY_MODELS 100

//------------------------------------------------------------------------------
//           NEW Nelson Calculations for 10hr fuels
//------------------------------------------------------------------------------
struct FMS_Cover
{
	    Fms 		**fms;            // Array of Nelsons Fms Sticks

/* change 10-29-09  add new DeadFuelMoisture Class, used to replace the Fms **fms */
     DeadFuelMoisture  **dfm;  /* New ver 1 Nelson sticks */

     long 	FuelSize;
     long 	LoVal;              // Low value of radiation (Millivolts)
     long 	HiVal;			// hi value of radiation	 (Millivolts)
     double	*SolRad;           	// solar radiation, W/m2
     double 	Aspectf;			// radians
     long 	NumFms;			// number of sticks (radiation categories)
     long 	Aspect, Elev, Fuel, Slope;	// Elevation and Fuel type for site
	    double 	*LastMx;			// Array of last moistures (fractions)
     double 	*NextMx;			// Array of current moistures
     double	*LastEq;			// Array of equilibrium moisture contents
     double	*NextEq;			// Array of equilibrium moisture contents

     FMS_Cover(long FuelSize, CI *_a_CI);
     bool AllocCover();
     void FreeCover();
	    CI *a_CI;
};

// **********************************************************************
struct FMS_Aspects
{
	  long Slope, Elev, Fuel;
	  long NumAspects, FuelSize;
   long LoVal, HiVal;    		// degrees
	  FMS_Cover **Fms_Aspect;
   FMS_Aspects(long FuelSize, CI *_a_CI);
   bool AllocAspects();
   void FreeAspects();
	  CI *a_CI;
};

/*************************************************************************/
struct FMS_Slopes
{
   long Elev, Fuel;
	  long NumSlopes, FuelSize;
  	long LoVal, HiVal;
	  FMS_Aspects **Fms_Slope;
   FMS_Slopes(long FuelSize, CI *_a_CI);
   bool AllocSlopes();
   void FreeSlopes();
	  CI *a_CI;
};

/****************************************************************************/
struct FMS_Elevations
{
     long 			NumElevs;    		// Number of elevational categories
     long 			Fuel, FuelSize; 	// fuel type for this site, and size class for this elev
     double 			FirstTime;		// starting time for moisture
     double 			LastTime;			// ending time for moisture
     FMS_Slopes 		**Fms_Elev;	// Array of elevations each containing an array of radiations

     FMS_Elevations(long FuelSize, CI *_a_CI);
     void SetFuel(long Fuel);
     bool AllocElevations();
     void FreeElevations();
	    CI *a_CI;
};

/*****************************************************************************/
struct FuelMoistureMap {

/* Fixit - hard coded numbers */
     long 				NumFuels[4];
     long 				FuelKey[4][256];
     long 				FmTolerance[4][4];
     double				CuumRain[NUM_FUEL_SIZES];           // centimeters
	    FMS_Elevations 		**FMS[4];
	    CI *a_CI;
     FuelMoistureMap(CI *_a_CI = NULL);
     ~FuelMoistureMap();
     void SearchCondenseFuels(long FuelSize);
     bool AllocFuels(long FuelSize);
     void FreeFuels(long FuelSize);
     bool ReAllocFuels(long FuelSize);
};

/***************************************************************************/
/* DeadMoisture History is storage for old moisture calculations        */
/* Month, Day, Military Time were implement along with the RAWS data    */
/*  they help avoid doing extra lookups into the RAWS table             */
/*  they are the date/time starting point of the weather observation */
typedef struct {

    int   i_Mth;              /* Month, Day, Military Time   */
    int   i_Day;              /* See Note-1 above */
    int   i_Time;

    double 	LastTime;         /* Start time of simulation slice */
    double 	SimTime;          /* End time */
    long		  Elevation;			       // feet
	   double 	AirTemperature;	 	  // fahrenheit
    double 	RelHumidity;        // percentage
    long		  CloudCover;		       // percentage
    double 	Rainfall;           // cm of rain
   	double 	*Moisture;	        	// array of 10hr moisture contents at SimTime
	   void   	*next;			           // pointer to next DeadMoistureHistory Struct
}  DeadMoistureHistory;

/****************************************************************************/
/* DeadMoistureDescription - is summary of old moisture calculations  */
/* Note-1 NumAlloc = # of moistures in values in the DeadMoistureHistory.Moisture[] */
/*        which is one for every permutation in the moist map which is  */
/*         (numfueltypes * elev * slope * asp * cover)             */

struct DeadMoistureDescription {
    unsigned long		NumAlloc[eC_Sta][NUM_FUEL_SIZES];		/* See Note-1 above */
    long 	NumStations;
    long  NumElevs  [eC_Sta] [NUM_FUEL_SIZES];
    long		NumSlopes [eC_Sta] [NUM_FUEL_SIZES];
    long		NumAspects[eC_Sta] [NUM_FUEL_SIZES];
    long		NumCovers [eC_Sta] [NUM_FUEL_SIZES];
    long  NumFuels  [eC_Sta] [NUM_FUEL_SIZES];			// for each fuel type
    long  FuelKey   [eC_Sta] [NUM_FUEL_SIZES] [256];
    long		NumHist   [eC_Sta] [NUM_FUEL_SIZES];
    double	EndTime  [eC_Sta] [NUM_FUEL_SIZES];

// change 10-29-09, remove this, only gets used for the FireEnvironment2 Import,
//  Export and Copy functions which don't really get used.
//  See comments in flmfms.cpp above the ExportMoistureData()
//     Fms		**fms;

    DeadMoistureDescription();
   ~DeadMoistureDescription();

    void ResetDescription(long Station, long FuelSize);
};


/*****************************************************************************/
class FmsThread {
	  bool 	FirstTime;
   unsigned 	ThreadID;
	  long 	StationNumber, FuelType, FuelSize,	Begin, End, Date, Hour;
   double 	SimTime, *temp;
   FuelMoistureMap	*Stations;
   DeadMoistureHistory **CurHist;
   DeadMoistureDescription *MxDesc;

   static unsigned  RunFmsThread(void *fmsthread);
   static unsigned  RunFmsThread_RAWS(void *fmsthread);

   void  UpdateMapMoisture_RAWS();
   void  UpdateMoistures_RAWS();

   void  UpdateMapMoisture();
   void  UpdateMapMoisture_NewDFM();

   void Stick07_Mngr (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc);
   void Stick07_First(long n, FMS_Cover *cov, double Radiant, double Ctemp, double humid, double Rain, long loc);
   void Stick07_Next (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc);


   void Stick10_Mngr (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc);

   void Stick10_First (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc);
   void Stick10_Next (long n, FMS_Cover *cov, double Radiate, double Ctemp, double humid, double Rain, long loc);

   long GetLoc (long FuelSize, long i, long j, long k, long m, long n, long sn);

public:
	  CI *a_CI;
   long ThreadOrder, LastTime;
   //HANDLE hFmsEvent;

	  FmsThread(CI *_a_CI = NULL);
   ~FmsThread();

   void 	StartFmsThread(long ID, long SizeClass, DeadMoistureDescription *mxdesc, bool FirstTime);
   void 	StartFmsThread_RAWS(long ID, long SizeClass, DeadMoistureDescription *mxdesc, bool FirstTime);

   void 	SetRange (double simTime, long date, long hour, long stationNumber,
     				          	long fuelType,	FuelMoistureMap *map, DeadMoistureHistory **hist,
                    long begin, long bnd);
   void 	SiteSpecific(long ElevDiff, double tempref, double *temp,	double humref, double *humid);
   double 	SimpleRadiation(long date, double hour, long cloud, long elev,	long slope, long aspect, long cover);
   void 	UpdateMoistures();
};

/****************************************************************************/

class  FE2 {
public:
   bool  b_Terminate;  /* Terminate Conditioning run */
   double d_Progress;  /* percent of conditioning completed  */

   long		HistoryCount, NumFmsThreads;
   long 	 CloudCount, NumStations;
   long		StationNumber, elevref;
   double 	tempref, humref, rain;
   double 	humidmx, humidmn, tempmx, tempmn;

/* it looks the part of the code that uses this never gets run */
   double SimStart;

/***********************************************************************/

   int Init ();

   void SiteSpecific (long elev, double *ad_airtemp, double *ad_relhumd);

   bool HaveFuelMoist (long station, long fuel);
   void ElevTempHum (long *al_elev, double *ad_temp, double *ad_hum);

   CI *a_CI;
   FmsThread	*fmsthread;

   FuelMoistureMap 	   	*Stations;
   DeadMoistureDescription	MxDesc;
   DeadMoistureHistory		*FirstHist[eC_Sta] [4];    // four types of fuels, 1hr 10hr, 100hr, 1000hr X 5 streams,
   DeadMoistureHistory		*CurHist  [eC_Sta] [4];
   DeadMoistureHistory		*NextHist [eC_Sta] [4];

   bool 	AllocData (double lo, double hi, long interval, long elev, long fuel, long StationNumber);

   int DeadMoistureHistory_Display (int FuelSize, int sn);

// -------------------------------------------------------------------------------------------------
   void 	HumTemp_Stu ( long date, long hour, double *tempref, double *humref, long *elevref, double *rain,
     				        double *humidmx, double *humidmn, double *tempmx, double *tempmn, long *tr1, long *tr2);

   void 	RunFmsThreads_Lar(double SimTime, long StationNumber, long FuelType, long FuelSize);

   void 	RunFmsThreads_Stu (double SimTime, long StationNumber, long FuelType, long FuelSize);

   void RunFmsThreads_RAWS (double SimTime, long sn, long FuelType, long FuelSize);

   void Load_CurHist (DeadMoistureHistory *CurHist, d_RAWS *a_RAWS, int i_Elev, double d_RaiAcm, long SimTim, long l_Intv);


    bool ExportMoistureDataText(char* FileName, char* LCP_FilNam);

    int Set_FarsiteTime (long l_Adj);

    void Disp_CurHistMoist (long FuelSize, long FuelType);

//______________________________________________________________________________________________

   bool		CheckMoistureHistory(long Station, long FuelSize, double SimTime);
   void		CopyMoistureHistory(long Station, long FuelSize);
   bool 	AllocHistory(long Station, long FuelSize);
   DeadMoistureHistory *DMH_FindRec (long FuelSize, double d_Time);


   void 	FreeHistory(long Station, long FuelSize);
   void		RefreshHistoryDescription(long Station, long FuelSize);
   long 	GetClouds(long date, double hour);
   bool		AllocFmsThreads();
   void 	FreeFmsThreads();
   void 	CloseFmsThreads();

   void 	ResetData(long FuelSize);
   void 	ResetAllThreads();
   bool 	AllocStations(long Num);
   void 	FreeStations();
   double 	GetMx(double Time, long fuel, long elev, long slope,
     	        			 	double aspectf, long cover, double *equil, double *solrad, long FuelSize);
   bool 	CalcMapFuelMoistures(double SimTime);
   int   CondMngr ();
   bool		CheckMoistureTimes(double SimTime);
   double Get_Progress ();
   void  Terminate ();
};
