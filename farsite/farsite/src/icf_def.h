/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: icf.h    Input Command File Settings Class Definitions
* Desc:
*
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
#pragma once 

/* Number of hours that we can extend the last RAWS record */
#define  e_RAWS_ExtHr  6  

/* max number minutes that can be between RAWS weather records, */
/*  the records will get checked for this */
#define  e_RAWS_MaxGap (12 * 60)     /* 12 hours */

void Get_Nxt_MDH(int *M, int *D, int *H);
int  DaysInMth (int Mth);


#define  e_Missing  -999 

#define  e_HrMin    0   /* Lower limit check on Hour time */
#define  e_HrMax 2359   /* Upper limit check on Hour time */

/* The psuedo wind weather year, second year will be 2002, as long as it's  */
/* not a leap year it doesn't matter what years we use, this will help    */
/* us tell when we are dealing with 2 years of input data, and setting    */
/* and using conditioning start and end dates                           */
#define  e_StrYear  2001
#define  e_DaysPerYear 365

#define  e_MinPerDay  1440   /* Minutes in a day, 24 * 60 */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                       Input File Test Switches                            */
#define  e_ICF_Sep  ':'  /* used on the end of every switch                  */

#define  e_ICF_FMD   "FUEL_MOISTURES_DATA"
#define  e_ICF_FMF   "FUEL_MOISTURE_FILE"
#define  e_ICF_CFF   "CUSTOM_FUELS_FILE"

#define  e_ICF_CFD   "CUSTOM_FUELS_DATA"
#define  e_ICF_CFDU  "CUSTOM_FUELS_UNITS"

#define  e_ICF_RAWS  "RAWS"
#define  e_ICF_RAWSElev  "RAWS_ELEVATION"
#define  e_ICF_RAWSUnits "RAWS_UNITS"    /* English or Metric */  
#define  e_ICF_RAWF  "RAWS_FILE"         /* data file name would be the argument */

#define  e_ICF_WeF   "WEATHER_FILE"
#define  e_ICF_WeD   "WEATHER_DATA"
#define  e_ICF_WeDU  "WEATHER_DATA_UNITS"

#define  e_ICF_WiF   "WIND_FILE"
#define  e_ICF_WiD   "WIND_DATA"
#define  e_ICF_WiDU  "WIND_DATA_UNITS"

#define  e_ICF_Cond  "CONDITIONING_DATA"     /* for use with FSPRo cond inputs */
#define  e_ICF_CPE   "CONDITIONING_PERIOD_END"
#define  e_ICF_CPS   "CONDITIONING_PERIOD_START"

#define  e_ICF_WD    "WIND_DIRECTION"
#define  e_ICF_WS    "WIND_SPEED"

#define  e_ICF_GWDF   "GRIDDED_WINDS_DIRECTION_FILE" 
#define  e_ICF_GWSF   "GRIDDED_WIND_SPEED_FILE"

#define  e_ICF_GGW   "GRIDDED_WINDS_GENERATE"
#define  e_ICF_GWR   "GRIDDED_WINDS_RESOLUTION"
#define  e_ICF_GWH   "GRIDDED_WINDS_HEIGHT"
#define  e_ICF_GWV   "GRIDDED_WINDS_VEG"

// New WindNinja Diurnal Inputs
#define  e_ICF_GWD  "GRIDDED_WINDS_DIURNAL"           //  Yes  No
#define  e_ICF_GWDD "GRIDDED_WINDS_DIURNAL_DATE"      //  4 6 2008   Month Day Year 
#define  e_ICF_GWDT "GRIDDED_WINDS_DIURNAL_TIME"      //  0  0 13 -7 Sec Min Hour Time Zone, -7 = Mountain
#define  e_ICF_GWAT "GRIDDED_WINDS_DIURNAL_AIRTEMP"    //  Fahrenheit
#define  e_ICF_GWCC "GRIDDED_WINDS_DIURNAL_CLOUDCOVER" //  Percent 0 -> 100 
#define  e_ICF_GWLO "GRIDDED_WINDS_DIURNAL_LONGITUDE"  // -180.0 -> 180.0

// Farsite-WindNinja.......................................................
// Farsite uses these when doing WindNinja  
#define e_ICF_FGWL   "GRIDDED_WINDS_LONGITUDE"  //  -133.983378   -180 -> 180      
#define e_ICF_FGWT   "GRIDDED_WINDS_TIMEZONE"   //   -7 = Mountain    

/* Farsite-WindNinja, used to set bin'ing size of wind speed & direction */
#define e_BinDirDef  20.0
#define e_BinSpdDef  5.0
#define e_ICF_FGWBD  "GRIDDED_WINDS_DIRECTION_BIN"
#define e_ICF_FGWBS  "GRIDDED_WINDS_SPEED_BIN"
//...................................................... 


#define e_ICF_CFM    "CROWN_FIRE_METHOD"
#define e_ICF_NP     "NUMBER_PROCESSORS"
#define e_ICF_FMC    "FOLIAR_MOISTURE_CONTENT"

#define e_ICF_SDN    "SPREAD_DIRECTION_FROM_NORTH"
#define e_ICF_SDM    "SPREAD_DIRECTION_FROM_MAX"

#define e_ICF_AREA_EAST    "ANALYSIS_AREA_EAST"
#define e_ICF_AREA_WEST    "ANALYSIS_AREA_WEST"
#define e_ICF_AREA_NORTH    "ANALYSIS_AREA_NORTH"
#define e_ICF_AREA_SOUTH    "ANALYSIS_AREA_SOUTH"

#define e_ICF_USE_MEM_LCP    "USE_MEMORY_LCP"
#define e_ICF_USE_MEM_OUTPUTS    "USE_MEMORY_OUTPUTS"
#define e_ICF_TEMP_STORAGE_PATH "TEMP_STORAGE_PATH"

/* MTT Minimum Travel Time....                                               */
#define e_ICF_RES   "MTT_RESOLUTION"
#define e_ICF_ST    "MTT_SIM_TIME"
#define e_ICF_TPI   "MTT_TRAVEL_PATH_INTERVAL"
#define e_ICF_IFN   "MTT_IGNITION_FILE"
#define e_ICF_MTTBARRIERFILENAME   "MTT_BARRIER_FILE"
#define e_ICF_MTTFILLBARRIERS   "MTT_FILL_BARRIERS"
#define e_ICF_MTTSPOT	"MTT_SPOT_PROBABILITY"

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Output Grid Switches and their argument defines                           */

#define e_ICF_FL    "FLAMELENGTH"
#define e_ICF_SP    "SPREADRATE"
#define e_ICF_IN    "INTENSITY"
#define e_ICF_HE    "HEATAREA"
#define e_ICF_CR    "CROWNSTATE"
#define e_ICF_SO    "SOLARRADIATION"
#define e_ICF_FU1   "FUELMOISTURE1"
#define e_ICF_FU10  "FUELMOISTURE10"
#define e_ICF_MI    "MIDFLAME"
#define e_ICF_HO    "HORIZRATE"
#define e_ICF_MAD   "MAXSPREADDIR"
#define e_ICF_ELA   "ELLIPSEDIM_A"
#define e_ICF_ELB   "ELLIPSEDIM_B"
#define e_ICF_ELC   "ELLIPSEDIM_C"
#define e_ICF_MAS   "MAXSPOT"
#define e_ICF_MASDIR  "MAXSPOT_DIR"
#define e_ICF_MASDX   "MAXSPOT_DX"

#define e_ICF_FU100  "FUELMOISTURE100"
#define e_ICF_FU1k   "FUELMOISTURE1000"
#define e_ICF_WDG    "WINDDIRGRID"
#define e_ICF_WSG    "WINDSPEEDGRID"


#define e_ICF_SPOTTINGSEED  "SPOTTING_SEED"
#define e_ICF_ROS_ADJUST_FILE						"ROS_ADJUST_FILE"

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* FARSITE ....                                                               */
#define e_ICF_FARSITE_IGNITION					    "FARSITE_IGNITION_FILE"
#define e_ICF_FARSITE_BARRIERFILENAME				"FARSITE_BARRIER_FILE"
#define e_ICF_FARSITE_FILL_BARRIER			  		"FARSITE_FILL_BARRIERS"

#define e_ICF_FARSITE_START_TIME			    	"FARSITE_START_TIME"
#define e_ICF_FARSITE_END_TIME				     	"FARSITE_END_TIME"

#define e_ICF_FARSITE_TIMESTEP			     		"FARSITE_TIMESTEP"
#define e_ICF_FARSITE_VISIBLESTEP		   			"FARSITE_VISIBLESTEP"
#define e_ICF_FARSITE_SECONDARY_VISIBLESTEP			"FARSITE_SECONDARY_VISIBLESTEP"
#define e_ICF_FARSITE_DIST_RES					    "FARSITE_DISTANCE_RES"
#define e_ICF_FARSITE_PERIM_RES				   		"FARSITE_PERIMETER_RES"
#define e_ICF_FARSITE_SPOT						       "FARSITE_SPOT_PROBABILITY"
#define e_ICF_FARSITE_BURN_PERIODS			 		"FARSITE_BURN_PERIODS"
#define e_ICF_FARSITE_SPOT_IGNITION_DELAY			"FARSITE_SPOT_IGNITION_DELAY"
#define e_ICF_FARSITE_ACCELERATION_ON				"FARSITE_ACCELERATION_ON"
#define e_ICF_FARSITE_MINIMUM_SPOT_DISTANCE			"FARSITE_MINIMUM_SPOT_DISTANCE"
#define e_ICF_FARSITE_SPOT_GRID_RESOLUTION			"FARSITE_SPOT_GRID_RESOLUTION"
#define e_ICF_FARSITE_MIN_IGNITION_VERTEX_DISTANCE	"FARSITE_MIN_IGNITION_VERTEX_DISTANCE"


//TOM additions
#define e_ICF_TOM_RESOLUTION					"TREAT_RESOLUTION"
#define e_ICF_TOM_IDEAL_LCP						"TREAT_IDEAL_LANDSCAPE"
#define e_ICF_TOM_IGNITION						"TREAT_IGNITION"
#define e_ICF_TOM_ITERATIONS					"TREAT_ITERATIONS"
#define e_ICF_TOM_DIMENSION						"TREAT_DIMENSION"
#define e_ICF_TOM_FRACTION						"TREAT_FRACTION"
#define e_ICF_TOM_OPPORTUNITES_ONLY				"TREAT_OPPORTUNITES_ONLY"


/*****************************************************************************/
/*FSIM ....                                                                   */
#define e_ICF_FSIM_LANDSCAPE					"landscape"
#define e_ICF_FSIM_CUSTOMFUELS					"customfmd"
#define e_ICF_FSIM_ROSADJUST					"ROSAdjust"
#define e_ICF_FSIM_FMS97						"FMS97"
#define e_ICF_FSIM_FMS90						"FMS90"
#define e_ICF_FSIM_FMS80						"FMS80"
#define e_ICF_FSIM_CROWNFIRE					"CrownFireMethod"
#define e_ICF_FSIM_IGNITPROBGRID				"IgnitionProbabilityGrid"
#define e_ICF_FSIM_RESOLUTION					"Resolution"
#define e_ICF_FSIM_THREADSPERFIRE				"ThreadsPerFire"
#define e_ICF_FSIM_NUMSIMS						"NumSimulations"
#define e_ICF_FSIM_FRISKFILE					"FriskFile"
#define e_ICF_FSIM_JULIANSTART					"JulianStart"
#define e_ICF_FSIM_JULAINSTART					"JulainStart"
#define e_ICF_FSIM_BURNPROBFILE					"OutputsName"
#define e_ICF_FSIM_GRIDUNITS					"GridDistanceUnits"
#define e_ICF_FSIM_BARRIERFILE					"BarrierFile"
#define e_ICF_FSIM_FIREDAYFILE					"FireDayDistributionFile"
#define e_ICF_FSIM_SUPPRESSION					"Suppression"
#define e_ICF_FSIM_FIRESIZELIMIT				"FireSizeLimit"
#define e_ICF_FSIM_ERCSTREAMFILE				"ErcStreamFile"
#define e_ICF_FSIM_RECORD						"Record"
#define e_ICF_FSIM_FIRESSHAPEFILE				"OutputFirePerims"
#define e_ICF_FSIM_PROGRESS_PATHNAME			"ProgressFilePathName"

//FCONST additions
#define e_ICF_FCONST_WINDSPEED				"WindSpeed"
#define e_ICF_FCONST_WINDDIR				"WindDirection"
#define e_ICF_FCONST_NUMFIRES				"NumFires"
#define e_ICF_FCONST_FMSFILE				"FmsFile"
#define e_ICF_FCONST_OUTPUTPATH				"OutputsName"
#define e_ICF_FCONST_FIRELISTFILE			"FireListFile"
#define e_ICF_FCONST_DURATION				"Duration"
#define e_ICF_FCONST_ENVISION_FIRELIST		"EnvisionFireListFile"
#define e_ICF_FCONST_IGNITION_MULTIPLIER	"IgnitionMultiplier"
#define e_ICF_FCONST_SPOT_PROBABILITY		"SpotProbability"

//NODESPREAD additions
#define e_ICF_NODESPREAD_NUMLAT				"NodeSpreadNumLat"
#define e_ICF_NODESPREAD_NUMVERT			"NodeSpreadNumVert"

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                         Error Message Struct                             */
typedef struct  {
    int i_ErrNum;
    char cr_Sw[200];
    char cr_ErrMes[100];
 } d_EMS;


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                         Fuel Moisture Data                                */
#define e_GFM  5               /* Number of Global Fuel Moistures            */
#define e_GFMlow 2             /* Moisture limits                            */
#define e_GLFMup 300
#define e_GDFMup 30


#define e_MaxFMD 2000          /* Max structs to allocate                    */

typedef struct {
#define e_DefFulMod  0         /* Default Model, user enters into cmd file   */
    int i_Model;
    int i_TL1;                 /* 1 hour                                     */
    int i_TL10;                /* 10 hour                                    */
    int i_TL100;               /* 100 hour                                   */
    int i_TLLH;                /* Herb                                       */
    int i_TLLW;                /* Woody, large 3+ ? I guess                  */
}  d_FMD;


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                RAWS (Remote Automated Weather Statins)                    */
/* The year helps in particular with data that spans from Dec -> Jan  */
/*  RAWS input has its own year but we'll replace with #define e_StrYear */
/*  Moist Cond DL will define the SKIP_RAWS_TYPEDEF - because it also has */
#ifndef SKIP_RAWS_TYPEDEF  
typedef struct {
   int i_Yr;             /* see note above */
   int i_Mth;
   int i_Day;
   int i_Time;           /* Military 0000->2359 */
   float f_Temp;         /* Temperature */
   float f_Humidity;
   float f_PerHou;        /* Perceipitation Hourly */
   float f_WinSpd;       /* Wind Speed */
   float f_WinDir;       /* Wind Direction */
   float f_CloCov;       /* Cloud Cover */
  } d_RAWS;
#endif 


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                        Weather Data Struct                                */
/* NOTE: Year is not in the inputs file, but we'll stuff it in here as we    */
/*  read in the data to get rid of any confusion - see #define e_StrYear.  */
/* NOTE: Data can come from two consecutive years, but in total we don't allow */
/*  more than 365 days - see #define e_StrYear                             */
typedef struct  {
  int   i_Year;              /* Year isn't in input, we stuff it in here */
  int   i_Mth;               /* Mth = month,                                     */
  int   i_Day;               /* Day = day,                                       */
  float f_Per;               /* Per = precip/rain in 100th of inch (ex 10 = 0.1 inches*/
  int   i_mTH;               /* mTH = min_temp_hour 0-2300,                      */
  int   i_xTH;               /* xTH = max_temp_hour 0 - 2300,                    */
  float f_mT ;               /* mT  = min_temp,                                  */
  float f_xT ;               /* xT  = max_temp,                                  */
  int   i_mH ;               /* mH  = min_humidity,                              */
  int   i_xH ;               /* xH  = max_humidity,                              */
  int   i_Elv;               /* Elv = elevation,                                 */
  int   i_PST;               /* PST = precip_start_time 0-2300,                  */
  int   i_PET;               /* PET = precip_end_time 0-2300)                    */
 } d_Wtr;


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                           Wind Stream Structure                           */
/* NOTE: we have the Speed & Direction in here but conditioning doesn't    */
/*  use them, but I'm leaving them in now                                  */
typedef struct {
  int i_Year;              /* See NOTE for Weather Data Struct          */
  int i_Mth;               /* Mth = month,                                   */
  int i_Day;               /* Day = day,                                     */
  int i_Hr;                /* Hour                                           */
  float f_Spd;             /* Wind Speed                                     */
  int i_Dir;               /* Direction                                      */
  int i_CloCov;            /* Cloud Cover                                    */
}  d_Wnd;
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                              Windninja Diurnal Inputs                     */
typedef struct {
  char cr_YesNo[10]; /* use defines e_GW_No, e_GW_Yes, says if we're using diurnals */
  int i_Sec; 
  int i_Min;
  int i_Hour; 
  int i_Mth;
  int i_Day;
  int i_Year;
  int i_TimeZone;    /* -12->12, -7 =Mountain -6 =Mountain daylight sav time */ 
  float f_AirTemp;      /* Farhenheit */ 
  float f_CloudCov;     /* 0 -> 100 percent */
  float f_Longitude;    /* -180.0 -> 180.0 */
} d_WDI; 

 
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                             Farsite-WindNinja Inputs                     */
typedef struct {
  char cr_GriWin[10];   /* use defines e_GW_No, e_GW_Yes = Run WindNinja */
  float f_GriWinRes;   /* Grid Wind Resolution*/
  float f_GriWinHei;   /* Wind Height */
  int i_TimeZone;      /* -12->12, -7 =Mountain -6 =Mountain daylight sav time */ 
  float f_Longitude;   /* -180.0 -> 180.0 */

  float f_BinWinDir;   /* Bining for Wind Direction */
  float f_BinWinSpd;   /* Bining for Wind Speed */
} d_FWNI; 




/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
// Custom fuels
typedef struct  {     //same format as FMD files
	 int i_Model;
	 char cr_code[32];
 	float f_h1;
 	float f_h10;
	 float f_h100;
	 float f_lh;
	 float f_lw;
	 char dynamic[32];
	 float f_sl;
	 float f_slh;
	 float f_slw;
 	float f_depth;
 	float f_xmext;
 	float f_heatd;
 	float f_heatl;
 	char cr_comment[256];
} d_CustomFuel;


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                           Burn Period Structure                           */
typedef struct {
  int i_Month;               /* Mth = month,                                   */
  int i_Day;               /* Day = day,                                     */
  int i_HourStart;                /* Hour                                           */
  int i_HourEnd;                /* Hour                                           */
  int i_Year;     /* Added - Mar-14-2012 */
}  d_BurnPeriod;



/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/

#define eC_PthFN  1000              /* Max len of a Path File Name           */
#define eC_InpLin eC_PthFN + 200    /* Max len of an input line from file    */

/* Default init values                                                       */
#define ei_ICFInit  -99               /* integer num init value             */
#define  ef_ICFInit (float)ei_ICFInit /* float                              */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
typedef class  ICF {

public:

	int i_SpottingSeed;

// custom fuels info
  	d_CustomFuel *a_customFuels;
	  int iN_CustomFuels;

/*...........................................................................*/
/*                      Fuel Moisture info                                   */
   d_FMD *a_FMD;                        /* FUEL_MOISTURE_DATA                */
   int   iN_FMD;                        /* # of structs allocated            */

   char cr_FMF [eC_PthFN];              /*  FUEL_MOISTURE_FILE               */
   char cr_CFF [eC_PthFN];              /*  CUSTOM_FUELS_FILE                */
   char cr_ROSAdjustFile [eC_PthFN];              /*  ROS_ADJUST_FILE                */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                     Weather Stream Data RAWS                              */
/* Struct contains Mth Day Time and weather and wind data                    */
/* Newer stream data implementation replaces sperate weather and wind rec data */
  d_RAWS *a_RAWS;
  int  iN_RAWS;            /* Number of records */
  int  i_RAWSElev;         /* Elevation */ 

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                             Weather Data                                  */                                                                         
/* Only 1 year of data is allowed, we only work with non-leap-years,         */
/*  365 days, but 1 extra day is need */

// Leap-Year-Fix
   bool b_LeapYear;   /* set to true when weather data has a Feb 29 day in it */
   int Set_NextLeapDay (int *ai_Mth, int *ai_Day); 
//-----------------------


#define e_MaxWD (365 + 1)      /* Max number of Data records - See Note above */
   d_Wtr *a_Wtr;                 /* Weather stream structures connect here     */
   int  iN_Wtr;                 /* number of structs                          */

   int iN_YearWeather;        /* 1 or 2, for number of years of data in a_Wtr */
   char cr_WeFN [eC_PthFN];   /*  WEATHER_FILE, weather stream file name    */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                             Wind Data                                     */
#define e_MaxWiD  (24 * e_MaxWD)    /* Max Wind Recs - 24 per day * max days  */
   d_Wnd *a_Wnd;
   int  iN_Wnd;
#define eC_WSU 50 
   char  cr_WiDU[eC_WSU];      /* use English  Metric                  */

   int  iN_YearWind; 
   char cr_WiFN [eC_PthFN];     /*  WIND_FILE     wind stream file name       */



 char  cr_CustFuelUnits[eC_WSU];      /* CUSTOM_FUEL_UNITS: English Metric   */


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
   char cr_GWDF [eC_PthFN];     /*  GRIDDED_WINDS_DIRECTION_FILE             */
   char cr_GWSF [eC_PthFN];     /*  GRIDDED_WINDS_SPEED_FILE                 */
   bool b_hasValidGriddedWindsFiles;   


/*...........................................................................*/
#define e_WD_Downhill  -2.0      /* Wind Directions                          */
#define e_WD_Uphill    -1.0
#define e_WD_Min        0.0
#define e_WD_Max      360.0
   float f_WinDir;             /* Wind Direction                             */

#define e_WS_Min        0.0
#define e_WS_Max      200.0
   float  f_WinSpe;             /* Wind Speed                                 */

#define e_GWR_Def   100.0      /* Default Resolution                        */
#define e_GWR_Min     1.0       /* will also check against native lcp res    */ 
#define e_GWR_Max   10000.0 
   float  f_GriWinRes;          /* Gridded Winds resolution                   */

#define e_GWH_Min   0.0
#define e_GWH_Max   10000.0
#define e_GWH_Def   20.0       /* Default - feet */
   float f_GriWinHei;          /* Gridded Winds Height                       */


   int   i_YearEnd;               /* Not inputed - Internal Program set  */
   int   i_MthEnd;                /* Conditioning Period End                    */
   int   i_DayEnd;                /* month day time,                            */
   int   i_HourEnd;               /* hour:minute = time = military              */

   int   i_YearStart;             /* Not inputed - Internal progam set */
   int   i_MthStart;                /* Conditioning Period Start                    */
   int   i_DayStart;                /* Start month day time,                            */
   int   i_HourStart;               /* Start hour:minute = time = military              */
  // int   i_Min;

#define  e_GWV_Grass  "Grass"
#define  e_GWV_Brush  "Brush"
#define  e_GWV_Forest "Forest"
#define  eC_GWV  15
   char  cr_GriWinVeg[eC_GWV]; /* Gridded Winds Veg                          */


#define e_GW_No   "No"
#define e_GW_Yes  "Yes" 
#define eC_GGW 4
   char cr_GGW[eC_GGW];        /* Generate Grid Winds, "Yes", "No"           */

d_WDI   s_WDI;       /* WindNinja Diurnal Inputs */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
#define e_CFM_Finney   "Finney"
#define e_CFM_ScotRein "ScottReinhardt"
#define eC_CFM 30
   char cr_CroFirMet[eC_CFM];  /* CROWN_FIRE_METHOD:                         */


#define e_NPMin                1
#define e_NPMax                100
   int i_NumPro;               /* Number of Processors, NUMBER_PROCESSORS:   */

#define e_FM_Min        0
#define e_FM_Max      200
#define e_FM_Def      100
   int i_FolMoi;               /* FOLIAR_MOISTURE_CONTENT:                   */

   int i_SDFN;                 /* SPREAD_DIRECTION_FROM_NORTH:               */
   int i_SDFM;                 /* SPREAD_DIRECTION_FROM_MAX:               */

   //analysis area 
   float f_analysisEast;	//ANALYSIS_AREA_EAST:
   float f_analysisWest;	//ANALYSIS_AREA_WEST:
   float f_analysisNorth;	//ANALYSIS_AREA_NORTH:
   float f_analysisSouth;	//ANALYSIS_AREA_SOUTH:
   bool b_eastSet;
   bool b_westSet;
   bool b_northSet;
   bool b_southSet;
   int i_useMemoryLCP;		//USE_MEMORY_LCP:
   int i_useMemoryOutputs;  //USE_MEMORY_OUTPUTS:
   char  cr_storagePath[eC_PthFN];     /* TEMP_STORAGE_PATH:   -TEMP Storage Path for file based outputs during run*/
/* MTT - Minimum Travel Time......                                           */
   float f_Res;                /* MTT_RESOLUTION:                            */
   int   i_SimTim;             /* MTT_SIM_TIME:                              */
   int   i_TraPth;             /* MTT_TRAVEL_PATH_INTERVAL:                  */
   char  cr_IFN[eC_PthFN];     /* MTT_IGNITION_FILE:                         */
   char  cr_BarrierFileName[eC_PthFN];     /* MTT_BARRIERS_FILE:                         */
   float f_MTT_SpotProbability;  //MTT_SPOT_PROBABILITY:
   int i_MTT_FillBarriers;  //MTT_FILL_BARRIERS:

/* Output Grid switch indicators, tell if switch is set to yes or no        */
#define e_Yes 1
#define e_No  0

   int i_FL;         /*  FLAMELENGTH     */
   int i_SP;         /*  SPREADRATE      */
   int i_IN;         /*  INTENSITY       */
   int i_HE;         /*  HEATAREA        */
   int i_CR;         /*  CROWNSTATE      */
   int i_SO;         /*  SOLARRADIATION  */
   int i_FU1;        /*  FUELMOISTURE1   */
   int i_FU10;       /*  FUELMOISTURE10  */
   int i_MI;         /*  MIDFLAME        */
   int i_HO;         /*  HORIZRATE       */
   int i_MAD;        /*  MAXSPREADDIR    */
   int i_ELA;        /*  ELLIPSEDIM_A    */
   int i_ELB;        /*  ELLIPSEDIM_B    */
   int i_ELC;        /*  ELLIPSEDIM_C    */
   int i_MAS;        /*  MAXSPOT         */
   int i_MASDir;     /*  MAXSPOT_DIR     */
   int i_MASDx;      /*  MAXSPOT_DX      */

   int i_FU100;     /* FUELMOISTURE100   */
   int i_FU1k;      /* FUELMOISTURE1000  */
   int i_WDG;       /* WINDDIRGRID       */
   int i_WSG;       /* WINDSPEEDGRID     */

/* When reading input txt fil, most curnt line with found switch & arg index  */
   char cr_Line[eC_InpLin+1];  /* entire line                                */
   int  iX;                    /* index, tell where end of switch is         */
   FILE *fh;

   char cr_ErrMes[2000];
   char cr_ErrExt[500];       /*  extra error message text is placed    */


/* Farsite...............................*/
/* NOTE: the start & end Year is determined by the program from the weather data table */ 
   int i_FarsiteStartMth;
   int i_FarsiteStartDay;
   int i_FarsiteStartHour;
   int i_FarsiteStartYear;  

   int i_FarsiteEndMth;
   int i_FarsiteEndDay;
   int i_FarsiteEndHour;
   int i_FarsiteEndYear;  


   char  cr_FarsiteIgnition[eC_PthFN];     /* MTT_IGNITION_FILE:                         */
   char  cr_FarsiteBarrier[eC_PthFN];     /* MTT_BARRIERS_FILE:                         */

   float f_FarsiteSpotGridResolution;    /* for burn spot grid BSG Class  */
   float f_FarsiteSpotProb;          /* Spotting probability (0.0 - 1.0)                       */
   float f_FarsiteDistanceRes;          /* Distance resolution                       */
   float f_FarsitePerimeterRes;          /* Perimeter resolution                       */
   float f_FarsiteActualTimeStep;          /* ActualTimeStep (minutes)                      */
   float f_FarsiteVisibleTimeStep;          /* VisibleTimeStep (minutes)                      */
   float f_FarsiteSecondaryVisibleTimeStep;          /* SecondaryVisibleTimeStep (minutes)                      */
   float f_FarsiteSpotIgnitionDelay;          /* SecondaryVisibleTimeStep (minutes)                      */
   int	 i_FarsiteAccelerationOn;          /* SecondaryVisibleTimeStep (minutes)                      */
   int	 i_FarsiteFillBarriers;  //FARSITE_FILL_BARRIERS
   float f_FarsiteMinSpotDistance;          /* Minimum spotting Distance resolution                       */
   float f_FarsiteMinIgnitionVertexDistance;    /* for thinning extremely dense ignitions  */

// Farsite-WindNinja...................
   int FarsiteWindNinja();
   int FarsiteWindNinja_Check (int *ai);
   d_FWNI  s_FWNI;    /* Farsite WindNinja Inputs */
//......................................

   int  iN_BurnPeriods;

	  d_BurnPeriod *a_BurnPeriods;


/*...........................................................................*/
//TOM Data
	char iTrt_idealLcpName[eC_PthFN]; //ideal landscape file
	char iTrt_ignitionFileName[eC_PthFN]; //treat ignition file
	int iTrt_iterations;				//TOM iterations
	float iTrt_dimension;
	float iTrt_fraction;
	float iTrt_resolution;
	int iTrtOpportunitiesOnly;
	/*int iTrt_treatOpportunities;
	int iTrt_treatmentGrid;
	int iTrt_rosGrid;
	int iTrt_influenceGrid;
	int iTrt_arrivalGrid;
	int iTrt_fliMap;
	int iTrt_flowPaths;
	int iTrt_majorPaths;
	int iTrt_arrivalContour;*/

   int  BurnPeriodData (int *ai);
//   int ChkFarsitePeriod (int *ai);
   int ChkFarsiteSpotDelay(int *ai);
   int ChkFarsiteSpotGridResolution (int *ai); 
   int ChkFarsiteSpotProb(int *ai);
   int ChkFarsiteResolutions(int *ai);
   int ChkFarsiteTimeStep(int *ai);
   int ChkBurnPeriods(int *ai);
   int ChkBurnPeriods_New(int *ai);

/*...........................................................................*/
/*FSIM  .....................................................................*/
	char cr_FSIM_lcpFile[eC_PthFN];
	char cr_FSIM_fmdFile[eC_PthFN];
	char cr_FSIM_rosAdjustFile[eC_PthFN];
	char cr_FSIM_fms97File[eC_PthFN];
	char cr_FSIM_fms90File[eC_PthFN];
	char cr_FSIM_fms80File[eC_PthFN];
	char cr_FSIM_igProbFile[eC_PthFN];
	char cr_FSIM_friskFile[eC_PthFN];
	char cr_FSIM_burnProbFile[eC_PthFN];
	char cr_FSIM_barrierFile[eC_PthFN];
	char cr_FSIM_fireDayDistFile[eC_PthFN];
	char cr_FSIM_ercStreamFile[eC_PthFN];
	char cr_FSIM_progressFile[eC_PthFN];
	char cr_FCONST_fmsFile[eC_PthFN];
	char cr_FCONST_outputsName[eC_PthFN];
	char cr_FCONST_fireListFile[eC_PthFN];
	char cr_FCONST_Envision_fireListFile[eC_PthFN];
	
	int i_FSIM_firesShapeFile;

	int i_FSIM_crownFireMethod;
	int i_FSIM_threadsPerFire;
	int i_FSIM_julianStart;
	int i_FSIM_gridDistanceUnits;
	int i_FSIM_suppression;
	int i_FSIM_record;
	int i_FSIM_numSims;

	float f_FSIM_resolution;
	float f_FSIM_fireSizeLimit;

	int i_FCONST_numFires;
	int i_FCONST_windSpeed;
	int i_FCONST_windDir;
	int i_FCONST_duration;
	int i_FCONST_ignitionMultiplier;
	float f_FCONST_spotProbability;

	int i_NodeSpread_NumLat;
	int i_NodeSpread_NumVert;


/*...........................................................................*/
/* Get inputs from file functions....                                        */

   int  FSPRO_ConditionInputs (char cr_PthFN[500]);
   int  InputFlamMap (char cr_PthFN[]);
   int  InputFarsite (char cr_PthFN[]);
   int  Input (char cr_PthFN[]);
   int  InputFSIM (char cr_PthFN[]);
   int  InputFCONST (char cr_PthFN[]);



   ICF();
   ~ICF();
   void Init ();
   void DeleteAll ();

   int  Set_Date (char cr_Sw[]); 

   
   int  Set_FilNam (char cr_Sw[], char cr_FN[]);
   int  Set_SinNumArg (char cr_Sw[], float *af);
   int  Set_SinTxtArg (char cr_Sw[], char *a, int iN);
   int  Set_SinIntArg (char cr_Sw[], int *ai);
   int  Set_ConPerEnd (char cr_Sw[]);
   int  Set_ConPerStart (char cr_Sw[]);
   int  Set_ConPerStartDef (); 
   int  NextDay (int i_Mth, int i_Day, int i_Yr,  int *ai_Mth, int *ai_Day, int *ai_Yr); 

   int  Set_FSIM_FilNam (char cr_Sw[], char a_ICF_FN[], bool mandatorySW, bool isInput);

   void Set_OGS (char cr_Sw[], int *ai);
   int  GetLongInt (char cr[], long *al, int iS);
   int  GetInt (char cr[], int *ai, int iS);
   int  GetFlo (char cr[], float *af, int iS);
   int  GetWeatherYear (int i_Month, int i_Day); 
   

   int  GetSw    (char cr_Sw[]);
   int  CloseRet(int i_Ret);
   char *ErrorMessage(int i_Num);

   int  WeatherDataCmd (int *ai);
   int   WeatherDataFile (int *ai);
   int  RAWS_WeatherData (int *ai); 
   int  RAWS_WeatherDataFile (int *ai); 

   int SetWtrYear ();

   void  ToEnglish (char cr_Unit[]);

   int WindDataFile (int *ai);
   int SetWindYear ();


   int  WindData  (int *ai);
   int  FuelMoistData  (int *ai);
   int  GetCharLine (char buf[], char cr_Err[], FILE *fh);
   int  GetLine (float fr[], int iN, char cr_Err[], FILE *fh);
   int  GetLineFl (float fr[], int iN, float f_Fil, char cr_Err[], FILE *fh);
   int  CustomFuelData  (int *ai);


/*...........................................................................*/
/* Validation functions                                                      */
	int CheckRAWSWeatherWind(int *ai);

   int ChkCustomFuels(int *ai);

   int ValidateFlamMap ();
   int ValidateFarsite () ;

   int ChkFueMoi (int *ai);
   int ChkFolMoi (int *ai);
   int ChkConPer (int *ai, int i_Month, int i_Day, int i_Hour, double *d_Date);
   int ChkCondDate (char cr_Mode[], int *ai);
   int Chk_FarsiteDate (char cr_Mode[], int *ai); 
   int Chk_FarsiteStartEnd (int *ai);


   int isInWeather(long month);


   int ChkWeather (int *ai);
   int ChkWind (int *ai);

   int EngMet  (char cr[], char cr_Err[]);
   int ChkWind_SD (int *ai);
   int ChkGridWinds (int *ai);

   int ChkWNDiurnal ();
   
   int ChkCroFirMet  (int *ai);
   int GLM_FulMoi();
   int ChkNumPro (int *ai);
   int ChkSpreadDir (int *ai); 

/*...........................................................................*/
/* General functions                                                         */
   void  Trim_LT (char cr[]);
   int   isBlank (char  cr[] );
   void  Left_Just ( char  cr[] );
   void  Blk_End_Line (char cr_Line[], int i_Len);
   void  StrRepChr (char cr[], char c_This,  char c_That);
   char  Get_NumTyp (char cr_Data[]);
   void  Remove_CrLf (char cr[]);


   void  FromSwNa (char out[], char in[]);
   void  RemoveSwMa (char cr[]);
   void  Display ();
   int   isFile(char cr_PthFN[]);
   int   GetFloE (char cr[], float *af, int iS);

    int GridWindGenerate();
    float GridWindResolution();
    float GridWindHeight() ;
    int  GridWindGrass();
    int  GridWindBrush (); 
 
    int GridWindDiurnal (d_WDI *a_WDI);
    void Init_WDI (d_WDI *a_WDI);


	   float WindDirection();
	   float WindSpeed (); 
  
    int CheckNextDay (int m, int d, int mm, int dd );
    int CheckMonthDay (int m, int  d);
 
    int ChkWeaWinDay (int *ai);
    int ChkCondInputs (int *ai);
    int Chk_FarsiteCondInputs (int *ai);
    int Chk_RAWSData (int *ai);
   
    int  isLeapYearDay(int i_Mth, int i_Day);
    int GetMCDate(int mth, int day, int year);
    int GetWthDate(int iX ,int *m, int *d, int *y, int *t);
    int MilToMin (int i_MilTim);
    int GetTotMin(int mth, int day, int hr, int year);

    int CondMthDay (long *a_conditmonth,long *a_conditday); 

} d_ICF;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
   int _ChkMthDay (int M, int D, char cr_Err[]);


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*                       Function Return Error Codes                         */
#define  e_EMS_Fil  -101
#define  e_EMS_FMD  -100
#define  e_EMS_WDA  -102
#define  e_EMS_WSA  -103

#define  e_EMS_GWDF -130     /* Gridded winds direction file */
#define  e_EMS_GWSF -140     /* Gridded winds speed file */
#define  e_EMS_GWW  -120
#define  e_EMS_GWR  -104
#define  e_EMS_GWH  -105
#define  e_EMS_GWV  -106
#define  e_EMS_GWD  -150


// Farsite-WindNinja
#define  e_EMS_FGWL -151 
#define  e_EMS_FGWT -152
#define  e_EMS_FGWBD -153   /* Gridded Wind WindNinja Bining error - wind direction */
#define  e_EMS_FGWBS -154   /* Gridded Wind WindNinja Bining error - wind speed  */
//.......................................



#define  e_EMS_GWDFREAD -131     /* Gridded winds direction file read error*/
#define  e_EMS_GWSFREAD -141     /* Gridded winds speed file read error*/
#define  e_EMS_GWEXTENT -145     /* Gridded winds extents*/
#define  e_EMS_GWLCP    -146     /* Gridded winds lcp coverage*/

#define  e_EMS_GWWN -160    /* Gridded winds windninja error */

#define  e_EMS_CPS  -126
#define  e_EMS_CPE  -107
#define  e_EMS_WeSt -108
#define  e_EMS_WiD  -109
#define  e_EMS_NP   -110
#define  e_EMS_FMC  -111
#define  e_EMS_SDN  -112
#define  e_EMS_RES  -113
#define  e_EMS_ST   -114
#define  e_EMS_TPI  -115
#define  e_EMS_WeF  -116
#define  e_EMS_MTT_SPOT  -117


#define  e_EMV_FM   -200
#define  e_EMV_WeDU -201
#define  e_EMV_WiSU -202
#define  e_EMV_CFM  -203
#define  e_EMS_WeD1 -204
#define  e_EMS_WeD2 -205
#define  e_EMS_WeD3 -206
#define  e_EMV_WiD  -207
#define  e_EMS_RAWS -210
#define  e_EMS_RAWF -211

#define  e_EMS_WiF  -208
#define  e_EMS_WiD1 -209

#define  e_EMS_CF_FM    -230
#define  e_EMS_CF_EM    -231
#define  e_EMS_CF_DF    -232
#define  e_EMS_CF_DEPTH -233
#define  e_EMS_CF_LH    -234
#define  e_EMS_CF_DH    -235
#define  e_EMS_CF_SAV   -236
#define  e_EMS_CFD      -237
#define  e_EMS_RAWS_WS  -238
#define  e_EMS_RAWS_WDIR  -239
#define  e_EMS_RAWS_TEMP  -240
#define  e_EMS_RAWS_RH  -241
#define  e_EMS_RAWS_PCP  -242
#define  e_EMS_RAWS_CLOUDCOVER  -242
#define  e_EMS_FUEL_UNDEFINED  -243

#define  e_EMS_BadMTTIgnition -250
#define  e_EMS_NoSpreadOnIgnition  -251
#define  e_EMS_MTTCancelled  -252
#define  e_EMS_NoSpreadToAdjacentCell -253
#define  e_EMS_SpottingSeed -254

#define  e_EMS_FARSITE_START  -260
#define  e_EMS_FARSITE_END  -261
#define  e_EMS_FARSITE_IGNITION -262
#define  e_EMS_FARSITE_BARRIER  -263
#define  e_EMS_FARSITE_TIMESTEP  -264
#define  e_EMS_FARSITE_VISIBLETIMESTEP  -265
#define  e_EMS_FARSITE_SECONDARYVISIBLETIMESTEP  -266
#define  e_EMS_FARSITE_DIST_RES  -267
#define  e_EMS_FARSITE_PERIM_RES  -268
#define  e_EMS_FARSITE_SPOT  -269
#define  e_EMS_FARSITE_BURN_PERIODS  -270
#define  e_EMS_FARSITE_SPOT_IGNITION_DELAY  -271
#define  e_EMS_FARSITE_ACCELERATION_ON  -272

#define  e_EMS_FARSITE_STARTEND  -273
#define  e_EMS_FARSITE_NO_IGNITION -274
#define  e_EMS_FARSITE_FILL_BARRIERS -275
#define  e_EMS_FARSITE_RESOLUTIONS  -276
#define  e_EMS_FMS  -277
#define  e_EMS_FARSITE_SPOT_GRID_RESOLUTION -278
#define  e_EMS_FARSITE_MIN_IGNITION_VERTEX_DISTANCE -279


#define  e_EMS_FSIM_CROWNFIRE	-280
#define  e_EMS_FSIM_RESOLUTION	-281
#define  e_EMS_FSIM_THREADSPERFIRE	-282
#define  e_EMS_FSIM_NUMSIMS	-283
#define  e_EMS_FSIM_JULIANSTART	-284
#define  e_EMS_FSIM_GRIDUNITS	-285
#define  e_EMS_FSIM_SUPPRESSION	-286
#define  e_EMS_FSIM_FIRESIZELIMIT	-287
#define  e_EMS_FSIM_RECORD	-288
#define  e_EMS_FSIM_OUTPUTSNAME	-289

#define  e_EMS_FCONST_NUMFIRES	-290
#define  e_EMS_FCONST_WINDSPEED	-291
#define  e_EMS_FCONST_WINDDIR	-292
#define  e_EMS_FCONST_DURATION	-293


#define  e_EMS_Aloc -300
#define  e_EMS_NODESPREAD_NUMLAT -301
#define  e_EMS_NODESPREAD_NUMVERT -302

#define	 e_EMS_TOM_ITERATIONS	-350
#define	 e_EMS_TOM_IGNITION	-351
#define	 e_EMS_TOM_IDEAL_LCP	-352
#define	 e_EMS_TOM_DIMENSION	-353
#define	 e_EMS_TOM_FRACTION	-354
#define	 e_EMS_TOM_RESOLUTION	-355
#define	 e_EMS_TOM_OPPORTUNITIES_ONLY	-356


#define  e_EMS_Gen  -400   /* General Error */

#define  e_EMS_FmpRunning  -500   /* Launch Error */
#define  e_EMS_FmpNoLCP  -501   /* Launch Error */
#define  e_EMS_FmpNoMoistures  -502   /* Launch Error */
#define  e_EMS_FmpNoIgnitionGrid  -503   /* Launch Error */
#define  e_EMS_FmpNoIgnition  -504   /* Launch Error */
#define  e_EMS_FmpDataNotAvailable  -505   /* Launch Error */
//#define  e_EMS_FmpNoIgnition  -504   /* Launch Error */


//output errors
#define e_EMS_FILE_OPEN_ERROR    -600
#define e_EMS_FILE_WRITE_ERROR    -601
#define e_EMS_OUTPUT_DOES_NOT_EXIST    -602
#define e_EMS_FILE_CLOSE_ERROR    -603

// Moisture Conditioning Run Error 
#define e_EMS_MoistCond_Error  -700

//no projection information (aded for kml output)
#define e_EMS_NO_PROJECTION		-800

#define e_EMS_FUELMODEL			-801

/*---------------------------------------------------- */
/*  FSPro Error code                   */

#define e_EMS_fFile   -901
#define e_EMS_fVers   -902
#define e_EMS_fReso   -904
#define e_EMS_fDura   -905
#define e_EMS_fNFir   -906
#define e_EMS_fMLag   -907
#define e_EMS_fPoly   -908
#define e_EMS_fThre   -909
#define e_EMS_fCFue   -910
#define e_EMS_fSpec   -9101
#define e_EMS_fCalm   -911
#define e_EMS_fNWDi   -912
#define e_EMS_fNWSp   -913
#define e_EMS_fNWCe   -914
#define e_EMS_fERCC   -915
#define e_EMS_fERCY   -916
#define e_EMS_fNWPY   -917
#define e_EMS_fHRC    -918       /*  -18, "Missing/Invalid HistoricERCValues: switch" },                               */
#define e_EMS_fAvgE   -919       /*  -19, "Missing/Invalid AvgERCValues: switch" },                                    */
#define e_EMS_fStdD   -920       /*  -20, "Missing/Invalid StdDevERCValues:  switch" },                                */
#define e_EMS_fNuCu   -921       /*  -21, "Missing/Invalid NumWxCurrYear:  switch" },                                  */
#define e_EMS_fERCV   -922       /*  -22, "Missing/Invalid CurrentERCValues: switch" },                                */
#define e_EMS_fNFor   -923       /*  -23, "Missing/Invalid NumForecast: switch" },                                     */
#define e_EMS_fBarr   -924       /*  -24, "Missing/Invalid BarrierFill: switch" },                                     */
#define e_EMS_fIgni   -925       /*  -25, "Missing/Invalid IgnitionFile: switch or file name argument." },             */
#define e_EMS_fBFil   -926       /*  -26, "Missing/Invalid BarriersFile: switch or file name argument." },             */
#define e_EMS_fTran   -927       /*  -27, "Missing/Invalid TransectsFile: switch or file name argument." },            */
#define e_EMS_fCFul   -928       /*  -28, "Missing/Invalid CustFuelsFile: switch or file name argument." },            */
#define e_EMS_fDuag   -929       /*  -29, "Duration: argument is to large in combination with other values." },        */
#define e_EMS_fNoIg   -930       /*  -30, "No Ignition Point Found - check Ignition and Barrier(if used) File(s)" },   */
#define e_EMS_fOIIF   -931       /*  -31, "Can't Open Ignition Input File" },                                          */
#define e_EMS_fOBIF   -932       /*  -32, "Can't Open Barrier Input File" },                                           */
#define e_EMS_fGWin   -940       /*  -40, "Missing/Invalid gridded winds input data specified" },                      */
#define e_EMS_fGWRe   -941       /*  -41, "Gridded wind resolution is less than FSPRO-Inputs Resolution:" },           */
#define e_EMS_fGWHi   -9411      /*  -411,"Gridded wind height is invalid" },                                          */
#define e_EMS_fGWYN   -9412      /*  -412,"Invalid/Missing gridded winds diurnal Yes/No switch" },                     */
#define e_EMS_fGWDa   -942       /*  -42, "Invalid/Missing gridded winds diurnal date, month day year " },             */
#define e_EMS_fGWMi   -943       /*  -43, "Invalid/Missing gridded winds time, second minute hour time-zone" },        */
#define e_EMS_fGWTe   -944       /*  -44, "Invalid/Missing gridded winds air temp,cloud cover or longitude" },         */
#define e_EMS_fFlLe   -950       /*  -50, "Invalid/Missing Flame Length Grid Stack switch or argument" },              */
#define e_EMS_fDHis   -951       /*  -51, "Invalid/Missing Daily Acres History switch, argument or values" },          */
#define e_EMS_fWMat   -960       /*  -60, "Invalid/Missing Wind Matrix Values switch or values" },                     */
#define e_EMS_fWSum   -961       /*  -61, "Wind matrix sum <= 0.0" },  */


#define e_EMS_Hist   -965 
#define e_EMS_Pshp   -966

/* these have to do with FSRPro File errors */
#define e_EMS_ffWWG   -970       /* WatershedGrid file */
#define e_EMS_ffATG   -971
#define e_EMS_ffShp   -972  
#define e_EMS_ffShD   -973 
#define e_EMS_ffDaA   -974   
#define e_EMS_ffWaS   -975
#define e_EMS_ffLay   -976
#define e_EMS_ffSuL   -977
#define e_EMS_ffPTF   -978
#define e_EMS_ffIgn   -979
#define e_EMS_ffSta   -980

#define e_EMS_CloFil  -999

/*---------------------------------------------*/
/* Moisture Condition DLL error codes          */
/*  the FuelCondition code will return these code */
#define e_EMS_cFMM   -2000
#define e_EMS_cBFM   -2001 

#define e_EMS_cMem   -2002   /* memory alloc */


#define e_EMS_cMiss  -2003   

#define e_EMS_cWtSt -2004
#define e_EMS_cWdOb -2005
#define e_EMS_cElv  -2006
#define e_EMS_cSlp  -2007
#define e_EMS_cCov  -2008
#define e_EMS_cSDt  -2009
#define e_EMS_cPro  -2010
#define e_EMS_cNoWt -2011
#define e_EMS_cNoWn -2012
#define e_EMS_cSqWt -2013
#define e_EMS_cInWn -2014
#define e_EMS_cInWt -2015

#define   e_EMS_cbMt   -2301
#define   e_EMS_cbDy   -2302
#define   e_EMS_cbPr   -2303
#define   e_EMS_cbMtt   -2304
#define   e_EMS_cbXt   -2305
#define   e_EMS_cbXh   -2306
#define   e_EMS_cbMh   -2307
#define   e_EMS_cbEl   -2308
#define   e_EMS_cbPs   -2309
#define   e_EMS_cbPe   -2310
#define   e_EMS_cbPrr   -2311
#define   e_EMS_cbPB   -2312
#define   e_EMS_cbP    -2313

#define   e_EMS_Abort  -2500






/* FlamMap input moisture condition error code */
#define  e_EMS_cStDa  -2100       /* start date error */
#define  e_EMS_cStEn  -2101      /* start end date */ 


#ifdef STU_GAVE_ME_THESE 
#define  e_EMS_FC_WX_STATIONS   -175
#define  e_EMS_FC_WIND_STATIONS   -176
#define  e_EMS_FC_ELEVATION   -177
#define  e_EMS_FC_SLOPE   -178
#define  e_EMS_FC_COVER   -179
#define  e_EMS_FC_START_DATETIME   -180
#define  e_EMS_FC_START_PROCESSOR   -181
#define  e_EMS_FC_NUM_PROCESSORS   -182
#define  e_EMS_FC_NO_WX   -183
#define  e_EMS_FC_NO_WIND   -184
#define  e_EMS_FC_BAD_RAWS_STREAM   -185
#endif 