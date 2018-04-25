/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: icf_chk.cpp
* Desc: Input Command File Check/Validate functions
*       Functions mainly validate the data previously read in. Checks
*       include, limit checking, consistance with switches, etc.
* Date: 1-4-08
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/

 #define _CRT_SECURE_NO_DEPRECATE

// #include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "icf_def.h"

/* Format statements for sprintf min max errors                             */
/* NOTE: one is for integers and one of floats....................           */
#define e_fMMFmt "%4.1f is out of limits ( %4.1f -> %4.1f )"
#define e_iMMFmt "%d is out of limits ( %d -> %d )"


/*.......................................................................... */
/*                           Error Message Table                             */
d_EMS sr_EMS[] =  {
/*        Err Num     Switch Nam  Error Message                               */
       {  e_EMS_Fil,  "",          "Can't Open file." },

/* Fuel Moisture                                                             */
       {  e_EMS_FMS,  e_ICF_FMD,   "Invalid fuel moisture" },
       {  e_EMS_FMD,  e_ICF_FMD,   "Invalid/Missing number of fuel moistures" },
       {  e_EMV_FM,   e_ICF_FMF ", " e_ICF_FMD,  "Invalid moisture switches"},

/* Weather Stream Data                                                       */
       {  e_EMS_WeD1, e_ICF_WeD,   "Invalid/Missing weather stream data" },
       {  e_EMS_WeD2, e_ICF_WeD,   "Invalid combination of weather switches" },
       {  e_EMV_WeDU, e_ICF_WeDU,  "Invalid Weather Stream Units" },
       {  e_EMS_WeF,  e_ICF_WeF,   "File Not found or invalid contents" },

/* New Weather Stream Data */
       {  e_EMS_RAWS, e_ICF_RAWS,  "Invalid/Missing RAWS weather stream data" },
       {  e_EMS_RAWF, e_ICF_RAWF,  "RAWS data file missing or invalid contents" },

/* Wind Stream Data                                                       */
       {  e_EMS_WiD,  e_ICF_WiD,   "Invalid/Missing wind stream data" },
       {  e_EMS_WiD1, e_ICF_WiD,   "Invalid combination of wind switches" },
       {  e_EMS_WiF,  e_ICF_WiF,   "Invalid combination of wind switches" },

       {  e_EMS_WDA,  e_ICF_WD,    "Invalid/Missing wind direction" },
       {  e_EMS_WSA,  e_ICF_WS,    "Invalid/Missing wind speed" },

/* Gridded Winds Files*/
	      {  e_EMS_GWDF,      e_ICF_GWDF,    "Invalid/Missing gridded wind dir file" },
	      {  e_EMS_GWSF,      e_ICF_GWSF,    "Invalid/Missing gridded wind speed file" },
	      {  e_EMS_GWDFREAD,  e_ICF_GWDF,    "Error reading wind dir file" },
	      {  e_EMS_GWSFREAD,  e_ICF_GWSF,    "Error reading wind speed file" },
	      {  e_EMS_GWEXTENT,  e_ICF_GWSF,    "Wind File extents not identical" },
	      {  e_EMS_GWLCP,     e_ICF_GWSF,    "Wind Files incomplete lcp coverage" },


/* Gridded Winds */
	      {  e_EMS_GWW,  e_ICF_GGW,    "Invalid/Missing gridded wind Yes No" },
	      {  e_EMS_GWR,  e_ICF_GWR,   "Invalid/Missing gridded wind resolution" },
       {  e_EMS_GWH,  e_ICF_GWH,   "Invalid/Missing gridded wind height" },
       {  e_EMS_GWV,  e_ICF_GWV,   "Invalid/Missing gridded wind veg type" },
       {  e_EMS_GWD, "",          "Invalid/Missing gridded wind diurnal data" },

// Farsite-WindNinja
       { e_EMS_FGWL, e_ICF_FGWL,   "Invalid/Missing gridded wind longitude" },
       { e_EMS_FGWT, e_ICF_FGWL,   "Invalid/Missing gridded wind time zone" },
       { e_EMS_FGWBD, e_ICF_FGWBD, "Invalid gridded wind Direction bining value" },
       { e_EMS_FGWBS, e_ICF_FGWBS, "Invalid gridded wind Speed bining value" },
// Farsite-WindNinja

	   {  e_EMS_GWWN, "",          "WindNinja Error" },

	   {  e_EMS_CPS,  e_ICF_CPS,   "Invalid condition period START date time" },
	   {  e_EMS_CPE,  e_ICF_CPE,   "Invalid condition period end date time" },
       {  e_EMS_NP,   e_ICF_NP,    "Invalid number of processors" },
       {  e_EMS_FMC,  e_ICF_FMC,   "Invalid foliar moisture content"},
       {  e_EMS_SDN,  e_ICF_SDN,   "Invalid spread direction offset, use _NORTH or _MAX or none"},
       {  e_EMS_RES,  e_ICF_RES,   "Invalid MTT resolution"},
       {  e_EMS_ST,   e_ICF_ST,    "Invalid MTT maximum simulation time"},
       {  e_EMS_TPI,  e_ICF_TPI,   "Invalid MTT path interval"},
       {  e_EMS_MTT_SPOT,  e_ICF_MTTSPOT,   "Invalid MTT Spot Probability"},

       {  e_EMV_WiD,  e_ICF_WiD,   "Invalid Wind Stream Unit" },
       {  e_EMV_CFM,  e_ICF_CFM,   "Invalid Crown Fire" },
       {  e_EMS_WeD3, e_ICF_WeF,   "Invalid combination of weather switches" },


       {  e_EMS_CF_FM, "",   "Invalid Fuel Model in FMD" },
       {  e_EMS_CF_EM, "",   "Invalid Extinction Moisture in FMD" },
       {  e_EMS_CF_DF, "",   "Fuel Model Has No Dead Fuel in FMD" },
       {  e_EMS_CF_DEPTH, "",   "Invalid Depth in FMD" },
       {  e_EMS_CF_LH, "",   "Live Heat Content Too Low in FMD" },
       {  e_EMS_CF_DH, "",   "Dead Heat Content Too Low in FMD" },
       {  e_EMS_CF_SAV, "",   "SAV Ratios Out of Range in FMD" },
       {  e_EMS_CFD,  e_ICF_CFD,   "Invalid/Missingnumber of custom fuels" },

	   {  e_EMS_RAWS_WS,  "",   "Invalid RAWS Wind Speed" },
	   {  e_EMS_RAWS_WDIR,  "",   "Invalid RAWS Wind Direction" },
	   {  e_EMS_RAWS_TEMP,  "",   "Invalid RAWS Temperature" },
	   {  e_EMS_RAWS_RH,  "",   "Invalid RAWS Relative Humidity" },
	   {  e_EMS_RAWS_PCP,  "",   "Invalid RAWS Precipitation" },
	   {  e_EMS_RAWS_CLOUDCOVER,  "",   "Invalid RAWS Cloud Cover" },

       {  e_EMS_FUEL_UNDEFINED,  "",   "Undefined custom fuel in landscape" },

	   {  e_EMS_BadMTTIgnition,     "",  "Invalid Ignition" },
       {  e_EMS_NoSpreadOnIgnition, "",  "No Spread Data for Ignition" },
       {  e_EMS_MTTCancelled,       "",		"MTT Cancelled" },
       {  e_EMS_NoSpreadToAdjacentCell, "",  "No Spread From Ignition" },
       {  e_EMS_SpottingSeed, "",  "Invalid Spotting Seed" },

       {  e_EMS_FARSITE_START,     "",			"Invalid start time" },
       {  e_EMS_FARSITE_END,       "",			"Invalid end time" },
       {  e_EMS_FARSITE_IGNITION,  "",			"Invalid ignition" },
       {  e_EMS_FARSITE_BARRIER,   "",			"Invalid barrier" },
       {  e_EMS_FARSITE_TIMESTEP,   "",			"Invalid/Missing Time Step" },
       {  e_EMS_FARSITE_VISIBLETIMESTEP,          "",			"Invalid/Missing Visible Time Step" },
       {  e_EMS_FARSITE_SECONDARYVISIBLETIMESTEP, "",			"Invalid Secondary Visible Time Step" },
       {  e_EMS_FARSITE_DIST_RES,     "",			"Invalid distance resolution" },
       {  e_EMS_FARSITE_PERIM_RES,    "",			"Invalid perimeter resolution" },
       {  e_EMS_FARSITE_SPOT,         "",			"Invalid spot probability (0.0 - 1.0)" },
       {  e_EMS_FARSITE_SPOT_GRID_RESOLUTION, "",   "Invalid spot grid resolution" },
       {  e_EMS_FARSITE_MIN_IGNITION_VERTEX_DISTANCE, "",   "Invalid min ignition vertex distance" },

       {  e_EMS_FARSITE_BURN_PERIODS, "",			"Invalid burn periods" },
       {  e_EMS_FARSITE_SPOT_IGNITION_DELAY, "",	"Invalid spot ignition delay" },
	   {  e_EMS_FARSITE_ACCELERATION_ON, "", "Invalid Acceleration" },
	   {  e_EMS_FARSITE_STARTEND,        "", "Invalid Sim Start or End" },
       {  e_EMS_FARSITE_NO_IGNITION,     "",			"No ignition" },
       {  e_EMS_FARSITE_FILL_BARRIERS,   "",			"Invalid Fill Barriers switch" },
       {  e_EMS_FARSITE_RESOLUTIONS,     "",			"Perimter res must be >= Distance res" },
       {  e_EMS_FARSITE_DIST_RES,        "",			"Invalid minimum spotting distance" },

 	   //fsim error messages
 	   {  e_EMS_FSIM_CROWNFIRE, "", "Invalid Crown Fire Method" },
 	   {  e_EMS_FSIM_RESOLUTION, "", "Invalid Resolution" },
 	   {  e_EMS_FSIM_THREADSPERFIRE, "", "Invalid ThreadsPerFire" },
 	   {  e_EMS_FSIM_NUMSIMS, "", "Invalid FSIM NumSimulations" },
 	   {  e_EMS_FSIM_JULIANSTART, "", "Invalid FSIM JulianStart" },
 	   {  e_EMS_FSIM_GRIDUNITS, "", "Invalid GridDistanceUnits" },
 	   {  e_EMS_FSIM_SUPPRESSION, "", "Invalid FSIM Suppression" },
 	   {  e_EMS_FSIM_FIRESIZELIMIT, "", "Invalid FSIM FireSizeLimit" },
 	   {  e_EMS_FSIM_RECORD, "", "Invalid FSIM Record argument" },
	   {  e_EMS_FSIM_OUTPUTSNAME, "", "Missing/Invalid OutputsName switch"},

	   //fconst
 	   {  e_EMS_FCONST_NUMFIRES, "", "Invalid FConst NumFires argument" },
 	   {  e_EMS_FCONST_WINDSPEED, "", "Invalid FConst WindSpeed argument" },
 	   {  e_EMS_FCONST_WINDDIR, "", "Invalid FConst WindDirection argument" },
 	   {  e_EMS_FCONST_DURATION, "", "Invalid FConst Duration argument" },

	   {  e_EMS_Aloc,               "",  "Error allocating Memory" },
	   {  e_EMS_NODESPREAD_NUMLAT,               "",  "Invalid NumLat" },
	   {  e_EMS_NODESPREAD_NUMVERT,               "",  "Invalid NumVert" },

		//TOM messages
       {  e_EMS_TOM_ITERATIONS,        "",			"Invalid Treatment Iterations" },
       {  e_EMS_TOM_IGNITION,        "",			"Invalid Treatment Ignitions" },
       {  e_EMS_TOM_IDEAL_LCP,        "",			"Invalid Ideal Landscape" },
       {  e_EMS_TOM_DIMENSION,        "",			"Invalid Treatment Dimension" },
       {  e_EMS_TOM_FRACTION,        "",			"Invalid Teatment Fraction" },
       {  e_EMS_TOM_RESOLUTION,        "",			"Invalid Treatment Resolution" },
       {  e_EMS_TOM_OPPORTUNITIES_ONLY,        "",	"Invalid Treatment Opportunities Only" },

	   {  e_EMS_Gen,                "",   "Error in FlamMap input command file"},

	   //Launch() error codes
	   {  e_EMS_FmpRunning,  "",   "FlamMap already running"},
	   {  e_EMS_FmpNoLCP,  "",   "No Landscape file loaded"},
	   {  e_EMS_FmpNoMoistures,  "",   "No Fuel Moistures set"},
	   {  e_EMS_FmpNoIgnitionGrid,  "",   "No Ignition loaded"},
	   {  e_EMS_FmpNoIgnition,  "",   "No Ignition on landscape"},
	   {  e_EMS_FmpDataNotAvailable,  "",   "Fire Behavior failed"},

	   {  e_EMS_FILE_OPEN_ERROR,  "",   "Error opening output file"},
	   {  e_EMS_FILE_WRITE_ERROR,  "",   "Error writing to output file"},
	   {  e_EMS_OUTPUT_DOES_NOT_EXIST,  "",   "Output data does not exist"},
	   {  e_EMS_FILE_CLOSE_ERROR,  "",   "Error closing output file"},

	    { e_EMS_MoistCond_Error,   "",   "MoistCond"},

	   { e_EMS_NO_PROJECTION, "", "No Projection information"},

 	   { e_EMS_FUELMODEL, "", "Invalid Fuel Model"},

/*------------------------------------------------------------------------*/
/*  FSPRo Error codes - message */

 //    { iC_MCErr,     "",  "Missing/Invalid Moisture Conditioning Inputs" },
     { e_EMS_fFile,  "",  "Can't open input file" },
     { e_EMS_fVers,  "",  "Missing/Invalid FSPRO-Inputs-File-Version-1 - FSPRO-Inputs-File-Version-2 header" },
//   { -3, "Missing/Invalid Dimension: switch" },
     { e_EMS_fReso,  "",  "Missing/Invalid Resolution: switch" },
     { e_EMS_fDura,  "",  "Missing/Invalid Duration: switch" },
     { e_EMS_fNFir,  "",  "Missing/Invalid NumFires: switch" },
     { e_EMS_fMLag,  "",  "Missing/Invalid MaxLag: switch" },
     { e_EMS_fPoly,  "",  "Missing/Invalid PolyDegree: switch" },
     { e_EMS_fThre,  "",  "Missing/Invalid ThreadsPerFire: switch" },
     { e_EMS_fCFue,  "",  "Missing/Invalid UseCustomFuels: switch and/or specified file name" },
     { e_EMS_fSpec,  "",  "Can't Find/Open specified custom fuels file" },
     { e_EMS_fCalm,  "",  "Missing/Invalid CalmValue: switch" },
     { e_EMS_fNWDi,  "",  "Missing/Invalid NumWindDirs: switch or wrong number of entries" },
     { e_EMS_fNWSp,  "",  "Missing/Invalid NumWindSpeeds: switch" },
     { e_EMS_fNWCe,  "",  "Missing/Invalid WindCellValues: switch" },
     { e_EMS_fERCC,  "",  "Missing/Invalid NumERCClasses:  switch" },
     { e_EMS_fERCY,  "",  "Missing/Invalid NumERCYears: switch" },
     { e_EMS_fNWPY,  "",  "Missing/Invalid NumWxPerYear: switch" },

     { e_EMS_fHRC,   "",  "Missing/Invalid HistoricERCValues: switch" },
     { e_EMS_fAvgE,  "",  "Missing/Invalid AvgERCValues: switch" },
     { e_EMS_fStdD,  "",  "Missing/Invalid StdDevERCValues:  switch" },
     { e_EMS_fNuCu,  "",  "Missing/Invalid NumWxCurrYear:  switch" },
     { e_EMS_fERCV,  "",  "Missing/Invalid CurrentERCValues: switch" },
     { e_EMS_fNFor,  "",  "Missing/Invalid NumForecast: switch" },
     { e_EMS_fBarr,  "",  "Missing/Invalid BarrierFill: switch" },
     { e_EMS_fIgni,  "",  "Missing/Invalid IgnitionFile: switch or file name argument." },

// These were not getting used
//   { -26, "Missing/Invalid BarriersFile: switch or file name argument." },
//   { -27, "Missing/Invalid TransectsFile: switch or file name argument." },
//   { -28, "Missing/Invalid CustFuelsFile: switch or file name argument." },

     { e_EMS_fDuag,  "",  "Duration: argument is too large in combination with other values." },
     { e_EMS_fNoIg,  "",  "No Ignition Point Found - check Ignition and Barrier(if used) File(s)" },
     { e_EMS_fOIIF,  "",  "Can't Open Ignition Input File" },
     { e_EMS_fOBIF,  "",  "Can't Open Barrier Input File" },
     { e_EMS_fGWin,  "",  "Missing/Invalid gridded winds input data specified" },
     { e_EMS_fGWRe,  "",  "Gridded wind resolution is less than FSPRO-Inputs Resolution:" },
     { e_EMS_fGWHi,  "",  "Gridded wind height is invalid" },
     { e_EMS_fGWYN,  "",  "Invalid/Missing gridded winds diurnal Yes/No switch" },
     { e_EMS_fGWDa,  "",  "Invalid/Missing gridded winds diurnal date, month day year " },
     { e_EMS_fGWMi,  "",  "Invalid/Missing gridded winds time, second minute hour time-zone" },
     { e_EMS_fGWTe,  "",  "Invalid/Missing gridded winds air temp,cloud cover or longitude" },
     { e_EMS_fFlLe,  "",  "Invalid/Missing Flame Length Grid Stack switch or argument" },
     { e_EMS_fDHis,  "",  "Invalid/Missing Daily Acres History switch, argument or values" },
     { e_EMS_fWMat,  "",  "Invalid/Missing Wind Matrix Values switch or values" },
     { e_EMS_fWSum,  "",  "Wind matrix sum <= 0.0" },


     { e_EMS_Hist,   "",  "Fire Size Historgram Processing Error" },
     { e_EMS_Pshp,   "",  "Perimeter Shape File Processing Error" },
     { e_EMS_ffWWG, "",   "Can't Open/Write/Close Watershed Grid File" },
     { e_EMS_ffATG, "",   "Can't Open/Write/Close Average Time Grid File" },
     { e_EMS_ffShp, "",   "Can't Create Shape File" },
     { e_EMS_ffShD, "",   "Can't Create Shape Database" },
     { e_EMS_ffDaA, "",   "Can't Open/Close Daily Acres File" },
     { e_EMS_ffWaS, "",   "Can't Open/Write/Close Watershed Severtiy CSV file" },
     { e_EMS_ffLay, "",   "Can't Open/Write/Close Layer File" },
     { e_EMS_ffSuL, "",   "Can't Open/Write/Close SuppresionLayer File", },
     { e_EMS_ffPTF, "",   "Can't Open/Write/Close Time File" },
     { e_EMS_ffIgn, "",  "Can't Open/Write/Close Ingition Grid File" },
     { e_EMS_ffSta, "",   "Can't Open/Write/Close Statistic Report file" },
      { e_EMS_CloFil, "",   "Generic Close file error CloseAndReturn() function" },


/*.....................................................*/
/* FuelCondition Error Messages - the fuel moisture conditioning */
/*   DLL code uses these */

   { e_EMS_cFMM,   "", "To many Fuel Moisture Models " },
   { e_EMS_cBFM,   "", "Bad Fuel Moisture Model" },

   { e_EMS_cMem,   "", "Cannot alloc Weather Memory " },
   { e_EMS_cMiss,  "", "Not All required inputs loaded " },

   { e_EMS_cWtSt,   "", "Bad number of weather stations " },
   { e_EMS_cWdOb,   "", "Bad number of wind observations " },
   { e_EMS_cElv,    "", "Bad Elevation or Unit" },
   { e_EMS_cSlp,    "", "Bad Slope or Unit " },
   { e_EMS_cCov,    "", "Bad Cover Value " },
   { e_EMS_cSDt,    "", "Bad Starting date/time " },
   { e_EMS_cPro,    "", "Bad number of processors " },
   { e_EMS_cNoWt,   "", "No Weather Data specified " },
   { e_EMS_cNoWn,   "", "No Wind Data specified " },

   { e_EMS_cSqWt,   "", "Weather data record(s) is out of sequence " },

   { e_EMS_cInWn,   "", "Invalid wind date"  },
   { e_EMS_cInWt,   "", "Invalid weather date " },


   { e_EMS_cbMt,   "",     "Bad weather month"         },
   { e_EMS_cbDy,   "",     "Bad weather  day"          },
   { e_EMS_cbPr,   "",     "Bad Precipitation amount"  },
   { e_EMS_cbMtt,   "",     "Bad Min temp time"         },
   { e_EMS_cbXt,   "",     "Bad Max temp time"         },
   { e_EMS_cbXh,   "",     "Bad Max Humidity"          },
   { e_EMS_cbMh,   "",     "Bad Min Humidity"          },
   { e_EMS_cbEl,   "",     "Bad Elevation"             },
   { e_EMS_cbPs,   "",     "Bad Precip start time"     },
   { e_EMS_cbPe,   "",     "Bad Precip end time"       },
   { e_EMS_cbPrr,   "",     "Precip no str/end time"    },
   { e_EMS_cbPB,   "",     "Bad Precip str/end time"   },
   { e_EMS_cbP,    "",     "0 Precip bad str/end time" },


   { e_EMS_Abort,  "",     "User Aborted Moisture Conditioning Run" },


/* FlamMap input moisture condition errors */
   { e_EMS_cStDa,  "",  "start date not in weather data" },
   { e_EMS_cStEn,  "",   "Bad start/ending dates" },

   {  0, "End", "End" } };


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ValidateFarsiteInputs
* Desc: Main function for validation all FARSITE inputs
*       Caller uses the ICF::ErrorMessage() function to get error message
*        text.
*  Ret: 1 No Error
*       else error number code, see ErrorMessage()
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ValidateFarsite ()
{
int i;

   if ( !ICF::ChkFueMoi (&i))            /* Fuel Moist, File & embed'd data  */
     return i;


  if ( !ICF::ChkFolMoi (&i)) /* Foliar Moisture Content     */
     return i;

   if ( !ICF::Chk_FarsiteStartEnd (&i))  /* start & end simulation date/times */
     return i;

  if (!ICF::Chk_FarsiteCondInputs (&i))  /* Moist Cond weather data */
     return i;

   if ( !ICF::ChkFarsiteTimeStep (&i)) /* Farsite Time Step              */
     return i;

   if ( !ICF::ChkFarsiteResolutions (&i)) /* Farsite Time Distance and Perimeter Resolutions              */
     return i;

   if ( !ICF::ChkFarsiteSpotProb (&i)) /* Farsite Spotting probability              */
     return i;

   if ( !ICF::ChkFarsiteSpotGridResolution (&i) ) /* Farsite Spot Grid Resolution */
     return i;

  if ( !ICF::ChkFarsiteSpotDelay (&i)) /* Farsite Spotting Ignition Delay              */
     return i;

// Mar 14 2012 - new function
   if ( !ICF::ChkBurnPeriods_New (&i))     /* Valid Burn Periods?                  */
     return i;

 //  if ( !ICF::ChkBurnPeriods (&i))     /* Valid Burn Periods?                  */
 //    return i;

   if(!ICF::ChkCustomFuels(&i))
	   return i;


   return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ValidateFlamMap
* Desc: Main function for validation all inputs
*
*       Caller uses the ICF::ErrorMessage() function to get error message
*        text.
*  Ret: 1 No Error
*       else error number code, see ErrorMessage()
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ValidateFlamMap ()
{
int i;

   if ( !ICF::ChkCondInputs (&i))      /* Weather/Wind - RAWS */
     return i;

   if ( !ICF::ChkGridWinds (&i))     /* Gridded Winds inputs                  */
     return i;

   if ( !ICF::ChkWind_SD (&i) )       /* Wind Direction & Speed      */
     return i;

   if ( !ICF::ChkFueMoi (&i))         /* Fuel Moist, File & embed'd data  */
     return i;

   if ( !ICF::ChkFolMoi (&i))         /* Foliar Moisture Content     */
     return i;

   if ( !ICF::ChkCroFirMet (&i))     /* Crown Fire Method                    */
     return i;

   if ( !ICF::ChkNumPro (&i))
     return i;

   if ( !ICF::ChkSpreadDir (&i))
     return i;


   if(!ICF::ChkCustomFuels(&i))
	   return i;


   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkCondInputs
* Desc: Validate any and all of the inputs that will be used for doing dead fuel
*        moisiture conditioning and and sent to the Cond DLL
* Note-1: Make sure the cond start and end date/times are far enough apart
*         When using RAWS hourly data I just set it to 240 minutes because
*         that is the time interval for FUEL_SIZE 3, which give it enought time
*
*  Out: ai....error number
*  Ret: 1 OK, 0 = error occured see error number - ai
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkCondInputs (int *ai)
{
int diff,str,end, allowed;

   *ai = e_EMS_WeD2;

/*--------------------------------------------------------------------------------*/
/* Check Constant Moist if there is NO Condition weather, wind, or Stream Data     */
  if ( this->a_Wtr == NULL && this->a_Wnd == NULL && this->a_RAWS == NULL ) {
    if ( this->i_MthEnd != ei_ICFInit ) {
      strcpy (this->cr_ErrExt, "Condition ending date specified but no weather data found");
      return 0; }

    if ( this->i_MthStart != ei_ICFInit ) {
      strcpy (this->cr_ErrExt, "Condition starting date specified but no weather data found");
      return 0; }
    *ai = 0;
    return 1; }

/* -------------------------------------------------------------------------- */
/* If we have weather data we must have wind data */
   if ( this->a_Wtr != NULL && this->a_Wnd == NULL ) {
      strcpy (this->cr_ErrExt,"Have weather data but no wind data");
      return 0; }
   if ( this->a_Wnd != NULL && this->a_Wtr == NULL ) {
      strcpy (this->cr_ErrExt,"Have wind data but no weather data");
      return 0; }

/* if we have weather/wind data than we cannot also allow Weather Stream Data */
  if ( this->a_Wtr != NULL && this->a_RAWS != NULL ) {
    sprintf (this->cr_ErrExt,"Use only one type of weather stream input: %s or %s or %s", e_ICF_RAWS, e_ICF_WeF, e_ICF_WeD);
    return 0; }

/* Validate Wind and Weather Stream */
  if ( this->a_Wtr != NULL ) {        /* We have weather data */
    if ( !ICF::ChkWeather (ai))       /* Check the Weather Data       */
      return 0;
    if ( !ICF::ChkWind (ai))          /* Check the Wind Stream data           */
      return 0;
    if ( !ICF::ChkWeaWinDay (ai))     /* See if Weather & Wind use same days */
      return 0; }

/* Validate RAWS Weather Stream Data */
   if ( this->a_RAWS != NULL ) {
     if ( !Chk_RAWSData (ai) )
        return 0;  }

/* Check Conditioning Starting and Ending dates */
    str = ChkCondDate ((char *)"Start", ai);     /* Check Start Date */
    if ( str == 0 )
      return 0;
    end = ChkCondDate ((char *)"End ", ai);      /* Check End date */
    if ( end == 0 )
      return 0;

/* Make sure start and end condition dates are far enough apart */
/* See Note-1 above */
    diff = end - str;                    /* take diff of  minutes */
    if ( this->iN_RAWS != NULL )         /* if have RAWS hourly data */
      allowed = 240 ;                    /* Set minimum minutes time condition can run */
    else
      allowed = ( e_MinPerDay );//* 2 );     /* daily weather data */

    if ( diff <  allowed ) {
      strcpy (this->cr_ErrExt, "Start/End dates too close or not chronological");
      return 0; }

   *ai = 0;
   return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkCondWSD
* Desc: Check the fuel conditioning Weather Stream Data (RAWS)
*       This just checks the record date/time sequence. More error checks
*        will get done by the Cond DLL once the data is loaded into it.
* NOTE: the d_RAWS.i_Yr should be setting using the e_StrYear(+1)
*  Out: ai....error number
*  Ret: 1 OK, 0 = error occured see error number - ai
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Chk_RAWSData (int *ai)
{
int i, j, A, B, C, date, min, iS;

  B = 0;
  for ( i = 0; i < this->iN_RAWS; i++ ) {
     A = B;
     date  = ICF::GetMCDate (this->a_RAWS[i].i_Mth, this->a_RAWS[i].i_Day, this->a_RAWS[i].i_Yr);

     if ( date == 0 ) {
       sprintf (this->cr_ErrExt,"Invalid Month: %d  Day: %d", this->a_RAWS[i].i_Mth, this->a_RAWS[i].i_Day);
       return 0; }

     if ( this->a_RAWS[i].i_Time < 0 ||  this->a_RAWS[i].i_Time > 2359 ){
        sprintf (this->cr_ErrExt,"Invalid Time: %d Month: %d  Day: %d", this->a_RAWS[i].i_Time, this->a_RAWS[i].i_Mth, this->a_RAWS[i].i_Day);
        return 0; }

     min = ICF::MilToMin(this->a_RAWS[i].i_Time);  /* Military Time to minutes */
     B = (date * e_MinPerDay) + min;               /* to total minutes */
     if ( A >= B ) {
       sprintf (this->cr_ErrExt,"Non-consecutive date found Month: %d  Day: %d  Time: %d", this->a_RAWS[i].i_Mth, this->a_RAWS[i].i_Day,this->a_RAWS[i].i_Time);
       return 0; } }

/*-------------------------------------------------------------------------------*/
/* Make sure we don't have more than one group of records for the same date */
  for ( i = 0; i < this->iN_RAWS; i++ ) {
    iS = 0;
    for ( j = i; j < this->iN_RAWS; j++ ) {
      if ( this->a_RAWS[i].i_Mth == this->a_RAWS[j].i_Mth && this->a_RAWS[i].i_Day == this->a_RAWS[j].i_Day ) {
        if ( iS == 1 ) {
          sprintf (this->cr_ErrExt,"Same Month %d, Day %d Found", this->a_RAWS[j].i_Mth, this->a_RAWS[j].i_Day);
          return 0; }
        continue; }
      iS = 1;  } }

/*-------------------------------------------------------------------------------*/
/* make sure there is no more than a 12 hour gap between each record */
   j = this->iN_RAWS - 1;
   for ( i = 0; i < j; i++ ) {

      A = this->GetTotMin(this->a_RAWS[i].i_Mth,this->a_RAWS[i].i_Day,
                     this->a_RAWS[i].i_Time, this->a_RAWS[i].i_Yr);

      B = this->GetTotMin(this->a_RAWS[i+1].i_Mth,this->a_RAWS[i+1].i_Day,
                     this->a_RAWS[i+1].i_Time, this->a_RAWS[i+1].i_Yr);
      C = B - A;
//      printf ("%d  %d   %d \n", A, B, C);
      if ( C >= e_RAWS_MaxGap ) {
        sprintf (this->cr_ErrExt,"Excessive time gap");
        *ai =e_EMS_RAWS;
        return 0; }
    }

    return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkSpreadDir
* Desc: Check Spread Direction North and Max inputs
*  Out: ai....error number
*  Ret: 1 OK, 0 = error occured see error number - ai
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkSpreadDir (int *ai)
{
   *ai = e_EMS_SDN;  /* set in case of error */

   if ( this->i_SDFM == ei_ICFInit && this->i_SDFN == ei_ICFInit )
      return 1;     /* if both are not found that's ok */

   if ( this->i_SDFM != ei_ICFInit && this->i_SDFN != ei_ICFInit )
      return 0;     /* Both switches present is an error */

   if ( this->i_SDFM != ei_ICFInit )
      if ( this->i_SDFM < 0 || this->i_SDFM > 360 )
         return 0;

   if ( this->i_SDFN != ei_ICFInit )
      if ( this->i_SDFN < 0 || this->i_SDFN > 360 )
         return 0;

   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkNumPro
* Desc: Check the Number of Processors switch argument
*  Out: ai....error number
*  Ret: 1 OK, 0 = error occured see error number - ai
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkNumPro (int *ai)
{
// For now ok, program will set automatically
     return 1;

	if ( this->i_NumPro >= e_NPMin && this->i_NumPro <= e_NPMax )
     return 1;

   sprintf (this->cr_ErrExt,e_iMMFmt, this->i_NumPro, e_NPMin, e_NPMax);
   *ai = e_EMS_NP;
   return 0;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkCroFirMet - check Crown Fire Method
* Desc: check the switch arguments for  CROWN_FIRE_METHOD:
*  Out: ai...error number
*  Ret: 1 ok, else 0 for error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkCroFirMet (int *ai)
{
  if ( !strcmp (this->cr_CroFirMet,"") )
     return 1;                         /* Ok if blank, we'll set default later  */

  if ( !strcmp (this->cr_CroFirMet,e_CFM_Finney))
     return 1;

  if ( !strcmp (this->cr_CroFirMet,e_CFM_ScotRein) )
     return 1;

  sprintf (this->cr_ErrExt,"Bad argument-> \"%s\"  use %s or %s",
           this->cr_CroFirMet, e_CFM_Finney, e_CFM_ScotRein);
  *ai = e_EMV_CFM;
  return 0;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkGridWinds
* Desc: Check the switches for the Gridded Winds
*  Out: *ai....error code
*  Ret: 1 = ok, 0 error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::ChkGridWinds (int *ai)
{

	bool hasWindDirFile = false, hasWindSpeedFile = false;
  if ( strcmp (this->cr_GWDF,"") )
  {
	  hasWindDirFile = true;
    // sprintf(this->cr_ErrExt, "Switch %s Not implemented yet",e_ICF_GWDF);
	 //*ai = e_EMS_Gen;
	 //return 0;
  }
  if ( strcmp (this->cr_GWSF,"") )
  {
	  hasWindSpeedFile = true;
     //sprintf(this->cr_ErrExt, "Switch %s Not implemented yet",e_ICF_GWSF);
	 //*ai = e_EMS_Gen;
	// return 0;
  }
  if((hasWindDirFile && !hasWindSpeedFile) || (hasWindSpeedFile && !hasWindDirFile))
  {//need them both
     sprintf(this->cr_ErrExt, "Need both Wind Dir and Speed files");
	 *ai = e_EMS_Gen;
	 return 0;
  }
  if(hasWindDirFile && hasWindSpeedFile)//has both, OK will load on launch
  {
	  b_hasValidGriddedWindsFiles = true;
	  return 1;
  }
  else
 	  b_hasValidGriddedWindsFiles = false;

//    if no gridded winds files, fall thru to WindNinja checks

/* ...........................................................................*/
/* Check individual Gridded Wind Switches, these cause WindNinja to be used */
  if ( !strcmp (this->cr_GGW,""))        /* No 'Generate' switch found      */
    return 1;
  if ( !strcmp (this->cr_GGW,e_GW_No))   /* 'Generate' switch is NO          */
    return 1;

  if ( strcmp (this->cr_GGW,e_GW_Yes)){  /* Then its gotta be YES           */
     sprintf (this->cr_ErrExt, "Use %s or %s", e_GW_Yes, e_GW_No);
     *ai = e_EMS_GWW;
	 return 0; }

/* -------------------------------------------------------------------------*/
/* NOTE: more checking and setting defaults will get done when we load these */
/* values and run WindNinja see FlamMap::LoadGridWinds                      */

/* Resolution .....................                                     */
  if ( this->f_GriWinRes != ef_ICFInit ) {
	  if ( this->f_GriWinRes < e_GWH_Min || this ->f_GriWinRes > e_GWH_Max ) {
		sprintf (this->cr_ErrExt, "Gridded wind resolution %4.1f is out of range ( %4.1f -> %4.1f )", this->f_GriWinRes, e_GWR_Min,e_GWR_Max);
        *ai = e_EMS_GWR;
		return 0; }}

/* Wind Height................                                         */
  if ( this->f_GriWinHei != ef_ICFInit ) {
    if ( this->f_GriWinHei < e_GWH_Min || this->f_GriWinHei > e_GWH_Max ){
	   sprintf (this->cr_ErrExt, "Gridded wind height %6.1f is out of range ( %4.0f -> %4.0f )", this->f_GriWinHei,e_GWH_Min, e_GWH_Max);
      *ai = e_EMS_GWH;
	  return 0; }}

/* Check optional diurnal inputs */
  if ( !ChkWNDiurnal() ){
     *ai = e_EMS_GWD;
     return 0; }


/* Gridded Winds Vegetation/Height                                        */
// This switch is now Obsolete....................
//  if ( strcmp (this->cr_GriWinVeg,"") &&
//	   strcmp (this->cr_GriWinVeg,e_GWV_Grass) &&
//       strcmp (this->cr_GriWinVeg,e_GWV_Brush) &&
//	   strcmp (this->cr_GriWinVeg,e_GWV_Forest) ) {
//       sprintf (this->cr_ErrExt, "Gridded wind veg type %s is invalid, use %s %s %s",
//		   this->cr_GriWinVeg,e_GWV_Grass,e_GWV_Brush,e_GWV_Forest);
//       *ai = e_EMS_GWV;
//        return 0; }

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkWNDiurnal
* Desc: Check the Windninja diurnal inputs. Diurnal inputs are optional.
*
*  Ret: 1 ok, 0 = error,
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::ChkWNDiurnal ()
{
   if ( strcmp (s_WDI.cr_YesNo, e_GW_Yes) )
      return 1;   /* not using diurnal info */

   if ( CheckMonthDay (s_WDI.i_Mth, s_WDI.i_Day ) == 0 ) {
      strcpy (cr_ErrExt,"Invalid Month Day Date");
      return 0; }

   if ( s_WDI.i_Year < 1900 || s_WDI.i_Year > 2020 ){ /* ballpark check */
     strcpy (cr_ErrExt,"Invalid Year Date");
     return 0; }

   if ( s_WDI.i_Sec < 0 || s_WDI.i_Sec > 60 ||
        s_WDI.i_Min < 0 || s_WDI.i_Min > 60 ||
        s_WDI.i_Hour < 0 || s_WDI.i_Hour > 24 ||
        s_WDI.i_TimeZone < -12 || s_WDI.i_TimeZone > 12 ) {
      strcpy (cr_ErrExt,"Invalid Second Minute Hour Time Zone");
      return 0; }

   if ( s_WDI.f_AirTemp < -50.0 || s_WDI.f_AirTemp > 130.0 ){
      strcpy (cr_ErrExt,"Invalid Air Temperature");
      return 0; }

/* percent is expressed as a whole number */
   if ( s_WDI.f_CloudCov < 0.0 || s_WDI.f_CloudCov > 100.0 ){
      strcpy (cr_ErrExt,"Invalid Cloud Cover Percent");
      return 0; }

/* Catch if someone trys to enter percent as a decimal */
   if ( s_WDI.f_CloudCov > 0.0 && s_WDI.f_CloudCov < 1.0 ){
      strcpy (cr_ErrExt,"Invalid Cloud Cover Percent, use integer value 0 -> 100");
      return 0; }

  if ( s_WDI.f_Longitude < -180.0 || s_WDI.f_Longitude > 180.0 ) {
     strcpy (cr_ErrExt,"Invalid Longitude");
     return 0; }

 return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkWeaWinDay
* Desc: Make sure that the Weather & Wind data used to do Conditioning have
*        the same starting and ending dates.
*  Ret: 1 ok, 0 = error,
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkWeaWinDay (int *ai)
{
int i,j;


  if ( this->iN_Wnd <= 0 )             /* is there any wind data read in    */
     return 1;                         /*  no                               */
  if ( this->iN_Wtr <= 0 )              /* is there weather data             */
     return 1;

   *ai = e_EMS_WiD;                    /* set error code in case of error   */


  if ( this->a_Wnd[0].i_Mth != this->a_Wtr[0].i_Mth ||
       this->a_Wnd[0].i_Day != this->a_Wtr[0].i_Day ) {
    strcpy (this->cr_ErrExt,"Weather - Wind beginning dates do not match");
    return 0; }

  i = this->iN_Wnd - 1;
  j = this->iN_Wtr - 1;
  if ( this->a_Wnd[i].i_Mth != this->a_Wtr[j].i_Mth ||
       this->a_Wnd[i].i_Day != this->a_Wtr[j].i_Day ) {
    strcpy (this->cr_ErrExt,"Weather - Wind ending dates do not match");
    return 0; }

   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkWind   Check Wind  Stream Data
*       switch = WIND_STREAM:
* Desc: Check the fields in the array of entries
*
# Mth Day Hour Speed Direction CloudCover
#   8  10    0     1    54        0
*
*  Out: ai...error number
*
*  Ret: 1 ok, 0 = error,
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkWind (int *ai)
{
int i;

   if ( this->iN_Wnd <= 0 )             /* is there any wind data read in    */
     return 1;                          /*  no                               */

   *ai = e_EMS_WiD;                     /* Set incase we get an error        */

   for ( i = 0; i < this->iN_Wnd; i++ ) {
     if ( !_ChkMthDay (this->a_Wnd[i].i_Mth, this->a_Wnd[i].i_Day,this->cr_ErrExt))
       return 0;

     if ( this->a_Wnd[i].i_Hr < e_HrMin || this->a_Wnd[i].i_Hr > e_HrMax ) {
       sprintf (this->cr_ErrExt, "Hour %d is out of limits %d -> %d",this->a_Wnd[i].i_Hr,e_HrMin,e_HrMax);
       return 0; }

     if ( this->a_Wnd[i].i_CloCov < 0 || this->a_Wnd[i].i_CloCov > 100 ) {
       sprintf (this->cr_ErrExt, "Cloud Cover %d is out of limits 0 > 100",this->a_Wnd[i].i_CloCov);
       return 0; }
   }

/* Chk for E English or Metric, and Set default if none     */
  if ( !ICF::EngMet (this->cr_WiDU,this->cr_ErrExt)){
     *ai = e_EMV_WiD;
     return 0;}

/* Check date/time sequence  */
int A, B, date, min;
  B = 0;
  for ( i = 0; i < this->iN_Wnd; i++ ) {
     A = B;
     date  = ICF::GetMCDate (this->a_Wnd[i].i_Mth, this->a_Wnd[i].i_Day, this->a_Wnd[i].i_Year);
     min = ICF::MilToMin(this->a_Wnd[i].i_Hr);
     B = (date * e_MinPerDay) + min;           /* to total minutes */
    if ( A >= B ) {
       sprintf (this->cr_ErrExt,"Non-consecutive date found Month: %d  Day: %d ", this->a_Wnd[i].i_Mth, this->a_Wnd[i].i_Day);
       return 0; } }

  return 1;
}
/*********************************************************************
* Name: MilToMin
* Desc: Convert Military time to total minutes.
*       Ex: 1030 = ( 10 * 60 ) + 30 = 630 minutes
*******************************************************************/
int ICF::MilToMin (int i_MilTim)
{
int hour,min,Tot;
    hour = i_MilTim / 100;
    min = i_MilTim % 100;
    Tot = hour * 60;
    Tot = Tot + min;
    return Tot;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkWeather   Check Weather inputs
* Desc: Check the fields in the array of entries
*       switch = WEATHER_STREAM or  WEATHER_DATA:
*       Both switches are optional but you can't have both
*
*   i_Mth   = month,
*   i_Day   = day,
*   f_Per   = precip in 100th of inch (ex 10 = 0.1 inches
*   i_mTH   = min_temp_hour 0-2359,
*   i_xTH   = max_temp_hour 0 - 2359,
*   f_mT    = min_temp,
*   f_xT    = max_temp,
*   i_xH    = max_humidity,
*   i_mH    = mix_humidity,
*   i_Elv   = elevation,
*   i_PST   = precip_start_time 0-2359, if f_Per > 0
*   i_PET   = precip_end_time 0-2359    if f_Per > 0
*
*  Out: ai...error number
*
*  Ret: 1 ok, 0 = error,
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkWeather (int *ai)
{
int i,j;

   if ( this->iN_Wtr <= 0 )              /* is there any data                 */
     return 1;                          /*  no                               */

   *ai = e_EMS_WeD1;                   /* Set incase we get an error        */

   for ( i = 0; i < this->iN_Wtr; i++ ) {
     if ( !_ChkMthDay (this->a_Wtr[i].i_Mth, this->a_Wtr[i].i_Day,this->cr_ErrExt))
       return 0;
     if ( this->a_Wtr[i].f_Per < 0 || this->a_Wtr[i].f_Per > 10000 ) {
       sprintf (this->cr_ErrExt,"Precip is out of limits %f",this->a_Wtr[i].f_Per);
       return 0; }
     if ( this->a_Wtr[i].i_mTH < 0 || this->a_Wtr[i].i_mTH > e_HrMax ) {
       sprintf (this->cr_ErrExt,"Min temp hour is out of limits %d",this->a_Wtr[i].i_mTH);
       return 0; }
     if ( this->a_Wtr[i].i_xTH < 0 || this->a_Wtr[i].i_xTH > e_HrMax ) {
       sprintf (this->cr_ErrExt,"Max temp hour is out of limits %d",this->a_Wtr[i].i_xTH);
       return 0; }

     if ( this->a_Wtr[i].i_xH > 100 || this->a_Wtr[i].i_xH < 1 ){
       sprintf (this->cr_ErrExt,"Humidity Max: %d is out of range(1->100)",this->a_Wtr[i].i_xH);
       return 0; }

     if ( this->a_Wtr[i].i_mH < 1 || this->a_Wtr[i].i_mH > 100 ) {
       sprintf (this->cr_ErrExt,"Humidity Min: %d is out of range(1->100) - Mth %d  Day %d",
                    this->a_Wtr[i].i_mH,this->a_Wtr[i].i_Mth, this->a_Wtr[i].i_Day);
       return 0; }

/*----------------------------------------------------------------------------------*/
/* Taking this check out for now - I think min can be greater than the max, because */
/*  I think the min humid is just the humid for the min temp, so it could be anything */
/*  I noticed this happening with the test RAWS data I summarized */
//     if ( this->a_Wtr[i].i_mH > this->a_Wtr[i].i_xH ) {
//	    	 sprintf (this->cr_ErrExt,"Humidity Min: %d is > Max: %d  Mth %d  Day %d",
//           this->a_Wtr[i].i_mH,this->a_Wtr[i].i_xH,this->a_Wtr[i].i_Mth, this->a_Wtr[i].i_Day);
//       return 0; }

     if ( this->a_Wtr[i].i_Elv < 0 || this->a_Wtr[i].i_Elv > 14000 ) {
       sprintf (this->cr_ErrExt,"Elevation %d is out of range (0 -> 14000)",this->a_Wtr[i].i_Elv);
       return 0; }

     if ( this->a_Wtr[i].i_PST < 0 || this->a_Wtr[i].i_PST > e_HrMax ) {
      sprintf (this->cr_ErrExt,"Precip start time out of range %d",this->a_Wtr[i].i_PST);
      return 0; }

     if ( this->a_Wtr[i].i_PET < 0 || this->a_Wtr[i].i_PET > e_HrMax ) {
      sprintf (this->cr_ErrExt,"Precip end time out of range %d",this->a_Wtr[i].i_PET);
      return 0; }

	    if ( this->a_Wtr[i].f_Per > 0 ) {
	      if (this->a_Wtr[i].i_PET == 0 ){
	        sprintf ( this->cr_ErrExt, "Precip %.2f has no start/end time - Mth %d Day %d", this->a_Wtr[i].f_Per, this->a_Wtr[i].i_Mth, this->a_Wtr[i].i_Day);
	      	 return 0; }
	    if (this->a_Wtr[i].i_PST >= this->a_Wtr[i].i_PET ) {
		     sprintf ( this->cr_ErrExt, "Precip start time %d > end time %d", this->a_Wtr[i].i_PST,this->a_Wtr[i].i_PET);
	      return 0; } }

	    if ( this->a_Wtr[i].f_Per == 0 ) {
	     	if ( this->a_Wtr[i].i_PET > 0 || this->a_Wtr[i].i_PST > 0 ){
	      	 sprintf (this->cr_ErrExt, "0 Precip contains invalid start/end time - %d  %d", this->a_Wtr[i].i_PST,this->a_Wtr[i].i_PET);
	        return 0; } }
	}

/* Check that all days are consecutive */
int A, B;
  B = 0;
  for ( i = 0; i < this->iN_Wtr; i++ ) {
     A = B;
     B  = ICF::GetMCDate (this->a_Wtr[i].i_Mth, this->a_Wtr[i].i_Day, this->a_Wtr[i].i_Year);
     if ( A >= B ) {
       sprintf (this->cr_ErrExt,"Non-consecutive date found Month: %d  Day: %d ", this->a_Wtr[i+1].i_Mth, this->a_Wtr[i+1].i_Day);
       return 0; } }

/* Only allow a years worth of data */
  if ( this->iN_Wtr > e_MaxWD ) {
    sprintf (this->cr_ErrExt,"Number of weather records exceeds limit %d", e_MaxWD);
    return 0; }

/* Make sure we don't have multiple recs with the same date, we only allow one year's */
/*  worth of data, which means no overlaping dates, this check is a little of kill...but it easy  */
  for ( i = 0; i < this->iN_Wtr; i++ ) {
    for ( j = i; j < this->iN_Wtr; j++ ) {
      if ( i == j )     /* don't compare to self */
         continue;
      if (  this->a_Wtr[i].i_Mth == this->a_Wtr[j].i_Mth && this->a_Wtr[i].i_Day == this->a_Wtr[j].i_Day ) {
        sprintf (this->cr_ErrExt,"two same dates found, Month: %d  Day: %d ", this->a_Wtr[i].i_Mth, this->a_Wtr[i].i_Day);
        return 0; } } }

  return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*  Ret:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int _ChkMthDay (int M, int D, char cr_Err[])
{
int Day[] = {0, 31,28,31,30,31,30,31,31,30,31,30,31};

  sprintf (cr_Err,"Wrong month (%d) and/or day (%d)",M,D);
  if ( M < 1 || M >  12 )
    return 0;
  if ( D < 1 )
    return 0;
  if ( D > Day[M] )
    return 0;

  strcpy (cr_Err,"");
  return 1;

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: EngMet
* Desc: Check for English or Metric unit text
*       if none set to English
*  Ret: 1 = ok, 0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::EngMet (char cr[],  char cr_Err[])
{

   strcpy (cr_Err,"");

   if ( !strcmp(cr,"" )) {
     strcpy (cr,"English");
     return 1; }

   if ( !strcmp(cr,"English") )
      return 1;

   if ( !strcmp(cr,"Metric") )
      return 1;

   sprintf (cr_Err,"\"%s\" - Use: %s or %s",cr,"English","Metric");
   return 0;

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkCondDate
* Desc: Check a Condtioning Period Start or End date that has come from
*        the inputs file switch.
*       The date must be within the entered weather data table dates but
*        in addition it is offset by a day - see code below
* Note-1:For Daily Weather Data:
*         Condition start day has to be at least one day after the first day
*         in the Weather data table, End day has to be at least one day before
*         the last date in the weather table. Not sure why but that's how the
*         old FlamMap did it.
*        For hourly RAWS Data:
*         the start cond date can be right at the very beginning of the RAWS
*          data, and the end cond date can be right up to the end of the RAWS
*          data
*
*
*  Out: ai....error number
*  Ret: 1 = ok, 0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkCondDate (char cr_Mode[], int *ai)
{
int i, cy,cm,cd,ch, wm,wd,wy,wt, i_ConDate, m;
int iS, i_ConTotMin, i_WtrDate, i_WtrTotMin ;

  *ai = e_EMS_CPE;            /* if no err occurs caller won't use           */

  if ( !strcmp (cr_Mode,"Start") ){  /* Get Start Conditioning Date */
    cy = this->i_YearStart;
    cm = this->i_MthStart;
    cd = this->i_DayStart;
    ch = this->i_HourStart;
    iS = 0; }                         /* Get 1st entry in Weather data */
  else {                              /* Get Ending Conditionng date */
    cy = this->i_YearEnd;
    cm = this->i_MthEnd;
    cd = this->i_DayEnd;
    ch = this->i_HourEnd;
    iS = -1; }                        /* get last entry in Weather data */

  if ( cm == ei_ICFInit ) {           /* If month is missing    */
    sprintf (this->cr_ErrExt, "Conditioning %s date is missing", cr_Mode);
    return 0; }

  if ( !_ChkMthDay (cm, cd,this->cr_ErrExt)) {
     sprintf (this->cr_ErrExt,"Invalid %s date %d %d", cr_Mode, cm, cd );
     return 0; }

  if ( ch < 0 || ch > 2359 ) {
    sprintf (this->cr_ErrExt,"Bad hour of day specified %d ( use 0->2359 )", ch);
    return 0; }

/* Form Conditioning Date (start/end) in total minutes... */
  i_ConDate = ICF::GetMCDate(cm,cd,cy);         /* psuedo Julian Moist Cond date */
  m = ICF::MilToMin(ch);                        /* Military time to minute */
  i_ConTotMin = (i_ConDate * e_MinPerDay) + m;  /* Cond Data in Total Minutes */

  ICF::GetWthDate(iS,&wm,&wd,&wy,&wt);           /* Get 1st or last rec in weathr tbl */
  i_WtrDate = ICF::GetMCDate(wm,wd,wy);          /* Get psuedo julian date */
  m = ICF::MilToMin(wt);
  i_WtrTotMin = (i_WtrDate * e_MinPerDay) + m;   /* total time in minutes  */

/*----------------------------------------------------------------------------------*/
/* Check Start condition date against the weather data we have to work with */
  if ( !strcmp (cr_Mode,"Start") ){
    if ( this->iN_RAWS > 0 ) {            /* are we working with RAWS data */
      if ( i_ConTotMin < i_WtrTotMin ) {
        strcpy (this->cr_ErrExt,"Conditioning Start Date is less than starting RAWS data");
        return 0; } }
/* working with Daily weather data */
    else {
      i = i_WtrTotMin + e_MinPerDay;
      if ( (i_ConTotMin ) < i ) {
        strcpy (this->cr_ErrExt,"Conditioning Start Date is NOT within usable range of weather data");
        return 0; } }

    return i_ConTotMin;
  }


/*------------------------------------------------------------------------------*/
/* For End Conditioning Date */

  if ( this->iN_RAWS > 0 ) {              /* if working with RAWS data */
    if ( i_ConTotMin > i_WtrTotMin ) {
      strcpy (this->cr_ErrExt,"Conditioning End Date is outside of RAWS data");
      return 0; }  }
  else {
// Changed 1-26-11 Fixed bug, it was causing errors for 2cd to last day dates
//     i = i_WtrTotMin - e_MinPerDay;
//     if ( i_ConTotMin > i ) {
      if ( i_ConTotMin > i_WtrTotMin ) {
      strcpy (this->cr_ErrExt,"Conditioning Ending Date out of usable range");
      return 0; } }

  return i_ConTotMin;
}


/********************************************************************
* Name: isInWeather
* Desc: See if there is any data for the specified month in the
*        weather data.
*   In: month....1->12
*  Ret: 0 = month not in data
*       1 = month is in data
*       2 = month is in data but it is in the 2cd year
*********************************************************************/
int ICF::isInWeather(long month)
{
int i, iS_Year;
   iS_Year = 0;
   for ( i = 0; i < iN_Wtr; i++ ) {
      if ( i > 0 ) {
         if ( a_Wtr[i-1].i_Mth == 12 && a_Wtr[i].i_Mth == 1 )  /* into next year */
           iS_Year++;  }
      if ( a_Wtr[i].i_Mth == month ) {   /* Found Mth we looking for */
        if ( iS_Year > 0 )
          return 2;               /* we have data in 2 diff years */
        else
          return 1; }             /* weather data is all in same year */
   }
   return 0;                      /* Month was in weather data */
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkFolMoi Check Foliar Moisture Content
* Desc:
*  Out: ai....error number
*  Ret: 1 = ok, 0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::ChkFolMoi (int *ai)
{
  if ( this->i_FolMoi == ei_ICFInit ) /* no val enterd in cmd fil */
    return 1;

  if ( this->i_FolMoi < e_FM_Min ||
       this->i_FolMoi > e_FM_Max ) {
    sprintf (this->cr_ErrExt,e_iMMFmt, this->i_FolMoi, e_FM_Min, e_FM_Max);
   *ai = e_EMS_FMC;
   return 0; }

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkWind_SD   Check Wind Inputs
* Desc: Wind Direction and Speed are Mandatory
*  Out: ai....error number
*       cr_Err...error message
*  Ret: 1 = ok, 0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::ChkWind_SD (int *ai)
{
  if (this->b_hasValidGriddedWindsFiles)
		   return 1;

 if ( this->f_WinDir == ei_ICFInit ){   /* if missing                        */
    strcpy (this->cr_ErrExt,"Mandatory Wind Direction not specified ");
    *ai = e_EMS_WDA;
    return 0;}                          /* error                             */

  if ( this->f_WinDir < e_WD_Min || this->f_WinDir > e_WD_Max ) {
    if ( this->f_WinDir != e_WD_Downhill && this->f_WinDir != e_WD_Uphill ) {
	   sprintf (this->cr_ErrExt,e_fMMFmt, this->f_WinDir,e_WD_Min, e_WD_Max);
      *ai = e_EMS_WDA;
      return 0; }
  }

  if ( this->f_WinSpe == ef_ICFInit ){   /* if missing                        */
    strcpy (this->cr_ErrExt,"Mandatory Wind Speed not specified");
    *ai = e_EMS_WSA;
    return 0;}                          /* error                             */

  if ( this->f_WinSpe < e_WS_Min ||     /* check limits                      */
       this->f_WinSpe > e_WS_Max ) {
    sprintf (this->cr_ErrExt,e_fMMFmt, this->f_WinSpe, e_WS_Min, e_WS_Max);
    *ai = e_EMS_WSA;
    return 0; }

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkFueMoi     Check Fuel Moisture
* Desc: We need either a fuel file name or all the individual global fuel
*       moistures. Every other condition is an error
*  Out: ai....error number
*  Ret: 1 ok, else 0
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkFueMoi (int *ai)
{
int i,j;
char sw1[100], sw2[100], cr[30];

  ICF::FromSwNa (sw1, (char *)e_ICF_FMF);            /* Form switch name with mark on end */
  ICF::FromSwNa (sw2, (char *)e_ICF_FMD);

  if ( strcmp (this->cr_FMF,"") &&      /* Have a file name                  */
       this->iN_FMD > 0 ) {             /* and also have fuels data          */
    sprintf (this->cr_ErrExt, "Use only one: %s or %s", sw1, sw2);
    *ai = e_EMV_FM;
    return 0; }                         /* Error - Can't have both           */

  if ( this->iN_FMD <= 0  &&            /* No moistures specified            */
      !strcmp (this->cr_FMF,"") ){      /* and No file name                  */
    sprintf (this->cr_ErrExt, "Must use one: %s or %s", sw1, sw2);
    *ai = e_EMV_FM;
    return 0; }                         /* Error - Can't have both           */

  if ( strcmp (this->cr_FMF,"") )       /* Have a file so beat it            */
     return 1;

  j = 0;
  strcpy (cr,"");
  for ( i = 0; i < this->iN_FMD; i++ ) {
    if (this->a_FMD[i].i_Model == e_DefFulMod) /* count Default Models       */
      j++;
    if (this->a_FMD[i].i_TL1   < e_GFMlow || this->a_FMD[i].i_TL1   > e_GLFMup) sprintf (cr, "1 Hr (%d) "  ,this->a_FMD[i].i_TL1  );
    if (this->a_FMD[i].i_TL10  < e_GFMlow || this->a_FMD[i].i_TL10  > e_GLFMup) sprintf (cr, "10 Hr (%d) " ,this->a_FMD[i].i_TL10 );
    if (this->a_FMD[i].i_TL100 < e_GFMlow || this->a_FMD[i].i_TL100 > e_GLFMup) sprintf (cr, "100 Hr (%d) ",this->a_FMD[i].i_TL100);
    if (this->a_FMD[i].i_TLLH  < e_GFMlow || this->a_FMD[i].i_TLLH  > e_GLFMup) sprintf (cr, "Herb (%d) "  ,this->a_FMD[i].i_TLLH );
    if (this->a_FMD[i].i_TLLW  < e_GFMlow || this->a_FMD[i].i_TLLW  > e_GLFMup) sprintf (cr, "Woody (%d) " ,this->a_FMD[i].i_TLLW );
    if (strcmp (cr,"")) {
      sprintf (this->cr_ErrExt,"Out of limit %s fuel moisture ( limit %d -> %d )",cr,e_GFMlow, e_GLFMup);
      *ai = e_EMS_FMS;
      return 0; }
  }

  if ( j != 1 ) {                       /* Only want one Default Model       */
    strcpy (this->cr_ErrExt,"No default Model 0 specified");
    *ai = e_EMS_FMD;
    return 0; }

  return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: FromSwNa
* Desc: Form a with name by putting the swithc mark on the end
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void  ICF::FromSwNa (char out[], char in[])
{
 sprintf (out,"%s%c",in,e_ICF_Sep);
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: RemoveSwMa
* Desc: Remove switch mark, "Switch:" --> "Switch"
* NOTE: this modifies the string you send in
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void  ICF::RemoveSwMa (char cr[])
{
 StrRepChr (cr,e_ICF_Sep, '\0' );   /* remov mark on end of Switch       */
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckNextDay
* Desc: See if days are consecutive, one day apart
* NOTE: THIS doesn't do a Feb 29, we're not worry about that though
*       This will do end of year wrap around though
*   In: month day combinations, mm dd is the day after combination
*  Ret: 1 ok,
*       0 days are not consectutive or bad month or day
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int   ICF::CheckNextDay (int m, int d, int mm, int dd )
{
int i;

   if ( m == 12 && d == 31 && mm == 1 && dd == 1 )
     return 1;                      /* Special Case - Year end wrap around */

   i = CheckMonthDay (mm, dd) - CheckMonthDay (m, d);
   if ( i != 1 )
     return 0;
   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckMthDay
* Desc: Return the day number - julian date
* NOTE: THIS doesn't do a Feb 29
*   In: month 1->12, day 1->N
*  Ret: julian date or 0 for invalid date
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int ICF::CheckMonthDay (int m, int  d)
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



/**************************************************************************
* Name: ChkBurnPeriods_New
* Desc: this function replaces the other ChkBurnPeriods which didn't know
*       handle year end wrap - See Note function heading note for the
*       old ChkBurnPeriods()
***************************************************************************/
int ICF::ChkBurnPeriods_New(int *ai)
{
int i, mth, day, HrSt, HrEn, iN, year ;
int  i_Date, i_Min, i_StrMin, i_EndMin, i_BPMin, i_PreMin;

  *ai = e_EMS_FARSITE_BURN_PERIODS;     /* if no err occurs caller won't use           */
   if ( this->iN_BurnPeriods <= 0 )
 	 return 1;                          // no burn periods


   i_Date = ICF::GetMCDate(this->i_FarsiteStartMth,this->i_FarsiteStartDay,this->i_FarsiteStartYear);          /* Get psuedo julian date */
   i_Min = ICF::MilToMin(this->i_FarsiteStartHour);
   i_StrMin = (i_Date * e_MinPerDay) + i_Min;   /* total time in minutes  */


   i_Date = ICF::GetMCDate(this->i_FarsiteEndMth,this->i_FarsiteEndDay,this->i_FarsiteEndYear);       /* Get psuedo julian date */
   i_Min = ICF::MilToMin(this->i_FarsiteEndHour);
   i_EndMin = (i_Date * e_MinPerDay) + i_Min;   /* total time in minutes  */

   i_PreMin = iN = 0;
   for ( i = 0; i < this->iN_BurnPeriods; i++) {  /* Each Burn Period */
      mth = this->a_BurnPeriods[i].i_Month;
      day = this->a_BurnPeriods[i].i_Day;
      year = this->a_BurnPeriods[i].i_Year;
      HrSt = this->a_BurnPeriods[i].i_HourStart;
      HrEn = this->a_BurnPeriods[i].i_HourEnd;

      if ( !_ChkMthDay (mth, day,this->cr_ErrExt)) {
        sprintf (this->cr_ErrExt,"Invalid date %d %d", mth, day );
        return 0; }

      if ( HrSt < 0 || HrSt > 2400 || HrEn < 0 || HrEn > 2400 ) {
        sprintf (this->cr_ErrExt,"Invalid Time %d %d", HrSt, HrEn );
        return 0; }

      if ( HrSt >= HrEn ) {
        sprintf (this->cr_ErrExt,"Bogus Start/End times %d %d", HrSt, HrEn );
        return 0; }

      i_Date = ICF::GetMCDate(mth,day,year);       /* Get psuedo julian date */
      i_Min = ICF::MilToMin(HrSt);
      i_BPMin = (i_Date * e_MinPerDay) + i_Min;   /* total time in minutes  */
      if ( i_BPMin < i_PreMin ) {
        sprintf (this->cr_ErrExt,"Burn Periods out of order %d %d", HrSt, HrEn );
        return 0; }
      i_PreMin = i_BPMin;

      if ( i_BPMin >= i_EndMin )
	  {
        sprintf (this->cr_ErrExt,"Burn period after end time");
        return 0;
	  }
      if ( i_BPMin < i_StrMin )
	  {
        sprintf (this->cr_ErrExt,"Burn period before start time");
        return 0;
	  }

      iN++;             /* Count Burn Periods inside Farsite Start End Times */


   } /* for i */

   if ( iN == 0 ) {
     sprintf (this->cr_ErrExt,"No burn periods within simulation time");
     return 0; }

   return 1;
}

/**************************************************************************
* This function is replaced by ChkBurnPeriods_New()
* This function wouldn't detect that the burn period was inside the
*       Farsite dates in the following type situtions
*       Farsite Start = Dec 20, FarSite End Jan 3
*       Burn Period  Jan 1
*  Mar 14 2012
***************************************************************************/
int ICF::ChkBurnPeriods(int *ai)
{
	*ai = e_EMS_FARSITE_BURN_PERIODS;            /* if no err occurs caller won't use           */
	if(this->iN_BurnPeriods <= 0)
 		return 1;                     //no burn periods

	//make sure valid months, days, years and at least one burn period is within simulation time
	int simStart = CheckMonthDay (this->i_FarsiteStartMth, this->i_FarsiteStartDay),
		simEnd = CheckMonthDay(this->i_FarsiteEndMth, this->i_FarsiteEndDay);
	int bpDay;
	bool willBurn = false;
	for(int i = 0; i < this->iN_BurnPeriods; i++)
	{
		//check month/day
		bpDay = this->CheckMonthDay(this->a_BurnPeriods[i].i_Month,this->a_BurnPeriods[i].i_Day);
		if(!bpDay)
		{
			sprintf (this->cr_ErrExt,"BP %d, Invalid Date %d/%d", i + 1,
				this->a_BurnPeriods[i].i_Month,  this->a_BurnPeriods[i].i_Day);
			return 0;
		}
		if(bpDay > simStart && bpDay < simEnd)
			willBurn = true;
		if(this->a_BurnPeriods[i].i_HourStart < 0 || this->a_BurnPeriods[i].i_HourStart > 2400)
		{
			sprintf (this->cr_ErrExt,"BP %d, Start Time %d (0-2400)", i + 1,
				this->a_BurnPeriods[i].i_HourStart);
			return 0;
		}
		if(this->a_BurnPeriods[i].i_HourEnd < 0 || this->a_BurnPeriods[i].i_HourEnd > 2400)
		{
			sprintf (this->cr_ErrExt,"BP %d, Start Time %d (0-2400)", i + 1,
				this->a_BurnPeriods[i].i_HourStart);
			return 0;
		}
		if(this->a_BurnPeriods[i].i_HourStart >= this->a_BurnPeriods[i].i_HourEnd)
		{
			sprintf (this->cr_ErrExt,"BP %d, End Time %d <= Start Time %d", i + 1,
				this->a_BurnPeriods[i].i_HourEnd,  this->a_BurnPeriods[i].i_HourStart);
			return 0;
		}
		if(!willBurn)
		{
			if(bpDay == simStart && this->a_BurnPeriods[i].i_HourEnd >= this->i_FarsiteStartHour)
				willBurn = true;
			if(bpDay == simEnd && this->a_BurnPeriods[i].i_HourStart <= this->i_FarsiteEndHour)
				willBurn = true;
		}
	}
	if(!willBurn)
	{
		sprintf (this->cr_ErrExt,"No burn periods within simulation time");
		return 0;
	}

	return 1;
}

int ICF::ChkFarsiteTimeStep(int *ai)
{
	if(this->f_FarsiteActualTimeStep <= 0 || this->f_FarsiteActualTimeStep > 360)
	{//timestep out of range, 1 - 360 minutes
		*ai = e_EMS_FARSITE_TIMESTEP;
		sprintf (this->cr_ErrExt,"Timestep %.0f (use 1->360)", this->f_FarsiteActualTimeStep );
		return 0;
	}
	return 1;
}


int ICF::ChkFarsiteResolutions(int *ai)
{
	if(this->f_FarsiteDistanceRes <= 0 || this->f_FarsiteDistanceRes > 500)
	{//Distance Res out of range, 1 - 360 m
		*ai = e_EMS_FARSITE_DIST_RES;
		sprintf (this->cr_ErrExt,"Distance Res %.0f (use 1->500)", this->f_FarsiteDistanceRes );
		return 0;
	}
	if(this->f_FarsitePerimeterRes <= 0 || this->f_FarsitePerimeterRes > 500)
	{//Distance Res out of range, 1 - 360 m
		*ai = e_EMS_FARSITE_PERIM_RES;
		sprintf (this->cr_ErrExt,"Perimeter Res %.0f (use 1->500)", this->f_FarsitePerimeterRes );
		return 0;
	}
	if(this->f_FarsitePerimeterRes <  this->f_FarsiteDistanceRes)
	{//Distance Res out of range, 1 - 360 m
		*ai = e_EMS_FARSITE_RESOLUTIONS;
		sprintf (this->cr_ErrExt,"Perimeter Res %.0f < Distance Res %.0f", this->f_FarsitePerimeterRes, this->f_FarsiteDistanceRes  );
		return 0;
	}
	return 1;
}

int ICF::ChkFarsiteSpotProb(int *ai)
{
	if(this->f_FarsiteSpotProb < 0.0 || this->f_FarsiteSpotProb > 1.0)
	{//spot prob out of range, 0.0 - 1.0 minutes
		*ai = e_EMS_FARSITE_SPOT;
		sprintf (this->cr_ErrExt,"Spot Probability %.3f (use 0->1.0)", this->f_FarsiteSpotProb );
		return 0;
	}
	return 1;
}

int ICF::ChkFarsiteSpotDelay(int *ai)
{
	if(this->f_FarsiteSpotIgnitionDelay < 0.0 || this->f_FarsiteSpotIgnitionDelay > 320.0)
	{//spot prob out of range, 0.0 - 1.0 minutes
		*ai = e_EMS_FARSITE_SPOT_IGNITION_DELAY;
		sprintf (this->cr_ErrExt,"Spot Ignition Delay %.1f (use 0->320.0)", this->f_FarsiteSpotIgnitionDelay );
		return 0;
	}
	return 1;
}


/**************************************************************************************/
int ICF::ChkFarsiteSpotGridResolution(int *ai)
{
/* Ok if switch wasn't found in inputs, we'll set a default later */
  if ( this->f_FarsiteSpotGridResolution == ei_ICFInit )
    return 1;

/* nothing specific about these limits, just make sure we don't get anything wild */
  if ( this->f_FarsiteSpotGridResolution > 5000.0 ){
    *ai = e_EMS_FARSITE_SPOT_GRID_RESOLUTION;
     sprintf (this->cr_ErrExt,"Spot Grid Resolution: %0.1f, use < 5000.0 ", this->f_FarsiteSpotGridResolution );
	 return 0;
	}


  return 1;
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Chk_FarsiteCondInputs
* Desc: Validate any and all of the inputs that will be used for doing dead fuel
*        moisiture conditioning and senting them to the Cond DLL
*  Out: ai....error number
*  Ret: 1 OK, 0 = error occured see error number - ai
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Chk_FarsiteCondInputs (int *ai)
{

   *ai = e_EMS_WeD2;

/* MUST have either Daily or RAWS weather data    */
/* Check this here because the following function won't */
  if ( this->a_Wtr == NULL && this->a_Wnd == NULL && this->a_RAWS == NULL ) {
     strcpy (this->cr_ErrExt, "No weather/wind data found");
     return 0;  }

/* -------------------------------------------------------------------------- */
/* If we have weather data we must have wind data */
   if ( this->a_Wtr != NULL && this->a_Wnd == NULL ) {
      strcpy (this->cr_ErrExt,"Have weather data but no wind data");
      return 0; }
   if ( this->a_Wnd != NULL && this->a_Wtr == NULL ) {
      strcpy (this->cr_ErrExt,"Have wind data but no weather data");
      return 0; }

/* if we have weather/wind data than we cannot also allow Weather Stream Data */
  if ( this->a_Wtr != NULL && this->a_RAWS != NULL ) {
    sprintf (this->cr_ErrExt,"Use only one type of weather stream input: %s or %s or %s", e_ICF_RAWS, e_ICF_WeF, e_ICF_WeD);
    return 0; }


/* Validate Wind and Weather Stream */
  if ( this->a_Wtr != NULL ) {        /* We have weather data */
    *ai = e_EMS_WeD1;
    if ( !ICF::ChkWeather (ai))       /* Check the Weather Data       */
      return 0;
    if ( !ICF::ChkWind (ai))          /* Check the Wind Stream data           */
      return 0;
    if ( !ICF::ChkWeaWinDay (ai))     /* See if Weather & Wind use same days */
      return 0; }

/* Validate RAWS Weather Stream Data */
   if ( this->a_RAWS != NULL ) {
     *ai = e_EMS_RAWS;
     if ( !Chk_RAWSData (ai) )
        return 0;  }

   *ai = 0;                    /* no error found */
   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Chk_FarsiteStartEnd
* Desc: Check the starting & end burn simulation date/times.
*       The date must be within the entered weather data table
*
*  Out: ai....error number
*  Ret: 1 = ok, 0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Chk_FarsiteStartEnd (int *ai)
{
int str, end, diff;
   str = Chk_FarsiteDate ((char *)"Start", ai);     /* Check Start Date */
   if ( str == 0 ) {
     *ai = e_EMS_FARSITE_START;
     return 0; }

   end = Chk_FarsiteDate ((char *)"End ", ai);      /* Check End date */
   if ( end == 0 ) {
     *ai = e_EMS_FARSITE_END;
     return 0; }

/* Take diff in minutes and check against and arbitrary small number of minutes */
   diff = end - str;
   if ( diff < 10 ) {
     *ai = e_EMS_FARSITE_STARTEND;
     strcpy (this->cr_ErrExt, "Start/End dates are equal, reversed or to close");
     return 0; }

   *ai = 0;
   return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Chk_FarsiteDate
* Desc: Check a Start or End date that has come from
*        the inputs file switch.
*       The date must be within the entered weather data table
* Note-1: Using RAWS data the weather conditioning can run going right up
*          up to very end of the weather data BUT Farsite (for whatever reason)
*          when it runs goes past the Farsite Ending date by the number of
*          minutes for the 'actual' time increment, so we need to make sure
*          the there is that much of a buffer of weather data on the end past
*          the Farsite ending date/time
*  Out: ai....error number
*  Ret: 1 = ok, 0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Chk_FarsiteDate (char cr_Mode[], int *ai)
{
int i, cy,cm,cd,ch, wm,wd,wy,wt, i_FarDate, m;
int iS, i_FarTotMin, i_WtrDate, i_WtrTotMin ;

  *ai = e_EMS_CPE;            /* if no err occurs caller won't use           */

  if ( !strcmp (cr_Mode,"Start") ){  /* Get Start Conditioning Date */
    cy = this->i_FarsiteStartYear;
    cm = this->i_FarsiteStartMth;
    cd = this->i_FarsiteStartDay;
    ch = this->i_FarsiteStartHour;
    iS = 0; }                         /* Get 1st entry in Weather data */
  else {                              /* Get Ending Conditionng date */
    cy = this->i_FarsiteEndYear;
    cm = this->i_FarsiteEndMth;
    cd = this->i_FarsiteEndDay;
    ch = this->i_FarsiteEndHour;
    iS = -1; }                        /* get last entry in Weather data */

  if ( cm == ei_ICFInit ) {           /* If month is missing    */
    sprintf (this->cr_ErrExt, "Farsite %s date is missing", cr_Mode);
    return 0; }

  if ( !_ChkMthDay (cm, cd,this->cr_ErrExt)) {
     sprintf (this->cr_ErrExt,"Invalid %s date %d %d", cr_Mode, cm, cd );
     return 0; }

  if ( ch < 0 || ch > 2359 ) {
    sprintf (this->cr_ErrExt,"Bad hour of day specified %d ( use 0->2359 )", ch);
    return 0; }

/* Form Date (start/end) in total minutes... */
  i_FarDate = ICF::GetMCDate(cm,cd,cy);         /* psuedo Julian Moist Cond date */
  m = ICF::MilToMin(ch);                        /* Military time to minute */
  i_FarTotMin = (i_FarDate * e_MinPerDay) + m;  /* Cond Data in Total Minutes */

  ICF::GetWthDate(iS,&wm,&wd,&wy,&wt);           /* Get 1st or last rec in weathr tbl */
  i_WtrDate = ICF::GetMCDate(wm,wd,wy);          /* Get psuedo julian date */
  m = ICF::MilToMin(wt);
  i_WtrTotMin = (i_WtrDate * e_MinPerDay) + m;   /* total time in minutes  */

/*----------------------------------------------------------------------------------*/
/* Check Start date against the weather data we have to work with */
  if ( !strcmp (cr_Mode,"Start") ){
    if ( this->iN_RAWS > 0 ) {            /* are we working with RAWS data */
      if ( i_FarTotMin < i_WtrTotMin ) {
        strcpy (this->cr_ErrExt," Start Date is less than starting RAWS data");
        return 0; } }

/* working with Daily weather data, so Farsite starting date has to be at least */
/*  what the start of the condition date that will get used, or more actually */
    else {
      i = i_WtrTotMin + e_MinPerDay;
      if ( (i_FarTotMin ) < i ) {
        strcpy (this->cr_ErrExt,"Start Date is NOT within usable range of daily weather data");
        return 0; } }

    return i_FarTotMin;
  }


/*------------------------------------------------------------------------------*/
/* For End Conditioning Date */

  if ( this->iN_RAWS > 0 ) {              /* if RAWS data - see Note-1 above*/
    i = i_WtrTotMin - (int) this->f_FarsiteActualTimeStep;
    if ( i_FarTotMin > i ) {
      sprintf (this->cr_ErrExt,"Ending Date to close or past end of RAWS data (%4.0f minute buffer required)",this->f_FarsiteActualTimeStep);
      return 0; }  }

/* When using Daily weather data, conditioning needs time buffer on end  */
/* i_WtrTotMin represents time at begin of day but weather data is for entire day */
/* as long as i_FarTotMin is before it, it's ok */
  else {
     if ( i_FarTotMin > i_WtrTotMin ) {
      strcpy (this->cr_ErrExt,"Ending Date out of usable range");
      return 0; } }

  return i_FarTotMin;
}




/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkCustomFuels     Check Custom Fuels
* Desc: If have custom fuels, make sure parameters are all valid
*  Out: ai....error number
*  Ret: 1 ok, else 0
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::ChkCustomFuels(int *ai)
{
	if(this->a_customFuels == NULL || this->iN_CustomFuels == 0)
		return 1;
	for(int f = 0; f < iN_CustomFuels; f++)
	{
		if(a_customFuels[f].i_Model < 14 || a_customFuels[f].i_Model > 256)
		{
			*ai = e_EMS_CF_FM;
			return 0;
		}
		if(a_customFuels[f].f_xmext <= 0.0)
		{
			*ai = e_EMS_CF_EM;
			return 0;
		}
		if(a_customFuels[f].f_h1 + a_customFuels[f].f_h10 + a_customFuels[f].f_h100 <= 0.0)
		{
			*ai = e_EMS_CF_DF;
			return 0;
		}
		if(a_customFuels[f].f_depth <= 0.0)
		{
			*ai = e_EMS_CF_DEPTH;
			return 0;
		}
		if(a_customFuels[f].f_heatl <= 6000.0)
		{
			*ai = e_EMS_CF_LH;
			return 0;
		}
		if(a_customFuels[f].f_heatd <= 4000.0)
		{
			*ai = e_EMS_CF_DH;
			return 0;
		}
		if(a_customFuels[f].f_sl > 4000.0 || a_customFuels[f].f_slh > 4000.0 || a_customFuels[f].f_slw > 4000.0)
		{
			*ai = e_EMS_CF_SAV;
			return 0;
		}
	}
	return 1;
}


// Farsite-WindNinja
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ChkGridWinds_Farsite
* Desc: Check the limits on the values readin for ththe Farsite Gridded Winds
*       Just doing some general limit checking here, other checks were done
*        when the data got read in from the inputs file.
*  Out: *ai....error code
*  Ret: 1 = ok, 0 error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::FarsiteWindNinja_Check (int *ai)
{

  if ( !strcmp (this->s_FWNI.cr_GriWin,e_GW_No))
    return 1;               /* Not doing gridded winds */

/* Resolution .....................                                     */
	 if ( this->s_FWNI.f_GriWinRes < e_GWH_Min || this->s_FWNI.f_GriWinRes > e_GWH_Max ) {
		   sprintf (this->cr_ErrExt, "Gridded wind resolution %4.1f is out of range ( %4.1f -> %4.1f )", this->f_GriWinRes, e_GWR_Min,e_GWR_Max);
     *ai = e_EMS_GWR;
	    	return 0; }

/* Wind Height................                                         */
    if ( this->s_FWNI.f_GriWinHei < e_GWH_Min || this->s_FWNI.f_GriWinHei > e_GWH_Max ){
	      sprintf (this->cr_ErrExt, "Gridded wind height %6.1f is out of range ( %4.0f -> %4.0f )", this->f_GriWinHei,e_GWH_Min, e_GWH_Max);
       *ai = e_EMS_GWH;
	      return 0; }

/* Time Zone */
    if ( this->s_FWNI.i_TimeZone < -12 || this->s_FWNI.i_TimeZone > 12 ) {
       sprintf (this->cr_ErrExt, "Out out limits, use -12 --> 12");
       *ai = e_EMS_FGWT;
	      return 0; }

    if ( this->s_FWNI.f_Longitude < -180.0 || this->s_FWNI.f_Longitude > 180.0 ) {
       sprintf (this->cr_ErrExt, "Out out limits, use -180 --> 180");
       *ai = e_EMS_FGWL;
	      return 0; }


/* Check the Ninja grid Bining values */
    if ( this->s_FWNI.f_BinWinDir < 0 || this->s_FWNI.f_BinWinDir > 180 ) {
       sprintf (this->cr_ErrExt, "Use 0 --> 180");
       *ai = e_EMS_FGWBD;
       return 0; }

    if ( this->s_FWNI.f_BinWinSpd < 0 || this->s_FWNI.f_BinWinSpd > 100 ) {
       sprintf (this->cr_ErrExt, "Use 0 --> 100");
       *ai = e_EMS_FGWBS;
       return 0; }

  return 1;
}


