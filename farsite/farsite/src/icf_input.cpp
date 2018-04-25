/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: icf_inp.cpp
* Desc: Input Command File Input functions
*       Functions mainly read the data from the cmd file and do
*       some minimum checks, like verify valid numbers from agrs, proper
*       number of agrs for switch.
* Date: 1-4-08
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
// #include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <io.h>
// #define _CRT_SECURE_NO_DEPRECATE
#include "icf_def.h"
#include <time.h>

extern d_EMS sr_EMS[];          /* Error Message Struct table */
void CelsToFahr (float *af);

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Input
* Desc: Read in the input file's specific data that is used by Farsite
*         or FlamMap.
*       The FlamMap and Farsite input files both have some of the same
*        switches.
*   In: cr_PthFN....path and file name.
*  Ret: 1 = no error, else error number, see error message defines and table
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Input (char cr_PthFN[])
{
int i;

  ICF::DeleteAll ();                         /* delete/free up any 'new'memory    */
  ICF::Init ();                              /* Init all class variables          */

  this->fh = fopen (cr_PthFN,"r");
  if ( this->fh == NULL ){
	  sprintf (this->cr_ErrExt,"%s\n",cr_PthFN);
	 return e_EMS_Fil;}

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Fuel Moistures Files Section                                              */

  if ( !ICF::FuelMoistData (&i))             /* FUEL_MOISTURES_DATA:         */
    return ICF::CloseRet(i);                 /*  it's an error             */

  if ( !ICF::Set_FilNam ((char *)e_ICF_FMF, (char *)this->cr_FMF)) /* FUEL_MOISTURE_FILE    */
    return ICF::CloseRet(e_EMS_Fil);

   if (!ICF::Set_FilNam ((char *)e_ICF_CFF, this->cr_CFF))  /* CUSTOM_FUELS_FILE     */
    return ICF::CloseRet(e_EMS_Fil);

  if (!ICF::CustomFuelData(&i))
    return ICF::CloseRet(i);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Newer RAWS (Remote Automated Weather Stations) data -                     */
   if ( !ICF::RAWS_WeatherData (&i) )     /* Data embedded in inputs file */
     return ICF::CloseRet (i);

   if ( !ICF::RAWS_WeatherDataFile(&i) )  /* Data in separate data file */
     return ICF::CloseRet (i);

   //ros adjustment
  if (!ICF::Set_FilNam ((char *)e_ICF_ROS_ADJUST_FILE, this->cr_ROSAdjustFile))  /* ROS_ADJUST_FILE     */
    return ICF::CloseRet(e_EMS_Fil);

  /*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Weather Stream Data, can be in embeded in cmd file or .wtr file, not both */
  if ( !ICF::WeatherDataFile (&i) )           /* WEATHER_FILE:               */
    return ICF::CloseRet(i);

  if ( !ICF::WeatherDataCmd (&i))             /* WEATHER_DATA               */
      return ICF::CloseRet(i);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Wind Stream Data, can be in embeded in cmd file or .wnd file, not both    */
  if ( !ICF::WindDataFile (&i) )              /* WIND_FILE:                  */
    return ICF::CloseRet(i);

  if ( !ICF::WindData  (&i) )                      /* WIND_DATA              */
     return ICF::CloseRet(i);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Canopy Characteristics Section                                            */
  if ( !ICF::Set_SinIntArg((char *)e_ICF_FMC,&this->i_FolMoi))/* FOLIAR_MOISTURE_CONTENT */
    return ICF::CloseRet(e_EMS_FMC);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Fire Behavior - Tab/Section                                               */
/* Crown Fire Method...Finney, Reinhardt                                     */
  ICF::Set_SinTxtArg((char *)e_ICF_CFM,this->cr_CroFirMet,eC_CFM); /* CROWN_FIRE_METHOD */

/* Analysis Area parameters section         - To run subset of an landscape  */
  if (1 == ICF::Set_SinNumArg ((char *)e_ICF_AREA_EAST, &this->f_analysisEast)) /* ANALYSIS_AREA_EAST        */
	   this->b_eastSet = true;
  if (1 == ICF::Set_SinNumArg ((char *)e_ICF_AREA_WEST, &this->f_analysisWest)) /* ANALYSIS_AREA_WEST        */
	   this->b_westSet = true;
  if (1 == ICF::Set_SinNumArg ((char *)e_ICF_AREA_NORTH, &this->f_analysisNorth)) /* ANALYSIS_AREA_NORTH        */
	   this->b_northSet = true;
  if (1 == ICF::Set_SinNumArg ((char *)e_ICF_AREA_SOUTH, &this->f_analysisSouth)) /* ANALYSIS_AREA_SOUTH        */
	   this->b_southSet = true;

	if(!ICF::Set_SinIntArg ((char *)e_ICF_SPOTTINGSEED, &this->i_SpottingSeed))
		return ICF::CloseRet(e_EMS_SpottingSeed);
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: InputFarsite
* Desc: Input Command File - open, read in and do some basic checking
*
*   In: cr_PthFN....path and file name.
*  Ret: 1 = no error, else error number, see error message defines and table
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::InputFarsite (char cr_PthFN[])
{
int i;

  i = this->Input(cr_PthFN);                /* read in Moist, Weathr, Wind, etc */
  if ( i != 1 )                             /*  if error  */
    return i;

// Farsite-WindNinja .................................................
  i = this->FarsiteWindNinja ();
  if ( i != 1 )
    return i;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Farsite start & end simulation dates                                 */
   if ( !ICF::Set_Date ((char *)e_ICF_FARSITE_END_TIME))
     return ICF::CloseRet(e_EMS_FARSITE_END);

   if ( !ICF::Set_Date ((char *)e_ICF_FARSITE_START_TIME))
     return ICF::CloseRet(e_EMS_FARSITE_START);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Actual Time Step                                            */
    if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_TIMESTEP, &this->f_FarsiteActualTimeStep))
	    	return ICF::CloseRet(e_EMS_FARSITE_TIMESTEP);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Visible Time Step                                            */
// if (!ICF::Set_SinNumArg (e_ICF_FARSITE_SECONDARY_VISIBLESTEP, &this->f_FarsiteSecondaryVisibleTimeStep))
//	  return ICF::CloseRet(e_EMS_FARSITE_SECONDARYVISIBLETIMESTEP);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Secondary Visible Time Step                                            */
// if (!ICF::Set_SinNumArg (e_ICF_FARSITE_VISIBLESTEP, &this->f_FarsiteVisibleTimeStep))
//	  return ICF::CloseRet(e_EMS_FARSITE_VISIBLETIMESTEP);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Distance Res                                            */
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_DIST_RES, &this->f_FarsiteDistanceRes))
		return ICF::CloseRet(e_EMS_FARSITE_DIST_RES);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Perimeter Res                                            */
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_PERIM_RES, &this->f_FarsitePerimeterRes))
		return ICF::CloseRet(e_EMS_FARSITE_PERIM_RES);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Spotting Burn Grid Resolution */
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_SPOT_GRID_RESOLUTION, &this->f_FarsiteSpotGridResolution))
		return ICF::CloseRet(e_EMS_FARSITE_SPOT_GRID_RESOLUTION);
//minimum ignition vertex distance

	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_MIN_IGNITION_VERTEX_DISTANCE, &this->f_FarsiteMinIgnitionVertexDistance))
		return ICF::CloseRet(e_EMS_FARSITE_MIN_IGNITION_VERTEX_DISTANCE);
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Spotting Ignition delay                                            */
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_SPOT_IGNITION_DELAY, &this->f_FarsiteSpotIgnitionDelay))
		return ICF::CloseRet(e_EMS_FARSITE_SPOT_IGNITION_DELAY);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Spotting probability                                            */
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_SPOT, &this->f_FarsiteSpotProb))
		return ICF::CloseRet(e_EMS_FARSITE_SPOT);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Acceleration                                            */
	if (!ICF::Set_SinIntArg ((char *)e_ICF_FARSITE_ACCELERATION_ON, &this->i_FarsiteAccelerationOn))
		return ICF::CloseRet(e_EMS_FARSITE_ACCELERATION_ON);

	/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Distance Res                                            */
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FARSITE_MINIMUM_SPOT_DISTANCE, &this->f_FarsiteMinSpotDistance))
		return ICF::CloseRet(e_EMS_FARSITE_DIST_RES);


// Burn Periods
	if ( !ICF::BurnPeriodData (&i) )                      /* BurnPeriodData              */
     return ICF::CloseRet(i);


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
  if (!ICF::Set_FilNam ((char *)e_ICF_FARSITE_IGNITION, this->cr_FarsiteIgnition))            /* FARSITE_IGNITION_FILE  */
    return ICF::CloseRet(e_EMS_FARSITE_IGNITION);

  if (!ICF::Set_FilNam ((char *)e_ICF_FARSITE_BARRIERFILENAME, this->cr_FarsiteBarrier))      /* FARSITE_BARRIER_FILE  */
    return ICF::CloseRet(e_EMS_FARSITE_BARRIER);

  if ( !ICF::Set_SinIntArg ((char *)e_ICF_FARSITE_FILL_BARRIER, &this->i_FarsiteFillBarriers)) /* FARSITE_FILL_BARRIERS      */
    return ICF::CloseRet(e_EMS_ST);

  return ICF::CloseRet(1);              /* Close file and ret ok code.       */

}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: InputFlamMap
* Desc: Input Command File - open, read in and do some basic checking
*
*   In: cr_PthFN....path and file name.
*  Ret: 1 = no error, else error number, see error message defines and table
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::InputFlamMap (char cr_PthFN[])
{
int i;

  i = this->Input(cr_PthFN);                /* read in Moist, Weathr, Wind, etc */
  if ( i != 1 )                             /*  if error  */
    return i;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Get the Conditioning Period End & Start dates  */
/* Change 10-7-10, switch to use the generic Set_Date() */
  if ( !ICF::Set_Date((char *)e_ICF_CPE))
    return ICF::CloseRet(e_EMS_CPE);

  ICF::Set_ConPerStartDef ();       /* set default, incase switch not found      */
  if ( !ICF::Set_Date((char *)e_ICF_CPS))
     return ICF::CloseRet(e_EMS_CPS);

/* Old   */
//   if ( !ICF::Set_ConPerEnd(e_ICF_CPE))                  /* CONDITIONING_PERIOD_END   */
//     return ICF::CloseRet(e_EMS_CPE);
//   if ( !ICF::Set_ConPerStart(e_ICF_CPS))                /* CONDITIONING_PERIOD_START   */
//     return ICF::CloseRet(e_EMS_CPS);


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Wind Section                                                              */
  if ( !ICF::Set_SinNumArg ((char *)e_ICF_WD, &this->f_WinDir)) /* WIND_DIRECTION    */
    return ICF::CloseRet(e_EMS_WDA);

  if (!ICF::Set_SinNumArg ((char *)e_ICF_WS, &this->f_WinSpe)) /* WIND_SPEED        */
    return ICF::CloseRet(e_EMS_WSA);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/

  if ( !ICF::Set_SinIntArg ((char *)e_ICF_NP ,&this->i_NumPro)) /* NUMBER_PROCESSORS */
    return ICF::CloseRet(e_EMS_NP );

/* These are Optional but exclusive if used, well check them later */
  if ( !ICF::Set_SinIntArg ((char *)e_ICF_SDN,&this->i_SDFN)) /* SPREAD_DIRECTION_FROM_NORTH  */
    return ICF::CloseRet(e_EMS_SDN);
  if ( !ICF::Set_SinIntArg ((char *)e_ICF_SDM,&this->i_SDFM)) /* SPREAD_DIRECTION_FROM_MAX  */
    return ICF::CloseRet(e_EMS_SDN);


  ICF::Set_SinIntArg((char *)e_ICF_USE_MEM_LCP, &this->i_useMemoryLCP); /* USE_MEMORY_LCP */
  ICF::Set_SinIntArg((char *)e_ICF_USE_MEM_OUTPUTS, &this->i_useMemoryOutputs); /* USE_MEMORY_OUTPUTS */
	 ICF::Set_FilNam ((char *)e_ICF_TEMP_STORAGE_PATH, this->cr_storagePath);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Gridded Winds                                                             */
  ICF::Set_SinTxtArg((char *)e_ICF_GGW,this->cr_GGW,eC_GGW);      /* GRIDDED_WINDS_GENERATE */

  if ( !ICF::Set_SinNumArg((char *)e_ICF_GWR,&this->f_GriWinRes)) /* GRIDDED_WINDS_RESOLUTION */
    return ICF::CloseRet(e_EMS_GWR);

  if ( !ICF::Set_SinNumArg((char *)e_ICF_GWH,&this->f_GriWinHei)) /* GRIDDED_WINDS_HEIGHT */
    return ICF::CloseRet(e_EMS_GWH);

 if ( !ICF::GridWindDiurnal (&this->s_WDI) )   /* Diurnal Ninja inputs */
    return ICF::CloseRet(e_EMS_GWD);

/* Gridded Winds Files.........................................................*/
 if (!ICF::Set_FilNam ((char *)e_ICF_GWDF, this->cr_GWDF))   /* GRIDDED_WIND_DIRECTION_FILE   */
    return ICF::CloseRet(e_EMS_Fil);

  if (!ICF::Set_FilNam ((char *)e_ICF_GWSF, this->cr_GWSF))   /* WIND_SPEED_GRID_FILE       */
    return ICF::CloseRet(e_EMS_Fil);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* MTT Minimum Travel Time.........                                          */
  if ( !ICF::Set_SinNumArg ((char *)e_ICF_RES,&this->f_Res))    /* MTT_RESOLUTION    */
    return ICF::CloseRet(e_EMS_RES);
  if ( !ICF::Set_SinIntArg ((char *)e_ICF_ST, &this->i_SimTim)) /* MTT_SIM_TIME      */
    return ICF::CloseRet(e_EMS_ST);
  if ( !ICF::Set_SinIntArg ((char *)e_ICF_TPI,&this->i_TraPth)) /* MTT_TRAVEL_PATH_INTERVAL */
    return ICF::CloseRet(e_EMS_TPI);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
  if (!ICF::Set_FilNam ((char *)e_ICF_IFN, this->cr_IFN))                               /* MTT_IGNITION_FILE  */
    return ICF::CloseRet(e_EMS_Fil);

  if (!ICF::Set_FilNam ((char *)e_ICF_MTTBARRIERFILENAME, this->cr_BarrierFileName))    /* MTT_BARRIER_FILE  */
    return ICF::CloseRet(e_EMS_Fil);

 if ( !ICF::Set_SinIntArg ((char *)e_ICF_MTTFILLBARRIERS, &this->i_MTT_FillBarriers))   /* MTT_FILL_BARRIERS      */
    return ICF::CloseRet(e_EMS_ST);


 if (!ICF::Set_SinNumArg ((char *)e_ICF_MTTSPOT, &this->f_MTT_SpotProbability))         /* MTT_SPOT_PROBABILITY  */
		  return ICF::CloseRet(e_EMS_Fil);

   //NumLat
  Set_SinIntArg ((char *)e_ICF_NODESPREAD_NUMLAT, &this->i_NodeSpread_NumLat);//) /* NodeSpreadNumLat    */

  //NumVert
  Set_SinIntArg ((char *)e_ICF_NODESPREAD_NUMVERT, &this->i_NodeSpread_NumVert);//) /* NodeSpreadNumVert    */

 //TOM ADDITIONS
  if (!ICF::Set_FilNam ((char *)e_ICF_TOM_IDEAL_LCP, this->iTrt_idealLcpName))                               /* MTT_IGNITION_FILE  */
    return ICF::CloseRet(e_EMS_TOM_IDEAL_LCP);

  if (!ICF::Set_FilNam ((char *)e_ICF_TOM_IGNITION, this->iTrt_ignitionFileName))                               /* MTT_IGNITION_FILE  */
    return ICF::CloseRet(e_EMS_TOM_IGNITION);

 if ( !ICF::Set_SinIntArg ((char *)e_ICF_TOM_ITERATIONS, &this->iTrt_iterations))   /* MTT_FILL_BARRIERS      */
    return ICF::CloseRet(e_EMS_TOM_ITERATIONS);

 if (!ICF::Set_SinNumArg ((char *)e_ICF_TOM_DIMENSION, &this->iTrt_dimension))         /* MTT_SPOT_PROBABILITY  */
		  return ICF::CloseRet(e_EMS_TOM_DIMENSION);

  if (!ICF::Set_SinNumArg ((char *)e_ICF_TOM_FRACTION, &this->iTrt_fraction))         /* MTT_SPOT_PROBABILITY  */
		  return ICF::CloseRet(e_EMS_TOM_FRACTION);

  if (!ICF::Set_SinNumArg ((char *)e_ICF_TOM_RESOLUTION, &this->iTrt_resolution))         /* MTT_SPOT_PROBABILITY  */
		  return ICF::CloseRet(e_EMS_TOM_RESOLUTION);

  if ( !ICF::Set_SinIntArg ((char *)e_ICF_TOM_OPPORTUNITES_ONLY, &this->iTrtOpportunitiesOnly))   /* Only want treat opportunities?      */
    return ICF::CloseRet(e_EMS_TOM_OPPORTUNITIES_ONLY);


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Output Grid Switches                                                      */
  ICF::Set_OGS ((char *)e_ICF_FL,  &this->i_FL  );   /*  FLAMELENGTH     */
  ICF::Set_OGS ((char *)e_ICF_SP,  &this->i_SP  );   /*  SPREADRATE      */
  ICF::Set_OGS ((char *)e_ICF_IN,  &this->i_IN  );   /*  INTENSITY       */
  ICF::Set_OGS ((char *)e_ICF_HE,  &this->i_HE  );   /*  HEATAREA        */
  ICF::Set_OGS ((char *)e_ICF_CR,  &this->i_CR  );   /*  CROWNSTATE      */
  ICF::Set_OGS ((char *)e_ICF_SO,  &this->i_SO  );   /*  SOLARRADIATION  */
  ICF::Set_OGS ((char *)e_ICF_FU1, &this->i_FU1 );   /*  FUELMOISTURE1   */
  ICF::Set_OGS ((char *)e_ICF_FU10,&this->i_FU10);   /*  FUELMOISTURE10  */
  ICF::Set_OGS ((char *)e_ICF_MI,  &this->i_MI  );   /*  MIDFLAME        */
  ICF::Set_OGS ((char *)e_ICF_HO,  &this->i_HO  );   /*  HORIZRATE       */
  ICF::Set_OGS ((char *)e_ICF_MAD, &this->i_MAD );   /*  MAXSPREADDIR    */
  ICF::Set_OGS ((char *)e_ICF_ELA, &this->i_ELA );   /*  ELLIPSEDIM_A    */
  ICF::Set_OGS ((char *)e_ICF_ELB, &this->i_ELB );   /*  ELLIPSEDIM_B    */
  ICF::Set_OGS ((char *)e_ICF_ELC, &this->i_ELC );   /*  ELLIPSEDIM_C    */
  ICF::Set_OGS ((char *)e_ICF_MAS, &this->i_MAS );   /*  MAXSPOT         */
  ICF::Set_OGS ((char *)e_ICF_MASDIR,&this->i_MASDir); /* MAXSPOT_DIR   */
  ICF::Set_OGS ((char *)e_ICF_MASDX, &this->i_MASDx);  /* MAXSPOT_DX      */
  ICF::Set_OGS ((char *)e_ICF_FU100, &this->i_FU100);  /* FUELMOISTURE100  */
  ICF::Set_OGS ((char *)(char *)e_ICF_FU1k, &this->i_FU1k);    /* FUELMOISTURE1000 */
  ICF::Set_OGS ((char *)e_ICF_WDG, &this->i_WDG);      /* WINDDIRGRID */
  ICF::Set_OGS ((char *)e_ICF_WSG, &this->i_WSG);      /* WINDSPEEDGRID */

  return ICF::CloseRet(1);              /* Close file and ret ok code.       */
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GridWindDiurnal
* Desc: Read in the gridded wind diurnal data that WindNinja will use
*       This data is optional but if switch GRIDDED_WINDS_DIURNAL is
*       "YES" than related data is required
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GridWindDiurnal (d_WDI *a_WDI)
{
int i;
char A[100], B[100], C[100], D[100];

   if ( !ICF::Set_SinTxtArg((char *)e_ICF_GWD,a_WDI->cr_YesNo,eC_GGW))
     return 1;         /* If switch not found - that's ok */

   if ( !strcmp (a_WDI->cr_YesNo,e_GW_No) )
     return 1;         /* Input set to "No" - not doing diurnal */
   if ( strcmp (a_WDI->cr_YesNo,e_GW_Yes) ) {
     strcpy (this->cr_ErrExt,this->cr_Line);
     return 0; }        /* if switch arg not "Yes" it's an error */

/*........................................................................*/
/* Get Month Day Year   */
   if ( !ICF::GetSw ((char *)e_ICF_GWDD)){    /* Find the Switch in input file     */
    sprintf (this->cr_ErrExt,"Can't find switch %s", e_ICF_GWDD);
    return 0; }                      /* if not found its error */

   A[0] = B[0] = C[0] = 0;
   sscanf (&this->cr_Line[this->iX],"%10s %10s %10s",A,B,C); /* mth day yr */

/* Make sure mth day year are at least valid numbers for now               */
   if ( Get_NumTyp (A) != 'I' || Get_NumTyp (B) != 'I' || Get_NumTyp (C) != 'I' ) {
     strcpy (this->cr_ErrExt,this->cr_Line);
     return 0; }

   a_WDI->i_Mth =  atoi(A);
   a_WDI->i_Day =  atoi(B);
   a_WDI->i_Year = atoi(C);

/*........................................................................*/
/* Get Sec Min Hour Time Zone  */
   if ( !ICF::GetSw ((char *)e_ICF_GWDT)){   /* Find the Switch in input file     */
     sprintf (this->cr_ErrExt,"Can't find switch %s", e_ICF_GWDT);
     return 0;}                        /* if not found its error */

   A[0] = B[0] = C[0] = D[0] = 0;
   sscanf (&this->cr_Line[this->iX],"%10s %10s %10s %10s",A,B,C,D); /* sec, min hr zone */

/* Make sure mth day year are at least valid numbers for now               */
   if ( Get_NumTyp (A) != 'I' || Get_NumTyp (B) != 'I' ||
        Get_NumTyp (C) != 'I' || Get_NumTyp (D) != 'I' ) {
      strcpy (this->cr_ErrExt,this->cr_Line);
      return 0; }

    a_WDI->i_Sec = atoi (A);
    a_WDI->i_Min = atoi (B);
    a_WDI->i_Hour = atoi (C);
    a_WDI->i_TimeZone = atoi (D);

/*..........................................................................*/
/* Get Air temp */
  i = ICF::Set_SinNumArg ((char *)e_ICF_GWAT, &a_WDI->f_AirTemp);
  if ( i == -1 )  {
    sprintf (this->cr_ErrExt,"Can't find switch %s", e_ICF_GWAT);
    return 0; }
  if ( i == 0 ) {   /* Invalid number format */
    strcpy (this->cr_ErrExt,this->cr_Line);
    return 0; }

/*..........................................................................*/
/* Get Cloud Cover */
  i = ICF::Set_SinNumArg ((char *)e_ICF_GWCC, &a_WDI->f_CloudCov);
  if ( i == -1 )  {
    sprintf (this->cr_ErrExt,"Can't find switch %s", e_ICF_GWCC);
    return 0; }
  if ( i == 0 ) {   /* Invalid number format */
    strcpy (this->cr_ErrExt,this->cr_Line);
    return 0; }

/* Get Longitude, note: we'll get latitude from the lcp file */
  i = ICF::Set_SinNumArg ((char *)e_ICF_GWLO, &a_WDI->f_Longitude);
  if ( i == -1 )  {
    sprintf (this->cr_ErrExt,"Can't find switch %s", e_ICF_GWLO);
    return 0; }
  if ( i == 0 ) {   /* Invalid number format */
    strcpy (this->cr_ErrExt,this->cr_Line);
    return 0; }

  return 1;
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: FuelMoistData
* Desc: Find the Fuel Moisture Model Data switch and if found load its data.
*       Checks for valid number values, and for the right amount of
*       data, doesn't check limits.
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::FuelMoistData (int *ai)
{
int i,j, i_Arg;

#define e_ffr 6
float fr[e_ffr];

   strcpy (this->cr_ErrExt,"");

/* find the switch and its arg......                                         */
   i = ICF::Set_SinIntArg ((char *)e_ICF_FMD ,&i_Arg); /* FUEL_MOISTURES_DATA:       */
   if ( i == -1)                        /* either/or switch so not an error  */
     return 1;                          /*  at this point, will chk mor latr */

  if ( i == 0 ) {                       /* Switch found with bad argument    */
    strcpy (this->cr_ErrExt,"Invalid/Missing Switch argument");
    *ai = e_EMS_FMD;
    return 0;}

/* Check the switch argument....                                             */
  if ( i_Arg < 1 || i_Arg > e_MaxFMD){   /* Arg is num lines of weather data  */
    sprintf (this->cr_ErrExt,"Invalid argument value: %d", i_Arg);
    *ai = e_EMS_FMD;
    return 0; }

  this->a_FMD = new d_FMD[i_Arg];         /* allocate weather sturts           */
  this->iN_FMD = i_Arg;                   /* number of                         */

/*.......................................................................... */
  for ( i = 0; i < i_Arg; i++ ) {     /* Read each line's data into fr      */
     j = ICF::GetLine (fr,e_ffr, this->cr_ErrExt, this->fh);
     if ( j  == 0 ) {                 /* If error                           */
       *ai = e_EMS_FMD;
       return 0; }

/* copy data into class                                                      */
     this->a_FMD[i].i_Model = (int) fr[0];
     this->a_FMD[i].i_TL1  =  (int) fr[1];
     this->a_FMD[i].i_TL10 =  (int) fr[2];
     this->a_FMD[i].i_TL100 = (int) fr[3];
     this->a_FMD[i].i_TLLH =  (int) fr[4];
     this->a_FMD[i].i_TLLW =  (int) fr[5];
  } /* for i */

/* Try to get one extra line, shouldn't get one, number of lines needs       */
/*  specified by the switch argument                                         */
    j = ICF::GetLine (fr,e_ffr, this->cr_ErrExt, this->fh);
    if ( j == 1 ){
      sprintf(this->cr_ErrExt,"Extra line of data found, switch specifies %d", i_Arg);
     *ai = e_EMS_FMD;
      return 0; }
    strcpy (this->cr_ErrExt,"");  /* not actual error, so blank this out */

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: WindData
* Desc: Find the Wind Data switch and if found load its data.
*       Checks for valid number values, and for the right amount of
*       data, doesn't check limits.
* Note-1: We don't deal with any leap year days.
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::WindData  (int *ai)
{
int i,j, i_Arg, i_Mth, i_Day, iN_Recs;

#define e_wifr 6
float fr[e_wifr];

   strcpy (this->cr_ErrExt,"");

/* find the switch and its arg......                                         */
   i = ICF::Set_SinIntArg ((char *)e_ICF_WiD,&i_Arg); /* WIND_DATA:                  */
   if ( i == -1 )                        /* Optional switch so not an error   */
     return 1;                          /*  if none found                    */

  if ( i == 0 ) {                       /* Switch found with bad argument    */
    strcpy (this->cr_ErrExt,"Invalid/Missing Switch argument");
    *ai = e_EMS_WiD1;
    return 0;}

/* Check the switch argument....                                             */
  if ( i_Arg < 1 || i_Arg > e_MaxWiD){   /* Arg is num lines of wind data  */
    sprintf (this->cr_ErrExt,"Invalid argument value: %d, Limit: %d", i_Arg,e_MaxWiD);
    *ai = e_EMS_WiD1;
    return 0; }

  this->a_Wnd = new d_Wnd[i_Arg];         /* allocate wind structs          */

/*.......................................................................... */
  iN_Recs = 0;
  for ( i = 0; i < i_Arg; i++ ) {     /* Read each line's data into fr      */
     j = ICF::GetLine (fr,e_wifr, this->cr_ErrExt, this->fh);
     if ( j  == 0 ) {                 /* If error                           */
       *ai = e_EMS_WiD1;
       return 0; }

    i_Mth = (int) fr[0];
    i_Day = (int) fr[1];

// Leap-year-fix - when we read in the Daily weather if a Feb 29 was detected
//  we would have set the leap year flag,
   if ( this->b_LeapYear == true )
      this->Set_NextLeapDay (&i_Mth, &i_Day); /* chk, move post Feb 28 days foward 1 day */
// old code if (ICF::isLeapYearDay(i_Mth,i_Day)) { /* Don't want any leap year days */
//     continue; }

/* copy data into class                                                      */
     this->a_Wnd[iN_Recs].i_Mth    = i_Mth;
     this->a_Wnd[iN_Recs].i_Day    = i_Day;
     this->a_Wnd[iN_Recs].i_Hr     = (int) fr[2];
     this->a_Wnd[iN_Recs].f_Spd    =       fr[3];
     this->a_Wnd[iN_Recs].i_Dir    = (int) fr[4];
     this->a_Wnd[iN_Recs].i_CloCov = (int) fr[5];
     iN_Recs++;                        /* # of recs we load */
  } /* for i */

  this->iN_Wnd = iN_Recs;            /* # of recs in table */

/* Try to get one extra line, shouldn't get one, number of lines needs       */
/*  specified by the switch argument                                         */
    j = ICF::GetLine (fr,e_wifr, this->cr_ErrExt, this->fh);
    if ( j == 1 ){
      sprintf(this->cr_ErrExt,"Extra line of data found, switch specifies %d", i_Arg);
     *ai = e_EMS_WiD1;
      return 0; }
    strcpy (this->cr_ErrExt,"");  /* not actual error, so blank this out */

/* Look for Unit switch, if none set to english                              */
   if ( !ICF::Set_SinTxtArg ((char *)e_ICF_WiDU,this->cr_WiDU,eC_WSU) )
     strcpy (this->cr_WiDU,"English");

   ICF::SetWindYear ();   /* Set Year(s) into table */
  return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: WindDataFile       WIND_FILE:
* Desc: Find the Wind Data File Name switch and if found, open
*       the file and loads its data.
*       The first line in the file may have "English" or "Metric"
*        if none - defaults to English
*
*  Out: ai.....error number
* Note-1: If address used 'new' memory for loading wind data it means
*          data was already load from the embedded cmd file wind data
*  Ret: 1 file found and loaded in ICF, or switch not found
*       0 Switch found but in error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::WindDataFile (int *ai)
{
int i,n,x,fpos,i_Ret,i_Mth,i_Day,iN_Recs;
float fr[12];
char cr[300];
char cr_Unit[30];
FILE *fhW;

   *ai = e_EMS_WiF;                     /* err mes num incase error occurs   */
    i_Ret = 1;                          /* assume OK return value            */

   if (!ICF::Set_FilNam ((char *)e_ICF_WiF, this->cr_WiFN)) {
     return 0; }                        /* Invalid or bad file name arg      */

   if ( !strcmp (this->cr_WiFN,"") )    /* If no switch found in cmd file   */
     return 1;                          /* not an error, just didn't fnd sw  */

   if ( this->a_Wnd != NULL ) {         /* See Note-1 above                  */
     sprintf (this->cr_ErrExt,"Use only one %s %s ", e_ICF_WiF,e_ICF_WiD);
     return 0; }

/* Make sure there's no Unit switch in file, unit switch if for embeded data */
   strcpy (cr,"");
   if ( ICF::Set_SinTxtArg ((char *)e_ICF_WiDU,cr,eC_WSU) ) {
     sprintf (this->cr_ErrExt,"Invalid combination of switches: %s - %s", e_ICF_WiDU,e_ICF_WiF);
     *ai = e_EMS_WiF;
     return 0; }

   fhW = fopen (this->cr_WiFN,"r");     /* open Wind File                    */
   if ( fhW == NULL ) {                 /* was checked, but make sure        */
     sprintf (this->cr_ErrExt,"can't open: %s",this->cr_WiFN);
     return 0; }

/* 1st line in file can contain the units                                    */
   fscanf (fhW, "%s", cr);              /* 1st line in file                  */
   if ( !strcmp (cr,"Metric")){         /* See what unit                     */
     strcpy (cr_Unit,"Metric");   /* save it                           */
     fpos = ftell(fhW); }               /* pos in file to look for num data  */
   else if ( !strcmp (cr,"English")){
     strcpy (this->cr_WiDU,"English");
     fpos = ftell(fhW); }
   else {                               /* None - default to English         */
     strcpy (this->cr_WiDU,"English");
     fpos = 0;
     fseek (fhW, fpos, SEEK_SET);}        /* to begin of file                  */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Go thru file twice once to check and count - then to assign data          */
   for ( n = 0; n < 2; n++ ) {
     iN_Recs = 0;
     for ( i = 0; i < 10000; i++ ) {    /* Each line in file                 */
       x = ICF::GetLineFl (fr, 6, 0, this->cr_ErrExt, fhW);
       if ( x == 0 ){                   /* if Error in line read in          */
         fclose (fhW);
         return 0; }
       if ( x == -1 )                   /* if end of file                    */
         break;

       i_Mth = (int) fr[0];
       i_Day = (int) fr[1];


// Leap-Year-Fix
     this->Set_NextLeapDay(&i_Mth, &i_Day);

//      if (ICF::isLeapYearDay(i_Mth,i_Day)) { /* Don't want any leap year days */
//         continue; }
// --------------------------

       if ( n == 1 ) {                         /* on 2cd pass save data             */
         this->a_Wnd[iN_Recs].i_Mth   = i_Mth;       /* Mth = month, */
         this->a_Wnd[iN_Recs].i_Day   = i_Day;       /* Day = day,   */
         this->a_Wnd[iN_Recs].i_Hr    = (int) fr[2]; /* Hour         */
         this->a_Wnd[iN_Recs].f_Spd   =       fr[3]; /* Speed        */
         this->a_Wnd[iN_Recs].i_Dir   = (int) fr[4]; /* Direction    */
         this->a_Wnd[iN_Recs].i_CloCov= (int) fr[5];}/* Cloud Cover  */
       iN_Recs++;
     } /* for i */

    if ( n == 0 ) {                     /* After 1st pass thur file        */
      this->a_Wnd = new d_Wnd[iN_Recs];  /* allocate weather sturts         */
      this->iN_Wnd = iN_Recs;            /* number of                       */
      fseek (fhW, fpos, SEEK_SET); }    /* back to start of data in file   */
   } /* for n */

   ICF::SetWindYear ();   /* Set Year(s) into table */

   fclose (fhW);
   return i_Ret;
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: WeatherDataFile       WEATHER_FILE:
* Desc: Find the Weather Data File Name switch and if found, open
*       the file and loads its data.
*       The first line in the file may have "English" or "Metric"
*        if none - defaults to English
* NOTE: This function is setup to allow for the Weather Stream input
*       file to have or not have the last to values on a line, which
*       are the percip start and end times. I wasn't sure if the file
*       format allows that or not or possible future use.
*
*  Out: ai.....error number
* Note-1: If address used 'new' memory for loading weather data it means
*          data was already load from the embedded cmd file weather data
*  Ret: 1 file found and loaded in ICF, or switch not found
*       0 Switch found but in error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::WeatherDataFile (int *ai)
{
int i,n,x,fpos,i_Ret,iN_Recs,i_Mth, i_Day;
float fr[12];
char cr[300];
char cr_Unit[30];
FILE *fhW;

   *ai = e_EMS_WeF;                     /* err mes num incase error occurs   */
    i_Ret = 1;                          /* assume OK return value            */

   if (!ICF::Set_FilNam ((char *)e_ICF_WeF, this->cr_WeFN)) {
     return 0; }                       /* Invalid or bad file name arg      */

   if ( !strcmp (this->cr_WeFN,"") )    /* If no switch found in cmd file   */
     return 1;                          /* not an error, just didn't fnd sw  */

   if ( this->a_Wtr != NULL ) {          /* See Note-1 above                  */
     sprintf (this->cr_ErrExt,"Use only one %s %s ", e_ICF_WeF,e_ICF_WeD);
     return 0; }

/* Make sure there's no Unit switch in file, unit switch if for embeded data */
   if ( ICF::Set_SinTxtArg ((char *)e_ICF_WeDU,cr,eC_WSU) ) {
     sprintf (this->cr_ErrExt,"Invalid combination of switches: %s - %s", e_ICF_WeDU,e_ICF_WeF);
     *ai = e_EMV_WeDU;
     return 0; }

   fhW = fopen (this->cr_WeFN,"r");     /* open Weather file                 */
   if ( fhW == NULL ) {                 /* was checked, but make sure        */
     sprintf (this->cr_ErrExt,"can't open: %s",this->cr_WeFN);
     return 0; }

/* 1st line in file can contain the units                                    */
   fscanf (fhW, "%s", cr);              /* 1st line in file                  */
   if ( !strcmp (cr,"Metric")){         /* See what unit                     */
     strcpy (cr_Unit,"Metric");   /* save it                           */
     fpos = ftell(fhW); }               /* pos in file to look for num data  */
   else if ( !strcmp (cr,"English")){
     strcpy (cr_Unit,"English");
     fpos = ftell(fhW); }
   else {                               /* None - default to English         */
     strcpy (cr_Unit,"English");
     fpos = 0;                        /* to begin of file                  */
     fseek (fhW, fpos, SEEK_SET);
    }
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Go thru file twice once to check and count - then to assign data          */
   for ( n = 0; n < 2; n++ ) {
     iN_Recs = 0;
     for ( i = 0; i < 10000; i++ ) {    /* Each line in file                 */

       x = ICF::GetLineFl (fr, 12, 0, this->cr_ErrExt, fhW);
       if ( x == 0 ){                   /* if Error in line read in          */
         fclose (fhW);
		       return 0; }

       if ( x == -1 )                   /* if end of file                    */
         break;

       i_Mth = (int) fr[0];
       i_Day = (int) fr[1];

// Leap-Year-Fix
     if ( ICF::isLeapYearDay(i_Mth,i_Day))
       this->b_LeapYear = true;

     if ( this->b_LeapYear == true )
       this->Set_NextLeapDay (&i_Mth, &i_Day);
//------------------------------

       if ( n == 1 ) {                  /* on 2cd pass save data             */
         this->a_Wtr[iN_Recs].i_Mth = i_Mth;     /* Mth = month,                                     */
         this->a_Wtr[iN_Recs].i_Day = i_Day;     /* Day = day,                                       */
         this->a_Wtr[iN_Recs].f_Per =       fr[2];     /* Per = precip in 100th of inch (ex 10 = 0.1 inches*/
         this->a_Wtr[iN_Recs].i_mTH = (int) fr[3];     /* mTH = min_temp_hour 0-2359,                      */
         this->a_Wtr[iN_Recs].i_xTH = (int) fr[4];     /* xTH = max_temp_hour 0 - 2359,                    */
         this->a_Wtr[iN_Recs].f_mT  = (int) fr[5];     /* mT  = min_temp,                                  */
         this->a_Wtr[iN_Recs].f_xT  = (int) fr[6];     /* xT  = max_temp,                                  */
         this->a_Wtr[iN_Recs].i_xH  = (int) fr[7];     /* xH  = max_humidity,                              */
         this->a_Wtr[iN_Recs].i_mH  = (int) fr[8];     /* mH  = min_humidity,                              */
         this->a_Wtr[iN_Recs].i_Elv = (int) fr[9];     /* Elv = elevation,                                 */
         this->a_Wtr[iN_Recs].i_PST = (int) fr[10];    /* PST = precip_start_time 0-2359,                  */
         this->a_Wtr[iN_Recs].i_PET = (int) fr[11]; }  /* PET = precip_end_time 0-2359                    */
      iN_Recs++;
    } /* for i */

    if ( n == 0 ) {                       /* After 1st pass thur file        */
      this->a_Wtr = new d_Wtr[iN_Recs];     /* allocate weather sturts         */
      this->iN_Wtr = iN_Recs;              /* number of                       */
      fseek (fhW, fpos, SEEK_SET); }      /* back to start of data in file   */

   } /* for n */

   ICF::SetWtrYear();                     /* Set the year into WS table */

/* We want it in English if it comes in as Metric */
   ToEnglish (cr_Unit);

   fclose (fhW);
   return i_Ret;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: RAWS_WeatherDataFile
* Desc: Read data from a RAWS data file (data is not embedded
*       in the inputs command file).
*       This is the newer Weather Stream Hourly Data.
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::RAWS_WeatherDataFile (int *ai)
{
int i;
char cr_FN[1000];
FILE *fh, *tmp;

   *ai = e_EMS_RAWF;   /* in case we get an error */

/* See if there's a RAWS file name switch, and see if it's there */
  if (!ICF::Set_FilNam ((char *)e_ICF_RAWF, cr_FN)) {
     return 0; }      /* Fil Nam found but error - opening/finding */

  if ( !strcmp (cr_FN,"") )    /* No Swithc found */
     return 1;                 /* That's ok, it's not an error */

  fh = fopen (cr_FN,"r");      /* open file, already chk'd it's there */

/* Do a little monkey business and swap FILE pointers for the  */
/*  inputs command file and the RAWS data file, then we can */
/*  just use the WeatherStreamData() which normally read the RAWS */
/*  data right from the inputs command file */
  tmp = this->fh;
  this->fh = fh;

  i = ICF::RAWS_WeatherData(ai);

  fclose (this->fh);
  this->fh = tmp;       /* restore inputs cmd file FILE ptr */

  if ( i == 2 ) {       /* Says no RAWS switch found, switch has # recs as it's arg */
    sprintf  (this->cr_ErrExt,"No %s: switch found in RAWS input file", e_ICF_RAWS);
    *ai = e_EMS_RAWS;
    i = 0; }

  return i;             /* 0 would be and error */

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: RAWS_WeatherData
* Desc: This is the newer Weather Stream Data. It basicaly will replace
*        the individual Weather and Wind input data
* Note-1 This was done in the orginal FlamMap code so do it here too
* Note-2 We don't use any leap year Feb 29 days
* Note-3 For now we're not doing anything with the year in the input
*         file. See code below where we stuff a year in.
*  Out: ai.....error number
*  Ret: 1 ok,
*       2 no RAWS switch found
*       0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::RAWS_WeatherData (int *ai)
{
int i, j,i_Arg, iN_Recs, i_Mth, i_Day,i_Year;
int M,D,H;
float f;
#define eC_Un 20
char cr_Un[eC_Un];

#define e_wr 10
float wr[e_wr];

/* find the switch and its arg......                                         */
   i = ICF::Set_SinIntArg ((char *)e_ICF_RAWS,&i_Arg); /* WEATHER_DATA:                */
   if ( i == -1 )
     return 2;                          /*  if none found                    */

  if ( i == 0 ) {                       /* Switch found with bad argument    */
    strcpy (this->cr_ErrExt,"Invalid/Missing Switch argument");
    *ai = e_EMS_RAWS;
    return 0;}

/* Check the switch argument....                                             */
  if ( i_Arg < 1 ){   /* Arg is num lines of weather data  */
    sprintf (this->cr_ErrExt,"Invalid argument value: %d", i_Arg);
    *ai = e_EMS_RAWS;
    return 0; }

/* Look for Unit switch, if none set to english               */
   strcpy (cr_Un,"");
   ICF::Set_SinTxtArg ((char *)e_ICF_RAWSUnits, cr_Un,eC_Un);
   if ( !ICF::EngMet (cr_Un,this->cr_ErrExt)) {   /* Chk & set defualt to English */
     *ai = e_EMS_RAWS;
     return 0; }

   i = ICF::Set_SinNumArg((char *)e_ICF_RAWSElev, &f);   /* Get elevation */
   if ( i <= 0 ) {
      strcpy (this->cr_ErrExt,"Missing/Invalid RAWS Elevation");
      *ai = e_EMS_RAWS;
      return 0; }

   if ( !strcmp (cr_Un,"Metric") )
     this->i_RAWSElev = (int) (f * 3.28);
   else
     this->i_RAWSElev = (int) (f + 0.5);  /* roundoff elev */

/* Alloc structs, might not use all if there is any Feb 29 data */
  this->a_RAWS = new d_RAWS[i_Arg + 1];  /* 1 extra for projected wethr rec, see below */
  if ( this->a_RAWS == NULL ) {
      strcpy (this->cr_ErrExt,"Can't Allocate Memory for RAWS Weather Stream Data");
      *ai = e_EMS_RAWS;
      return 0; }

/*.......................................................................... */
/* Read in each line stream data  */
  ICF::Set_SinIntArg ((char *)e_ICF_RAWS,&i_Arg); /* put fil pointr before stream recs  */
  iN_Recs = 0;
  for ( i = 0; i < i_Arg; i++ ) {     /* Read each line's data into fr      */
     j = ICF::GetLineFl (wr,e_wr, 0, this->cr_ErrExt, this->fh);
     if ( j  <= 0 ) {                 /* If error or end of file              */
       sprintf (this->cr_ErrExt, "Number of RAWS records doesn't match number specified, %d", i_Arg);
       *ai = e_EMS_RAWS;
       return 0; }

/* copy data into class                                 */
     i_Year = (int) wr[0];  /* See Note-3 above */
     i_Mth = (int) wr[1];
     i_Day = (int) wr[2];
    // if (ICF::isLeapYearDay(i_Mth,i_Day) )  /* See Note-2 above */
     //  continue;
 // Leap-Year-Fix
     if ( ICF::isLeapYearDay(i_Mth,i_Day))
       this->b_LeapYear = true;

     if ( this->b_LeapYear == true )
       this->Set_NextLeapDay (&i_Mth, &i_Day);

     this->a_RAWS[iN_Recs].i_Mth = i_Mth;        /* month    */
     this->a_RAWS[iN_Recs].i_Day = i_Day;        /* day    */
     this->a_RAWS[iN_Recs].i_Time = (int) wr[3];    /* Time */
     this->a_RAWS[iN_Recs].f_Temp = wr[4];          /* Temperature */
     this->a_RAWS[iN_Recs].f_Humidity = wr[5];      /* Humidity */
     this->a_RAWS[iN_Recs].f_PerHou = wr[6];        /* Perceipitation Hourly */
     this->a_RAWS[iN_Recs].f_WinSpd = wr[7];        /* Wind Speed */
     this->a_RAWS[iN_Recs].f_WinDir = wr[8];        /* Wind Direction */
     this->a_RAWS[iN_Recs].f_CloCov = wr[9];        /* Cloud Cover */

     if ( !strcmp (cr_Un,"Metric")) {             /* Get it to English units */
        CelsToFahr ( &this->a_RAWS[iN_Recs].f_Temp); /* Temp to Fahrenheit */
        this->a_RAWS[iN_Recs].f_PerHou *= 3.93;      /* percip to inches  */
        this->a_RAWS[iN_Recs].f_WinSpd *= 0.5402; }  /* (0.62125/1.15) 10m wind kph to 20ft wind mph*/

     if ( this->a_RAWS[iN_Recs].f_Humidity > 99.0 )   /* See Note-1 above */
        this->a_RAWS[iN_Recs].f_Humidity = 99.0;
     iN_Recs++;                        /* Count actual recs loaded */
  } /* for i */

/* Try to get one extra line, shouldn't get one, number of lines needs       */
/*  specified by the switch argument                                         */
    j = ICF::GetLine (wr,e_wr, this->cr_ErrExt, this->fh);
    if ( j == 1 ){
      sprintf(this->cr_ErrExt,"Extra line of data found, the specified number is %d", i_Arg);
     *ai = e_EMS_RAWS;
      return 0; }
    strcpy (this->cr_ErrExt,"");  /* not actual error, so blank this out */

/*-----------------------------------------------------------------------*/
  j = iN_Recs - 1;               /* index to last record in tbl */
  M = this->a_RAWS[j].i_Mth;
  D = this->a_RAWS[j].i_Day;
  H = this->a_RAWS[j].i_Time ;

  for( i = 1; i <= e_RAWS_ExtHr; i++ )  /* Advance date/time by 6 hours */
    Get_Nxt_MDH (&M,&D,&H);

/* Create an extra rec 6 hours later on end of table */
/* this will make last weather rec appear to have 6 hrs of projected data */
  this->a_RAWS[iN_Recs].i_Mth = M;       /* month         */
  this->a_RAWS[iN_Recs].i_Day = D;       /* day           */
  this->a_RAWS[iN_Recs].i_Time = H;      /* Time Military */
  this->a_RAWS[iN_Recs].f_Temp = this->a_RAWS[j].f_Temp;          /* Temperature */
  this->a_RAWS[iN_Recs].f_Humidity = this->a_RAWS[j].f_Humidity;  /* Humidity */
  this->a_RAWS[iN_Recs].f_PerHou = 0;                             /* Perceipitation Hourly */
  this->a_RAWS[iN_Recs].f_WinSpd = this->a_RAWS[j].f_WinSpd;      /* Wind Speed */
  this->a_RAWS[iN_Recs].f_WinDir = this->a_RAWS[j].f_WinDir;      /* Wind Direction */
  this->a_RAWS[iN_Recs].f_CloCov = this->a_RAWS[j].f_CloCov ;     /* Cloud Cover */
  iN_Recs++;

  this->iN_RAWS = iN_Recs;  /* actual # of recs we saved */

/* OK - now let's fill in a year, it doesn't matter what year, this will really help */
/*  to check date sequence and might turn out to be usefull for locating records  */
  int Year = e_StrYear;
  this->a_RAWS[0].i_Yr = Year;
  for ( i = 1; i < this->iN_RAWS; i++ ) {
    if ( this->a_RAWS[i].i_Mth == 1 && this->a_RAWS[i - 1].i_Mth == 12 )
       Year++;
    a_RAWS[i].i_Yr = Year; }
	int ret = CheckRAWSWeatherWind(ai);
	if(ret != 1)
	{
		*ai = ret;
		return 0;
	}

  return 1;
}

/******************************************************************
* Name: Get_Nxt_MDH
* Desc: Add one hour to the Month Day Hour sent in
*       M = 1 -> 12
*       D = day of month - febuary only use up to 28
*       H = 0 -> 2359, military time, any hour minutes is ok,
*           ex 630, 1000, 1201, 1415, all
******************************************************************/
void Get_Nxt_MDH(int *M, int *D, int *H)
{
int h, m, d, dm;

/* Add one hour (60 minutes) */
   h = *H / 100;       /* get hour */
   h++;                /* add an hour */
   h = h * 100;        /* get it back militiary 0000 -> 2300 */
   m = *H % 100;       /* orginal minute */
   h = h + m;          /* add them back on */

   if ( h < 2359 ) {
     *H = h;
     return; }

/* Need to go to next day */
   h = h - 2400;   /* Set hour to next morning */
  *H = h;

   m = *M;             /* Month */
   d = *D;             /* Currnet day */
   d++;                /* Next Day */
   dm = DaysInMth (m); /* Get days in a month */
   if ( d <= dm ) {    /* day is still in same mth */
     *D = d;
      return ; }

/* Go to 1st day of next month */
   *D = 1;
   m = *M;
   m++;             /* Next mth */
   if ( m > 12 ) {  /* go to January */
     *M = 1;
      return ; }

  *M = m;
}

/********************************************************************/
int  DaysInMth (int Mth)
{
int i;    /*   j  f  m  a  m  j  j  a  s  o  n  d */
int ir[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
   if ( Mth > 12 )
     Mth = 12;
   i = Mth - 1;
   return  ir [i];
}

/***************************************************************************
* Name: CelsToFahr
* Desc: Convert Celsius to Fahrenheit
****************************************************************************/
void CelsToFahr (float *af)
{
float f;
    f = *af * 1.8;
    f = f + 32.0;
    *af = f;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: WeatherDataCmd
* Desc: Weather Data that is embedded in the command file.
*       Find the Weather Data switch and if found load its data.
*       Checks for valid number values, and for the right amount of
*       data, doesn't check limits.
* Note-1: This would mean that the Weather Stream File switch was found
*         and its data loaded, and we can't have both.
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::WeatherDataCmd  (int *ai)
{
int i,j, i_Arg, iN_Recs, i_Mth, i_Day;
char cr_Unit[50];

#define e_fr 12
float fr[e_fr];

/* find the switch and its arg......                                         */
   i = ICF::Set_SinIntArg ((char *)e_ICF_WeD,&i_Arg); /* WEATHER_DATA:                */
   if ( i == -1 )                        /* Optional switch so not an error   */
     return 1;                          /*  if none found                    */

  if ( i == 0 ) {                       /* Switch found with bad argument    */
    strcpy (this->cr_ErrExt,"Invalid/Missing Switch argument");
    *ai = e_EMS_WeD1;
    return 0;}

   if ( this->a_Wtr != NULL ) {          /* See Note-1 above                  */
     *ai = e_EMS_WeD2;
     sprintf (this->cr_ErrExt, "Use only one %s %s ", e_ICF_WeF,e_ICF_WeD);
     return 0; }

/* Check the switch argument....                                             */
  if ( i_Arg < 1 || i_Arg > e_MaxWD){   /* Arg is num lines of weather data  */
    sprintf (this->cr_ErrExt,"Invalid argument value: %d, Limit: %d", i_Arg,e_MaxWD);
    *ai = e_EMS_WeD1;
    return 0; }

  this->a_Wtr = new d_Wtr[i_Arg];         /* allocate weather sturts           */

/*.......................................................................... */
  iN_Recs = 0;
  for ( i = 0; i < i_Arg; i++ ) {     /* Read each line's data into fr      */
     j = ICF::GetLineFl (fr,e_fr, 0, this->cr_ErrExt, this->fh);
     if ( j  == 0 ) {                 /* If error                           */
       *ai = e_EMS_WeD1;
       return 0; }

     i_Mth = (int) fr[0];
     i_Day = (int) fr[1];

// Leap-Year-Fix, if we see a Feb 29, set leap year flag,
// The way we deal with leap year is by moving all post Feb 28 dates
//  ahead by one day, ie. Feb 29 becomes mar 1, mar 1 becomes mar 2, etc
     if ( ICF::isLeapYearDay(i_Mth,i_Day))
       this->b_LeapYear = true;

     if ( this->b_LeapYear == true ) /* need to move post Feb 28 days ahead 1 day */
       this->Set_NextLeapDay (&i_Mth, &i_Day);
// ------------------------------


/* copy data into class                                                      */
     this->a_Wtr[iN_Recs].i_Mth = i_Mth;           /* Mth = month,                                     */
     this->a_Wtr[iN_Recs].i_Day = i_Day;           /* Day = day,                                       */
     this->a_Wtr[iN_Recs].f_Per =       fr[2];     /* Per = precip in 100th of inch (ex 10 = 0.1 inches*/
     this->a_Wtr[iN_Recs].i_mTH = (int) fr[3];     /* mTH = min_temp_hour 0-2359                      */
     this->a_Wtr[iN_Recs].i_xTH = (int) fr[4];     /* xTH = max_temp_hour 0 - 2359,                    */
     this->a_Wtr[iN_Recs].f_mT  = (int) fr[5];     /* mT  = min_temp,                                  */
     this->a_Wtr[iN_Recs].f_xT  = (int) fr[6];     /* xT  = max_temp,                                  */
     this->a_Wtr[iN_Recs].i_xH  = (int) fr[7];     /* xH  = max_humidity,                              */
     this->a_Wtr[iN_Recs].i_mH  = (int) fr[8];     /* mH  = min_humidity,                              */
     this->a_Wtr[iN_Recs].i_Elv = (int) fr[9];     /* Elv = elevation,                                 */
     this->a_Wtr[iN_Recs].i_PST = (int) fr[10];    /* PST = precip_start_time 0-2359                  */
     this->a_Wtr[iN_Recs].i_PET = (int) fr[11];    /* PET = precip_end_time 0-2359                   */
     iN_Recs++;
  } /* for i */

/* Try to get one extra line, shouldn't get one, number of lines needs       */
/*  specified by the switch argument                                         */
    j = ICF::GetLine (fr,e_fr, this->cr_ErrExt, this->fh);
    if ( j == 1 ){
      sprintf(this->cr_ErrExt,"Extra line of data found, switch specifies %d", i_Arg);
     *ai = e_EMS_WeD1;
      return 0; }
    strcpy (this->cr_ErrExt,"");  /* not actual error, so blank this out */


   this->iN_Wtr = iN_Recs;       /* # of records we actually loaded */
   ICF::SetWtrYear ();          /* Set year dates into the WS weather array */

/* Look for Unit switch, if none set to english                              */
   if ( !ICF::Set_SinTxtArg ((char *)e_ICF_WeDU,cr_Unit,eC_WSU) )
     strcpy (cr_Unit,"English");

/* We want it in English if it comes in as Metric */
   ToEnglish (cr_Unit);

  return 1;
}

/*******************************************************************************
* Name: SetWtrYear
* Desc: Set the year into the weather data struct array.
*       The actual year doesn't matter (as long as it's not a leap year) we
*        just need something to help distinguish when there is more than
*        one year of data
*       Also set the number of years in the weather data.
******************************************************************************/
int ICF::SetWtrYear ()
{
int i, i_Year;

   i_Year = e_StrYear;
   for ( i = 0;i < this->iN_Wtr; i++ ) {
     if ( i > 0 && this->a_Wtr[i].i_Mth == 1 && this->a_Wtr[i].i_Day == 1 )
       i_Year++;
     this->a_Wtr[i].i_Year = i_Year;
   }

   if ( i_Year == e_StrYear ) {
     this->iN_YearWeather = 1;
     return 1; }

   this->iN_YearWeather = 2;
   return 2;
}

/*******************************************************************************
* Name: SetWtrYear
* Desc: Set the year into the wind data struct array.
*       The actual year doesn't matter (as long as it's not a leap year) we
*        just need something to help distinguish when there is more than
*        one year of data
*       Also set the number of years in the wind data.
******************************************************************************/
int ICF::SetWindYear ()
{
int i, i_Year;

   if ( this->a_Wnd == NULL )    /* We should be here if this is NULL */
     return 0;                   /*  but check just in case */

   i_Year = e_StrYear;
   this->a_Wnd[0].i_Year = i_Year;

   for ( i = 1; i < this->iN_Wnd; i++ ) {
     if ( this->a_Wnd[i-1].i_Mth == 12 && this->a_Wnd[i].i_Mth == 1 )
         i_Year++;
     this->a_Wnd[i].i_Year = i_Year;
   }

   if ( i_Year == e_StrYear ) {
     this->iN_YearWind = 1;
     return 1; }

   this->iN_YearWind = 2;
   return 2;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ToEnglish
* Desc: Check and get the Units to English if they come in
*        as Metric
*       ALSO to a check and set on the Humidty - see code below
*       cr_Unit...."English" "Metric"
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void  ICF::ToEnglish (char cr_Unit[])
{
int i;
d_Wtr *aWS;

  if ( this->iN_Wtr <= 0 )
     return;

  for ( i = 0; i < this->iN_Wtr; i++ ) {
    aWS = &this->a_Wtr[i];
    if ( !strcmp (cr_Unit,"Metric")){    /* if in Metric convt to Englis */

/* percipitation from centimeters to hundredths of inches */
       aWS->f_Per *= 3.93;                 /*  ppt  *= 3.93;     */

/* Min temperature */
       aWS->f_mT  *= 1.8;                  /*  Tmin *= 1.8;      */
       aWS->f_mT  += 32;                   /*  Tmin += 32;       */

/* Max termperature */
       aWS->f_xT  *= 1.8;                  /*  Tmax *= 1.8;      */
       aWS->f_xT  += 32;                   /*  Tmax += 32;       */

       aWS->i_Elv *= 3.28; }               /*  elref *= 3.28;    */

/* Not sure about this but it was done in FlamMap code                       */
     if ( aWS->i_mH > 99 )                  /*  if ( Hmax > 99 )  */
       aWS->i_mH = 99;                      /*    Hmax = 99;      */
     if ( aWS->i_xH > 99 )                  /*  if ( Hmin > 99)   */
       aWS->i_xH = 99;                       /*    Hmin = 99;      */
   } /* for i */

}

//function retrieves the next input line from the input file
//for specialized blocks that need to parse a line themselves
int  ICF::GetCharLine (char buf[], char cr_Err[], FILE *fh)
{
	char *a, cr_Sav[500], cr[500];
	strcpy (cr_Err,"");
/* Loop until we find the next line of data, skip blank and comment lines    */
   while (1)
   {
     a = fgets (cr,300,fh);             /* Read a line from file             */
     if ( a == NULL )                   /* end of file,                      */
       return 0;

     Remove_CrLf (cr);                  /* Replc any car-ret lin-fed with Nul*/
     Trim_LT(cr);                       /* strip lead and tail blanks        */
     strcpy (cr_Sav, cr);               /* Save for err message              */
	// blank = ICF::isBlank(cr);
     if (ICF::isBlank (cr))             /* skip blank lines                  */
       continue;
    if ( cr[0] == '#' )                /* Skip comment lines                */
       continue;
	break;
   }
   strcpy(buf, cr);
   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetLine
* Desc: Get one line of data (the next line in the file) and put the
*        values in the fr[]
*       This is used for switch like WEATHER_DATA: that have multiple
*        lines of number data following them.
*   In: iN....number of fields on line
*  Out: fr[]..values from line
*       cr_Err..error message
*  Ret: 1 = OK,
*      -1 = End of file
*       0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetLine (float fr[], int iN, char cr_Err[], FILE *fh)
{
int j;
float f;
char *a, cr_Sav[500], cr[500];

   strcpy (cr_Err,"");

/* Loop until we find the next line of data, skip blank and comment lines    */
   while (1) {
     a = fgets (cr,300,fh);             /* Read a line from file             */
     if ( a == NULL )                   /* end of file,                      */
       return -1;

     Remove_CrLf (cr);                  /* Replc any car-ret lin-fed with Nul*/
     Trim_LT(cr);                       /* strip lead and tail blanks        */
     strcpy (cr_Sav, cr);               /* Save for err message              */
     if (ICF::isBlank (cr))             /* skip blank lines                  */
       continue;
     if ( cr[0] == '#' )                /* Skip comment lines                */
       continue;

     for ( j = 0; j < iN; j++ ){        /* Get each val from the line        */

       if ( ICF::GetFlo (cr,&fr[j],j))  /* Get, each value on line           */
         continue;                      /* Go value                          */

       sprintf (cr_Err, "Bad or extra line of data found:\nLine in Error-> %s", cr_Sav);
       return 0;

     } /* for j */
     break;

   } /* while */


/* Try to get one value off end of line, we should shoudn't find any         */

// Original code  if ( ICF::GetFlo (cr,&fr[j],j)) {  /* Get, chk and put into Struct      */
// Bug Fix
  if ( ICF::GetFlo (cr,&f,j)) {  /* Get, chk and put into Struct      */
    sprintf (cr_Err, "Extra data found on line:\n%s", cr_Sav);
    return 0; }

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetLineFl
* Desc: Get one line of data (the next line in the file) and put the
*        values in the fr[]
* Note-1: IF THE REQUESTED number of values are not found on the line
*       then the f_Fil char is returned in that array position
*   In: iN....number of fields on line
*  Out: fr[]..values from line
*       cr_Err..error message
*  Ret: 1 = OK,
*      -1 = End of file
*       0 = Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetLineFl (float fr[], int iN, float f_Fil, char cr_Err[], FILE *fh)
{
int j, x;
char *a, cr_Sav[500], cr[500];

   strcpy (cr_Err,"");
   strcpy (cr_Sav,"");

   for ( j = 0; j < iN; j++ )
     fr[j] = f_Fil;

/* Loop until we find the next line of data, skip blank and comment lines    */
   while (1) {
     strcpy (cr,"");
     a = fgets (cr,300,fh);             /* Read a line from file             */
     if ( a == NULL )                   /* end of file,                      */
       return -1;

     Remove_CrLf (cr);                  /* Replc any car-ret lin-fed with Nul*/
     Trim_LT(cr);                       /* strip lead and tail blanks        */
     strcpy (cr_Sav, cr);               /* Save for err message              */
     if (ICF::isBlank (cr))             /* skip blank lines                  */
       continue;
     if ( cr[0] == '#' )                /* Skip comment lines                */
       continue;

     for ( j = 0; j < iN; j++ ){        /* Get each val from the line        */

       x = ICF::GetFloE (cr,&fr[j],j);  /* Get, each value on line           */
       if ( x == 1 )                    /* if a good number found            */
         continue;                      /* keep going                        */
       if ( x == -1 )                   /* No more numbers on line           */
         return 1;                      /* thats ok, see Note-1 above        */

       sprintf (cr_Err, "Bad line of data found:\nLine in Error-> %s", cr_Sav);
       return 0;
     } /* for j */

     break;

   } /* while */

/* Try to get one value off end of line, we should shoudn't find any         */
  if ( ICF::GetFlo (cr,&fr[j],j)) {  /* Get, chk and put into Struct      */
    sprintf (cr_Err, "Extra data found on line:\n%s", cr_Sav);
    return 0; }

  return 1;
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetFlo
* Desc: Parse out token from a string, check, convert and send back
*       Token delimter is a blank
*   In: cr....string of test
*       iS....0 must be sent in to signal 1st token is wanted
*  Out: ai....int token from string
*  Ret: 1 0k
*       0 not a numberic arg OR at end of string
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetFlo (char cr[], float *af, int iS)
{
char *a;
  if ( iS == 0 )                        /* 1st token, sets string to use     */
    a = strtok (cr," ");
  else
    a = strtok (NULL," ");              /* get folling tokens                */

  if ( a == NULL )                      /* happens when no token found, like  */
     return 0;                          /* trying to get more than there are  */

  if (Get_NumTyp (a) == 'X' )           /* make sure it's a number           */
    return 0;

  *af = atof (a);                       /* Strint long int                   */
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetFloE
* Desc: Parse out token from a string, check, convert and send back
*       Token delimter is a blank
* NOTE: this functions will distinguish between and error and an
*        end of line.
*   In: cr....string of test
*       iS....0 must be sent in to signal 1st token is wanted
*  Out: ai....int token from string
*  Ret: 1 0k,
*      -1 hit end of line
*       0 not a numberic arg
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetFloE (char cr[], float *af, int iS)
{
char *a;
  if ( iS == 0 )                        /* 1st token, sets string to use     */
    a = strtok (cr," ");
  else
    a = strtok (NULL," ");              /* get folling tokens                */

  if ( a == NULL )                      /* happens when no token found, like  */
     return -1;                         /* trying to get more than there are  */

  if (Get_NumTyp (a) == 'X' )           /* make sure it's a number           */
    return 0;

  *af = atof (a);                       /* Strint long int                   */
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_OGS - Output Grid Switches
* Desc: Find the specifed switch in the file and mark accordingly
*   In: cr_Sw....switch define
*  Out: ai......ICF Class variable
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void ICF::Set_OGS (char cr_Sw[], int *ai)
{
  if (ICF::GetSw (cr_Sw) )
    *ai = e_Yes;
  else
    *ai = e_No;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_ConPerStart
* Desc: Set the Conditioning Period Starting date
* NOTE: Optional switch - if not found then the a beggining date based
*        on the entered weather data will get set later.
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_ConPerStart (char cr_Sw[])
{
int i;
char Mth[15],Day[15],Hour[15];
char cr[eC_InpLin];

	 if ( !ICF::GetSw (cr_Sw) ) {    /* Find the Switch in input file     */
    ICF::Set_ConPerStartDef ();       /* if not found set the default      */
		  return 1; }

 	Mth[0] = Day[0] = Hour[0] = 0;

/* Just take out the separators if they a there, if they are already blank   */
/*  chars separating them - ok                                               */
 	strcpy (cr,&this->cr_Line[this->iX]);
 	StrRepChr (cr, '/', ' ');   /* replace mth day separator         */
 	sscanf (cr,"%10s %10s %10s",Mth,Day,Hour);

/* Here we just make sure that they are valid numbers                        */
 	i = 0;
  if ( Get_NumTyp (Mth) != 'I' )
		  i++;
 	if ( Get_NumTyp (Day) != 'I' )
		  i++;
 	if ( Get_NumTyp (Hour) != 'I' )
		  i++;

  if ( i > 0 ) {
    strcpy (this->cr_ErrExt,&this->cr_Line[this->iX]);
    return 0; }

  this->i_MthStart =  atoi(Mth);
  this->i_DayStart =  atoi(Day);
  this->i_HourStart = atoi(Hour);

  this->i_YearStart = ICF::GetWeatherYear(this->i_MthStart,this->i_DayStart);

  if ( this->i_YearStart == 0 ) {   /* Date was not found in weather data */
    strcpy (this->cr_ErrExt,&this->cr_Line[this->iX]);
    return 0; }

  return 1;

}
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_ConPerStartDef
* Desc: Set the Conditioning Period Starting default date which is at
*        the start of the weather data we have.
*       Daily data we set to one day after start of data because of
*        how data will get interpolated.
*       Hourly RAWS data we can set to the very beginning of the
*        weather data.
*  Ret: 1 ok, 0 error bad date in weather table, or NO weather data found
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_ConPerStartDef ()
{
int m,d,y, mm, dd, yy, t;

/* Daily Weather data - start at begining plus one day */
   if ( this->a_Wtr != NULL ) {               /* if we have this daily of weather data */
     m = this->a_Wtr[0].i_Mth;
     d = this->a_Wtr[0].i_Day;
     y = this->a_Wtr[0].i_Year;
     t = 0;                                   /* daily weather data starts at begin of day */
     if ( !ICF::NextDay(m,d,y,&mm,&dd,&yy) )  /* date of the very next day */
        return 0; }                           /* invalid date send in */

/* RAWS weather data - start right at very beginning */
   else if ( this->a_RAWS != NULL ) {         /* if we have the new RAWS data */
     mm = this->a_RAWS[0].i_Mth;
     dd = this->a_RAWS[0].i_Day;
     yy = this->a_RAWS[0].i_Yr;
     t = this->a_RAWS[0].i_Time; }
   else
     return 0;

   this->i_MthStart  = mm;
   this->i_DayStart  = dd;
   this->i_YearStart = yy;
   this->i_HourStart = t;
   return 1;
}

/************************************************************************
* Name:
* Desc:
*
*
************************************************************************/
int ICF::CondMthDay (long *al_conditmonth,long *al_conditday)
{
int m,d,y, mm, dd, yy, t;

/* Daily Weather data - start at begining plus one day */
   if ( this->a_Wtr != NULL ) {               /* if we have this daily of weather data */
     m = this->a_Wtr[0].i_Mth;
     d = this->a_Wtr[0].i_Day;
     y = this->a_Wtr[0].i_Year;
     t = 0;                                   /* daily weather data starts at begin of day */
     if ( !ICF::NextDay(m,d,y,&mm,&dd,&yy) )  /* date of the very next day */
        return 0; }                           /* invalid date send in */

/* RAWS weather data - start right at very beginning */
   else if ( this->a_RAWS != NULL ) {         /* if we have the new RAWS data */
     mm = this->a_RAWS[0].i_Mth;
     dd = this->a_RAWS[0].i_Day;
     yy = this->a_RAWS[0].i_Yr;
     t = this->a_RAWS[0].i_Time; }
   else
     return 0;

   *al_conditmonth = mm;
   *al_conditday = dd;
   return 1;
}





/************************************************************************
* Name: NextDay
* Desc: Get the following day
*   In: i_Mth, i_Day
*  Out: the next month/day
*  Ret: 1 ok, 0 invalid month/day sent in
************************************************************************/
int ICF::NextDay (int i_Mth, int i_Day, int i_Yr, int *ai_Mth, int *ai_Day, int *ai_Yr)
{
int rd[] = {0, 31, 28, 31, 30, 31,30, 31, 31, 30, 31, 30, 31};

   if ( !ICF::CheckMonthDay (i_Mth, i_Day ) )
     return 0;                      /* Invalid date */

   *ai_Yr = i_Yr;

   if ( i_Mth == 12 && i_Day == 31 ) {  /* Last day of year */
      *ai_Mth = 1;
      *ai_Day = 1;
      *ai_Yr = i_Yr + 1;
      return 1; }

   i_Day++;                      /* Next Day */
   if ( i_Day > rd[i_Mth] ){     /* Exceeds days in that month */
     *ai_Mth = i_Mth + 1;        /* set to first day of next mth */
     *ai_Day = 1;
     return 1;}

   *ai_Mth = i_Mth;
   *ai_Day = i_Day;
   return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_ConPerEnd - Conditioning Period End
* Desc: Find switch then following arguments in format as
*         Mt/Dy Hr:Mn   or   Mt Dy Hr Min
*         12/19 1020    or   12 19 10 20  are both ok
* NOTE: values are set into the approbriate class variables but only
*       check for being valid number not check for limits
*  Ret: 1 = switch and arg found ok, or switch not found
*       0 = error - bad numeric arg found, or not correct number of args
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_ConPerEnd (char cr_Sw[])
{
int i;
char Mth[15],Day[15],Hour[15];
char cr[eC_InpLin];

   if ( !ICF::GetSw (cr_Sw) )       /* Find the Switch in input file     */
     return 1;                          /* Didn't find it,                   */

  Mth[0] = Day[0] = Hour[0] = 0;

/* Just take out the separators if they a there, if they are already blank   */
/*  chars separating them - ok                                               */
  strcpy (cr,&this->cr_Line[this->iX]);
  StrRepChr (cr, '/', ' ');   /* replace mth day separator         */

  sscanf (cr,"%10s %10s %10s",Mth,Day,Hour);

/* Here we just make sure that they are valid numbers                        */
  i = 0;
  if ( Get_NumTyp (Mth) != 'I' )
    i++;
  if ( Get_NumTyp (Day) != 'I' )
    i++;
  if ( Get_NumTyp (Hour) != 'I' )
    i++;

  if ( i > 0 ) {
    strcpy (this->cr_ErrExt,&this->cr_Line[this->iX]);
    return 0; }

  this->i_MthEnd =  atoi(Mth);
  this->i_DayEnd =  atoi(Day);
  this->i_HourEnd = atoi(Hour);

  this->i_YearEnd = ICF::GetWeatherYear(this->i_MthEnd,this->i_DayEnd); /* Get year from Weather table */
  if ( this->i_YearEnd == 0 ) {   /* Date was not found in weather data */
    strcpy (this->cr_ErrExt,&this->cr_Line[this->iX]);
    return 0; }

  return 1;
}





/******************************************************************
* Name: GetWeatherYear
* Desc: Get the year from either the Weather or RAWS table for the
*        specified month and day. One one table will have data in it.
* NOTE: THE Weather data needs to get loaded and the year set before this
*        function can get called
* NOTE: Weather data should of had a psdudo year stuff in the
*        records.
*       RAWS - comes in with a year but then has it replaces with
*        a pseudo year
*  Ret: year or 0 for not found
*****************************************************************/
int  ICF::GetWeatherYear ( int i_Mth, int i_Day)
{
int i;
/* Weather data could be either in the old table or the new RAWS type */

  if ( this->a_Wtr != NULL ) {             /* if data in old style table */
    for ( i = 0; i < this->iN_Wtr; i++ ) {
       if ( this->a_Wtr[i].i_Mth == i_Mth && this->a_Wtr[i].i_Day == i_Day )
         return this->a_Wtr[i].i_Year;
    }
    return 0;
  }

  if ( this->a_RAWS != NULL ) {      /* If data is in the RAWS table */
    for ( i = 0; i < this->iN_RAWS; i++ ) {
      if ( this->a_RAWS[i].i_Mth == i_Mth && this->a_RAWS[i].i_Day == i_Day )
         return this->a_RAWS[i].i_Yr; }
    return 0;
  }

  return 0; /* error - mth day not found */
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CloseRet
* Desc: Function just provides a way for the caller to close the file
*        and return a function value using one statement.
*  Ret: small value as sent in
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::CloseRet(int i_Ret)
{
  fclose (this->fh);
  return i_Ret;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetInt
* Desc: Parse out token from a string, check, convert and send back
*       Token delimter is a blank
*   In: cr....string of test
*       iS....0 must be sent in to signal 1st token is wanted
*  Out: ai....int token from string
*  Ret: 1 0k, 0 = not a numberic arg or at end of string
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetInt (char cr[], int *ai, int iS)
{
char *a;
  if ( iS == 0 )                        /* 1st token, sets string to use     */
    a = strtok (cr," ");
  else
    a = strtok (NULL," ");              /* get folling tokens                */

  if ( a == NULL )                      /* happens when no token found, like  */
     return 0;                          /* trying to get more than there are  */

  if (Get_NumTyp (a) == 'X' )           /* make sure it's a number           */
    return 0;

  *ai = atoi (a);                       /* Strint long int                   */
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetLongInt
* Desc: Parse out token from a string, check, convert and send back
*       Token delimter is a blank
*   In: cr....string of test
*       iS....0 must be sent in to signal 1st token is wanted
*  Out: al....long int toke from string
*  Ret: 1 0k, 0 = not a numberic arg or at end of string
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetLongInt (char cr[], long *al, int iS)
{
char *a;
  if ( iS == 0 )                        /* 1st token, sets string to use     */
    a = strtok (cr," ");
  else
    a = strtok (NULL," ");              /* get folling tokens                */

  if ( a == NULL )                      /* happens when no token found, like  */
     return 0;                          /* trying to get more than there are  */

  if (Get_NumTyp (a) == 'X' )           /* make sure it's a number           */
    return 0;

  *al = atol (a);                       /* Strint long int                   */
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_FilNam
* Desc: Find the specified path file name switch in the input file,
*       check the switch argument (file name) to make sure the file is
*       there, then set path file name into the ICF class
*   In: cr_Sw......switch to look for
*  Out: a_ICF_FN...address in the ICF class to put path file name
*  Ret: 1 OK = switch found and file name is good or switch not found
*       0 Error - switch found but can't find file
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_FilNam (char cr_Sw[], char a_ICF_FN[])
{

  strcpy (a_ICF_FN,"");

  if ( !ICF::GetSw (cr_Sw) )            /* find the switch                   */
    return 1;                           /* Switch not found                  */

  strcpy (a_ICF_FN, &this->cr_Line[iX]);/* This is where the arg now is */
  Trim_LT (a_ICF_FN);                   /* trim any lead and tail blnks     */

  if ( !isFile (a_ICF_FN)) {            /* see if file is there */
    sprintf (this->cr_ErrExt,"command file switch and file name:\n%s\n",this->cr_Line);
    return 0; }

  return 1;

}
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: isFile
* Desc: See if a file exists
*   In: path and/or file name
* Ret: 1 = file is there, else 0
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::isFile(char cr_PthFN[])
{
    FILE *fd;
    if((fd = fopen(cr_PthFN, "r")) == NULL)
    return 0;
    fclose(fd);
    return 1;
/*int i;
  i = access (cr_PthFN,0);
  if ( i == 0 )
     return 1;
  return 0;*/
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_SinIntArg
* Desc: Find the specified switch in the input file and set it's
*        integer numeric argument to the specifed variable
*        in ICF class
*   In: cr_Sw.....switch to look for
*       *ai.......address of variable in ICF class to put the arg
*  Ret: 1 switch found and has good integer value argument
       -1 switch not found
*       0 switch has invalid argument
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_SinIntArg (char cr_Sw[], int *ai)
{
char  cr[50];

  if ( !ICF::GetSw (cr_Sw) )           /* Didn't find switch, this may not  */
    return -1;                         /* maybe optional sw we'll chck latr */

  strcpy (cr,"");
  sscanf (&this->cr_Line[this->iX],"%10s",cr);

  if ( Get_NumTyp (cr) == 'X' )
    return 0;

  *ai = atoi (cr);

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_SinNumArg
* Desc: Find the specified switch in the input file and set it's
*        floating point numeric argument to the specifed variable
*        in ICF class
*   In: cr_Sw.....switch to look for
*       *af.......address of variable in ICF class to put the arg
*  Ret: 1 ok or switch and has good argument
*      -1 switch not found
*       0 switch has invalid numeric argument
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_SinNumArg (char cr_Sw[], float *af)
{
char  cr[50];

  if ( !ICF::GetSw (cr_Sw) )           /* Didn't find switch, this may not  */
    return -1;                         /* maybe optional sw we'll chck latr */

  strcpy (cr,"");
  sscanf (&this->cr_Line[this->iX],"%20s",cr);

  if ( Get_NumTyp (cr) == 'X' )
    return 0;

  *af = atof (cr);

  return 1;

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_SinTxtArg
* Desc: Find the specified switch in the input file and set it's
*        txt argument to the specifed variable in ICF class
*   In: cr_Sw.....switch to look for
*       *a..... ..where to put the text arg
*  Ret: 1 ok, 0 switch not found
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_SinTxtArg (char cr_Sw[], char *a, int iN)
{
char cr_Fmt[30];


  sprintf(cr_Fmt,"%c%ds",'%',iN);          /* make format string with max chars */

  if ( !ICF::GetSw (cr_Sw) )           /* Didn't find switch, this may not  */
    return 0;                          /* maybe optional sw we'll chck latr */

  strcpy (a,"");
  sscanf (&this->cr_Line[this->iX],cr_Fmt,a);

  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GetSw
* Desc: Look for a specifed  switch in the input file
* NOTE: The input line from the file is saved in ICF.cr_Line and
*        the index ICF.iX will get set to point to just after the switch
*        so any args can get pulled off
*   In: cr_Sw.....switch to look for
*  Ret: 1 Ok,    0 Not Found
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::GetSw (char cr_Sw[])
{
char  *a, cr[eC_InpLin], cr_Arg[10];

   sprintf(cr_Arg,"%c%ds",'%',eC_InpLin);   /* make format string with max chars */
   strcpy (this->cr_Line,"");

   fseek (this->fh,0L,SEEK_SET);              /* start search at begin of file     */

   while (1) {
      a = fgets (cr, eC_InpLin,this->fh);  /* Read a line from file             */
      if ( a == NULL )                  /* End of File                       */
         break;

      Remove_CrLf(cr);       /* Remove and Car Ret or Line Feed   */
      Trim_LT (cr);          /* Remove and leading blanks         */

      if (cr[0] == '#' )
        continue;

      strcpy (this->cr_Line,cr);        /* Save line as it was read in      */

      RemoveSwMa(cr);                   /* remov mark on end of Switch       */

      if ( strcmp (cr,cr_Sw) )        /* Is it the switch we want          */
         continue;                      /* Nope                              */

      this->iX = strlen (cr_Sw) + 1 ;   /* len of fnd swtch + 1 for ':'      */
      return 1;
   }

   strcpy (this->cr_Line,"");
   this->iX = 0;
   return 0;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Trim_LT
* Desc: Remove Leading & Trailing blanks from a string.
* NOTE: This will leave any embedded blanks intact.
*       see the Rem_LT_Blanks function
*   In: cr.......String
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void  ICF::Trim_LT (char cr[])
{
int i,j;
   Left_Just (cr);                /* remove leading */
   j = strlen(cr);
   if ( j == 0 )
     return;
   j--;
   for ( i = j; i > 0; i--) {     /* go to end of string and work */
     if ( cr[i] != ' ' )          /* back until a char is hit */
       break;                     /* then null term the string */
     if ( cr[i] == ' ' )
       cr[i] = 0;
   }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
*  Name: Left_Just
*  Desc: Left Justify a null terminated string
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void ICF::Left_Just ( char  cr[] )
{
int i, j, len;
   len = strlen(cr);
   if ( cr[1] == 0 ) return;                 /* Empty String */
   for ( i = 0; i <= len; i++ ) {            /* Find first non blank */
     if ( cr[i] != ' ' )
        break ;   }
   if ( i == 0 )  return;                    /* Already Left Justified */
   for ( j = 0; j <= len; j++ ) {            /* Left Justify it          */
    cr [j]  =  cr [i++];
    if ( cr[j] == 0 )
       break;     }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Blk_End_Line
* Desc: Go thru a string looking for a carriage ret or other char that
*         specifies the end of a line (line of text from a file)
*         from that point fill the reset of the string with blank chars
*         and null term the string
*   In: cr_Line......String
*       i_Len........Length
*  Out: cr_Line......String with end blanked out
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void  ICF::Blk_End_Line (char cr_Line[], int i_Len)
{
int i,j;
   j = 0;
   for ( i = 0; i < i_Len; i++ ) {
     if ( cr_Line[i] < 25 )
       j = 1;
     if ( j == 1 )
        cr_Line[i] = ' '; }
   cr_Line[i-1] = 0;                /* null term at last position in string  */
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: StrRepChr
* Desc: Replace all the characters in a string with another character
*       String must be NULL terminated
*   In: c_This....find this char in string
*       c_That....new char to be put into string
* In/Out:  cr.....null term string
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void  ICF::StrRepChr (char cr[], char c_This,  char c_That )
{
int i,j;
    j = strlen(cr);
    for ( i = 0; i <= j; i++ ) {
      if ( cr[i] == 0 )
        break;
      if ( cr[i] == c_This )
        cr[i] = c_That; }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Get_NumTyp
* Desc: See if a char string is an Integer or Float or Invalid
*   In: cr_Data....String to be checked
*  Ret: F,I, or X
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
char ICF::Get_NumTyp (char cr_Data[])
{
char cr[30];
int  i_CntDig,i_CntDec, i, j;

   if ( strlen(cr_Data) >= sizeof (cr) )
      return 'X';

   sscanf (cr_Data,"%s",cr);	      /* remove any lead or trail blanks */

   i_CntDig = 0;
   i_CntDec = 0;
   if ( cr[0] == 0 )                    /* String is empty                   */
      return 'X';

   j = 0;
   if ( cr[j] == '+' || cr[j] == '-' )  /* ok on front of string             */
       j++;
   if ( cr[j] == '.' ) {
       i_CntDec++;
       j++; }
   for ( i = j; i < 1000; i++ ) {       /* go thru string                    */
     if ( cr[i] == 0 )                  /* end of string                     */
        break;
     if ( cr[i] >= '0' && cr[i] <= '9' ) {
        i_CntDig++;                     /* Count digits                      */
        continue; }
     if ( cr[i] == '.' ) {
        i_CntDec++;                     /* Count Decimal Points              */
        continue; }
     return 'X';  }                     /* Bad Char in string                */

   if ( i_CntDig == 0 )                 /* Need at least on digit            */
     return 'X';
   if ( i_CntDec > 1 )                  /* Can only have one Decimal         */
     return 'X';
   if ( i_CntDec == 1 )                 /* One Decimal Points, was Float     */
      return 'F' ;
   else
      return 'I' ;                      /* Was Integer                       */
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Display
* Desc: Used for testing
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void  ICF::Display ()
{
int i;
  printf ("------------------------------------------------\n");
  printf ("\n");
  printf ("Fuel Moistures Data \n");
  printf ("Model   1   10   100   LH   LW\n");
  for ( i = 0; i < iN_FMD; i++ ) {
    printf (" %3d  %3d  %3d  %3d  %3d  %3d\n",
    a_FMD[i].i_Model, a_FMD[i].i_TL1,  a_FMD[i].i_TL10,
    a_FMD[i].i_TL100, a_FMD[i].i_TLLH, a_FMD[i].i_TLLW );
  }

  printf ("\n");
  printf ("cr_CFF  %s %s \n", e_ICF_CFF, cr_CFF );
  printf ("cr_FMF  %s %s \n", e_ICF_FMF, cr_FMF );

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
  printf ("\n");
  printf ("-----------------------------------------------------------\n");
  printf ("       Weather  \n");

  printf ("cr_WeFN:  %s = %s \n", e_ICF_WeF, cr_WeFN);

  printf ("Weather Stream %d\n",iN_Wtr);
  printf ("    Mth  Day  Per  mTH  xTH   mT   xT  mH   xH   Elv  PST   PET\n");

  for ( i = 0; i < iN_Wtr; i++ ) {
    printf ("%2d %4d %4d %f  %4d %4d %f %f %4d %4d %4d %4d %4d\n",
    i,
    this->a_Wtr[i].i_Mth,
    this->a_Wtr[i].i_Day,
    this->a_Wtr[i].f_Per,
    this->a_Wtr[i].i_mTH,
    this->a_Wtr[i].i_xTH,
    this->a_Wtr[i].f_mT,
    this->a_Wtr[i].f_xT,
    this->a_Wtr[i].i_xH,
    this->a_Wtr[i].i_mH,
    this->a_Wtr[i].i_Elv,
    this->a_Wtr[i].i_PST,
    this->a_Wtr[i].i_PET);  }


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
  printf ("----------------------------------------------------\n");
  printf ("\n");
  printf ("      Wind \n");
  printf ("\n");
  printf ("cr_WiFN %s >%s< \n", e_ICF_WiF, cr_WiFN);
  printf ("\n");
  printf ("cr_WiDU %s >%s< \n", e_ICF_WiDU, cr_WiDU);
  printf ("\n");

  printf ("Wind Stream %d \n", iN_Wnd);
    printf ("      Mth  Day   Hr  Spd  Dir CloCov\n");
  for ( i = 0; i < iN_Wnd; i++ ) {
    printf ("%3d   %2d   %2d  %4d  %f   %3d   %d\n",
    i,
    this->a_Wnd[i].i_Mth,
    this->a_Wnd[i].i_Day,
    this->a_Wnd[i].i_Hr,
    this->a_Wnd[i].f_Spd,
    this->a_Wnd[i].i_Dir,
    this->a_Wnd[i].i_CloCov);
  }


  printf ("------------------------------------------------\n");

  printf ("f_WinDir %f \n",     f_WinDir );
  printf ("f_WinSpe %f \n",     f_WinSpe );

  printf ("i_GriWinRes %s %f\n", e_ICF_GWR,  f_GriWinRes);
  printf ("f_GriWinHei %s %f\n", e_ICF_GWH,  f_GriWinHei);
  printf ("cr_GriWinVeg %s %s\n", e_ICF_GWV, cr_GriWinVeg);

  printf ("\n");
  printf ("Conditioning Period End\n");
  printf ("i_Mth  %d\n", i_MthEnd);
  printf ("i_Day  %d\n", i_DayEnd);
  printf ("i_Hour %d\n", i_HourEnd);
  //printf ("i_Min  %d\n", i_Min);

  printf ("\n");
  printf ("\n");

  printf ("cr_WDF  %s %s \n", e_ICF_GWDF, cr_GWDF );
  printf ("cr_WSF  %s %s \n", e_ICF_GWSF, cr_GWSF );


  printf ("cr_GGW  %s %s \n", e_ICF_GGW, cr_GGW );


  printf ("\n");
  printf ("cr_Line  %s \n",     cr_Line  );
  printf ("iX       %d \n",     iX       );


   printf ("\n");
   printf ("cr_CroFirMet: %s, %s\n",e_ICF_CFM,  cr_CroFirMet);
   printf ("i_NumPro    : %s, %d\n",e_ICF_NP ,  i_NumPro    );
   printf ("i_FolMoi    : %s, %d\n",e_ICF_FMC,  i_FolMoi    );
   printf ("i_SDFN      : %s, %d\n",e_ICF_SDN,  i_SDFN      );
   printf ("i_SDFM      : %s, %d\n",e_ICF_SDM,  i_SDFM      );

   printf ("\n");
   printf ("MTT Minimum Travel Time....\n");
   printf ("f_Res       : %s, %f\n", e_ICF_RES, f_Res   );
   printf ("i_SimTim    : %s, %d\n", e_ICF_ST , i_SimTim);
   printf ("i_TraPth    : %s, %d\n", e_ICF_TPI, i_TraPth);
   printf ("cr_IFN      : %s, %s\n", e_ICF_IFN, cr_IFN  );


   printf ("\n");
   printf ("Output Grids.........\n");
   printf ("FLAMELENGTH    %d \n", i_FL);
   printf ("SPREADRATE     %d \n",  i_SP);
   printf ("INTENSITY      %d \n",  i_IN);
   printf ("HEATAREA       %d \n",  i_HE);
   printf ("CROWNSTATE     %d \n",  i_CR);
   printf ("SOLARRADIATION %d \n",  i_SO);
   printf ("FUELMOISTURE1  %d \n",  i_FU1);
   printf ("FUELMOISTURE10 %d \n",  i_FU10);
   printf ("MIDFLAME       %d \n",  i_MI);
   printf ("HORIZRATE      %d \n",  i_HO);
   printf ("MAXSPREADDIR   %d \n",  i_MAD);
   printf ("ELLIPSEDIM_A   %d \n",  i_ELA);
   printf ("ELLIPSEDIM_B   %d \n",  i_ELB);
   printf ("ELLIPSEDIM_C   %d \n",  i_ELC);
   printf ("MAXSPOT        %d \n",  i_MAS);
   printf ("MAXSPOT_DIR    %d \n",  i_MASDir);
   printf ("MAXSPOT_DX     %d \n",  i_MASDx);

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Init
* Desc: Just init everything, do this before doing anything else
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void  ICF::Init ()
{
	//time_t now;
	//time( &now );
	i_SpottingSeed = -1;

// Leap-Year-Fix
   this->b_LeapYear = false ;


   b_hasValidGriddedWindsFiles = false;
	  a_customFuels = NULL;
	  iN_CustomFuels = 0;

   strcpy (cr_CustFuelUnits,""); /* Custom Fuels Units              */


   f_WinDir = ef_ICFInit;
   f_WinSpe = ef_ICFInit;

   f_GriWinRes = ef_ICFInit;
   f_GriWinHei = ef_ICFInit;
   strcpy (cr_GriWinVeg,"");

   i_MthEnd =  ei_ICFInit;
   i_DayEnd =  ei_ICFInit;
   i_HourEnd = ei_ICFInit;
   i_MthStart =  ei_ICFInit;
   i_DayStart =  ei_ICFInit;
   i_HourStart = ei_ICFInit;
//   i_Min =  ei_ICFInit;
   strcpy (cr_ROSAdjustFile,"");
   strcpy (cr_CFF,"");
   strcpy (cr_GWDF,"");
   strcpy (cr_GWSF,"");
   strcpy (cr_FMF,"");
   strcpy (cr_WiFN,"");
   strcpy (cr_WeFN,"");

   strcpy (cr_GGW ,"");

   strcpy (cr_WiDU,"");          /* Wind Strem Units                  */

   a_Wtr = NULL;                  /* Pointer to Weather stream array structs  */
   iN_Wtr = 0;

   a_Wnd = NULL;                 /* Pointer to Wind stream array structs  */
   iN_Wnd = 0;

   a_FMD = NULL;                 /* Pointer to Fuel Moisture array structs  */
   iN_FMD = 0;

/* Newer type weather stream input data */
   a_RAWS = NULL;                /* Weather/Wind Stream Data */
   iN_RAWS = 0;                  /* # of records */
   i_RAWSElev = 0;

   strcpy (cr_CroFirMet,"");       /* CROWN_FIRE_METHOD:                       */
   i_NumPro =  -1;                 /* Number of Processors, NUMBER_PROCESSORS: */
   i_FolMoi = ei_ICFInit;          /* FOLIAR_MOISTURE_CONTENT:                 */
   i_SDFN = ei_ICFInit;            /* SPREAD_DIRECTION_FROM_NORTH:             */
   i_SDFM = ei_ICFInit;            /* SPREAD_DIRECTION_FROM_MAX:               */

   f_Res = 90;             //ef_ICFInit;        /* MTT_RESOLUTION:                            */
   i_SimTim = 0;          //ei_ICFInit;         /* MTT_SIM_TIME:                              */
   i_TraPth = 500;        //ei_ICFInit;         /* MTT_TRAVEL_PATH_INTERVAL:                  */
   strcpy (cr_IFN,"");			            	   /* MTT_IGNITION_FILE:                      */
   strcpy(cr_BarrierFileName, "");	      /* MTT_BARRIER_FILE:                       */
   i_MTT_FillBarriers = true;		          /* MTT_FILL_BARRIERS:                      */
  	f_MTT_SpotProbability = 0.1;	         /* MTT_SPOT_PROBABILITY:                   */

	//TOM stuff
	strcpy(iTrt_idealLcpName, "");
	strcpy(iTrt_ignitionFileName, "");
	iTrt_iterations = 1;				//TOM iterations
	iTrt_dimension = 500.0;
	iTrt_fraction = 1.0;
	iTrt_resolution = 30.0;
	iTrtOpportunitiesOnly = 0;


	i_FL  = e_No;                 /* Output Grid Layers                       */
   i_SP  = e_No;
   i_IN  = e_No;
   i_HE  = e_No;
   i_CR  = e_No;
   i_SO  = e_No;
   i_FU1  = e_No;
   i_FU10  = e_No;
   i_MI  = e_No;
   i_HO  = e_No;
   i_MAD  = e_No;
   i_ELA = e_No;
   i_ELB = e_No;
   i_ELC = e_No;
   i_MAS  = e_No;
   i_MASDir  = e_No;
   i_MASDx  = e_No;

   strcpy (cr_Line,"");

   iX = 0;
   fh = NULL;
   strcpy (cr_ErrMes,"");
   strcpy (cr_ErrExt,"");
   f_analysisEast = f_analysisWest = f_analysisNorth = f_analysisSouth = 0.0;
   b_eastSet = b_westSet = b_northSet = b_southSet = false;
   i_useMemoryLCP = i_useMemoryOutputs = 1;
   strcpy(cr_storagePath, "");


   strcpy (cr_FarsiteIgnition,"");    /* MTT_IGNITION_FILE:           */
   strcpy (cr_FarsiteBarrier,"");     /* MTT_BARRIERS_FILE:            */

   f_FarsiteSpotGridResolution = ei_ICFInit;       /* Spot Grid Resolution */
   f_FarsiteSpotProb = 0.0;          /* Spotting probability (0.0 - 1.0)                       */
   f_FarsiteDistanceRes = 30;          /* Distance resolution                       */
   f_FarsitePerimeterRes = 60;          /* Perimeter resolution                       */
   f_FarsiteActualTimeStep = 60;          /* ActualTimeStep (minutes)                      */
   f_FarsiteVisibleTimeStep = 120;          /* VisibleTimeStep (minutes)                      */
   f_FarsiteSecondaryVisibleTimeStep = 240;          /* SecondaryVisibleTimeStep (minutes)                      */
   i_FarsiteStartMth = ei_ICFInit;                /* Farsite Start                    */
   i_FarsiteStartDay = ei_ICFInit;                /* month day time,                            */
   i_FarsiteStartHour = ei_ICFInit;               /* hour:minute = time = military              */
   i_FarsiteEndMth = ei_ICFInit;                /* Farsite End                    */
   i_FarsiteEndDay = ei_ICFInit;                /* month day time,                            */
   i_FarsiteEndHour = ei_ICFInit;               /* hour:minute = time = military              */
  	f_FarsiteSpotIgnitionDelay = 0.0;            //ignition delay for spots
	  i_FarsiteAccelerationOn = 1;				//Acceleration on by default
	  i_FarsiteFillBarriers = 1;					//Barrier fill on by default
	  f_FarsiteMinSpotDistance = 0.0;
	f_FarsiteMinIgnitionVertexDistance = 1.0;
// Farsite-WindNinja
   strcpy (this->s_FWNI.cr_GriWin,e_GW_No);
   this->s_FWNI.f_GriWinHei = ef_ICFInit;
   this->s_FWNI.f_GriWinRes = ef_ICFInit;
   this->s_FWNI.f_Longitude = ef_ICFInit;
   this->s_FWNI.i_TimeZone  = ei_ICFInit;

   iN_BurnPeriods = 0;
	a_BurnPeriods = NULL;
//FSIM additions
	strcpy(cr_FSIM_lcpFile,"");
	strcpy(cr_FSIM_fmdFile,"");
	strcpy(cr_FSIM_rosAdjustFile,"");
	strcpy(cr_FSIM_fms97File,"");
	strcpy(cr_FSIM_fms90File,"");
	strcpy(cr_FSIM_fms80File,"");
	strcpy(cr_FSIM_igProbFile,"");
	strcpy(cr_FSIM_friskFile,"");
	strcpy(cr_FSIM_burnProbFile,"");
	strcpy(cr_FSIM_barrierFile,"");
	strcpy(cr_FSIM_fireDayDistFile,"");
	strcpy(cr_FSIM_ercStreamFile,"");
	strcpy(cr_FSIM_progressFile, "");
	//strcpy(cr_FSIM_firesShapeFile,"");
	i_FSIM_firesShapeFile = 0;

	i_FSIM_crownFireMethod = 0;//0 = Finney, 1 = ScottRheinhart
	f_FSIM_resolution = 30.0;
	i_FSIM_threadsPerFire = 1;
	i_FSIM_numSims = 0;
	i_FSIM_julianStart = 30;
	i_FSIM_gridDistanceUnits = 0;
	i_FSIM_suppression = 0; //0 = false, 1 = true
	f_FSIM_fireSizeLimit = -1.0;
	i_FSIM_record = 0; // 0 = false, 1 = true

	//FCONST additions
	strcpy(cr_FCONST_fmsFile, "");
	strcpy(cr_FCONST_outputsName, "");
	strcpy(cr_FCONST_fireListFile, "");
	strcpy(cr_FCONST_Envision_fireListFile, "");
	i_FCONST_numFires = 0;
	i_FCONST_windSpeed = 0;
	i_FCONST_windDir = 0;
	i_FCONST_ignitionMultiplier = 1;
	f_FCONST_spotProbability = 0.0;

	i_NodeSpread_NumLat = 6;
	i_NodeSpread_NumVert = 3;

	Init_WDI (&s_WDI);  /* Init Diurnal inputs */
}



/****************************************************************************/
/* Init the Windninja Diurnal Inputs Struct */
void ICF::Init_WDI (d_WDI *a_WDI)
{
  strcpy (a_WDI->cr_YesNo, e_GW_No);
  a_WDI->i_Sec = ei_ICFInit;
  a_WDI->i_Min = ei_ICFInit;
  a_WDI->i_Hour = ei_ICFInit;
  a_WDI->i_Mth = ei_ICFInit;
  a_WDI->i_Day = ei_ICFInit;
  a_WDI->i_Year = ei_ICFInit;
  a_WDI->i_Year = ei_ICFInit;
  a_WDI->i_TimeZone = ei_ICFInit;
  a_WDI->f_AirTemp = ef_ICFInit;
  a_WDI->f_CloudCov = ef_ICFInit;
  a_WDI->f_Longitude = ef_ICFInit;

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ErrorMessage
* Desc: Form the error message for the specified number sent in
*
*   In: i_Num...error number
*  Ret: pointer to error messaage text
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
char *ICF::ErrorMessage (int i_Num)
{
int i;

  sprintf(this->cr_ErrMes,"%d = Unknown Error Message Number",i_Num);       /* in case bad err num sent in  */

  for ( i = 0; i < 1000; i++ ) {             /* look for the err text in tbl */
    if ( !strcmp (sr_EMS[i].cr_Sw,"End") )
      break;

    if ( sr_EMS[i].i_ErrNum != i_Num )
      continue;

	//block removed SB 2009/08/28 at Hans'request
    //if ( strcmp (sr_EMS[i].cr_Sw,"")){   /* use switch text when present    */
      ///FromSwNa (sw1, sr_EMS[i].cr_Sw);
      //sprintf (this->cr_ErrMes, "Switch = %s\n%s", sw1,sr_EMS[i].cr_ErrMes); }
    //else
      sprintf (this->cr_ErrMes,"%s",sr_EMS[i].cr_ErrMes);

    if ( strcmp (this->cr_ErrExt,""))  /* app any extra detail err mess txt */
       strcat (this->cr_ErrMes," ");
       strcat (this->cr_ErrMes,this->cr_ErrExt);

    break;
   }

   return this->cr_ErrMes;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Remove_CrLf
* Desc: Remove and carrage returns of line feeds by nulling them out.
*        it doesn't matter which one comes first it the string
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void ICF::Remove_CrLf (char cr[])
{
   StrRepChr (cr, '\n', '\0'); /* replace any car ret, line feeds   */
   StrRepChr (cr, '\r', '\0'); /*  with nulls                       */
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: isBlank
* Desc: See if a line contains only blanks or is empty.
*   In: cr......string
*  Ret: 1....Blank Line
*       0....not Empty
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int   ICF::isBlank (char  cr[] )
{
int i,j;
    j = strlen (cr);
	for ( i = 0; i <= j; i++ ) {
      if ( cr[i] == 0 )
        return 1;
      if ( cr[i] == ' ' )
        continue ;
      if ( cr[i] == '\n' )
        continue ;
      return 0;
    }
    return 0;   /* Shouldn't get here */
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: DeleteAll
* Desc: Delete any 'new' memory
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void ICF::DeleteAll ()
{
  if ( a_Wtr != NULL ) {
   delete[] a_Wtr;
   a_Wtr = NULL; }

  if ( a_Wnd != NULL ) {
   delete[] a_Wnd;
   a_Wnd = NULL; }

  if ( a_FMD != NULL ) {
   delete[] a_FMD;
   a_FMD = NULL; }

  if ( a_RAWS != NULL ) {       /* Weather Stream Data */
    delete[] a_RAWS;
    a_RAWS = NULL; }

  if (a_customFuels != NULL) {
	   delete[] a_customFuels;
	   a_customFuels = NULL; }

	if( a_BurnPeriods != NULL)
	{
		delete[] a_BurnPeriods;
		a_BurnPeriods = NULL;
		iN_BurnPeriods = 0;
	}

}
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
ICF::ICF()
{
 ICF::Init();
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
ICF::~ICF()
{
   ICF::DeleteAll ();
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GridWindGenerate
* Desc: see if the the Gridded Winds Generate switch is set to yes
*  Ret: 1 = set to yes, else 0
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int ICF::GridWindGenerate()
{
  if ( strcmp (this->cr_GGW,e_GW_Yes))
    return 1;
  return 0;
}
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GridWindResolution
* Desc: Get the gridded wind resolution or tell caller that it is set to
*        missing - meaning it wasn't set by the user in the command file
*  Ret: resolution or = ei_ICFInit if missing/no set by user
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
float ICF::GridWindResolution()
{
   return this->f_GriWinRes;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: GridWindHeight
* Desc: Get the gridded wind Height or tell caller that it is set to
*        missing - meaning it wasn't set by the user in the command file
*  Ret: resolution,  or = ef_ICFInit  if missing/no set by user
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
float  ICF::GridWindHeight()
{
   return this->f_GriWinHei;
}

/* Get the Wind Direction and Wind speed */
float ICF::WindDirection (){  return this->f_WinDir;}
float ICF::WindSpeed () { return this->f_WinSpe; }

int  ICF::GridWindGrass()
{
  if ( !strcmp(this->cr_GriWinVeg, e_GWV_Grass))
	return 1;
  return 0;
}
int  ICF::GridWindBrush()
{
  if ( !strcmp(this->cr_GriWinVeg, e_GWV_Brush))
	return 1;
  return 0;
}

/***********************************************************************
* Desc: See if it's Feb 29 the leap year day
*        This is a common place for functions to use as a check, the
*        current thinking is to not allow Feb 29, but if we rethink
*        this function might come in handle to allow Feb 29 using
*        the return value
*   Ret: 1 yes it's leap year day
*        0 NO it's not
**********************************************************************/
int  ICF::isLeapYearDay (int i_Mth, int i_Day)
{
  if ( i_Mth == 2 && i_Day == 29 )
    return 1;
  return 0;
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
int ICF::GetMCDate(int mth, int day, int year)
{
int i;
   i = ICF::CheckMonthDay(mth, day);
   if ( i == 0 )
     return 0;

  if ( year != e_StrYear )
      i += e_DaysPerYear;
   return i;
}

/*************************************************************
* Name: GetTotMin - Get Total Minutes
* Desc: for the given date/time get the total number of minutes
*       e_StrYear.
*   In: mth, day,
*       hr - military time 0000 -> 2359
*       year - e_StrYear or e_StrYear + 1
************************************************************/
int ICF::GetTotMin(int mth, int day, int hr, int year)
{
int i, i_Min, i_Time;
 i =  GetMCDate(mth,day,year); /* Day in a 2 year period */
 i_Time = i * e_MinPerDay;
 i_Min = MilToMin (hr);      /* Military time to total minutes */
 i = i_Time + i_Min;               /* minutes from start of 2 yr period */
 return i;
}

/*****************************************************************
* Name: GetWthTblDate
* Desc: From the loaded weather table get the specifed date
*       Either weather table or none could be load. But you're
*       probably coming here becuase you know you have weather data.
*   In: iX...0 -> N, N = last index
*            To get last entry in table, send in a neg number
*  Out: month, day, year
*  Ret: 1 ok, 0 no weather data found
******************************************************************/
int ICF::GetWthDate(int iX ,int *m, int *d, int *y, int *t )
{
int i;
  *m = *d = *y = *t = 0;
  i = iX;
  if ( this->a_Wtr != NULL ) {
     if ( i < 0 || i >= this->iN_Wtr )
        i = this->iN_Wtr - 1;
     *m = this->a_Wtr[i].i_Mth;
     *d = this->a_Wtr[i].i_Day;
     *y = this->a_Wtr[i].i_Year;
     *t = 0;     /* daily data doesn't have an exact time */
     return 1; }

  if ( this->a_RAWS != NULL ) {
     if ( i < 0 || i >= this->iN_RAWS )
        i = this->iN_RAWS - 1;
     *m = this->a_RAWS[i].i_Mth;
     *d = this->a_RAWS[i].i_Day;
     *y = this->a_RAWS[i].i_Yr;
     *t = this->a_RAWS[i].i_Time;
     return 1; }

  return 0;

}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BurnPeriodData
* Desc: Find the Burn Periods switch and if found load its data.
*       Checks for valid number values, and for the right amount of
*       data, doesn't check limits.
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::BurnPeriodData(int *ai)
{
	#define e_bpfr 4
	float fr[e_bpfr];
	int i,j, i_Arg;
	/* find the switch and its arg......                                         */
	i = ICF::Set_SinIntArg ((char *)e_ICF_FARSITE_BURN_PERIODS,&i_Arg); /* FARSITE_BURN_PERIODS:                  */
	if ( i == -1 )                        /* Optional switch so not an error   */
		return 1;                          /*  if none found                    */
   strcpy (this->cr_ErrExt,"");

  if ( i == 0 )
  {                       /* Switch found with bad argument    */
    strcpy (this->cr_ErrExt,"Invalid/Missing Switch argument");
    *ai = e_EMS_FARSITE_BURN_PERIODS;
    return 0;
  }
/* Check the switch argument....                                             */
  if ( i_Arg < 1)// || i_Arg > e_MaxWiD){   /* Arg is num lines of wind data  */
  {
    sprintf (this->cr_ErrExt,"Invalid argument value: %d", i_Arg);
    *ai = e_EMS_FARSITE_BURN_PERIODS;
    return 0;
  }

  this->iN_BurnPeriods = i_Arg;                   /* number of                         */
  this->a_BurnPeriods = new d_BurnPeriod[iN_BurnPeriods];         /* allocate BurnPeriods           */

/*.......................................................................... */
  for ( i = 0; i < i_Arg; i++ ) {     /* Read each line's data into fr      */
     j = ICF::GetLine (fr,e_bpfr, this->cr_ErrExt, this->fh);
     if ( j  == 0 ) {                 /* If error                           */
       *ai = e_EMS_FARSITE_BURN_PERIODS;
       return 0; }

/* copy data into class                                                      */
	 this->a_BurnPeriods[i].i_Month      = (int) fr[0];
     this->a_BurnPeriods[i].i_Day        = (int) fr[1];

// Leap-Year-Fix, move post Feb 28 days ahead one day on leap year
     this->Set_NextLeapDay(& this->a_BurnPeriods[i].i_Month, &this->a_BurnPeriods[i].i_Day);
//------------------

	 this->a_BurnPeriods[i].i_HourStart  = (int) fr[2];
	 this->a_BurnPeriods[i].i_HourEnd    = (int) fr[3];

/* Added Mar-14-2012 - get the Year by finding this mth day in the weather data table */
     this->a_BurnPeriods[i].i_Year = this->GetWeatherYear(this->a_BurnPeriods[i].i_Month,this->a_BurnPeriods[i].i_Day);
  } /* for i */

/* Try to get one extra line, shouldn't get one, number of lines needs       */
/*  specified by the switch argument                                         */
   j = ICF::GetLine (fr,e_bpfr, this->cr_ErrExt, this->fh);
   if ( j == 1 ){
     sprintf(this->cr_ErrExt,"Extra line of data found, switch specifies %d", i_Arg);
     *ai = e_EMS_FARSITE_BURN_PERIODS;
     return 0; }
   strcpy (this->cr_ErrExt,"");  /* not actual error, so blank this out */
   return 1;
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_Date
* Desc: Find switch then following arguments in format as
*         Mt/Dy Hr:Mn   or   Mt Dy Hr Min
*         12/19 1020    or   12 19 10 20  are both ok
* NOTE: values are set into the approbriate class variables but only
*       check for being valid number not check for limits
*  Ret: 1 = switch and arg found ok, or switch not found
*       0 = error - bad numeric arg found, or not correct number of args
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_Date (char cr_Sw[])
{
int i, i_Mth, i_Day, i_Year;
char Mth[15],Day[15],Hour[15];
char cr[eC_InpLin];

   if ( !ICF::GetSw (cr_Sw) )       /* Find the Switch in input file     */
     return 1;                          /* Didn't find it,                   */

  Mth[0] = Day[0] = Hour[0] = 0;

/* Just take out the separators if they a there, if they are already blank   */
/*  chars separating them - ok                                               */
  strcpy (cr,&this->cr_Line[this->iX]);
  StrRepChr (cr, '/', ' ');   /* replace mth day separator         */

  sscanf (cr,"%10s %10s %10s",Mth,Day,Hour);

/* Here we just make sure that they are valid numbers                        */
  i = 0;
  if ( Get_NumTyp (Mth) != 'I' )
    i++;
  if ( Get_NumTyp (Day) != 'I' )
    i++;
  if ( Get_NumTyp (Hour) != 'I' )
    i++;

  if ( i > 0 ) {
    strcpy (this->cr_ErrExt,&this->cr_Line[this->iX]);
    return 0; }

  i_Mth  =  atoi(Mth);
  i_Day =  atoi(Day);

// Leap-Year-Fix NOTE NOTE NOTE --- I also change all those atoi(Mth) atoi(Day) below
  this->Set_NextLeapDay(&i_Mth, &i_Day);


  i_Year = ICF::GetWeatherYear(i_Mth,i_Day); /* Get year from Weather table */
  if ( i_Year == 0 ) {   /* Date was not found in weather data */
    strcpy (this->cr_ErrExt,&this->cr_Line[this->iX]);
    return 0; }

/* Set the request date/time */
  if ( !strcmp (cr_Sw,e_ICF_FARSITE_START_TIME	)){
     this->i_FarsiteStartMth = i_Mth;    //  atoi(Mth);
     this->i_FarsiteStartDay = i_Day;    //   atoi(Day);
     this->i_FarsiteStartHour = atoi(Hour);
     this->i_FarsiteStartYear = i_Year; }

  else if ( !strcmp (cr_Sw,e_ICF_FARSITE_END_TIME )){
     this->i_FarsiteEndMth = i_Mth;  // atoi(Mth);
     this->i_FarsiteEndDay = i_Day;  //  atoi(Day);
     this->i_FarsiteEndHour = atoi(Hour);
     this->i_FarsiteEndYear = i_Year; }

  else if ( !strcmp (cr_Sw,e_ICF_CPE)){            /* FlamMap Ending Conditioning */
     this->i_MthEnd = i_Mth;   //  atoi(Mth);
     this->i_DayEnd = i_Day;   //  atoi(Day);
     this->i_HourEnd = atoi(Hour);
     this->i_YearEnd = i_Year; }

  else if ( !strcmp (cr_Sw,e_ICF_CPS)){           /* FlamMap Start Conditioning */
     this->i_MthStart = i_Mth;    // atoi(Mth);
     this->i_DayStart =  i_Day;    // atoi(Day);
     this->i_HourStart = atoi(Hour);
     this->i_YearStart = i_Year; }

  return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CustomFuelData
* Desc: Find the Custom Fuels Data switch and if found load its data.
*       Checks for valid number values, and for the right amount of
*       data, doesn't check limits.
*  Out: ai.....error number
*  Ret: 1 ok, 0 = error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::CustomFuelData  (int *ai)
{
	/* Look for Unit switch, if none set to english                              */
		if ( !ICF::Set_SinTxtArg ((char *)e_ICF_CFDU,this->cr_CustFuelUnits, eC_WSU) )
		 strcpy (this->cr_CustFuelUnits,"English");
	int i, i_Arg;
	strcpy (this->cr_ErrExt,"");
	i = ICF::Set_SinIntArg ((char *)e_ICF_CFD ,&i_Arg); // FUEL_MOISTURES_DATA:       //
	if ( i == -1)                        // either/or switch so not an error  //
		return 1;                          //  at this point, will chk mor latr //
	if ( i == 0 )
	{                       // Switch found with bad argument  //
		strcpy (this->cr_ErrExt,"Invalid/Missing Switch argument");
		*ai = e_EMS_CFD;
		return 0;
	}
	// Check the switch argument....                                            //
	if ( i_Arg < 1 || i_Arg > 256)
	{   // Arg is num lines of custom fuel models  //
		sprintf (this->cr_ErrExt,"Invalid argument value: %d", i_Arg);
		*ai = e_EMS_CFD;
		return 0;
	}


	iN_CustomFuels = i_Arg;
	a_customFuels = new d_CustomFuel[iN_CustomFuels];
	bool Metric = false;
	if(cr_CustFuelUnits[0] == 'M' || cr_CustFuelUnits[0] == 'm')
		Metric = true;
	//.......................................................................... //
	char buf[512];
	int j, nRead;
	  for ( i = 0; i < i_Arg; i++ )
	  {     /* Attempt to get a line of data      */
		 j = ICF::GetCharLine (buf, this->cr_ErrExt, this->fh);
		 if ( j  == 0 )
		 {                 /* If error                           */
		   *ai = e_EMS_CFD;
		   return 0;
		 }

		nRead = sscanf(buf, "%d %s %f %f %f %f %f %s %f %f %f %f %f %f %f %s",
			&a_customFuels[i].i_Model,
			a_customFuels[i].cr_code,
			&a_customFuels[i].f_h1,
			&a_customFuels[i].f_h10,
			&a_customFuels[i].f_h100,
			&a_customFuels[i].f_lh,
			&a_customFuels[i].f_lw,
			a_customFuels[i].dynamic,
			&a_customFuels[i].f_sl,
			&a_customFuels[i].f_slh,
			&a_customFuels[i].f_slw,
			&a_customFuels[i].f_depth,
			&a_customFuels[i].f_xmext,
			&a_customFuels[i].f_heatd,
			&a_customFuels[i].f_heatl,
			a_customFuels[i].cr_comment);
		if(nRead < 15)//comment is Optional!
		{
			*ai = e_EMS_CFD;
			return 0;
		}
		if(Metric)//go ahead and convert now...
		{
			a_customFuels[i].f_h1/=2.2417;
			a_customFuels[i].f_h10/=2.2417;
			a_customFuels[i].f_h100/=2.2417;
			a_customFuels[i].f_lh/=2.2417;
			a_customFuels[i].f_lw/=2.2417;
			a_customFuels[i].f_sl*=30.480060960;
			a_customFuels[i].f_slh*=30.480060960;
			a_customFuels[i].f_slw*=30.480060960;
			a_customFuels[i].f_depth/=30.480060960;
			a_customFuels[i].f_heatd/=2.324375;
			a_customFuels[i].f_heatl/=2.324375;
		}
	}
	  //check for extra line of custom fuels data...
	j = ICF::GetCharLine (buf, this->cr_ErrExt, this->fh);
	if ( j  != 0 )
	{                 /* If error                           */
		d_CustomFuel dummy;
		nRead = sscanf(buf, "%d %s %f %f %f %f %f %s %f %f %f %f %f %f %f %s",
			&dummy.i_Model,
			dummy.cr_code,
			&dummy.f_h1,
			&dummy.f_h10,
			&dummy.f_h100,
			&dummy.f_lh,
			&dummy.f_lw,
			dummy.dynamic,
			&dummy.f_sl,
			&dummy.f_slh,
			&dummy.f_slw,
			&dummy.f_depth,
			&dummy.f_xmext,
			&dummy.f_heatd,
			&dummy.f_heatl,
			dummy.cr_comment);
		if(nRead >= 15)//comment is Optional!
		{
			sprintf(this->cr_ErrExt,"Extra line of data found, switch specifies %d", i_Arg);
			*ai = e_EMS_CFD;
			return 0;
		}
	}
	return 1;
}

int ICF::CheckRAWSWeatherWind(int *ai)
{
	for(int i = 0; i < iN_RAWS; i++)
	{
		if(a_RAWS[i].f_Temp < -50.0 || a_RAWS[i].f_Temp > 150.0)
		{
			return e_EMS_RAWS_TEMP;
		}
		if(a_RAWS[i].f_Humidity < 0.0 || a_RAWS[i].f_Humidity > 100.0)
		{
			return e_EMS_RAWS_RH;
		}
		if(a_RAWS[i].f_PerHou < 0.0 || a_RAWS[i].f_PerHou > 10.0)
		{
			return e_EMS_RAWS_PCP;
		}
		if(a_RAWS[i].f_WinSpd < 0.0 || a_RAWS[i].f_WinSpd > 200.0)
		{
			return e_EMS_RAWS_WS;
		}
		if(a_RAWS[i].f_WinDir < 0.0 || a_RAWS[i].f_WinDir > 360.0)
		{
			return e_EMS_RAWS_WDIR;
		}
		if(a_RAWS[i].f_CloCov < 0.0 || a_RAWS[i].f_CloCov > 100.0)
		{
			return e_EMS_RAWS_CLOUDCOVER;
		}
	}
	return 1;
}

// Farsite-WindNinja
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: FarsiteWindNinja
* Desc: Get the Farsite switches need to run WindNinja - gridded winds
*       This function will return an error if a required switch isn't
*        found or has a bad argument
*       Limit checks are done later.
*   Ret: 1 = ok, else error code
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::FarsiteWindNinja ()
{
int i;

  strcpy (this->cr_ErrExt,"");  /* make sure this is blank */

/* 1st see if there is a switch to do gridded winds - run WindNinja, switch is optional */
  if ( !ICF::Set_SinTxtArg ((char *)e_ICF_GGW,this->s_FWNI.cr_GriWin,eC_GGW)){  /* GRIDDED_WINDS_GENERATE */
    strcpy (this->s_FWNI.cr_GriWin,e_GW_No);
    return 1; }
  if ( !strcmp (this->s_FWNI.cr_GriWin,e_GW_No) )  /* Not doing Grid Winds */
     return 1;
  if ( strcmp (this->s_FWNI.cr_GriWin,e_GW_Yes))    /* needs to be Yes or No */
    return ICF::CloseRet(e_EMS_GWW);                 /* error */

/* Winds Height switch is optional */
  i = ICF::Set_SinNumArg((char *)e_ICF_GWH,&this->s_FWNI.f_GriWinHei);  /* GRIDDED_WINDS_HEIGHT */
  if ( i == 0 )                        /* Switch found has invalid argument */
    return ICF::CloseRet(e_EMS_GWH);   /* return error */
  if ( i == -1 )
    this->s_FWNI.f_GriWinHei = e_GWH_Def;   /* Switch not found, so set default value */

/* Must have a Resolution */
   i = ICF::Set_SinNumArg((char *)e_ICF_GWR,&this->s_FWNI.f_GriWinRes); /* GRIDDED_WINDS_RESOLUTION */
   if ( i != 1 )
      return ICF::CloseRet(e_EMS_GWR);

/* Must have Longitude  -180.0 -> 180.0 */
//   i = ICF::Set_SinNumArg(e_ICF_FGWL,&this->s_FWNI.f_Longitude); /* GRIDDED_WINDS_LONGITUDE */
//   if ( i != 1 )
//     return ICF::CloseRet(e_EMS_FGWL);

/* Must have  -12->12, -7 =Mountain -6 =Mountain daylight sav time */
//  i = ICF::Set_SinIntArg (e_ICF_FGWT, &this->s_FWNI.i_TimeZone);
//  if ( i != 1 )
//    return ICF::CloseRet(e_EMS_FGWT);


/* Optional, Bining Range/Tolerances for wind speed & direction */
/* used to not rerun 'Ninja when inputs are close enough together */
  i = ICF::Set_SinNumArg((char *)e_ICF_FGWBD, &this->s_FWNI.f_BinWinDir);
  if ( i == -1 )               /* If No Bin Direction switch found use default */
    this->s_FWNI.f_BinWinDir = e_BinDirDef;

  i = ICF::Set_SinNumArg((char *)e_ICF_FGWBS, &this->s_FWNI.f_BinWinSpd);
  if ( i == -1 )               /* switch not found, so use default */
    this->s_FWNI.f_BinWinSpd = e_BinSpdDef;

  return 1;
}

/********************************************************************
*
*
********************************************************************/
int ICF::Set_NextLeapDay (int *ai_Mth, int *ai_Day)
{
int rd[] = {0, 31, 28, 31, 30, 31,30, 31, 31, 30, 31, 30, 31};

  if ( this->b_LeapYear == false ) /* Make sure we are dealing with a */
    return 0;                      /* Leap year, caller might not check*/

  if ( *ai_Mth == 1 )
    return 0;

  if ( *ai_Mth == 2 && *ai_Day <= 28 )
    return 0;

  if ( *ai_Mth == 2 && *ai_Day == 29 ) {
    *ai_Mth = 3;
    *ai_Day = 1;
     return 0; }

   *ai_Day = *ai_Day + 1;                      /* Next Day */
   if ( *ai_Day > rd[*ai_Mth] ){     /* Exceeds days in that month */
     *ai_Mth = *ai_Mth + 1;        /* set to first day of next mth */
     *ai_Day = 1; }


  return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: InputFCONST
* Desc: Input Command File - open, read in and do some basic checking
*
*   In: cr_PthFN....path and file name.
*  Ret: 1 = no error, else error number, see error message defines and table
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::InputFCONST (char cr_PthFN[])
{
  this->fh = fopen (cr_PthFN,"r");
  if ( this->fh == NULL ){
	  sprintf (this->cr_ErrExt,"%s\n",cr_PthFN);
	 return e_EMS_Fil;}

//landscape file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_LANDSCAPE, this->cr_FSIM_lcpFile, true, true)) /* landscape    */
    return ICF::CloseRet(e_EMS_Fil);

  ICF::Set_FSIM_FilNam ((char *)e_ICF_FCONST_ENVISION_FIRELIST, this->cr_FCONST_Envision_fireListFile, false, true); /* FireListFile    */

	 bool isEnvision = false;
	if(strlen(cr_FCONST_Envision_fireListFile) > 0)
	{
		if(!isFile(cr_FCONST_Envision_fireListFile))
			return e_EMS_Fil;
		isEnvision = true;
	}

	//Ignition Multiplier (only used for Envision Firelist files!!!!!
	if(isEnvision)
		ICF::Set_SinIntArg ((char *)e_ICF_FCONST_IGNITION_MULTIPLIER, &this->i_FCONST_ignitionMultiplier); //Ignition Multiplier

	ICF::Set_SinNumArg ((char *)e_ICF_FCONST_SPOT_PROBABILITY, &this->f_FCONST_spotProbability);         /* SpotProbability  */
	//fmd file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_CUSTOMFUELS, this->cr_FSIM_fmdFile, false, true)) /* customfmd    */
    return ICF::CloseRet(e_EMS_Fil);

//FMS file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FCONST_FMSFILE, this->cr_FCONST_fmsFile, !isEnvision, true)) /* FmsFile    */
    return ICF::CloseRet(e_EMS_Fil);

//FireList file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FCONST_FIRELISTFILE, this->cr_FCONST_fireListFile, false, true)) /* FireListFile    */
    return ICF::CloseRet(e_EMS_Fil);

  //Outputs Path
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FCONST_OUTPUTPATH, this->cr_FCONST_outputsName, !isEnvision, false)) /* OutputsName    */
    return ICF::CloseRet(e_EMS_Fil);

  //CrownFireMethod
	if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_CROWNFIRE, &this->i_FSIM_crownFireMethod)) //CrownFireMethod
		return ICF::CloseRet(e_EMS_FSIM_CROWNFIRE);

	//Resolution
    if (!ICF::Set_SinNumArg ((char *)e_ICF_FSIM_RESOLUTION, &this->f_FSIM_resolution)) //Resolution
	    	return ICF::CloseRet(e_EMS_FSIM_RESOLUTION);

	//ThreadsPerFire
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_THREADSPERFIRE, &this->i_FSIM_threadsPerFire)) //ThreadsPerFire
	    	return ICF::CloseRet(e_EMS_FSIM_THREADSPERFIRE);

	//Fires
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FCONST_NUMFIRES, &this->i_FCONST_numFires)) //NumFires
	    	return ICF::CloseRet(e_EMS_FCONST_NUMFIRES);

	//WindSpeed
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FCONST_WINDSPEED, &this->i_FCONST_windSpeed)) //WindSpeed
	    	return ICF::CloseRet(e_EMS_FCONST_WINDSPEED);

	//WindDir
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FCONST_WINDDIR, &this->i_FCONST_windDir)) //WindSpeed
	    	return ICF::CloseRet(e_EMS_FCONST_WINDDIR);

	//Duration
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FCONST_DURATION, &this->i_FCONST_duration)) //Duration
	    	return ICF::CloseRet(e_EMS_FCONST_DURATION);

//progress file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_PROGRESS_PATHNAME, this->cr_FSIM_progressFile, false, false)) /* ProgressFileName    */
    return ICF::CloseRet(e_EMS_Fil);

  //FiresShapeFile
  Set_SinIntArg ((char *)e_ICF_FSIM_FIRESSHAPEFILE, &this->i_FSIM_firesShapeFile);//) /* FiresShapeFile    */

  //NumLat
  Set_SinIntArg ((char *)e_ICF_NODESPREAD_NUMLAT, &this->i_NodeSpread_NumLat);//) /* NodeSpreadNumLat    */

  //NumVert
  Set_SinIntArg ((char *)e_ICF_NODESPREAD_NUMVERT, &this->i_NodeSpread_NumVert);//) /* NodeSpreadNumVert    */

  return ICF::CloseRet(1);              /* Close file and ret ok code.       */
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: InputFSIM
* Desc: Input Command File - open, read in and do some basic checking
*
*   In: cr_PthFN....path and file name.
*  Ret: 1 = no error, else error number, see error message defines and table
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  ICF::InputFSIM (char cr_PthFN[])
{
//int i;

  this->fh = fopen (cr_PthFN,"r");
  if ( this->fh == NULL ){
	  sprintf (this->cr_ErrExt,"%s\n",cr_PthFN);
	 return e_EMS_Fil;}

//landscape file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_LANDSCAPE, this->cr_FSIM_lcpFile, true, true)) /* landscape    */
    return ICF::CloseRet(e_EMS_Fil);

//fmd file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_CUSTOMFUELS, this->cr_FSIM_fmdFile, false, true)) /* customfmd    */
    return ICF::CloseRet(e_EMS_Fil);

//ROSAdjust file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_ROSADJUST, this->cr_FSIM_rosAdjustFile, false, true)) /* ROSAdjust    */
    return ICF::CloseRet(e_EMS_Fil);

//FMS97 file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_FMS97, this->cr_FSIM_fms97File, false, true)) /* FMS97    */
    return ICF::CloseRet(e_EMS_Fil);

//FMS90 file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_FMS90, this->cr_FSIM_fms90File, false, true)) /* FMS90    */
    return ICF::CloseRet(e_EMS_Fil);

//FMS80 file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_FMS80, this->cr_FSIM_fms80File, false, true)) /* FMS80    */
    return ICF::CloseRet(e_EMS_Fil);

//IgnitionProbabilityGrid file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_IGNITPROBGRID, this->cr_FSIM_igProbFile, false, true)) /* IgnitionProbabilityGrid    */
    return ICF::CloseRet(e_EMS_Fil);

//FriskFile
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_FRISKFILE, this->cr_FSIM_friskFile, true, true)) /* FriskFile    */
    return ICF::CloseRet(e_EMS_Fil);

//BurnProbabilityFile
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_BURNPROBFILE, this->cr_FSIM_burnProbFile, true, false)) /* BurnProbabilityFile    */
    return ICF::CloseRet(e_EMS_FSIM_OUTPUTSNAME);

//BarrierFile
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_BARRIERFILE, this->cr_FSIM_barrierFile, false, true)) /* BarrierFile    */
    return ICF::CloseRet(e_EMS_Fil);

//FireDayDistributionFile
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_FIREDAYFILE, this->cr_FSIM_fireDayDistFile, true, true)) /* FireDayDistributionFile    */
    return ICF::CloseRet(e_EMS_Fil);

//ErcStreamFile
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_ERCSTREAMFILE, this->cr_FSIM_ercStreamFile, false, true)) /* ErcStreamFile    */
    return ICF::CloseRet(e_EMS_Fil);

//progress file
  if ( !ICF::Set_FSIM_FilNam ((char *)e_ICF_FSIM_PROGRESS_PATHNAME, this->cr_FSIM_progressFile, false, false)) /* ProgressFileName    */
    return ICF::CloseRet(e_EMS_Fil);

  //FiresShapeFile
  Set_SinIntArg ((char *)e_ICF_FSIM_FIRESSHAPEFILE, &this->i_FSIM_firesShapeFile);//) /* FiresShapeFile    */
    //return ICF::CloseRet(e_EMS_Fil);

  //CrownFireMethod
	if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_CROWNFIRE, &this->i_FSIM_crownFireMethod)) //CrownFireMethod
		return ICF::CloseRet(e_EMS_FSIM_CROWNFIRE);

	//Resolution
    if (!ICF::Set_SinNumArg ((char *)e_ICF_FSIM_RESOLUTION, &this->f_FSIM_resolution)) //Resolution
	    	return ICF::CloseRet(e_EMS_FSIM_RESOLUTION);

	//ThreadsPerFire
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_THREADSPERFIRE, &this->i_FSIM_threadsPerFire)) //ThreadsPerFire
	    	return ICF::CloseRet(e_EMS_FSIM_THREADSPERFIRE);

	//NumSimulations
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_NUMSIMS, &this->i_FSIM_numSims)) //NumSimulations
	    	return ICF::CloseRet(e_EMS_FSIM_NUMSIMS);

	//JulianStart
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_JULIANSTART, &this->i_FSIM_julianStart)
		&& !ICF::Set_SinIntArg ((char *)e_ICF_FSIM_JULAINSTART, &this->i_FSIM_julianStart)) //JulianStart
	    	return ICF::CloseRet(e_EMS_FSIM_JULIANSTART);

	//GridDistanceUnits
    if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_GRIDUNITS, &this->i_FSIM_gridDistanceUnits)) //GridDistanceUnits
	    	return ICF::CloseRet(e_EMS_FSIM_GRIDUNITS);

	//Suppression
	if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_SUPPRESSION, &this->i_FSIM_suppression)) //Suppression
	    	return ICF::CloseRet(e_EMS_FSIM_SUPPRESSION);

	//FireSizeLimit
	if (!ICF::Set_SinNumArg ((char *)e_ICF_FSIM_FIRESIZELIMIT, &this->f_FSIM_fireSizeLimit)) //FireSizeLimit
	    	return ICF::CloseRet(e_EMS_FSIM_FIRESIZELIMIT);

	//Record
	if (!ICF::Set_SinIntArg ((char *)e_ICF_FSIM_RECORD, &this->i_FSIM_record)) //Record
	    	return ICF::CloseRet(e_EMS_FSIM_RECORD);

  //NumLat
  Set_SinIntArg ((char *)e_ICF_NODESPREAD_NUMLAT, &this->i_NodeSpread_NumLat);//) /* NodeSpreadNumLat    */

  //NumVert
  Set_SinIntArg ((char *)e_ICF_NODESPREAD_NUMVERT, &this->i_NodeSpread_NumVert);//) /* NodeSpreadNumVert    */

  return ICF::CloseRet(1);              /* Close file and ret ok code.       */

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Set_FSIM_FilNam
* Desc: Find the specified path file name switch in the input file,
*       check the switch argument (file name) to make sure the file is
*       there, then set path file name into the ICF class
*   In: cr_Sw......switch to look for
*	In: mandatorySW .... is switch mandatory?
*	In: isInput .... is switch input (0) or output (1)?
*  Out: a_ICF_FN...address in the ICF class to put path file name
*  Ret: 1 OK = mandatory switch found and input file name is good or non-mandatory switch not found
*       0 Error - mandatory switch not found or switch found but can't find input file
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int ICF::Set_FSIM_FilNam (char cr_Sw[], char a_ICF_FN[], bool mandatorySW, bool isInput)
{
  strcpy (a_ICF_FN,"");

  if ( !ICF::GetSw (cr_Sw) )            /* find the switch                   */
  {
	  if(mandatorySW)
		  return 0;						//mandatory Switch not found!
	  else
		return 1;                           /* Switch not found                  */
  }

  strcpy (a_ICF_FN, &this->cr_Line[iX]);/* This is where the arg now is */
  Trim_LT (a_ICF_FN);                   /* trim any lead and tail blnks     */

  if(!mandatorySW && strcmp(a_ICF_FN, "-1") == 0)
	  return 1;
  if ( isInput && !isFile (a_ICF_FN)) {            /* see if file is there */
    sprintf (this->cr_ErrExt,"command file switch and file name:\n%s\n",this->cr_Line);
    return 0; }

  return 1;
}
