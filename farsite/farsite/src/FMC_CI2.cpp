/*********************************************************************************
* Name: fmc_ci2.cpp - FlamMap Input Class
* Desc: This file is just an extension of fmc_ci.cpp.
*       A place to put the bigger and/or misc. functions.
*
*
**********************************************************************************/

#define _CRT_SECURE_NO_WARNINGS

#ifdef WIN32
#include <windows.h>
#include <conio.h>
#else
#include <stdio.h>
#endif

#include "cdtlib.h"
#include "newfms.h"

#include "semtime.h"
#include "deadfuelmoisture.h"

#include "FMC_CI.h"    /* FlamMap Input Class */
#include "FMC_FE2.h"    /* FireEnvironment2 Class */

#include "FMC_CFMC.h"



/***********************************************************************
* Name: GetMoistCalcInteraval
* Desc: MoistCalcInterval[][]  gets loaded in FMI::Init()
*       These intervals are use break the moisture map into groups
*       and timing for fuel size class simulations
*  These are the Categories
*  FM_INTERVAL_TIME  0
*  FM_INTERVAL_ELEV 1
*  FM_INTERVAL_SLOPE 2
*  FM_INTERVAL_ASP  3
*  FM_INTERVAL_COV  4
*
*  NOTE----> See FMI::Init() where table gets loaded.
************************************************************************/
long  CI::GetMoistCalcInterval (long FM_SIZECLASS, long CATEGORY)
{
long l;

   l = MoistCalcInterval [FM_SIZECLASS] [CATEGORY];
   return l;
 }

/***********************************************************************
* Name:
* Desc:
*
************************************************************************/
long CI::GetTheme_HiValue(short DataTheme)
{
long hi;

  switch ( DataTheme ) {
     	case 0: hi = l_ElevHigh; break;
     	case 1: hi = l_SlopeHigh; break;


//     	case 2: hi = Header.hiaspect; break;
//     	case 3: hi = Header.hifuel; break;
      case 4: hi = l_CoverHigh; break;
//     	case 5: hi = Header.hiheight; break;
//    	case 6: hi = Header.hibase; break;
//     	case 7: hi = Header.hidensity; break;
//     	case 8: hi = Header.hiduff; break;
//     	case 9: hi = Header.hiwoody; break; }

     default:
        LogicError ((char *)"CI::GetTheme_HiValue() fix this");
        return 0;
        break;
    }

     return hi;
}
/********************************************************************/
long CI::GetTheme_LoValue (short DataTheme)
{
long lo;

  switch (DataTheme) {
	     case 0: lo = l_ElevLow; break;
     	case 1: lo = l_SlopeLow; break;


//     	case 2: lo=Header.loaspect; break;
//     	case 3: lo=Header.lofuel; break;
    	case 4: lo = l_CoverLow; break;
//     	case 5: lo=Header.loheight; break;
//     	case 6: lo=Header.lobase; break;
//     	case 7: lo=Header.lodensity; break;
//     	case 8: lo=Header.loduff; break;
//     	case 9: lo=Header.lowoody; break;}

      default:
        LogicError ((char *)"CI::GetTheme_LoValue() fix this");
        return 0;
        break;
   }


     return lo;
}

/***********************************************************************
* Name:
* Desc:
*
************************************************************************/
int   CI::Set_Elev (long High, long Low, short ThemeUnit)
{
  l_ElevHigh = High;
  l_ElevLow = Low;
  s_EUnits =  ThemeUnit;
  return 1;   /* for now return ok, might do some err chkin later */
}

/***********************************************************************
* Name:
* Desc:
*
************************************************************************/
int CI::Set_Slope (long High, long Low, short ThemeUnit)
{
  l_SlopeHigh = High;
  l_SlopeLow = Low;
  s_SUnits =  ThemeUnit;
  return 1;   /* for now return ok, might do some err chkin later */
}

int CI::Set_Cover (long High, long Low, short ThemeUnit)
{
  l_CoverHigh = High;
  l_CoverLow = Low;
  s_CUnits = ThemeUnit;
  return 1;
}

/************************************************************************/
short	CI::GetTheme_Units (short DataTheme)
{

		switch (DataTheme) {
    case 0: return s_EUnits;
			 case 1: return s_SUnits;
//			 case 2: return Header.AUnits;
//			 case 3: return Header.FOptions;
			 case 4: return s_CUnits;
//			 case 5: return Header.HUnits;
//			 case 6: return Header.BUnits;
//			 case 7: return Header.PUnits;
//			 case 8: return Header.DUnits;
//			 case 9: return Header.WOptions;
		 }
 printf ("FIX THIS CI::GetTheme_Units() \n");
 // _getch();

		 return 0;
	}



/*************************************************************************
* Name: SetMoistures
* Desc: Set a Fuel Moisture Model into fm[],
*       These are the Fuel Models in the FlamMap Inputs file
* NOTE: SEE CI::SetAllMoistures()
**************************************************************************/
int CI::SetMoistures(int fuelModel, int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody)
{
  if(fuelModel < 1 || fuelModel > 256)
		  return 0;
	 fm[fuelModel - 1].TL1 = _fm1;
	 fm[fuelModel - 1].TL10 = _fm10;
  fm[fuelModel - 1].TL100 = _fm100;
	 fm[fuelModel - 1].TLLH = _fmHerb;
	 fm[fuelModel - 1].TLLW = _fmWoody;
	 if (fm[fuelModel - 1].TL1 > 1 && fm[fuelModel - 1].TL10 > 1)
		  fm[fuelModel - 1].FuelMoistureIsHere=true;
	 else
	  	fm[fuelModel - 1].FuelMoistureIsHere=false;

 	for (int k=0; k<4; k++)  {     // only up to 1000 hr fuels [3]
                  for( int j=0; j<eC_Sta; j++)
		   	EnvtChanged[k][j]=true; }
	return 1;
}

/*************************************************************************
* Name: SetAllMoistures
* Desc: Set every record in the fm[] with the given values, this is
*        used to set the defualt values into the entire table before
*        inserting the individual Fuel Models
*
**************************************************************************/
void CI::SetAllMoistures(int _fm1, int _fm10, int _fm100,	int _fmHerb, int _fmWoody)
{
	long i, j, k;
	for(i = 0; i <= 256; i++)	{
		 fm[i].TL1 = _fm1;
		 fm[i].TL10 = _fm10;
		 fm[i].TL100 = _fm100;
		 fm[i].TLLH = _fmHerb;
		 fm[i].TLLW = _fmWoody;
		 if (fm[i].TL1 > 1 && fm[i].TL10 > 1 )
			  fm[i].FuelMoistureIsHere=true;
 		else
		  	fm[i].FuelMoistureIsHere=false;

		for ( k = 0; k<4; k++)  {     // only up to 1000 hr fuels [3]
                         for (j=0; j<eC_Sta; j++)
				  EnvtChanged[k][j]=true;} }
}





/**************************************************************************
* Name: InitializeNewFuel
* Desc:
*
***************************************************************************/
/*void CI::InitializeNewFuel()
{
     newfuels[1].number=1;
     strcpy(newfuels[1].code, "FM1");
     newfuels[1].h1=0.740000;
     newfuels[1].h10=0.000000;
     newfuels[1].h100=0.000000;
     newfuels[1].lh=0.000000;
     newfuels[1].lw=0.000000;
     newfuels[1].dynamic=0;
     newfuels[1].sav1=3500;
     newfuels[1].savlh=1800;
     newfuels[1].savlw=1500;
     newfuels[1].depth=1.000000;
     newfuels[1].xmext=0.120000;
     newfuels[1].heatd=8000.000000;
     newfuels[1].heatl=8000.000000;
     strcpy(newfuels[1].desc, "Short Grass");

     newfuels[2].number=2;
     strcpy(newfuels[2].code, "FM2");
     newfuels[2].h1=2.000000;
     newfuels[2].h10=1.000000;
     newfuels[2].h100=0.500000;
     newfuels[2].lh=0.000000;
     newfuels[2].lw=0.500000;
     newfuels[2].dynamic=0;
     newfuels[2].sav1=3000;
     newfuels[2].savlh=1800;
     newfuels[2].savlw=1500;
     newfuels[2].depth=1.000000;
     newfuels[2].xmext=0.150000;
     newfuels[2].heatd=8000.000000;
     newfuels[2].heatl=8000.000000;
     strcpy(newfuels[2].desc, "Timber Grass/Understory");

     newfuels[3].number=3;
     strcpy(newfuels[3].code, "FM3");
     newfuels[3].h1=3.010000;
     newfuels[3].h10=0.000000;
     newfuels[3].h100=0.000000;
     newfuels[3].lh=0.000000;
     newfuels[3].lw=0.000000;
     newfuels[3].dynamic=0;
     newfuels[3].sav1=1500;
     newfuels[3].savlh=1800;
     newfuels[3].savlw=1500;
     newfuels[3].depth=2.500000;
     newfuels[3].xmext=0.250000;
     newfuels[3].heatd=8000.000000;
     newfuels[3].heatl=8000.000000;
     strcpy(newfuels[3].desc, "Tall Grass");

     newfuels[4].number=4;
     strcpy(newfuels[4].code, "FM4");
     newfuels[4].h1=5.010000;
     newfuels[4].h10=4.010000;
     newfuels[4].h100=2.000000;
     newfuels[4].lh=0.000000;
     newfuels[4].lw=5.010000;
     newfuels[4].dynamic=0;
     newfuels[4].sav1=2000;
     newfuels[4].savlh=1800;
     newfuels[4].savlw=1500;
     newfuels[4].depth=6.000000;
     newfuels[4].xmext=0.200000;
     newfuels[4].heatd=8000.000000;
     newfuels[4].heatl=8000.000000;
     strcpy(newfuels[4].desc, "Chaparral");

     newfuels[5].number=5;
     strcpy(newfuels[5].code, "FM5");
     newfuels[5].h1=1.000000;
     newfuels[5].h10=0.500000;
     newfuels[5].h100=0.000000;
     newfuels[5].lh=0.000000;
     newfuels[5].lw=2.000000;
     newfuels[5].dynamic=0;
     newfuels[5].sav1=2000;
     newfuels[5].savlh=1800;
     newfuels[5].savlw=1500;
     newfuels[5].depth=2.000000;
     newfuels[5].xmext=0.200000;
     newfuels[5].heatd=8000.000000;
     newfuels[5].heatl=8000.000000;
     strcpy(newfuels[5].desc, "Short Brush");

     newfuels[6].number=6;
     strcpy(newfuels[6].code, "FM6");
     newfuels[6].h1=1.500000;
     newfuels[6].h10=2.500000;
     newfuels[6].h100=2.000000;
     newfuels[6].lh=0.000000;
     newfuels[6].lw=0.000000;
     newfuels[6].dynamic=0;
     newfuels[6].sav1=1750;
     newfuels[6].savlh=1800;
     newfuels[6].savlw=1500;
     newfuels[6].depth=2.500000;
     newfuels[6].xmext=0.250000;
     newfuels[6].heatd=8000.000000;
     newfuels[6].heatl=8000.000000;
     strcpy(newfuels[6].desc, "Dormant Brush");

     newfuels[7].number=7;
     strcpy(newfuels[7].code, "FM7");
     newfuels[7].h1=1.130000;
     newfuels[7].h10=1.870000;
     newfuels[7].h100=1.500000;
     newfuels[7].lh=0.000000;
     newfuels[7].lw=0.370000;
     newfuels[7].dynamic=0;
     newfuels[7].sav1=1550;
     newfuels[7].savlh=1800;
     newfuels[7].savlw=1500;
     newfuels[7].depth=2.500000;
     newfuels[7].xmext=0.400000;
     newfuels[7].heatd=8000.000000;
     newfuels[7].heatl=8000.000000;
     strcpy(newfuels[7].desc, "Southern Rough");

     newfuels[8].number=8;
     strcpy(newfuels[8].code, "FM8");
     newfuels[8].h1=1.500000;
     newfuels[8].h10=1.000000;
     newfuels[8].h100=2.500000;
     newfuels[8].lh=0.000000;
     newfuels[8].lw=0.000000;
     newfuels[8].dynamic=0;
     newfuels[8].sav1=2000;
     newfuels[8].savlh=1800;
     newfuels[8].savlw=1500;
     newfuels[8].depth=0.200000;
     newfuels[8].xmext=0.300000;
     newfuels[8].heatd=8000.000000;
     newfuels[8].heatl=8000.000000;
     strcpy(newfuels[8].desc, "Closed Timber Litter");

     newfuels[9].number=9;
     strcpy(newfuels[9].code, "FM9");
     newfuels[9].h1=2.920000;
     newfuels[9].h10=0.410000;
     newfuels[9].h100=0.150000;
     newfuels[9].lh=0.000000;
     newfuels[9].lw=0.000000;
     newfuels[9].dynamic=0;
     newfuels[9].sav1=2500;
     newfuels[9].savlh=1800;
     newfuels[9].savlw=1500;
     newfuels[9].depth=0.200000;
     newfuels[9].xmext=0.250000;
     newfuels[9].heatd=8000.000000;
     newfuels[9].heatl=8000.000000;
     strcpy(newfuels[9].desc, "Hardwood Litter");

     newfuels[10].number=10;
     strcpy(newfuels[10].code, "FM10");
     newfuels[10].h1=3.010000;
     newfuels[10].h10=2.000000;
     newfuels[10].h100=5.010000;
     newfuels[10].lh=0.000000;
     newfuels[10].lw=2.000000;
     newfuels[10].dynamic=0;
     newfuels[10].sav1=2000;
     newfuels[10].savlh=1800;
     newfuels[10].savlw=1500;
     newfuels[10].depth=1.000000;
     newfuels[10].xmext=0.250000;
     newfuels[10].heatd=8000.000000;
     newfuels[10].heatl=8000.000000;
     strcpy(newfuels[10].desc, "Timber Litter/Understory");

     newfuels[11].number=11;
     strcpy(newfuels[11].code, "FM11");
     newfuels[11].h1=1.500000;
     newfuels[11].h10=4.510000;
     newfuels[11].h100=5.510000;
     newfuels[11].lh=0.000000;
     newfuels[11].lw=0.000000;
     newfuels[11].dynamic=0;
     newfuels[11].sav1=1500;
     newfuels[11].savlh=1800;
     newfuels[11].savlw=1500;
     newfuels[11].depth=1.000000;
     newfuels[11].xmext=0.150000;
     newfuels[11].heatd=8000.000000;
     newfuels[11].heatl=8000.000000;
     strcpy(newfuels[11].desc, "Light Slash");

     newfuels[12].number=12;
     strcpy(newfuels[12].code, "FM12");
     newfuels[12].h1=4.010000;
     newfuels[12].h10=14.030000;
     newfuels[12].h100=16.530000;
     newfuels[12].lh=0.000000;
     newfuels[12].lw=0.000000;
     newfuels[12].dynamic=0;
     newfuels[12].sav1=1500;
     newfuels[12].savlh=1800;
     newfuels[12].savlw=1500;
     newfuels[12].depth=2.300000;
     newfuels[12].xmext=0.200000;
     newfuels[12].heatd=8000.000000;
     newfuels[12].heatl=8000.000000;
     strcpy(newfuels[12].desc, "Medium Slash");

     newfuels[13].number=13;
     strcpy(newfuels[13].code, "FM13");
     newfuels[13].h1=7.010000;
     newfuels[13].h10=23.040000;
     newfuels[13].h100=28.050000;
     newfuels[13].lh=0.000000;
     newfuels[13].lw=0.000000;
     newfuels[13].dynamic=0;
     newfuels[13].sav1=1500;
     newfuels[13].savlh=1800;
     newfuels[13].savlw=1500;
     newfuels[13].depth=3.000000;
     newfuels[13].xmext=0.250000;
     newfuels[13].heatd=8000.000000;
     newfuels[13].heatl=8000.000000;
     strcpy(newfuels[13].desc, "Heavy Slash");

	newfuels[99].number=99;
     strcpy(newfuels[99].code, "NB9");
     strcpy(newfuels[99].desc, "Barren");

	newfuels[98].number=98;
     strcpy(newfuels[98].code, "NB8");
     strcpy(newfuels[98].desc, "Water");

	newfuels[93].number=93;
     strcpy(newfuels[93].code, "NB3");
     strcpy(newfuels[93].desc, "Snow or Ice");

	newfuels[92].number=92;
     strcpy(newfuels[92].code, "NB2");
     strcpy(newfuels[92].desc, "Agricultural or Cropland");

	newfuels[91].number=91;
     strcpy(newfuels[91].code, "NB1");
     strcpy(newfuels[91].desc, "Urban or Developed");

     newfuels[101].number=101;
     strcpy(newfuels[101].code, "GR1");
     newfuels[101].h1=0.100000;
     newfuels[101].h10=0.000000;
     newfuels[101].h100=0.000000;
     newfuels[101].lh=0.300000;
     newfuels[101].lw=0.000000;
     newfuels[101].dynamic=1;
     newfuels[101].sav1=2200;
     newfuels[101].savlh=2000;
     newfuels[101].savlw=1500;
     newfuels[101].depth=0.400000;
     newfuels[101].xmext=0.150000;
     newfuels[101].heatd=8000.000000;
     newfuels[101].heatl=8000.000000;
     strcpy(newfuels[101].desc, "Short, sparse, dry climate grass");

     newfuels[102].number=102;
     strcpy(newfuels[102].code, "GR2");
     newfuels[102].h1=0.100000;
     newfuels[102].h10=0.000000;
     newfuels[102].h100=0.000000;
     newfuels[102].lh=1.000000;
     newfuels[102].lw=0.000000;
     newfuels[102].dynamic=1;
     newfuels[102].sav1=2000;
     newfuels[102].savlh=1800;
     newfuels[102].savlw=1500;
     newfuels[102].depth=1.000000;
     newfuels[102].xmext=0.150000;
     newfuels[102].heatd=8000.000000;
     newfuels[102].heatl=8000.000000;
     strcpy(newfuels[102].desc, "Low load, dry climate grass");

     newfuels[103].number=103;
     strcpy(newfuels[103].code, "GR3");
     newfuels[103].h1=0.100000;
     newfuels[103].h10=0.400000;
     newfuels[103].h100=0.000000;
     newfuels[103].lh=1.500000;
     newfuels[103].lw=0.000000;
     newfuels[103].dynamic=1;
     newfuels[103].sav1=1500;
     newfuels[103].savlh=1300;
     newfuels[103].savlw=1500;
     newfuels[103].depth=2.000000;
     newfuels[103].xmext=0.300000;
     newfuels[103].heatd=8000.000000;
     newfuels[103].heatl=8000.000000;
     strcpy(newfuels[103].desc, "Low load, very coarse, humid climate grass");

     newfuels[104].number=104;
     strcpy(newfuels[104].code, "GR4");
     newfuels[104].h1=0.250000;
     newfuels[104].h10=0.000000;
     newfuels[104].h100=0.000000;
     newfuels[104].lh=1.900000;
     newfuels[104].lw=0.000000;
     newfuels[104].dynamic=1;
     newfuels[104].sav1=2000;
     newfuels[104].savlh=1800;
     newfuels[104].savlw=1500;
     newfuels[104].depth=2.000000;
     newfuels[104].xmext=0.150000;
     newfuels[104].heatd=8000.000000;
     newfuels[104].heatl=8000.000000;
     strcpy(newfuels[104].desc, "Moderate load, dry climate grass");

     newfuels[105].number=105;
     strcpy(newfuels[105].code, "GR5");
     newfuels[105].h1=0.400000;
     newfuels[105].h10=0.000000;
     newfuels[105].h100=0.000000;
     newfuels[105].lh=2.500000;
     newfuels[105].lw=0.000000;
     newfuels[105].dynamic=1;
     newfuels[105].sav1=1800;
     newfuels[105].savlh=1600;
     newfuels[105].savlw=1500;
     newfuels[105].depth=1.500000;
     newfuels[105].xmext=0.400000;
     newfuels[105].heatd=8000.000000;
     newfuels[105].heatl=8000.000000;
     strcpy(newfuels[105].desc, "Low load, humid climate grass");

     newfuels[106].number=106;
     strcpy(newfuels[106].code, "GR6");
     newfuels[106].h1=0.100000;
     newfuels[106].h10=0.000000;
     newfuels[106].h100=0.000000;
     newfuels[106].lh=3.400000;
     newfuels[106].lw=0.000000;
     newfuels[106].dynamic=1;
     newfuels[106].sav1=2200;
     newfuels[106].savlh=2000;
     newfuels[106].savlw=1500;
     newfuels[106].depth=1.500000;
     newfuels[106].xmext=0.400000;
     newfuels[106].heatd=9000.000000;
     newfuels[106].heatl=9000.000000;
     strcpy(newfuels[106].desc, "Moderate load, humid climate grass");

     newfuels[107].number=107;
     strcpy(newfuels[107].code, "GR7");
     newfuels[107].h1=1.000000;
     newfuels[107].h10=0.000000;
     newfuels[107].h100=0.000000;
     newfuels[107].lh=5.400000;
     newfuels[107].lw=0.000000;
     newfuels[107].dynamic=1;
     newfuels[107].sav1=2000;
     newfuels[107].savlh=1800;
     newfuels[107].savlw=1500;
     newfuels[107].depth=3.000000;
     newfuels[107].xmext=0.150000;
     newfuels[107].heatd=8000.000000;
     newfuels[107].heatl=8000.000000;
     strcpy(newfuels[107].desc, "High load, dry climate grass");

     newfuels[108].number=108;
     strcpy(newfuels[108].code, "GR8");
     newfuels[108].h1=0.500000;
     newfuels[108].h10=1.000000;
     newfuels[108].h100=0.000000;
     newfuels[108].lh=7.300000;
     newfuels[108].lw=0.000000;
     newfuels[108].dynamic=1;
     newfuels[108].sav1=1500;
     newfuels[108].savlh=1300;
     newfuels[108].savlw=1500;
     newfuels[108].depth=4.000000;
     newfuels[108].xmext=0.300000;
     newfuels[108].heatd=8000.000000;
     newfuels[108].heatl=8000.000000;
     strcpy(newfuels[108].desc, "High load, very coarse, humid climate grass");

     newfuels[109].number=109;
     strcpy(newfuels[109].code, "GR9");
     newfuels[109].h1=1.000000;
     newfuels[109].h10=1.000000;
     newfuels[109].h100=0.000000;
     newfuels[109].lh=9.000000;
     newfuels[109].lw=0.000000;
     newfuels[109].dynamic=1;
     newfuels[109].sav1=1800;
     newfuels[109].savlh=1600;
     newfuels[109].savlw=1500;
     newfuels[109].depth=5.000000;
     newfuels[109].xmext=0.400000;
     newfuels[109].heatd=8000.000000;
     newfuels[109].heatl=8000.000000;
     strcpy(newfuels[109].desc, "Very high load, humid climate grass");

     newfuels[121].number=121;
     strcpy(newfuels[121].code, "GS1");
     newfuels[121].h1=0.200000;
     newfuels[121].h10=0.000000;
     newfuels[121].h100=0.000000;
     newfuels[121].lh=0.500000;
     newfuels[121].lw=0.650000;
     newfuels[121].dynamic=1;
     newfuels[121].sav1=2000;
     newfuels[121].savlh=1800;
     newfuels[121].savlw=1800;
     newfuels[121].depth=0.900000;
     newfuels[121].xmext=0.150000;
     newfuels[121].heatd=8000.000000;
     newfuels[121].heatl=8000.000000;
     strcpy(newfuels[121].desc, "Low load, dry climate grass-shrub");

     newfuels[122].number=122;
     strcpy(newfuels[122].code, "GS2");
     newfuels[122].h1=0.500000;
     newfuels[122].h10=0.500000;
     newfuels[122].h100=0.000000;
     newfuels[122].lh=0.600000;
     newfuels[122].lw=1.000000;
     newfuels[122].dynamic=1;
     newfuels[122].sav1=2000;
     newfuels[122].savlh=1800;
     newfuels[122].savlw=1800;
     newfuels[122].depth=1.500000;
     newfuels[122].xmext=0.150000;
     newfuels[122].heatd=8000.000000;
     newfuels[122].heatl=8000.000000;
     strcpy(newfuels[122].desc, "Moderate load, dry climate grass-shrub");

     newfuels[123].number=123;
     strcpy(newfuels[123].code, "GS3");
     newfuels[123].h1=0.300000;
     newfuels[123].h10=0.250000;
     newfuels[123].h100=0.000000;
     newfuels[123].lh=1.450000;
     newfuels[123].lw=1.250000;
     newfuels[123].dynamic=1;
     newfuels[123].sav1=1800;
     newfuels[123].savlh=1600;
     newfuels[123].savlw=1600;
     newfuels[123].depth=1.800000;
     newfuels[123].xmext=0.400000;
     newfuels[123].heatd=8000.000000;
     newfuels[123].heatl=8000.000000;
     strcpy(newfuels[123].desc, "Moderate load, humid climate grass-shrub");

     newfuels[124].number=124;
     strcpy(newfuels[124].code, "GS4");
     newfuels[124].h1=1.900000;
     newfuels[124].h10=0.300000;
     newfuels[124].h100=0.100000;
     newfuels[124].lh=3.400000;
     newfuels[124].lw=7.100000;
     newfuels[124].dynamic=1;
     newfuels[124].sav1=1800;
     newfuels[124].savlh=1600;
     newfuels[124].savlw=1600;
     newfuels[124].depth=2.100000;
     newfuels[124].xmext=0.400000;
     newfuels[124].heatd=8000.000000;
     newfuels[124].heatl=8000.000000;
     strcpy(newfuels[124].desc, "High load, humid climate grass-shrub");

     newfuels[141].number=141;
     strcpy(newfuels[141].code, "SH1");
     newfuels[141].h1=0.250000;
     newfuels[141].h10=0.250000;
     newfuels[141].h100=0.000000;
     newfuels[141].lh=0.150000;
     newfuels[141].lw=1.300000;
     newfuels[141].dynamic=1;
     newfuels[141].sav1=2000;
     newfuels[141].savlh=1800;
     newfuels[141].savlw=1600;
     newfuels[141].depth=1.000000;
     newfuels[141].xmext=0.150000;
     newfuels[141].heatd=8000.000000;
     newfuels[141].heatl=8000.000000;
     strcpy(newfuels[141].desc, "Low load, dry climate shrub");

     newfuels[142].number=142;
     strcpy(newfuels[142].code, "SH2");
     newfuels[142].h1=1.350000;
     newfuels[142].h10=2.400000;
     newfuels[142].h100=0.750000;
     newfuels[142].lh=0.000000;
     newfuels[142].lw=3.850000;
     newfuels[142].dynamic=0;
     newfuels[142].sav1=2000;
     newfuels[142].savlh=1800;
     newfuels[142].savlw=1600;
     newfuels[142].depth=1.000000;
     newfuels[142].xmext=0.150000;
     newfuels[142].heatd=8000.000000;
     newfuels[142].heatl=8000.000000;
     strcpy(newfuels[142].desc, "Moderate load, dry climate shrub");

     newfuels[143].number=143;
     strcpy(newfuels[143].code, "SH3");
     newfuels[143].h1=0.450000;
     newfuels[143].h10=3.000000;
     newfuels[143].h100=0.000000;
     newfuels[143].lh=0.000000;
     newfuels[143].lw=6.200000;
     newfuels[143].dynamic=0;
     newfuels[143].sav1=1600;
     newfuels[143].savlh=1800;
     newfuels[143].savlw=1400;
     newfuels[143].depth=2.400000;
     newfuels[143].xmext=0.400000;
     newfuels[143].heatd=8000.000000;
     newfuels[143].heatl=8000.000000;
     strcpy(newfuels[143].desc, "Moderate load, humid climate shrub");

     newfuels[144].number=144;
     strcpy(newfuels[144].code, "SH4");
     newfuels[144].h1=0.850000;
     newfuels[144].h10=1.150000;
     newfuels[144].h100=0.200000;
     newfuels[144].lh=0.000000;
     newfuels[144].lw=2.550000;
     newfuels[144].dynamic=0;
     newfuels[144].sav1=2000;
     newfuels[144].savlh=1800;
     newfuels[144].savlw=1600;
     newfuels[144].depth=3.000000;
     newfuels[144].xmext=0.300000;
     newfuels[144].heatd=8000.000000;
     newfuels[144].heatl=8000.000000;
     strcpy(newfuels[144].desc, "Low load, humid climate timber-shrub");

     newfuels[145].number=145;
     strcpy(newfuels[145].code, "SH5");
     newfuels[145].h1=3.600000;
     newfuels[145].h10=2.100000;
     newfuels[145].h100=0.000000;
     newfuels[145].lh=0.000000;
     newfuels[145].lw=2.900000;
     newfuels[145].dynamic=0;
     newfuels[145].sav1=750;
     newfuels[145].savlh=1800;
     newfuels[145].savlw=1600;
     newfuels[145].depth=6.000000;
     newfuels[145].xmext=0.150000;
     newfuels[145].heatd=8000.000000;
     newfuels[145].heatl=8000.000000;
     strcpy(newfuels[145].desc, "High load, dry climate shrub");

     newfuels[146].number=146;
     strcpy(newfuels[146].code, "SH6");
     newfuels[146].h1=2.900000;
     newfuels[146].h10=1.450000;
     newfuels[146].h100=0.000000;
     newfuels[146].lh=0.000000;
     newfuels[146].lw=1.400000;
     newfuels[146].dynamic=0;
     newfuels[146].sav1=750;
     newfuels[146].savlh=1800;
     newfuels[146].savlw=1600;
     newfuels[146].depth=2.000000;
     newfuels[146].xmext=0.300000;
     newfuels[146].heatd=8000.000000;
     newfuels[146].heatl=8000.000000;
     strcpy(newfuels[146].desc, "Low load, humid climate shrub");

     newfuels[147].number=147;
     strcpy(newfuels[147].code, "SH7");
     newfuels[147].h1=3.500000;
     newfuels[147].h10=5.300000;
     newfuels[147].h100=2.200000;
     newfuels[147].lh=0.000000;
     newfuels[147].lw=3.400000;
     newfuels[147].dynamic=0;
     newfuels[147].sav1=750;
     newfuels[147].savlh=1800;
     newfuels[147].savlw=1600;
     newfuels[147].depth=6.000000;
     newfuels[147].xmext=0.150000;
     newfuels[147].heatd=8000.000000;
     newfuels[147].heatl=8000.000000;
     strcpy(newfuels[147].desc, "Very high load, dry climate shrub");

     newfuels[148].number=148;
     strcpy(newfuels[148].code, "SH8");
     newfuels[148].h1=2.050000;
     newfuels[148].h10=3.400000;
     newfuels[148].h100=0.850000;
     newfuels[148].lh=0.000000;
     newfuels[148].lw=4.350000;
     newfuels[148].dynamic=0;
     newfuels[148].sav1=750;
     newfuels[148].savlh=1800;
     newfuels[148].savlw=1600;
     newfuels[148].depth=3.000000;
     newfuels[148].xmext=0.400000;
     newfuels[148].heatd=8000.000000;
     newfuels[148].heatl=8000.000000;
     strcpy(newfuels[148].desc, "High load, humid climate shrub");

     newfuels[149].number=149;
     strcpy(newfuels[149].code, "SH9");
     newfuels[149].h1=4.500000;
     newfuels[149].h10=2.450000;
     newfuels[149].h100=0.000000;
     newfuels[149].lh=1.550000;
     newfuels[149].lw=7.000000;
     newfuels[149].dynamic=1;
     newfuels[149].sav1=750;
     newfuels[149].savlh=1800;
     newfuels[149].savlw=1500;
     newfuels[149].depth=4.400000;
     newfuels[149].xmext=0.400000;
     newfuels[149].heatd=8000.000000;
     newfuels[149].heatl=8000.000000;
     strcpy(newfuels[149].desc, "Very high load, humid climate shrub");

     newfuels[161].number=161;
     strcpy(newfuels[161].code, "TU1");
     newfuels[161].h1=0.200000;
     newfuels[161].h10=0.900000;
     newfuels[161].h100=1.500000;
     newfuels[161].lh=0.200000;
     newfuels[161].lw=0.900000;
     newfuels[161].dynamic=1;
     newfuels[161].sav1=2000;
     newfuels[161].savlh=1800;
     newfuels[161].savlw=1600;
     newfuels[161].depth=0.600000;
     newfuels[161].xmext=0.200000;
     newfuels[161].heatd=8000.000000;
     newfuels[161].heatl=8000.000000;
     strcpy(newfuels[161].desc, "Light load, dry climate timber-grass-shrub");

     newfuels[162].number=162;
     strcpy(newfuels[162].code, "TU2");
     newfuels[162].h1=0.950000;
     newfuels[162].h10=1.800000;
     newfuels[162].h100=1.250000;
     newfuels[162].lh=0.000000;
     newfuels[162].lw=0.200000;
     newfuels[162].dynamic=0;
     newfuels[162].sav1=2000;
     newfuels[162].savlh=1800;
     newfuels[162].savlw=1600;
     newfuels[162].depth=1.000000;
     newfuels[162].xmext=0.300000;
     newfuels[162].heatd=8000.000000;
     newfuels[162].heatl=8000.000000;
     strcpy(newfuels[162].desc, "Moderate load, humid climate timber-shrub");

     newfuels[163].number=163;
     strcpy(newfuels[163].code, "TU3");
     newfuels[163].h1=1.100000;
     newfuels[163].h10=0.150000;
     newfuels[163].h100=0.250000;
     newfuels[163].lh=0.650000;
     newfuels[163].lw=1.100000;
     newfuels[163].dynamic=1;
     newfuels[163].sav1=1800;
     newfuels[163].savlh=1600;
     newfuels[163].savlw=1400;
     newfuels[163].depth=1.300000;
     newfuels[163].xmext=0.300000;
     newfuels[163].heatd=8000.000000;
     newfuels[163].heatl=8000.000000;
     strcpy(newfuels[163].desc, "Moderate load, humid climate timber-grass-shrub");

     newfuels[164].number=164;
     strcpy(newfuels[164].code, "TU4");
     newfuels[164].h1=4.500000;
     newfuels[164].h10=0.000000;
     newfuels[164].h100=0.000000;
     newfuels[164].lh=0.000000;
     newfuels[164].lw=2.000000;
     newfuels[164].dynamic=0;
     newfuels[164].sav1=2300;
     newfuels[164].savlh=1800;
     newfuels[164].savlw=2000;
     newfuels[164].depth=0.500000;
     newfuels[164].xmext=0.120000;
     newfuels[164].heatd=8000.000000;
     newfuels[164].heatl=8000.000000;
     strcpy(newfuels[164].desc, "Dwarf conifer with understory");

     newfuels[165].number=165;
     strcpy(newfuels[165].code, "TU5");
     newfuels[165].h1=4.000000;
     newfuels[165].h10=4.000000;
     newfuels[165].h100=3.000000;
     newfuels[165].lh=0.000000;
     newfuels[165].lw=3.000000;
     newfuels[165].dynamic=0;
     newfuels[165].sav1=1500;
     newfuels[165].savlh=1800;
     newfuels[165].savlw=750;
     newfuels[165].depth=1.000000;
     newfuels[165].xmext=0.250000;
     newfuels[165].heatd=8000.000000;
     newfuels[165].heatl=8000.000000;
     strcpy(newfuels[165].desc, "Very high load, dry climate timber-shrub");

     newfuels[181].number=181;
     strcpy(newfuels[181].code, "TL1");
     newfuels[181].h1=1.000000;
     newfuels[181].h10=2.200000;
     newfuels[181].h100=3.600000;
     newfuels[181].lh=0.000000;
     newfuels[181].lw=0.000000;
     newfuels[181].dynamic=0;
     newfuels[181].sav1=2000;
     newfuels[181].savlh=1800;
     newfuels[181].savlw=1600;
     newfuels[181].depth=0.200000;
     newfuels[181].xmext=0.300000;
     newfuels[181].heatd=8000.000000;
     newfuels[181].heatl=8000.000000;
     strcpy(newfuels[181].desc, "Low load, compact conifer litter");

     newfuels[182].number=182;
     strcpy(newfuels[182].code, "TL2");
     newfuels[182].h1=1.400000;
     newfuels[182].h10=2.300000;
     newfuels[182].h100=2.200000;
     newfuels[182].lh=0.000000;
     newfuels[182].lw=0.000000;
     newfuels[182].dynamic=0;
     newfuels[182].sav1=2000;
     newfuels[182].savlh=1800;
     newfuels[182].savlw=1600;
     newfuels[182].depth=0.200000;
     newfuels[182].xmext=0.250000;
     newfuels[182].heatd=8000.000000;
     newfuels[182].heatl=8000.000000;
     strcpy(newfuels[182].desc, "Low load broadleaf litter");

     newfuels[183].number=183;
     strcpy(newfuels[183].code, "TL3");
     newfuels[183].h1=0.500000;
     newfuels[183].h10=2.200000;
     newfuels[183].h100=2.800000;
     newfuels[183].lh=0.000000;
     newfuels[183].lw=0.000000;
     newfuels[183].dynamic=0;
     newfuels[183].sav1=2000;
     newfuels[183].savlh=1800;
     newfuels[183].savlw=1600;
     newfuels[183].depth=0.300000;
     newfuels[183].xmext=0.200000;
     newfuels[183].heatd=8000.000000;
     newfuels[183].heatl=8000.000000;
     strcpy(newfuels[183].desc, "Moderate load confier litter");

     newfuels[184].number=184;
     strcpy(newfuels[184].code, "TL4");
     newfuels[184].h1=0.500000;
     newfuels[184].h10=1.500000;
     newfuels[184].h100=4.200000;
     newfuels[184].lh=0.000000;
     newfuels[184].lw=0.000000;
     newfuels[184].dynamic=0;
     newfuels[184].sav1=2000;
     newfuels[184].savlh=1800;
     newfuels[184].savlw=1600;
     newfuels[184].depth=0.400000;
     newfuels[184].xmext=0.250000;
     newfuels[184].heatd=8000.000000;
     newfuels[184].heatl=8000.000000;
     strcpy(newfuels[184].desc, "Small downed logs");

     newfuels[185].number=185;
     strcpy(newfuels[185].code, "TL5");
     newfuels[185].h1=1.150000;
     newfuels[185].h10=2.500000;
     newfuels[185].h100=4.400000;
     newfuels[185].lh=0.000000;
     newfuels[185].lw=0.000000;
     newfuels[185].dynamic=0;
     newfuels[185].sav1=2000;
     newfuels[185].savlh=1800;
     newfuels[185].savlw=1600;
     newfuels[185].depth=0.600000;
     newfuels[185].xmext=0.250000;
     newfuels[185].heatd=8000.000000;
     newfuels[185].heatl=8000.000000;
     strcpy(newfuels[185].desc, "High load conifer litter");

     newfuels[186].number=186;
     strcpy(newfuels[186].code, "TL6");
     newfuels[186].h1=2.400000;
     newfuels[186].h10=1.200000;
     newfuels[186].h100=1.200000;
     newfuels[186].lh=0.000000;
     newfuels[186].lw=0.000000;
     newfuels[186].dynamic=0;
     newfuels[186].sav1=2000;
     newfuels[186].savlh=1800;
     newfuels[186].savlw=1600;
     newfuels[186].depth=0.300000;
     newfuels[186].xmext=0.250000;
     newfuels[186].heatd=8000.000000;
     newfuels[186].heatl=8000.000000;
     strcpy(newfuels[186].desc, "High load broadleaf litter");

     newfuels[187].number=187;
     strcpy(newfuels[187].code, "TL7");
     newfuels[187].h1=0.300000;
     newfuels[187].h10=1.400000;
     newfuels[187].h100=8.100000;
     newfuels[187].lh=0.000000;
     newfuels[187].lw=0.000000;
     newfuels[187].dynamic=0;
     newfuels[187].sav1=2000;
     newfuels[187].savlh=1800;
     newfuels[187].savlw=1600;
     newfuels[187].depth=0.400000;
     newfuels[187].xmext=0.250000;
     newfuels[187].heatd=8000.000000;
     newfuels[187].heatl=8000.000000;
     strcpy(newfuels[187].desc, "Large downed logs");

     newfuels[188].number=188;
     strcpy(newfuels[188].code, "TL8");
     newfuels[188].h1=5.800000;
     newfuels[188].h10=1.400000;
     newfuels[188].h100=1.100000;
     newfuels[188].lh=0.000000;
     newfuels[188].lw=0.000000;
     newfuels[188].dynamic=0;
     newfuels[188].sav1=1800;
     newfuels[188].savlh=1800;
     newfuels[188].savlw=1600;
     newfuels[188].depth=0.300000;
     newfuels[188].xmext=0.350000;
     newfuels[188].heatd=8000.000000;
     newfuels[188].heatl=8000.000000;
     strcpy(newfuels[188].desc, "Long-needle litter");

     newfuels[189].number=189;
     strcpy(newfuels[189].code, "TL9");
     newfuels[189].h1=6.650000;
     newfuels[189].h10=3.300000;
     newfuels[189].h100=4.150000;
     newfuels[189].lh=0.000000;
     newfuels[189].lw=0.000000;
     newfuels[189].dynamic=0;
     newfuels[189].sav1=1800;
     newfuels[189].savlh=1800;
     newfuels[189].savlw=1600;
     newfuels[189].depth=0.600000;
     newfuels[189].xmext=0.350000;
     newfuels[189].heatd=8000.000000;
     newfuels[189].heatl=8000.000000;
     strcpy(newfuels[189].desc, "Very high load broadleaf litter");

     newfuels[201].number=201;
     strcpy(newfuels[201].code, "SB1");
     newfuels[201].h1=1.500000;
     newfuels[201].h10=3.000000;
     newfuels[201].h100=11.000000;
     newfuels[201].lh=0.000000;
     newfuels[201].lw=0.000000;
     newfuels[201].dynamic=0;
     newfuels[201].sav1=2000;
     newfuels[201].savlh=1800;
     newfuels[201].savlw=1600;
     newfuels[201].depth=1.000000;
     newfuels[201].xmext=0.250000;
     newfuels[201].heatd=8000.000000;
     newfuels[201].heatl=8000.000000;
     strcpy(newfuels[201].desc, "Low load activity fuel");

     newfuels[202].number=202;
     strcpy(newfuels[202].code, "SB2");
     newfuels[202].h1=4.500000;
     newfuels[202].h10=4.250000;
     newfuels[202].h100=4.000000;
     newfuels[202].lh=0.000000;
     newfuels[202].lw=0.000000;
     newfuels[202].dynamic=0;
     newfuels[202].sav1=2000;
     newfuels[202].savlh=1800;
     newfuels[202].savlw=1600;
     newfuels[202].depth=1.000000;
     newfuels[202].xmext=0.250000;
     newfuels[202].heatd=8000.000000;
     newfuels[202].heatl=8000.000000;
     strcpy(newfuels[202].desc, "Moderate load activity or low load blowdown");

     newfuels[203].number=203;
     strcpy(newfuels[203].code, "SB3");
     newfuels[203].h1=5.500000;
     newfuels[203].h10=2.750000;
     newfuels[203].h100=3.000000;
     newfuels[203].lh=0.000000;
     newfuels[203].lw=0.000000;
     newfuels[203].dynamic=0;
     newfuels[203].sav1=2000;
     newfuels[203].savlh=1800;
     newfuels[203].savlw=1600;
     newfuels[203].depth=1.200000;
     newfuels[203].xmext=0.250000;
     newfuels[203].heatd=8000.000000;
     newfuels[203].heatl=8000.000000;
     strcpy(newfuels[203].desc, "High load activity fuel or moderate load blowdown");

     newfuels[204].number=204;
     strcpy(newfuels[204].code, "SB4");
     newfuels[204].h1=5.250000;
     newfuels[204].h10=3.500000;
     newfuels[204].h100=5.250000;
     newfuels[204].lh=0.000000;
     newfuels[204].lw=0.000000;
     newfuels[204].dynamic=0;
     newfuels[204].sav1=2000;
     newfuels[204].savlh=1800;
     newfuels[204].savlw=1600;
     newfuels[204].depth=2.700000;
     newfuels[204].xmext=0.250000;
     newfuels[204].heatd=8000.000000;
     newfuels[204].heatl=8000.000000;
     strcpy(newfuels[204].desc, "High load blowdown");

}
*/
