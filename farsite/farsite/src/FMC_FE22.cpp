/******************************************************************
* Name: FMC_FE22.cpp
* Desc: This file is just an extention of the FMC_FE2.cpp
*
********************************************************************/
#ifdef WIN32
#include <windows.h>
#include <process.h>
#endif


#include "cdtlib.h"
#include "newfms.h"

#include "semtime.h"
#include "deadfuelmoisture.h"

#include "FMC_CI.h"
#include "FMC_FE2.h"

// extern  double PI;

static const double PI=acos(-1.0);

/******************************************************************************/
double FE2::GetMx(double Time, long fuel, long elev, long slope,	double aspectf,
                  long cover, double *equil, double *solrad, long FuelSize)
{
// Time = minutes
// fuel = index to fuel model, not zero-based
// elev = land elevation, meters
// radiation = millivolts
// equil =equilibrium moisture content

long i, Aspect;
long sn=StationNumber;
long ElevIndex, ElevIndexN, SlopeIndex, SlopeIndexN;
long AspIndex, AspIndexN, CovIndex, CovIndexN;
double loc;
double efract, sfract, afract, cfract, ipart;
double mx[16];
double mxavg[8], timeavg1, timeavg2;
double tmxavg, timefract, LastTime, FirstTime;

   if (fuel<1)
     return 0.0;

   if (MxDesc.NumFuels[sn][FuelSize] == 0)
     return 0.0;

   if (*solrad == 1 ) {
     CheckMoistureHistory(sn, FuelSize, Time);
     if ( FuelSize == SIZECLASS_10HR)
        CheckMoistureHistory(sn, SIZECLASS_1HR, Time); }
   else if (Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->FirstTime > Time ||
            Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->LastTime < Time)	{
      CheckMoistureHistory(sn, FuelSize, Time);
      if ( FuelSize == SIZECLASS_10HR )
            CheckMoistureHistory(sn, SIZECLASS_1HR, Time);
     }

 	 if (a_CI->GetTheme_Units(E_DATA)==0)
	    loc = ((double)(elev-a_CI->GetLoElev())) / (double) a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV);
   else
	    loc = ((double)(elev-a_CI->GetLoElev()*0.3048)) /	(double) a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ELEV);

   efract = modf(loc, &ipart);
   ElevIndex=(long) ipart;

   if (ElevIndex<0)
     ElevIndex=0;
   ElevIndexN = ElevIndex;

	  if ( ElevIndex>MxDesc.NumElevs[sn][FuelSize] - 1 )
     	ElevIndex=MxDesc.NumElevs[sn][FuelSize]-1;
   else
     	ElevIndexN++;
   if (ElevIndexN>MxDesc.NumElevs[sn][FuelSize]-1)
     ElevIndexN=MxDesc.NumElevs[sn][FuelSize]-1;

   SlopeIndexN = SlopeIndex = slope / a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE);
   if (SlopeIndex>MxDesc.NumSlopes[sn][FuelSize]-1)
      SlopeIndex=MxDesc.NumSlopes[sn][FuelSize]-1;
   else
      SlopeIndexN++;

   if (SlopeIndexN>MxDesc.NumSlopes[sn][FuelSize] - 1 )
     	SlopeIndexN=MxDesc.NumSlopes[sn][FuelSize]-1;
   loc = (double) (slope - (SlopeIndex * a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE)))/
        	(double) a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_SLOPE);
   sfract = modf(loc, &ipart);

	  Aspect = (long) (aspectf/PI*180.0);
   AspIndexN = AspIndex = Aspect / a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP);

     if(AspIndex>MxDesc.NumAspects[sn][FuelSize]-1)
     	AspIndex=MxDesc.NumAspects[sn][FuelSize]-1;
     else
     	AspIndexN++;
	if(AspIndexN>MxDesc.NumAspects[sn][FuelSize]-1)
     	AspIndexN=MxDesc.NumAspects[sn][FuelSize]-1;

	loc=(double) (Aspect-(AspIndex*a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP)))/
     	(double) a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_ASP);
     afract=modf(loc, &ipart);

	CovIndexN=CovIndex=cover/a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV);
     if(CovIndex>MxDesc.NumCovers[sn][FuelSize]-1)
     	CovIndex=MxDesc.NumCovers[sn][FuelSize]-1;
     else
     	CovIndexN++;
     if(CovIndexN>MxDesc.NumCovers[sn][FuelSize]-1)
     	CovIndexN=MxDesc.NumCovers[sn][FuelSize]-1;
	loc=(double) (cover-(CovIndex*a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV)))/
     	(double) a_CI->GetMoistCalcInterval(FuelSize, FM_INTERVAL_COV);
     cfract=modf(loc, &ipart);

// test..............
// int iii = Stations[sn].FuelKey [FuelSize] [fuel-1];
// test.............

	mx[0] = Stations[sn].FMS [FuelSize] [Stations[sn].FuelKey [FuelSize] [fuel-1] -1 ]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastMx[CovIndex];

	mx[1] =Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastMx[CovIndexN];
	mx[2] =Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastMx[CovIndex];
	mx[3] =Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastMx[CovIndexN];
	mx[4]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastMx[CovIndex];
	mx[5]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastMx[CovIndexN];
	mx[6]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastMx[CovIndex];
	mx[7]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastMx[CovIndexN];
	mx[8]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastMx[CovIndex];
	mx[9]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastMx[CovIndexN];
	mx[10]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastMx[CovIndex];
	mx[11]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastMx[CovIndexN];
	mx[12]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastMx[CovIndex];
	mx[13]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastMx[CovIndexN];
	mx[14]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastMx[CovIndex];
	mx[15]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastMx[CovIndexN];

     // interpolate Cover
     for(i=0; i<8; i++)
     {    mxavg[i]=mx[i*2]-cfract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Aspect
     for(i=0; i<4; i++)
     {    mxavg[i]=mx[i*2]-afract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Slope
     for(i=0; i<2; i++)
     {    mxavg[i]=mx[i*2]-sfract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     timeavg1=mx[i]-efract*(mx[i]-mx[i+1]);
// A A A
	mx[0]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextMx[CovIndex];
	mx[1]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextMx[CovIndexN];
	mx[2]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextMx[CovIndex];
	mx[3]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextMx[CovIndexN];
	mx[4]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextMx[CovIndex];
	mx[5]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextMx[CovIndexN];
	mx[6]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextMx[CovIndex];
	mx[7]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextMx[CovIndexN];
	mx[8]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextMx[CovIndex];
	mx[9]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextMx[CovIndexN];
	mx[10]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextMx[CovIndex];
	mx[11]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextMx[CovIndexN];
	mx[12]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextMx[CovIndex];
	mx[13]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextMx[CovIndexN];
	mx[14]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextMx[CovIndex];
	mx[15]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextMx[CovIndexN];

     // interpolate Cover
     for(i=0; i<8; i++)
     {    mxavg[i]=mx[i*2]-cfract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Aspect
     for(i=0; i<4; i++)
     {    mxavg[i]=mx[i*2]-afract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Slope
     for(i=0; i<2; i++)
     {    mxavg[i]=mx[i*2]-sfract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Elevation
     timeavg2=mx[i]-efract*(mx[i]-mx[i+1]);

     // interpolate over timeinterval
     FirstTime=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->FirstTime;
     LastTime=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->LastTime;
     if(LastTime-FirstTime>1e-6)
     	timefract=(Time-FirstTime)/(LastTime-FirstTime);
     else
     	timefract=1.0;
     if(timefract>1.0)
     	timefract=1.0;
     tmxavg=timeavg1-timefract*(timeavg1-timeavg2);

     if(FuelSize!=SIZECLASS_10HR)
     	return tmxavg;

     // finds mean equilibrium moisture content for use in 1-hour fuels
// B B B
  mx[0]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastEq[CovIndex];
	mx[1]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastEq[CovIndexN];
	mx[2]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastEq[CovIndex];
	mx[3]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastEq[CovIndexN];
	mx[4]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastEq[CovIndex];
	mx[5]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastEq[CovIndexN];
	mx[6]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastEq[CovIndex];
	mx[7]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastEq[CovIndexN];
	mx[8]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastEq[CovIndex];
	mx[9]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->LastEq[CovIndexN];
	mx[10]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastEq[CovIndex];
	mx[11]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->LastEq[CovIndexN];
	mx[12]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastEq[CovIndex];
	mx[13]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->LastEq[CovIndexN];
	mx[14]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastEq[CovIndex];
	mx[15]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->LastEq[CovIndexN];


     // interpolate Cover
     for(i=0; i<8; i++)
     {    mxavg[i]=mx[i*2]-cfract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Aspect
     for(i=0; i<4; i++)
     {    mxavg[i]=mx[i*2]-afract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Slope
     for(i=0; i<2; i++)
     {    mxavg[i]=mx[i*2]-sfract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     timeavg1=mx[i]-efract*(mx[i]-mx[i+1]);

	mx[0]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextEq[CovIndex];
	mx[1]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextEq[CovIndexN];
	mx[2]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextEq[CovIndex];
	mx[3]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextEq[CovIndexN];
	mx[4]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextEq[CovIndex];
	mx[5]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextEq[CovIndexN];
	mx[6]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextEq[CovIndex];
	mx[7]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextEq[CovIndexN];
	mx[8]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextEq[CovIndex];
	mx[9]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->NextEq[CovIndexN];
	mx[10]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextEq[CovIndex];
	mx[11]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->NextEq[CovIndexN];
	mx[12]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextEq[CovIndex];
	mx[13]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->NextEq[CovIndexN];
	mx[14]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextEq[CovIndex];
	mx[15]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->NextEq[CovIndexN];

     // interpolate Cover
     for(i=0; i<8; i++)
     {    mxavg[i]=mx[i*2]-cfract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Aspect
     for(i=0; i<4; i++)
     {    mxavg[i]=mx[i*2]-afract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Slope
     for(i=0; i<2; i++)
     {    mxavg[i]=mx[i*2]-sfract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Elevation
     timeavg2=mx[i]-efract*(mx[i]-mx[i+1]);
	*equil=timeavg1-timefract*(timeavg1-timeavg2);	// average over elevation

     if(*solrad==0)
     	return tmxavg;

     // interpolate solarradiation
	mx[0]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[1]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[2]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[3]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];
	mx[4]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[5]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[6]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[7]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndex]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];
	//added SB 04/25/2009
	mx[8]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[9]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[10]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[11]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];
	mx[12]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[13]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[14]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[15]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];

     // Added SB 4/25/2009
     for(i=0; i<8; i++)
     {    mxavg[i]=mx[i*2]-cfract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Cover
     for(i=0; i<4; i++)
     {    mxavg[i]=mx[i*2]-afract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Aspect
     for(i=0; i<2; i++)
     {    mxavg[i]=mx[i*2]-sfract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Slope
     timeavg1=mx[i*2]-efract*(mx[i*2]-mx[i*2+1]);

	mx[0]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[1]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[2]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[3]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];
	mx[4]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[5]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[6]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[7]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];
	//added SB 04/25/2009
	mx[8]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[9]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[10]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[11]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndex]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];
	mx[12]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndex];
	mx[13]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndex]->SolRad[CovIndexN];
	mx[14]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndex];
	mx[15]=Stations[sn].FMS[FuelSize][Stations[sn].FuelKey[FuelSize][fuel-1]-1]->Fms_Elev[ElevIndexN]->Fms_Slope[SlopeIndexN]->Fms_Aspect[AspIndexN]->SolRad[CovIndexN];

     // Added SB 4/25/2009
     for(i=0; i<8; i++)
     {    mxavg[i]=mx[i*2]-cfract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Cover
     for(i=0; i<4; i++)
     {    mxavg[i]=mx[i*2]-afract*(mx[i*2]-mx[i*2+1]);
		mx[i]=mxavg[i];
     }
     // interpolate Aspect
     for(i=0; i<2; i++)
     {    mxavg[i]=mx[i*2]-sfract*(mx[i*2]-mx[i*2+1]);
          mx[i]=mxavg[i];
     }
     // interpolate Slope
     timeavg2=mx[i*2]-efract*(mx[i*2]-mx[i*2+1]);

     *solrad=timeavg1-timefract*(timeavg1-timeavg2);
	 //TRACE1("GetMx() internals: ElevIndex = %ld ", ElevIndex);
	 //TRACE1("SlopeIndex = %ld ", SlopeIndex, );
	// TRACE1("AspIndex = %ld\n", AspIndex);
	 //TRACE2("GetMx() internals: ElevIndex = %ld, ElevIndexN = %ld ", ElevIndex, ElevIndexN);
	 //TRACE2("SlopeIndex = %ld, SlopeIndexN = %ld ", SlopeIndex, SlopeIndexN);
	 //TRACE2("AspIndex = %ld, AspIndexN = %ld ", AspIndex, AspIndexN);
	// TRACE2("CovIndex = %ld, CovIndexN = %ld\n", CovIndex, CovIndexN);
	//TRACE1("Setting solrad in GetMx() to %lf\n", *solrad);
     return tmxavg;
}

