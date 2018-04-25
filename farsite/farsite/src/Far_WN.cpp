/********************************************************************************
* Name: Far_WN.cpp
* Desc:  
*  > Include this file and .h files into project, had to put nn_def.h in here, but
*      should figure out a way so that there aren't 2 of them 
*
*
*  > look thru other coder for WN-Test to see other changes
*
*  > I took out the __int64 in FlamMap::GetLayerValueByCell() below in
*     this file 
*
*
*  > icf_Def.h   new  gridded winds switches 
*
*********************************************************************************/
#include <stdio.h> 
#include "Farsite5.h"

#include "..\WindNinja2\CWindNinja2.h" 

#include    "nn_def.h" 
#define e_FtToMt 0.3048 
const double PI=acos(-1.0);

long    DirDiff (long a, long b);
double _Similar (long l_DirA, long l_DirB, long l_DirTol, 
                 double d_SpdA, double d_SpdB, double d_SpdTol);

int  _ChkSetRes (float f_CmdFilRes, double d_lcpRes, long lcpRows, long lcpCols, double *ad, char cr_ErrMes[]);
void SiteSpecific(long ElevDiff, double tempref, double* temp,	double humref, double* humid);

int Get_MinDate (int i_Mth, int i_Day, int i_Yr, int i_MilTim);
int _Overlap (int W1, int W2, int B1, int B2);

/*********************************************************************************
* Name: Load_WindNinja
* Desc: Do all needed Ninja runs.
* Note-1: take 1 minute off the ending time so it doesn't overlap with a 
*          staring time, example a burn period 100->200, would over lap with
*          wind records 100->200 and 200->300, but it should only overlap
*          with the 100->200 wind record,  
*       
*********************************************************************************/
int Farsite5::Load_WindNinja(char cr_ErrMes[])
{   
int i, j, x, i_Err, iN_WN, iN_Pts;
int k, bS, bE, wS, wE;
double d_ElevRes; 
long l_NumNorthRows, l_NumEastCols;

  strcpy (cr_ErrMes,""); 

  if ( !stricmp (this->icf.s_FWNI.cr_GriWin,e_GW_No)) /* Check Inputs File Switch */
      return 1;                      /* Not doing WindNinja grid winds */
   
  iN_WN = iN_Pts = 0;                /* counters, Ninja runs */
/*---------------------------------------------------------------------------------------*/
  for (j = 0; j < MaxWindObs[0]; j++ ) {  /* for each wind record */
     wddt[0][j].a_FWN == NULL;   /* Init these */
     wddt[0][j].iS_WN = e_NA;  }

/*--------------------------------------------------------------------*/
/* Mark each rec in wnd tbl that corresponds/overlaps with a burn period  */
 k = MaxWindObs[0] - 2;  /* # of wind recs minus the last 2 rec in table */
 for ( i = 0; i < NumRelativeData; i++)	{  /* For each Burn Period */ 
    /* Minute dates of the burn period */
    bS = Get_MinDate (abp[i].Month, abp[i].Day, abp[i].Year, abp[i].Start);
    bE = Get_MinDate (abp[i].Month, abp[i].Day, abp[i].Year, abp[i].End);
    bE = bE - 1;   /* See Note-1 above */

   for (j = 0; j < k; j++ ) {  /* for each wind record, don't do last 13 mth rec */
     /* minute date this and following rec */
     wS = Get_MinDate (wddt[0][j].mo,   wddt[0][j].dy,   wddt[0][j].yr,   wddt[0][j].hr);
     wE = Get_MinDate (wddt[0][j+1].mo, wddt[0][j+1].dy, wddt[0][j+1].yr, wddt[0][j+1].hr);
     wE = wE - 1; /* See Note-1 above */
  
     x = _Overlap (wS, wE, bS, bE);
     if ( x == 1 ) 
       wddt[0][j].iS_WN = e_BP;
//       printf ("OverLap \n");
//       printf (" WT-1-> %d %d %d  %d \n", wddt[0][j].mo,   wddt[0][j].dy,   wddt[0][j].yr,   wddt[0][j].hr);
//       printf (" WT-2-> %d %d %d  %d \n",wddt[0][j+1].mo, wddt[0][j+1].dy, wddt[0][j+1].yr, wddt[0][j+1].hr); 
//       printf (" PB-> %d %d %d  %d -> %d \n", abp[i].Month, abp[i].Day, abp[i].Year, abp[i].Start, abp[i].End);
//       printf ("\n\n\n");  }
 
  }} 

/*--------------------------------------------------------------------------*/
/* Recs in burn period, determine what can be bined/resued & what to run Ninja on  */
   fN_WNToDo = 0; 
   for (j = 0; j < MaxWindObs[0]; j++ ) {
//     printf (" Look At %d %d %d %d\n", wddt[0][j].mo, wddt[0][j].dy, wddt[0][j].hr,wddt[0][j].iS_WN); 
  
     if ( wddt[0][j].iS_WN != e_BP )   /* not in the burn period */
       continue;  

//     printf ("\n----> %d,  Mth: %d Day: %d Hr: %d, Dir: %d Spd: %6.1f \n",  j, wddt[0][j].mo, wddt[0][j].dy, wddt[0][j].hr, wddt[0][j].wd, wddt[0][j].ws); 

/* Is there an existing Ninja run we can use, if YES than set it up */
     if ( WindNinja_SetExisting(j) ) {
        iN_Pts++;                       /* Count # of reuse */
        continue;}                      /* found and using an exisint ninja run */

     wddt[0][j].a_FWN = new d_FWN;      /* will hold ninja wind grids & other stuff */
     wddt[0][j].a_FWN->windDirGrid = NULL;
     wddt[0][j].a_FWN->windSpeedGrid =  NULL; 
     wddt[0][j].iS_WN = e_WN;           /* Says Ninja was run on this wind record */  
     wddt[0][j].a_FWN->f_Temperature = Temperature (wddt[0][j].mo, wddt[0][j].dy, wddt[0][j].hr); 
     iN_WN++; 
     this->fN_WNToDo++;                  /* Ninja runs to be done */ 
   }  /* for j */
 
/* Get the WindNinja class ready */
   i = WindNinjaInit (&WN2,  &d_ElevRes, &l_NumNorthRows, &l_NumEastCols, cr_ErrMes);
   if ( i == 0 ) 
     return 0;            /* error init'ing ninja */

/*-------------------------------------------------------------*/
/* Run Ninja on the selected Wind records                      */
   this->FPC.SetTimeSlice(iN_WN,this);   /* tells Progress Class how to time slice */  
    
   this->FPC.Set_NinjaRunning(iN_WN); 

   for ( i = 0; i < MaxWindObs[0]; i++ ) {
     if ( wddt[0][i].iS_WN != e_WN )
        continue; 
     j = WindNinjaRun (&WN2, &wddt[0][i], d_ElevRes, l_NumNorthRows, l_NumEastCols, cr_ErrMes);
     if ( j == 0 )
       return 0;                   /* Error or User Abort */
     this->FPC.Set_NinjaInc();     /* tell Progress Class another ninja is completed */
   }

// Test display 
//  Disp_WindTbl_WN();

 return 1; 
}

/*************************************************************************
*
*
**************************************************************************/
int _Overlap (int W1, int W2, int B1, int B2)
{
int a,b, c, d;

/* Make sure inputs are proper */
    if ( W1 >= W2 )
      return 0;

    if ( B1 >= B2 )
      return 0;

    if ( W1 < B1 ) /* Find lowest point */
     a = W1;
    else
     a = B1;

    if ( W2 > B2 ) /* Find highest point */
     b = W2;
    else
     b = B2;

    c = b - a;     /* Take diff */

    d = (W2 - W1) + (B2 - B1);  /* add both spreads */

    if ( d >= c )
      return 1;   /* there is overlap */

    return 0;

}

/********************************************************************
*
*
******************************************************************/
int Get_MinDate (int i_Mth, int i_Day, int i_Yr, int i_MilTim)
{
int i,  i_Time, i_Min ;
   
 //  for ( i = 0; i < 50000; i++ ) {
 //   i_Yr = this->wddt[0][i].yr; 

 //   if ( this->wddt[0][i].mo == 13 ) 
 //     break; 
       
 //   if ( this->wddt[0][i].mo == i_Mth )      
 //      break; 
 //  }

 i =  GetMCDate(i_Mth,i_Day,i_Yr); /* Day in a 2 year period */
 i_Time = i * e_MinPerDay;       
 i_Min = MilToMin (i_MilTim);      /* Military time to total minutes */
 i = i_Time + i_Min;               /* minutes from start of 2 yr period */

 return i; 
}



/***************************************************************************
* Name: WindNinja_SetExisting
* Desc: See if there is an existing 'Ninja run that can be used. 
*       Ninja wind grids are stored in the wddt[][].a_FWN
*   In: iX...index of Wind Table record 
***************************************************************************/
int Farsite5::WindNinja_SetExisting(int iX)
{
int i, iX_Fnd, iN;
double  d, d_Least, d_SpdBin;
long l_DirBin;

  iN = 0;
 
/* bin sizes */
  l_DirBin = (long) this->icf.s_FWNI.f_BinWinDir;  
  d_SpdBin = this->icf.s_FWNI.f_BinWinSpd;

  iX_Fnd = -1; 
  d_Least = 9999.0; 

  for (i = 0; i < MaxWindObs[0]; i++ ) {  /* Each Rec in the Wind Tbl */
    if ( wddt[0][i].iS_WN == e_NA )
      continue;              /* Not a Rec in burn period */
    if ( iX == i )           /* Don't want to compare this */
      continue;              /* Rec to itself, so skip to nxt */
    if ( wddt[0][i].iS_WN != e_WN )
      continue;              /* Rec didn't have Ninja run on it */ 

/* This record has previously run Ninja grids stored in it */
    d = _Similar ( wddt[0][i].wd, wddt[0][iX].wd, l_DirBin, 
                   wddt[0][i].ws, wddt[0][iX].ws, d_SpdBin);

    if ( d < 0 )             /* not within bining range */
      continue; 

/* Rec is in the bining range of previous Ninja run, the smaller the  */
/*  value the closer it is, so look for the closest match */ 
    if ( d < d_Least ) {    
      iN++;
      d_Least = d; 
      iX_Fnd = i; }
  }

  if ( iX_Fnd != -1 ) {
 //    printf ("Assign to %d \n", iX_Fnd ); 
     wddt[0][iX].a_FWN = wddt[0][iX_Fnd].a_FWN; /* Contains wind grids of existing Ninja run */
     wddt[0][iX].iS_WN = e_PT;    /* Says were using an existing Ninja run */
     return 1; }   
   
  return 0; 
}

/*************************************************************
* Name _Similar
* Desc: Compare two wind directions and speeds and return
*        a number that reflects how similar they are.
*
* NOTE NOTE: Currently this the smaller the number the more 
*             similar, 
*            I just to a simple adding of the differences for
*            now,  
**************************************************************/
double _Similar (long l_DirA, long l_DirB, long l_DirBin, 
                 double d_SpdA, double d_SpdB, double d_SpdBin)
{
long l_Dir;
double d, d_Spd;

  l_Dir =  DirDiff (l_DirA, l_DirB);  
  if ( l_Dir > l_DirBin )
    return -1.0; 

  d = d_SpdA - d_SpdB;              /* difference */
  d_Spd = abs(d);             /* absolulte value */
  if ( d_Spd > d_SpdBin )
    return -1.0;

  d = (double) l_Dir + d_Spd;
  return d; 
}

/********************************************************************
* Name: DirDiff
* Desc: find the different between to wind degrees
* NOTE: the returned difference with never exceed 180
*       Ex: the difference of 0 and 300 is 60 degrees NOT 300
*   In: A, B  0 -> 359
*  Ret: 0 -> 180
********************************************************************/
long DirDiff (long a, long b)
{
long A,B, C,D,Diff;
   A = a;
   B = b;
   if ( A > 359 || A < 0 )
     A = 0;

   if ( B > 359 || B < 0 )
     B = 0;

   if ( A < B )
     C = B - A;
   else
     C = A - B;

   D = 360 - C;

   if ( D < C )
     Diff = D;
   else
     Diff = C;
   return Diff;
}

/***************************************************************************
* Name: Temperature
* Desc: Get the temperature from the daily weather table or the RAWS 
*        tables. 
*       If the temperature comes from the daily table it gets interpolated. 
*   In: l_Hr is in military time
****************************************************************************/
float Farsite5::Temperature (long l_Mth, long l_Day, long l_Hr)
{
int i; 
double d_tempref, d_humref; 
long l_elevref; 
double d_rain, d_humidmx, d_humidmn, d_tempmx, d_tempmn;
int sn = 0; 
float f; 

  if ( this->icf.a_RAWS != NULL ) {  
    f = Get_RAWS_Temp (l_Mth, l_Day, l_Hr);
    return f; }

  HumTemp (l_Mth, l_Day, l_Hr, &d_tempref); 

   return (float) d_tempref; 
}

/*****************************************************************************
* Name: Get_RAWS_Temp
* Desc: Get the temperature for the specified date/time in the RAWS weather
*        table. 
* NOTE: Caller must check for existance of RAWS data before calling 
*        this function
******************************************************************************/
float Farsite5::Get_RAWS_Temp (long l_Mth, long l_Day, long l_Hr)
{
int i,j,ix;

/* NOTE  Caller should have made sure there was RAWS data before coming here */
  if ( this->icf.a_RAWS == NULL )  
    return 0;                 

/* Find first record for month day */
  for ( i = 0; i < this->icf.iN_RAWS; i++ ) {
    if ( icf.a_RAWS[i].i_Mth == l_Mth && icf.a_RAWS[i].i_Day == l_Day ) {   
      ix = i; 
      break; } }

/* just in case hour is less than the first hourly record of the day */
  if ( l_Hr <= icf.a_RAWS[ix].i_Time )   
    return icf.a_RAWS[ix].f_Temp;

/* Look at each record for the desired month day */
  for ( i = ix; i < this->icf.iN_RAWS; i++ ) {

    if ( i == this->icf.iN_RAWS - 1 )       /* Shouldn't hit last record */
      return this->icf.a_RAWS[i].f_Temp;    /* but just in case */

    if ( this->icf.a_RAWS[i+1].i_Day != l_Day ) /* if last record of the */
      return this->icf.a_RAWS[i].f_Temp;        /*  desired day  */
 
   if ( l_Hr >= this->icf.a_RAWS[i].i_Time && l_Hr < this->icf.a_RAWS[i+1].i_Time )
        return this->icf.a_RAWS[i].f_Temp;          
  } 
 
  return 0;   /* SHouldn't get here */
}


/*****************************************************************************
* Name: HumTemp
* Desc: For the specificed date/time use the daily weather data to interpolate
*        the temperature.
* NOTE: this code was modified from the orginal FireEnvironment2::HumTemp()
* 
******************************************************************************/
void Farsite5::HumTemp(long l_Mth, long l_Day, long l_Hr, double* tempref)
{

int i, StationNumber, sn;
long count = -1, month, day, hmorn, haft, Tmin, Tmax, Hmax, Hmin;
long elref, hx, garbage, ppt;
double h1, h2, dtprime, dtmxmn, humid, temp, tempf, humf, sign, Thalf, Hhalf;
long hour, LastCount, LastDate;

  StationNumber = sn = 0; 
  hour = l_Hr; 

  for ( i = 0; i < this->MaxWeatherObs[sn]; i++ ) {
    if ( l_Mth == this->wtrdt[sn][i].mo && l_Day == this->wtrdt[sn][i].dy )
      break; }

 LastCount = i; 

	hmorn = GetWeatherTime1(StationNumber, LastCount);
	haft = GetWeatherTime2(StationNumber, LastCount);
	ppt = (long)GetWeatherRain(StationNumber, LastCount);
	Tmin = (long)GetWeatherTemp1(StationNumber, LastCount);
	Tmax = (long)GetWeatherTemp2(StationNumber, LastCount);
	Hmax = (long)GetWeatherHumid1(StationNumber, LastCount);
	Hmin = (long)GetWeatherHumid2(StationNumber, LastCount);
	elref = (long)GetWeatherElev(StationNumber, LastCount);
	ppt = (long)GetWeatherRain(StationNumber, LastCount);
	count = LastCount;	// only meaningful if LastDate==date;

	if (hour > haft){
	 	count++;
		 garbage = (long)GetWeatherElev(StationNumber, count);
		 hmorn = (long)GetWeatherTime1(StationNumber, count);
		 Tmin = (long)GetWeatherTemp1(StationNumber, count);
		 Hmax = (long)GetWeatherHumid1(StationNumber, count);
	 	if (garbage != elref) {
		  	tempf = (double) Tmin;
		  	humf = (double) Hmax;
		  	SiteSpecific(elref - garbage, tempf, &temp, humf,	&humid);	/* adjusts to site specific temp and relhum at a given pixel */
		  	Tmin = (int) temp;
		  	Hmax = (int) humid;}
	}
	else if (hour < hmorn)	{
	 	count--;
		 if (count < 0)
			  count = 0;
		 garbage = (long)GetWeatherElev(StationNumber, count);
	 	haft = (long)GetWeatherTime2(StationNumber, count);
		 Tmax = (long)GetWeatherTemp2(StationNumber, count);
	 	Hmin = (long)GetWeatherHumid2(StationNumber, count);
	 	if (garbage != elref)	{
			  tempf = Tmax;
			  humf = Hmin;
			  SiteSpecific(elref - garbage, tempf, &temp, humf, &humid);	/* adjusts to site specific temp and relhum at a given pixel */
		  	Tmax = (int) temp;
		  	Hmin = (int) humid;}
	}

 hx = hmorn / 100;
	hx *= 100;
	if (hx < hmorn)
		 h1 = (double) hx + (10.0 * (hmorn - hx) / 6.0);
	else
	 	h1 = (double) hmorn;
	hx = haft / 100;
	hx *= 100;
	if (hx < haft)
	 	h2 = (double) hx + (10.0 * (haft - hx) / 6.0);
	else
	 	h2 = (double) haft;

	dtmxmn = (2400 - h2) + h1;		/* this section interpolates temperature */
	if (hour >= h1 && hour <= h2)	{	/* and humidity from high and low obs */	
		dtprime = (double) hour - h1;	/* and their time of observation */
		dtmxmn = 2400 - dtmxmn;
		sign = 1.0; }
	else	{
	 	if (hour > h2)
		  	dtprime = double (hour) - h2;   // time since the maximum
	 	else
			  dtprime = (2400 - h2) + hour;    // time since the maximum
		sign = -1.0;
	}

	Thalf = ((double) Tmax - (double) Tmin) / 2.0 * sign;
	Hhalf = ((double) Hmin - (double) Hmax) / 2.0 * sign;

	*tempref = ((double) (Tmax + Tmin)) / 	2.0 +	(Thalf * sin(PI * (dtprime / dtmxmn - 0.5)));
}

/**************************************************************************************/
void SiteSpecific(long ElevDiff, double tempref, double* temp,	double humref, double* humid)
{
	// FROM ROTHERMEL, WILSON, MORRIS, & SACKETT  1986
	double dewptref, dewpt;
	// ElevDiff is always in feet, temp is in Fahrenheit, humid is %

	dewptref = -398.0 - 7469.0 /
		(log(humref / 100.0) - 7469.0 / (tempref + 398.0));
	*temp = tempref + (double) ElevDiff / 1000.0 * 5.5; 	  	// Stephenson 1988 found summer adiabat at 3.07 F/1000ft
	dewpt = dewptref + (double) ElevDiff / 1000.0 * 1.1; 		// new humidity, new dewpt, and new humidity
	*humid = 7469.0 * (1.0 / (*temp + 398.0) - 1.0 / (dewpt + 398.0));
	*humid = exp(*humid) * 100.0;				// convert from ln units
	if (*humid > 99.0)
		*humid = 99.0;
}

/************************************************************************
* Name: WindNinjaInit
* Desc: Get the windninja class ready to be run.
*       This sets the things that only need to be set once, after which
*        Ninja can be run as many times as you want.
*  Ret: 1 = OK,  
*       < 0 Error, error message in cr_ErrMes 
************************************************************************/
int Farsite5::WindNinjaInit ( CWindNinja2 *a_WN2,  double *ad_ElevRes, 
       long *al_NumNorthRows, long *al_NumEastCols, char cr_ErrMes[])
{
bool b; 
long  i,j,c,r, l_NumNorthRows,l_NumEastCols, wnRows, wnCols, lN_EMiss, xl ; 
float f, gwRes; 
double d, d_Elev, yExtent, d_ElevRes, d_resol, unitsConv, minValue, maxValue; 
double   wnyExtent, xExtent, d_OutRes, *dr_Elev, *dr_Rough, wnxExtent;  
 c_NN NN;           /* Nearest Neighbor, for filling in grids */
 
   strcpy (cr_ErrMes,""); 
   unitsConv = 1.0;
   if ( Farsite5::GetTheme_Units (E_DATA) != 0)
      unitsConv = 0.3048;               //convert feet to meters
   d_ElevRes = Farsite5::GetCellResolutionX(); 
   
   gwRes = this->icf.s_FWNI.f_GriWinRes;
//  old way   gwRes = Farsite5::icf.GridWindResolution();
   yExtent = d_ElevRes * Farsite5::GetNumNorth();
   l_NumNorthRows = Farsite5::GetNumNorth(); 
   wnRows = yExtent / gwRes;
   wnyExtent = wnRows * gwRes;
 
   while(wnyExtent < yExtent) {
		   wnRows++;
		   wnyExtent = wnRows * gwRes; }
  
   l_NumNorthRows = wnyExtent / d_ElevRes;
   while ( l_NumNorthRows * d_ElevRes < wnyExtent)
	     l_NumNorthRows++;
       
   xExtent = d_ElevRes * Farsite5::GetNumEast();
   wnCols = xExtent / gwRes;
   wnxExtent = wnCols * gwRes;
   while ( wnxExtent < xExtent ){
	    	wnCols++;
	    	wnxExtent = wnCols * gwRes;  }
  
   l_NumEastCols = wnxExtent / d_ElevRes;
   while ( l_NumEastCols * d_ElevRes < wnxExtent )
	     l_NumEastCols++;

/* Setup WindNinja elev & rough height grids */
   a_WN2->SetHeadData( l_NumEastCols, l_NumNorthRows,  /* size of grid */
                    Farsite5::GetWestUtm(), (Farsite5::GetNorthUtm() - wnyExtent),
                    Farsite5::GetCellResolutionX(), 
                    NODATA_VAL,                     /* Miss data value */
                    NODATA_VAL);                    /* Init grid with this */
 
  dr_Elev = new double [l_NumNorthRows * l_NumEastCols]; 
  dr_Rough = new double [l_NumNorthRows * l_NumEastCols]; 
    
  minValue = 99999.9;  /* set for finding min and max */
  maxValue = -1.0;    
  lN_EMiss = 0;      
  xl = 0; 

/* Get elevations from landscape data */
  for ( r = 0;  r < l_NumNorthRows; r++ ) {
    for ( c = 0; c < l_NumEastCols; c++ ) {
     d_Elev = Farsite5::GetLayerValueByCell(E_DATA, c, r);	/* returns 1- if out of grid */ 		       
       if ( d_Elev == (double) NODATA_VAL || d_Elev < 0.0 ) {    
         dr_Elev[xl++] = NODATA_VAL; 
         lN_EMiss++; 
         continue; }             
 
       d_Elev = d_Elev * unitsConv; /* want meters */
       dr_Elev[xl++] = d_Elev;       
 
       if ( d_Elev < minValue )  /* Find Min Max elevs */
         minValue = d_Elev;
       if ( d_Elev > maxValue )
         maxValue = d_Elev;  }}

  a_WN2->Set_ElevMinMax (minValue,maxValue); 

/* -------------------------------------------------------------------*/
/* 7 rough height arrarys need to be load, checked for missing values */
/*  and then loaded to windninja, here we do array at a time */
 int ir[] = { e_Rough_h, e_Rough_d, e_Roughness, e_Albedo, e_Bowen, e_Cg, e_Anthropog, -1 }; 
      
   for ( i = 0; i < 100; i++ ) { /* Do each of the rough types, see above */
     if ( ir[i] == -1 )          /* All done */
       break; 
     xl = -1;
     for ( r = 0;  r < l_NumNorthRows; r++ ) {
       for ( c = 0; c < l_NumEastCols; c++ ) {
         xl++;
         if ( dr_Elev[xl] == NODATA_VAL )  /* look at elev for corspnd cell */
           dr_Rough[xl] = NODATA_VAL;      /* if miss also set rough to mis */
         else                              /* go determin which kind rough  */
           dr_Rough[xl] = GetRough (r,c,ir[i]);  /* and return it */
      }}  /* for r c */

/* Check and fill in any missing values */
     NN.Init(l_NumNorthRows, l_NumEastCols,dr_Rough,NODATA_VAL); 
     if ( !NN.Fill_Mngr()){
	      sprintf (cr_ErrMes, "%s - Error while filling missing elevations", NN.cr_Message);
       return 0;}

/* Give data to Windninja */    
     a_WN2->LoadGrid (dr_Rough,ir[i],l_NumNorthRows, l_NumEastCols); 

   } /* for i - do each type of rough array */

/* After all Rough arrays are done, fill elevation and give to windninja */
	  NN.Init(l_NumNorthRows, l_NumEastCols,dr_Elev,NODATA_VAL); 
   if ( !NN.Fill_Mngr()){
	     sprintf (cr_ErrMes, "%s - Error while filling missing elevations", NN.cr_Message);
      return 0; }

   a_WN2->LoadGrid (dr_Elev,e_Elev,l_NumNorthRows, l_NumEastCols); 
  
   delete[] dr_Elev;                   /* won't need these anymore */
   delete[] dr_Rough; 
 
   *ad_ElevRes = d_ElevRes;
   *al_NumNorthRows = l_NumNorthRows;
   *al_NumEastCols = l_NumEastCols;

  return 1; 
}

/*******************************************************************************
* Name: WindNinjaRun
* Desc: Run WindNinja for the specified wind speed and direction 
*   In: a_WD.....record from wind table with speed & direction
*  Out: *ai.....Error code, see code below
*  Ret: 1 ok 
*       0 = error or user terminated Ninja while it was running
*       if user terminates *ai == 0
*******************************************************************************/
int Farsite5::WindNinjaRun (CWindNinja2 *a_WN2, WindData *a_WD, double d_ElevRes, 
                             long l_NumNorthRows, long l_NumEastCols, char cr_ErrMes[])
{
bool b; 
int i_Min, i_Hr;
long  i,j,c,r, wnRows, wnCols, lN_EMiss, xl ; 
float f, gwRes; 
double d, d_Elev, yExtent, d_resol, unitsConv, minValue, maxValue; 
double   wnyExtent, xExtent, d_OutRes, *dr_Elev, *dr_Rough, wnxExtent;  

    strcpy (cr_ErrMes,""); 

//   a_WN2->Set_SpeedDirection(icf.WindSpeed(),icf.WindDirection());
   a_WN2->Set_SpeedDirection((float)a_WD->ws, (float)a_WD->wd);

/* Input/Output Wind Heights............................*/
   f = Farsite5::icf.s_FWNI.f_GriWinHei; /* Gridded Wind Height inputs file */ 
   if ( f == ef_ICFInit )                /* Not set in cmd file           */
     f = e_GWH_Def;                      /* defualt gridded wind height   */ 
   f = f * e_FtToMt; 		
   a_WN2->Set_InOutWindHeight (f,f);        /* set both the same */

/* Do Diurnal using time from the Wind table record */
//  a_WN2->Diurnal(true);            /* Yes, we have Diurnal data */
//	 a_WN2->Diurnal_Date (a_WD->dy,   /* Day, from WindData tbl rec */
 //                      a_WD->mo,   /* Month  "  */
 //                      2010);      /* Year - any one is ok */
//  i_Hr = a_WD->hr / 100;           /* Get Min & Hour from WindData rec */
//  i_Min = a_WD->hr - (i_Hr * 100);
//	 a_WN2->Diurnal_Time (1, i_Min, i_Hr, /* use 1 for second */
 //                      icf.s_FWNI.i_TimeZone); 	/* Time Zone from input cmd file */
	 
/* Use temperature we previously set into the Wind table record */
//  a_WN2->Diurnal_TempCloud (a_WD->a_FWN->f_Temperature, (float)a_WD->cl);

 // a_WN2->Diurnal_LatLong ( (double)GetLatitude(), 
//                          (double) icf.s_FWNI.f_Longitude); 
    
   if ( !_ChkSetRes (icf.s_FWNI.f_GriWinRes, d_ElevRes, 
                     l_NumNorthRows, l_NumEastCols, 
                     &d_OutRes, icf.cr_ErrExt)) {
      sprintf (cr_ErrMes, "Bad Gridded Wind Resolution input %f ",icf.s_FWNI.f_GriWinRes);
      return 0; }
	
  a_WN2->Set_MeshResolution (d_OutRes);
	  
   a_WN2->Set_NumVertLayers(20);  /* ninja team said to use 20 */
 
   i = GetMaxThreads(); 
   a_WN2->Set_NumberCPUs(i);

   a_WN2->Set_OutFiles (false);    /* Turn off Windninja's output */
   
   a_WN2->Compute_Domain_Height();   

   if ( !a_WN2->Check_Errors (cr_ErrMes) ){ /* chk inputs bfor run */ 
	     return 0; }
 
/* Run Wind Simulation */
   if ( !a_WN2->Simulate_Wind(cr_ErrMes)){
	     return 0; }  /* Error or User aborted */

/*..................................................... */ 
/* Windninja loaded 2 grid classes - angle and velocity */ 
double A,V;
long Rows, Cols; 

  Rows = a_WN2->Get_AngleGrid_Row();
  Cols = a_WN2->Get_AngleGrid_Col(); 

// Test dumps output grids windninja created.
// These calls use windninja grid functions to write output file
//   a_WN2->WriteAngleFile    ("c:\\LarryF\\WN2-Test\\wn-ang.asc");
//   a_WN2->WriteVelocityFile ("c:\\LarryF\\Vel-out.txt");

// Test dump windninja grid pulling values from it. 
//   _PutNinjaFile ( &WN2, "c:\\LarryF\\ang.asc", "Angle" );

/* Store the WindNinja grid info into the wind table record  */
  a_WD->a_FWN->nWindRows = Rows;
  a_WD->a_FWN->nWindCols = Cols;
  a_WD->a_FWN->windsResolution = a_WN2->Get_AngleGrid_CellSize();
 	a_WD->a_FWN->windsXllCorner =  a_WN2->Get_AngleGrid_xllCorner();
 	a_WD->a_FWN->windsYllCorner =  a_WN2->Get_AngleGrid_yllCorner();

/* allocate memory in Wind Table recrd for grids */
	 a_WD->a_FWN->windSpeedGrid = new short *[Rows];
 	a_WD->a_FWN->windDirGrid = new short *[Rows];
  	for(long r = 0; r < 	a_WD->a_FWN->nWindRows; r++)	{
 	 		a_WD->a_FWN->windSpeedGrid[r] = new short[Cols];
	 	 	a_WD->a_FWN->windDirGrid[r] = new short[Cols];
  		 for(long c = 0; c < 	a_WD->a_FWN->nWindCols; c++)	{
  			  	a_WD->a_FWN->windSpeedGrid[r][c] = 	a_WD->a_FWN->windDirGrid[r][c] = NODATA_VAL;}
 }

/* Copy the grids into the wind table record */
	 // long wnRow = Rows;        /* ninja grids keep 1st row last */
   for (r = 0; r < Rows; r++){
    // wnRow--; 
		   for (c = 0; c < Cols; c++){
		     A = a_WN2->Get_Angle (r,c);
       V = a_WN2->Get_Velocity (r,c); 
      	a_WD->a_FWN->windSpeedGrid[r][c] = (short)V;
      	a_WD->a_FWN->windDirGrid  [r][c] = (short)A;
    }}    

  return 1; 
}

 
/*******************************************************************************
*  Saving this it might come in handy 
*
*******************************************************************************/
#ifdef wowow

/* -------------------------------------------------------------
* Name: _PutNinjaFile
* Desc: Used for testing.  
*       Dump out the angle or velocity grid that comes 
*       from windninja. 
*   In: FN...file name
*       Type..."Angle" or "Velocity"
* ---------------------------------------------------------------*/
int _PutNinjaFile (CWindNinja2 *wn2, char FN[], char Type[])
{
FILE *fh;
long l_row, l_col,r,c; 
double d,d_xll, d_yll, d_CS;  
   fh = fopen (FN,"w");
   if ( fh == NULL )
     return 0; 

/* Can use these for angle or vel, both are same sizes */
    l_row = wn2->Get_AngleGrid_Row();
    l_col = wn2->Get_AngleGrid_Col(); 
 
		  	for(r = 0; r < l_row; r++ ){ 
       fprintf (fh,"Row: %d \n", r); 
		     for (c = 0; c < l_col; c++){
		       if ( !_stricmp (Type, "Angle" ))
           d = wn2->Get_Angle (r,c);       /* Do one or the other */
         else 
           d = wn2->Get_Velocity (r,c);
        fprintf (fh, "%5.1f ", d);
      }
      fprintf (fh,"\n");  
   }    
   fclose (fh); 
   return 1; 
}

#endif 
 	
/* ***********************************************************************
* Name: GetRough
* Desc: Set WN roughness height values  
*  Ret: 1 ok set,
*       0 = all need FlamMap landscape values were missing 
*********************************************************************** */
float Farsite5::GetRough (int r, int c, int i_Type)
{
int i,j; 
bool b; 
float f_Conv, f_CanHgt, f_CanCov, f_FueMod, f_FueBed; 
float f;

/* Canopy Height..........................................................*/
  i = Farsite5::GetTheme_Units (H_DATA);      /* Canopy Height units ?        */
  if ( i == 1 || i == 3 )               /* if Canopy Height is in meters  */
				f_Conv =  e_FtToMt / 10.0 ;       
  else                                
    f_Conv = 1.0 / 10.0;  
  
  f_CanHgt = Farsite5::GetLayerValueByCell(H_DATA, c, r);   /* Canopy Height      */
  if ( f_CanHgt != NODATA_VAL )
    f_CanHgt = f_CanHgt * f_Conv;    /* Get to Meters */

/* Canopy Cover ........................................................*/
   f_CanCov = Farsite5::GetLayerValueByCell(C_DATA, c, r);   /* % Canopy Cover    */
  	if ( f_CanCov != NODATA_VAL ) {
     if ( !Farsite5::GetTheme_Units (C_DATA) )      /*if Canopy Cover is in Classes   */
				   f_CanCov = Farsite5::ConvertUnits(f_CanCov, 0, 1, C_DATA); /* Class --> percent */
			}
	 
/* Fuel Model .......................................................*/
   f_FueMod = Farsite5::GetLayerValueByCell(F_DATA, c, r);   /* Fuel Model        */

   if ( f_CanHgt == NODATA_VAL && f_CanCov == NODATA_VAL && f_FueMod == NODATA_VAL )
     return 0; 
				
/* Fuel Bed Depth ..................................................*/
    f_FueBed = Farsite5::GetFuelDepth((int)f_FueMod) * e_FtToMt;   /* Always comes back as feet, so convert */
     
/* Call Jason's WN to set all rough values into WN Grids,  */
/*  would only get an error back if send in a bad unit     */
   f = computeSurfPropForCell( i_Type,              /* Rough Value to get */
                             (double) f_CanHgt,    /* meters */ 
                             (int) f_CanCov,       /* percent 1 -> 100 */     
                             (int) f_FueMod, 
                             (double) f_FueBed);  /* meters */
 
  return f; 
}



/**********************************************************************
* Name: GetLayerValueByCell 
* Desc: Get the landscape layer value from the specific Column & Row
* NOTE - I took this function from FlamMap so I had to change a few
*        things. see notes below
*  
*  Ret: -1 if out side grid
***********************************************************************/	 
float Farsite5::GetLayerValueByCell (int _layer, int col, int row)
{
float ret = -1;
celldata tCell;
crowndata tCrown;
grounddata tGround;
long  analysisRect_left, analysisRect_top, analysisRect_bottom;
long posit; 

// analysisRect was a struct in flammap
  analysisRect_left = 0; 
  analysisRect_top = 0; 
	 analysisRect_bottom = Header.numnorth - 1;

 	int tCol = col + analysisRect_left; 
  int tRow = row + analysisRect_top;
	 long northi = tRow;   
	 if (tCol < 0 || tCol >= Header.numeast)
		  return -1;
	 if (northi < 0 || northi >= Header.numnorth)
	  	return -1;

// Code from FlamMap used the __int64
// 	__int64 posit = ((__int64)northi * (__int64) Header.numeast + col);
 	 posit = northi * Header.numeast + col;

 	if( !CantAllocLCP)
		  GetCellDataFromMemory (posit, tCell, tCrown, tGround);

 	else	{
	  	double e = analysisRect_left + col * GetCellResolutionX();
		  double n = analysisRect_bottom + row * GetCellResolutionY();
	  	CellData (e, n, tCell, tCrown, tGround, &posit);	}

	 switch(_layer)	{
	   case E_DATA:
		    ret = tCell.e;
		    break;
	case S_DATA:
		ret = tCell.s;
		break;
	case A_DATA:
		ret = tCell.a;
		break;
	case F_DATA:
		ret = tCell.f;
		break;
	case C_DATA:
		ret = tCell.c;
		break;
	case H_DATA:
		ret = tCrown.h;
		break;
	case B_DATA:
		ret = tCrown.b;
		break;
	case P_DATA:
		ret = tCrown.p;
		break;
	case D_DATA:
		ret = tGround.d;
		break;
	case W_DATA:
		ret = tGround.w;
		break;
	}
	return ret;
}


/* **************************************************************************
* Name: computeSurfPropForCell
* This function computes surface properties for WindNinja, based on information
* available in a FARSITE .lcp file.
* If any of the input parameters (canopyHeight, canopyCover, fuelModel, fuelBedDepth)
* are not available (NO_DATA), enter a value less than 0.
* If sufficient data is not passed to this function to determine the surface properties,
* default values (rangeland) are used
* In:
*    canopyHeight = meters
*    canopyCover  = percent 0 -> 100 
*    fuelModel
*    fuelBedDepth = meters
*
*    i_Type: e_Rough_h, e_Rough_d, e_Roughness, e_Albedo
*            e_Bowen, e_Cg, e_Anthropog
*
* Ret: return the requested i_Type
*
*************************************************************************** */
float  Farsite5::computeSurfPropForCell ( int i_Type, double canopyHeight, int canopyCover,
                              int fuelModel, double fuelBedDepth)                              
{
double f = NODATA_VAL; 
double Rough_h, Rough_d, Roughness, Albedo;
double Bowen, Cg, Anthropog  ;


  if ( canopyHeight < 0 && canopyCover < 0 &&
        fuelModel    < 0 && fuelBedDepth < 0 )
     return NODATA_VAL;

// Go through logic of determining surface properties, depending on what data is available at this cell

  if (canopyCover >= 5 && canopyHeight > 0 )  {
    Rough_h   = canopyHeight;
    Rough_d   = canopyHeight * 0.63;
    Roughness = canopyHeight * 0.13;
    Albedo    = 0.1;   //assuming forest land cover for heat transfer parameters
    Bowen     = 1.0;
    Cg        = 0.15;
    Anthropog     = 0.0; }
		
// See if it's an unburnable Fuel Model
  else if ( fuelModel==90 || fuelModel==91 || fuelModel==92 ||
       fuelModel==93 || fuelModel==98)   {
     switch ( fuelModel )   {
       case 90:    // Barren
         Rough_h   = 0.00230769;
         Rough_d   = 0.00230769 * 0.63;
         Roughness   = 0.00230769 * 0.13;
         Albedo   = 0.3;
         Bowen   = 1.0;
         Cg   = 0.15;
         Anthropog     = 0.0;
         break;
       case 91:    // Urban Roughness
         Rough_h   = 5.0;
         Rough_d   = 5.0 * 0.63;
         Roughness   = 5.0 * 0.13;
         Albedo   = 0.18;
         Bowen   = 1.5;
         Cg   = 0.25;
         Anthropog     = 0.0;
         break;
       case 92:        // Snow Ice
         Rough_h   = 0.00076923;
         Rough_d   = 0.00076923 * 0.63;
         Roughness   = 0.00076923 * 0.13;
         Albedo   = 0.7;
         Bowen   = 0.5;
         Cg   = 0.15;
         Anthropog     = 0.0;
         break;
       case 93:        // Agriculture
         Rough_h   = 1.0;
         Rough_d   = 1.0 * 0.63;
         Roughness   = 1.0 * 0.13;
         Albedo   = 0.15;
         Bowen   = 1.0;
         Cg   = 0.15;
         Anthropog     = 0.0;
         break;
       case 98:        // Water
         Rough_h   = 0.00153846;
         Rough_d   = 0.00153846 * 0.63;
         Roughness   = 0.00153846 * 0.13;
         Albedo   = 0.1;
         Bowen   = 0.0;
         Cg   = 1.0;
         Anthropog     = 0.0;
         break;
      } /* switch */
    } /* if */
	
// just use fuel bed depth
  else if ( fuelBedDepth > 0.0 )  {
     Rough_h   = fuelBedDepth;
     Rough_d   = fuelBedDepth * 0.63;
     Roughness   = fuelBedDepth * 0.13;
     Albedo   = 0.25;  //use rangeland values for heat flux parameters
     Bowen   = 1.0;
     Cg   = 0.15;
     Anthropog     = 0.0;  }

// if there is a canopy height (no fuel model though)
  else if ( canopyHeight > 0.0) {
     Rough_h   = canopyHeight;
     Rough_d   = canopyHeight * 0.63;
     Roughness   = canopyHeight * 0.13;
     Albedo   = 0.1;      // assume forest land for heat flux parameters
     Bowen   = 1.0;
     Cg   = 0.15;
     Anthropog     = 0.0;  }

// If we make it to here, we'll just choose parameters based on rangeland...
   else {
     Rough_h   = 0.384615;
     Rough_d   = 0.384615 * 0.63;
     Roughness   = 0.384615 * 0.13;
     Albedo   = 0.25;
     Bowen   = 1.0;
     Cg   = 0.15;
     Anthropog     = 0.0; }

   if      ( i_Type == e_Rough_h   ) f =  Rough_h;
   else if ( i_Type ==  e_Rough_d   ) f =  Rough_d;
   else if ( i_Type ==  e_Roughness ) f =  Roughness;
   else if ( i_Type ==  e_Albedo    ) f =  Albedo;
   else if ( i_Type ==  e_Bowen     ) f =  Bowen;
   else if ( i_Type ==  e_Cg        ) f =  Cg;
   else if ( i_Type ==  e_Anthropog ) f =  Anthropog;
   else f = NODATA_VAL; /* shouldn't happen */
 
   return f;
}
     

/*************************************************************************/
float Farsite5::ConvertUnits(float val, int srcUnits, int destUnits, int themeType)
{
	float tVal = val;
	if(val != NODATA_VAL)//&& dataUnits)// != destUnits)
	{
		switch(themeType)
		{
		case E_DATA:
			if(srcUnits == 0 && destUnits == 1)
				tVal *= 3.2808;
			else if(srcUnits == 1 && destUnits == 0)
				tVal /= 3.2808;
			break;
		case S_DATA:
			if(srcUnits == 0 && destUnits == 1)
			{//convert to percent
				tVal = tan((PI * tVal)/ 180.0) * 100.0;
			}
			else if(srcUnits == 1 && destUnits == 0)
			{//convert percent to degrees
				double fraction, ipart;
				double slopef;

				tVal = slopef = atan((double) tVal / 100.0) / PI * 180.0;
				fraction = modf(slopef, &ipart);
				if(fraction>=0.5)
					tVal++;
			}
			break;
		case A_DATA://always convert to degrees if necessary
			if(srcUnits == 0)// grass 1-25 counterclockwise from east to degrees
			{
				if(tVal != 25.0)
				{
					tVal = (tVal - 1.0) * 15.0 - 90;
					if(tVal <= 0)
						tVal = fabs(tVal);
					else
						tVal = 360 - tVal;
				}
			}
			else if(srcUnits == 1)// degrees 0 to 360 counterclockwise from east to degrees
			{
				tVal = tVal - 90;
				if(tVal <= 0)
					tVal = fabs(tVal);
				else
					tVal = 360 - tVal;
			}
			//otherwise already in degrees
			break;
		case F_DATA://no conversions
			break;
		case C_DATA://can be in classes or percent
			if(srcUnits == 0 && destUnits == 1)//convert class to percent
			{
				if(val == 99.0) 
					tVal=0; 
				else if(val == 1.0)  
					tVal=10; 
				else if(val == 2.0)  
					tVal=30; 
				else if(val == 3.0)  
					tVal=60; 
				else if(val == 4.0)  
					tVal=75; 
				else 
					tVal=0; 

			}
			else if(srcUnits == 1 && destUnits == 0)//convert percent to class
			{
				if(val >= 70)
					tVal = 4;
				else if(val >= 45)
					tVal = 3;
				else if(val >= 20)
					tVal = 2;
				else if(val > 0)
					tVal = 1;
				else
					tVal = 0;
			}
			break;
		case H_DATA://always divide by 10...
  			//tVal /= 10.0;
			if((srcUnits == 0 || srcUnits == 2) && destUnits == 1)
			{//convert meters to feet
				tVal *= 3.2808;
			}
			else if((srcUnits == 1 || srcUnits == 3) && destUnits == 0)
			{//convert feet to meters
				tVal /= 3.2808;
			}
			break;
		case B_DATA:
			//tVal /= 10.0;
			if((srcUnits == 0 || srcUnits == 2) && destUnits == 1)
			{//convert meters to feet
				tVal *= 3.2808;
			}
			else if((srcUnits == 1 || srcUnits == 3) && destUnits == 0)
			{//convert feet to meters
				tVal /= 3.2808;
			}
			break;
		case P_DATA:
			//tVal /= 100.0;
			if(srcUnits == 0 || srcUnits == 2)//kg/m3
			{
				if(destUnits == 1)
					tVal *= 100.0;
				else if(destUnits == 2 || destUnits == 3)
				{
					tVal /= 16.0185;
					if(destUnits == 3)
						tVal *= 1000.0;
				}
			}

			else if(srcUnits == 1 || srcUnits == 3)
			{
				if(destUnits == 3)
					tVal *= 1000.0;
				else if(destUnits == 0 || destUnits == 1)
				{
					tVal *= 16.0185;
					if(destUnits == 1)
						tVal *= 100.0;
				}
			}

			break;
		case D_DATA:
			//tVal /= 10.0;
			if(srcUnits == 1 && destUnits == 1)
			{//convert Mg/Ha to Tons/Acre
				tVal /= 2.2417088978002777;
			}
			else if(srcUnits == 2 && destUnits == 0)
			{//convert Tons/Acre to Mg/Ha
				tVal *= 2.2417088978002777;
			}
			break;
		case W_DATA://classes, don't change
			break;

		}
	}
	return tVal;

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: _ChekSetRes
* Desc: Check the requested Gridded Wind Resolution from the cmd file 
*       against the native lcp elevation resolution and assign the resolution
*       we need to send into WindNinja
*       We don't allow the requested cmd file res to be less than the lcp
*        because of how memory is allocate in the WindNinja class for pass
*        array values in and out. 
* Ret: 1 = ok,  0 err - see error message
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  _ChkSetRes (float f_CmdFilRes, double d_lcpRes, long lcpRows, long lcpCols, double *ad, char cr_ErrMes[])
{
	double d; 
	strcpy (cr_ErrMes,""); 

	if ( d_lcpRes > e_GWR_Max )
	{  /* prevent this, else screws up our logic  */ 
		sprintf (cr_ErrMes,"Native LCP resolution %6.1f exceeds Gridded Wind Resolution Limit %.1f",d_lcpRes,e_GWR_Max);
		return 0; 
	}

	if ( f_CmdFilRes != ef_ICFInit )
	{        /* a res was in cmd file        */
		if ( (double)f_CmdFilRes < d_lcpRes )
		{
			sprintf (cr_ErrMes,"Specified Gridded Winds Resolution %4.1f is below native lcp %.1f", f_CmdFilRes, d_lcpRes);
			return 0; 
		}
		if ( f_CmdFilRes > e_GWR_Max )
		{
			sprintf (cr_ErrMes,"Resolution %4.1f exceeds limit of %.1f", f_CmdFilRes,e_GWR_Max);
			return 0; 
		}    
		//Now make adjustments to output resolution so that: 
		//1) output resolution is a multiple of the native lvp resolution
		// and has same extent as the lcp
		//2) Minimum of 3 output rows and output cols, else WindNinja fails
		/*long lcd = LCD(lcpRows, lcpCols);
		double tRes = d_lcpRes;
		if(lcd > 1)
		{
			while((tRes * lcd) <= f_CmdFilRes)
				tRes *= lcd;
		}
		//now make sure at least 3 rows and columns, else set to native lcp resolution as default!
		if((lcpRows * d_lcpRes) / tRes < 3.0 || (lcpCols * d_lcpRes) / tRes < 3.0)
		{
			tRes = d_lcpRes;
		}
		*ad = tRes; */
		*ad = f_CmdFilRes;
		return 1; 
	}

	if ( f_CmdFilRes == ef_ICFInit )    /* if no Resol set in cmd file        */  
		d = e_GWR_Def;                    /*  use default                       */
	if ( d < d_lcpRes )                 /* make sure not less than Elev lcp res */
		d = d_lcpRes; 
	*ad = d;
	return 1;    
}

 
/********************************************************************************
* Name: DeleteWindGrids
* Desc: Delete the wind speed and direction grids that were created 
*        when/if we ran WindNinja
********************************************************************************/
void Farsite5::DeleteWindGrids(d_FWN *a_FWN, int iS_WN)
{
  if ( a_FWN == NULL )  /* Farsite WindNinja Struct never assigned */ 
    return; 

  if ( iS_WN != e_WN ) {  /* this is not the master, just a pointer */
    a_FWN = NULL;         /* to the actual FWN that did the Ninja run */
    return; }

/* Free up allocated wind grids, could be NULL if user aborted Ninja run */
	 if (a_FWN->windSpeedGrid){
	  	for (long i = 0; i < a_FWN->nWindRows; i++) {
		   	delete[] a_FWN->windSpeedGrid[i]; }
	
	   delete[] a_FWN->windSpeedGrid;
		  a_FWN->windSpeedGrid = NULL;	}

	 if (a_FWN->windDirGrid)	{
	 	for(long i = 0; i < a_FWN->nWindRows; i++)
			  delete[] a_FWN->windDirGrid[i];
	 	delete[] a_FWN->windDirGrid;
	 	a_FWN->windDirGrid = NULL;	}

 	a_FWN->windsXllCorner = a_FWN->windsYllCorner = a_FWN->windsResolution = 0.0;
 	a_FWN->nWindRows = a_FWN->nWindCols = 0L;
  delete a_FWN;
  a_FWN = NULL; 
}

/******************************************************************************
* Name: GetWindGridSpeedByCoord
* Desc: get the wind speed that was created by windninja and stored in
*
******************************************************************************/
short Farsite5::GetWindGridSpeedByCoord(double xCoord, long yCoord, d_FWN *a_FWN)
{
short s; 
	if (a_FWN->windsResolution <= 0)
		return NODATA_VAL;
	long r = (yCoord - a_FWN->windsYllCorner) / a_FWN->windsResolution;
	long c = (xCoord - a_FWN->windsXllCorner) / a_FWN->windsResolution;
	s =  GetWindGridSpeed(r, c, a_FWN);
 return s; 
}

short Farsite5::GetWindGridSpeed(long wndRow, long wndCol, d_FWN *a_FWN)
{
short s;
	if(wndRow < 0 || wndRow >= a_FWN->nWindRows || wndCol < 0 || wndCol >= a_FWN->nWindCols)
		return 0;
	s = a_FWN->windSpeedGrid[wndRow][wndCol];
 return s;
}


/******************************************************************************
* Name:
*
*
******************************************************************************/
short Farsite5::GetWindGridDirByCoord(double xCoord, long yCoord, d_FWN *a_FWN)
{
short s; 

	if ( a_FWN->windsResolution <= 0)
	 	return NODATA_VAL;
	long r = (yCoord - a_FWN->windsYllCorner) / a_FWN->windsResolution;
	long c = (xCoord - a_FWN->windsXllCorner) / a_FWN->windsResolution;
	s = GetWindGridDir(r, c,a_FWN);
 return s; 
}

short Farsite5::GetWindGridDir (long wndRow, long wndCol, d_FWN *a_FWN)
{
	if (wndRow < 0 || wndRow >= a_FWN->nWindRows || wndCol < 0 || wndCol >= a_FWN->nWindCols)
		return 0;
	return a_FWN->windDirGrid[wndRow][wndCol];
}


/*********************************************************************************
* Name: Disp_WindTbl_WN   - for TESTING 
* Desc: Display some WindNinja related info in the Wind Table
*
*********************************************************************************/
void Farsite5::Disp_WindTbl_WN()
{
int i;
  printf ("_____________________________________________________________\n");
  for (i = 0; i < MaxWindObs[0]; i++ ) {  /* for each wind record */
    if ( wddt[0][i].iS_WN == e_NA )
      continue;                /* only want recs within burn periods */

    printf ("%d, Mth: %2d Day: %2d Hr: %4d, Dir: %3d  Spd: %4.1f ", 
             i, wddt[0][i].mo, wddt[0][i].dy, wddt[0][i].hr,
             wddt[0][i].wd, wddt[0][i].ws ); 
    if ( wddt[0][i].iS_WN == e_WN )
         printf ("WindNinja Master"); 
    else if ( wddt[0][i].iS_WN == e_PT )
        printf ("Pointer");
    else 
        printf ("Shit Bug "); 
    
    printf ("\n"); 

   }

   printf ("--------------------------------------------------------------\n"); 

}

#ifdef Larry-out
/******************************************************************************
* Name: WindNinja_Progress
* Desc: 
*
*  Out: afN_WNToDo...the number of Ninja runs that are needed to be run
*                    this won't get set until right before Ninja is actual
*                    runs. 
*******************************************************************************/
float Farsite5::WindNinja_Progress(float *afN_WNToDo)
{
int i;
float pc, f_Slice,g; 

 *afN_WNToDo = this->fN_WNToDo;  /* total # of runs needed */

/* Ninja not even requested to run */
 if ( !stricmp (this->icf.s_FWNI.cr_GriWin,e_GW_No) )
    return -1.0; 
 
  if ( this->fN_WNToDo == 0 )   /* Ninja did start yet */
     return 0; 
 
  if ( this->fN_WNToDo == this->fN_WNDone )  /* all runs of Ninja completed */
     return 1.0; 

  if ( this->i_RunStatus != e_WindNinja )   /* Shouldn't happen but might so */
     return 1.0;                            /* would mean Ninja had run */

  pc = this->WN2.Get_Progress();  /* comes back as 0 -> 1.0 */

 if ( pc > 0 )
   printf (" ---> %f \n", pc); 

  f_Slice = 1.0 / this->fN_WNToDo; /* each run gets this percent of total time */

  g = f_Slice * this->fN_WNDone;   /* The finished Ninja runs */

  g = g + pc;                      /* add percent of current Ninja that is running */

  if ( g > 1.0 )
   printf ("Oh Shit \n"); 

  return g; 

}
#endif 


#ifdef wwwwwww

  old function 
/*********************************************************************************
* Name: Load_WindNinja
* Desc: Do all needed Ninja runs.
*       
*********************************************************************************/
int Farsite5::Load_WindNinja(char cr_ErrMes[])
{   
int i, j, i_Err, iN_WN, iN_Pts;
double d_ElevRes; 
long l_NumNorthRows, l_NumEastCols;

  strcpy (cr_ErrMes,""); 

  if ( !stricmp (this->icf.s_FWNI.cr_GriWin,e_GW_No)) /* Check Inputs File Switch */
      return 1;                      /* Not doing WindNinja grid winds */
   

  iN_WN = iN_Pts = 0;                /* counters, Ninja runs */
/*---------------------------------------------------------------------------------------*/
/* First go thru and mark all the Wind recs that are within the burn period */
  for ( i = 0; i < NumRelativeData; i++)	{  /* For each Burn Period */ 
   for (j = 0; j < MaxWindObs[0]; j++ ) {  /* for each wind record */
  
     wddt[0][j].a_FWN == NULL;   /* Init these */
     wddt[0][j].iS_WN = e_NA;   

     if ( abp[i].Month != wddt[0][j].mo || abp[i].Day != wddt[0][j].dy )
         continue;              /* Not in month or day of burn perios */
  
     if (  wddt[0][j].hr >= this->abp[i].Start && wddt[0][j].hr <= this->abp[i].End ) 
       wddt[0][j].iS_WN = e_BP; /* Mark Record as being within the Burn Period */
   }}       


/*--------------------------------------------------------------------------*/
/* Recs in burn period, determine what can be bined/resued & what to run Ninja on  */
   fN_WNToDo = 0; 
   for (j = 0; j < MaxWindObs[0]; j++ ) {
     if ( wddt[0][j].iS_WN != e_BP )   /* not in the burn period */
       continue;  

//     printf ("\n----> %d,  Mth: %d Day: %d Hr: %d, Dir: %d Spd: %6.1f \n",  j, wddt[0][j].mo, wddt[0][j].dy, wddt[0][j].hr, wddt[0][j].wd, wddt[0][j].ws); 

/* Is there an existing Ninja run we can use, if YES than set it up */
     if ( WindNinja_SetExisting(j) ) {
        iN_Pts++;                       /* Count # of reuse */
        continue;}                      /* found and using an exisint ninja run */

     wddt[0][j].a_FWN = new d_FWN;      /* This will hold wind grids and other stuff */
     wddt[0][j].a_FWN->windDirGrid = NULL;
     wddt[0][j].a_FWN->windSpeedGrid =  NULL; 
     wddt[0][j].iS_WN = e_WN;  /* Says Ninja was run on this wind record */  
     wddt[0][j].a_FWN->f_Temperature = Temperature (wddt[0][j].mo, wddt[0][j].dy, wddt[0][j].hr); 
     iN_WN++; 
     this->fN_WNToDo++;           /* Ninja runs to be done */
//     printf ("Ninja %d,  Mth: %d Day: %d Hr: %d, Dir: %d  Spd: %6.1f \n", j, wddt[0][j].mo, wddt[0][j].dy, wddt[0][j].hr,  wddt[0][j].wd, wddt[0][j].ws); 
    }  /* for j */

//    printf (" %d  Bined Grids used \n", iN_Pts); 
//    printf ("%d WindNinja Runs - Starting \n", iN_WN); 
/* Get the WindNinja class ready */
   i = WindNinjaInit (&WN2,  &d_ElevRes, &l_NumNorthRows, &l_NumEastCols, cr_ErrMes);
   if ( i == 0 ) 
     return 0;            /* error init'ing ninja */

/*-------------------------------------------------------------*/
/* Run Ninja on the selected Wind records                      */
/* NOTE: this could go into an Open MP loop at some point, would have to think about how */
/*  it might work, might need to do multiple WindNinja objects.  */

    this->FPC.SetTimeSlice(iN_WN,this);   /* tells Progress Class how to time slice */  
    
    this->FPC.Set_NinjaRunning(iN_WN); 

   for ( i = 0; i < MaxWindObs[0]; i++ ) {
     if ( wddt[0][i].iS_WN != e_WN )
        continue; 
//     printf ("Run Ninja %d \n",i );
     j = WindNinjaRun (&WN2, &wddt[0][i], d_ElevRes, l_NumNorthRows, l_NumEastCols, cr_ErrMes);
     if ( j == 0 )
       return 0;                   /* Error or User Abort */
     this->FPC.Set_NinjaInc();     /* tell Progress Class another ninja is completed */
   }

// Test display 
//  Disp_WindTbl_WN();


 return 1; 
}
#endif

