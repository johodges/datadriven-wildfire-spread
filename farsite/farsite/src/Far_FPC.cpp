/********************************************************************
* Name: Far_FPC.cpp
* Desc: Farsite Progress Class,
*       Handles the percent of completion for the program.
*
*********************************************************************/
#include "Farsite5.h"


/*******************************************************************/
FPC::FPC()
{
  Init();
}

FPC::~FPC()
{
 // DeleteCriticalSection(&CriSec);
}

/**********************************************************************/
void FPC::Init ()
{

 i_State = e_Start;
 f_PreProg = 0;

 f_pcNinja = 0;    /* percentages of time that each section */
 f_pcCond = 0;     /* is approximated to run */
 f_pcFar = 0;

	//InitializeCriticalSection(&CriSec);

}

/***********************************************************************/
void FPC::Set_NinjaRunning (float f)
{
	//  EnterCriticalSection(&CriSec);

   this->i_State = e_Ninja;
   this->f_WNComplete = 0;
   this->f_WNToDo = f;

	//  LeaveCriticalSection(&CriSec);
  }

/***********************************************************************/
void FPC::Set_NinjaPrep ()
{
	  //EnterCriticalSection(&CriSec);
   this->i_State = e_NinjaPrep;
	 // LeaveCriticalSection(&CriSec);
  }

/***********************************************************************/
void FPC::Set_NinjaInc ()
{
	 // EnterCriticalSection(&CriSec);
   this->f_WNComplete++;
	 // LeaveCriticalSection(&CriSec);
 }

/***********************************************************************/
void FPC::Set_CondRunning ()
{
	 // EnterCriticalSection(&CriSec);
   this->i_State = e_Cond;
	 // LeaveCriticalSection(&CriSec);
}

/***********************************************************************/
void FPC::Set_FarsiteRunning ()
{
	 // EnterCriticalSection(&CriSec);
   this->i_State = e_Far;
	//  LeaveCriticalSection(&CriSec);
}

/**********************************************************************
* Name:
* Desc:
*
*  Out: cr.....which section of the program is currently running
*  Ret: percent to completeion
***********************************************************************/
//float  FPC::GetProgress (Farsite5 *a_F5, CWindNinja2 *a_WN2, CFMC *a_cfmc, char cr[])
float  FPC::GetProgress (Farsite5 *a_F5, CFMC *a_cfmc, char cr[])
{
float f,pc;

  //	EnterCriticalSection(&CriSec);

   strcpy (cr,"");

/* NOTE things run in this order Start, Ninja-Prep, Ninja, Cond, Farsite */

/* Haven't even got to ninja yet */
  if ( this->i_State == e_Start ) {
     strcpy (cr,"Startup");
     return LeaCriSec(0); }

/* Ninja Prep only takes a few seconds, mostly to run the Nearest Neighbot stuff once */
  if ( this->i_State == e_NinjaPrep ) {
     strcpy (cr, "WindNinja-Prep");
     return LeaCriSec(0); }


  if ( f_pcFar == 0 ) {    /* this might occur, means ninja didn't start yet */
     strcpy (cr,"Startup");
     return LeaCriSec(0); }


  if ( i_State == e_Ninja ) {
    sprintf (cr,"WindNinja %1.0f-%1.0f",this->f_WNComplete+1,this->f_WNToDo);
    pc = 1.0;//WindNinja_Progress (a_WN2);
    f = pc * this->f_pcNinja;
    return LeaCriSec (f); }


  if ( i_State == e_Cond ) {
     strcpy (cr,"Conditiong");
     pc = a_cfmc->Get_ProgressF();   /* Comes back 0 -> 1.0 */
     pc = pc * this->f_pcCond;       /* allocted portion of total run time */
     f = this->f_pcNinja + pc;       /* add it to previous time spent running */
     return LeaCriSec (f); }

  if ( i_State == e_Far ) {
     strcpy (cr, "FarsiteBurning");
     pc = a_F5->GetFarsiteProgress();
     f = this->f_pcNinja + this->f_pcCond;
     f = f + ( pc * this->f_pcFar );
     return LeaCriSec (f); }

/* Shouldn't get here, everthing should have been taken care of above */
  strcpy (cr,"Unknown process state");
 // LeaveCriticalSection(&CriSec);
  return 0;
}

/*******************************************************************************
* Convenient way for caller to leave crit section and return a value
*
*******************************************************************************/
float FPC::LeaCriSec (float f)
{
  //LeaveCriticalSection(&CriSec);
  return f;
}


/******************************************************************************
* Name: WindNinja_Progress
* Desc: Percentage of ninja's work done so far.
*       Ninja will be doing 1 or more runs.
* Return: 0 -> 1.0
*******************************************************************************/
/*float FPC::WindNinja_Progress (CWindNinja2 *a_WN2)
{
float f,g, pc, f_One, f_Done;

   if ( this->f_WNToDo == 0 )  // humm should even be calling this //
      return 0;

   f_One = 1.0 / this->f_WNToDo;
   f_Done = f_One * this->f_WNComplete;

   if ( f_Done > 1.0 )
     return 1.0;

   pc = a_WN2->Get_Progress();  //comes back as 0 -> 1.0 //

   pc = pc * f_One;

  f =  f_Done + pc;
   return f;
}
*/

/********************************************************************************
* Name: SetTimeSlice
* Desc: Set the percentage of time slices that each component running in
*        Farsite will approximately take up.
*       For example:  Ninja will run 25 percent of the time, Cond will do 25%
*        and Farsite will do the other 50 %
********************************************************************************/
int FPC::SetTimeSlice (int i_NumNinRun, Farsite5 *a_F5)
{
float f;

	 // EnterCriticalSection(&CriSec);


/* NOTE - not using this yet - this is how many minutes Conditioning will run */
/* I'm thinking of using it do help determine what portion of time Conditioning */
/* will take up, the more minutes it needs to Condition the longer it takes */
 //  f_CondMin = GetCondTime (a_F5);

/* Ninja could run serveral times, so set a rough percent based on number of */
/*  that will need to be run */
/* FOR NOW - this is just a wild guess */
  if ( i_NumNinRun <= 0 )
    f_pcNinja = 0;
  else if ( i_NumNinRun == 1 )
    f_pcNinja = 0.25;
  else if ( i_NumNinRun == 2 )
    f_pcNinja = 0.40;
  else if ( i_NumNinRun == 3 )
    f_pcNinja = 0.60;
  else
    f_pcNinja = 0.80;

  f = 1.0 - f_pcNinja;  /* Get remaining percentage */

  f_pcCond = f * 0.25;  /* Give Condtioning 25 percent of remaining */

  f_pcFar = 1.0 - ( f_pcCond + f_pcNinja ); /* Farsite gets the rest */

	// LeaveCriticalSection(&CriSec);
	 return 0;
}

/************************************************************************
*
*
*
*************************************************************************/
float FPC::GetCondTime (Farsite5 *a_F5)
{
int i_StrYr, i_MthStart, i_DayStart, i_HourStart,date, min;
int EndMin, StrMin, MaxMin;
    i_StrYr     = a_F5->icf.a_Wtr[1].i_Year;     /* Start Conditioning For Daily Weather */
    i_MthStart  = a_F5->icf.a_Wtr[1].i_Mth;    /*  on the second day */
    i_DayStart  = a_F5->icf.a_Wtr[1].i_Day;
    i_HourStart = 0;


//   cfmc->Set_DateStart (i_StrYr, i_MthStart, i_DayStart, i_HourStart);

   date = GetMCDate (i_MthStart, i_DayStart, i_StrYr);
   min = MilToMin (i_HourStart);              /* Military time to minutes */
   StrMin = (date * e_MinPerDay) + min;       /* to total minutes */

   date = GetMCDate (a_F5->icf.i_FarsiteEndMth, a_F5->icf.i_FarsiteEndDay, a_F5->icf.i_FarsiteEndYear);
   min = MilToMin (a_F5->icf.i_FarsiteEndHour);
   EndMin = (date * e_MinPerDay) + min;      /* to total minutes */

   MaxMin = EndMin - StrMin;

   MaxMin = MaxMin + (int) a_F5->actual;
//   d = (double) MaxMin;

//   d += this->actual;


  return MaxMin;
}




#ifdef werwerew
=======================================================================================
    i_StrYr =  icf->a_Wtr[1].i_Year;     /* Start Conditioning For Daily Weather */
    i_MthStart = icf->a_Wtr[1].i_Mth;    /*  on the second day */
    i_DayStart = icf->a_Wtr[1].i_Day;
    i_HourStart = 0;

/*--------------------------------------------------------------------*/
CondDate:

   cfmc->Set_DateStart (i_StrYr, i_MthStart, i_DayStart, i_HourStart);

   date = GetMCDate (i_MthStart, i_DayStart, i_StrYr);
   min = MilToMin (i_HourStart);              /* Military time to minutes */
   StrMin = (date * e_MinPerDay) + min;       /* to total minutes */

   date = GetMCDate (icf->i_FarsiteEndMth, icf->i_FarsiteEndDay, icf->i_FarsiteEndYear);
   min = MilToMin (icf->i_FarsiteEndHour);
   EndMin = (date * e_MinPerDay) + min;      /* to total minutes */

   MaxMin = EndMin - StrMin;
   d = (double) MaxMin;

   d += this->actual;
==========================================================================================
#endif
