/*********************************************************************************
* Name: fmc_ci3.cpp - FlamMap Input Class
* Desc: This class holds and makes available all needed inputs.
*
*
*
**********************************************************************************/
#define _CRT_SECURE_NO_WARNINGS    /* Get rid of sprintf Warnings */

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




/****************************************************************************
* Name: AllocWindData_Sta0
* Desc: I wrote this to replace the orginal AllocWindData() which had some
*        issues and was setup to do multiple stations in an odd way.
* NOTE: This only is for doing a Station 0, which is all we need right now
****************************************************************************/
int CI::AllocWindData_Sta0 (int iN)
{
   if ( wddt[0] )
     delete[] wddt[0];

   wddt[0] = new WindData[iN];
   if ( wddt[0] == NULL )
     return 0;
   NumWindStations = 1;
   MaxWindObs[0] = iN;

   return 1;
}

/************************************************************************
* Name: SetWindData_Sta0
* Desc: New version of the orginal SetWindData() which only does
*       Station 0 and doesn't load wind speed or direction which
*       aren't needed to do Conditioning
*************************************************************************/
int CI::SetWindData_Sta0 (long NumObs, long year, long month, long day, long hour, long cloudy)
{
//int i;

	 if ( NumObs < 0 || NumObs > MaxWindObs[0] ) /* caller should know better */
    return 0;                                 /* but just in case */
  wddt[0][NumObs].yr = year;
  wddt[0][NumObs].mo=month;
		wddt[0][NumObs].dy=day;
		wddt[0][NumObs].hr=hour;
		wddt[0][NumObs].cl=cloudy;
 	return 1;
}


/*********************************************************************
* Name: AllocWeatherData_Sta0
* Desc: Alloc Weather Data structs just for Station 0
*       Currently only Station 0 is implemented for Conditioning
**********************************************************************/
long CI::AllocWeatherData_Sta0 ( long NumObs)
{

  if ( NumObs < 1 )
    return 0;

	 if ( wtrdt[0] )
    delete[] wtrdt[0];

  if ( ( wtrdt[0] = new WeatherData [NumObs] ) == NULL )
	   return 0;

  MaxWeatherObs[0] = NumObs;
  NumWeatherStations = 1;
  return 1;
}


/*************************************************************************
* Name: SetWeatherData_Sta0
* Desc: I just modified the existing FlamMap Func to do Station 0 only
* NOTE: use english units
*************************************************************************/
int CI::SetWeatherData_Sta0 (long NumObs, long month, long day, double rain, long time1, long time2,
				double temp1, double temp2, long humid1, long humid2, double elevation, long tr1, long tr2)
{

 if ( NumObs >= MaxWeatherObs[0] )  /* Error trying to add more than we have room for */
    return 0;

   wtrdt[0][NumObs].mo = month;
	  wtrdt[0][NumObs].dy = day;
	 	wtrdt[0][NumObs].rn = rain;
	 	wtrdt[0][NumObs].t1 = time1;
	 	wtrdt[0][NumObs].t2 = time2;
	 	wtrdt[0][NumObs].T1 = temp1;
	 	wtrdt[0][NumObs].T2 = temp2;
	 	wtrdt[0][NumObs].H1 = humid1;
	 	wtrdt[0][NumObs].H2 = humid2;
	 	wtrdt[0][NumObs].el = elevation;
	 	wtrdt[0][NumObs].tr1 = tr1;
	 	wtrdt[0][NumObs].tr2 = tr2;

   EnvtChanged[0][0]=true;   // 1hr fuels
   EnvtChanged[1][0]=true;   // 10hr fuels
   EnvtChanged[2][0]=true;   // 100hr fuels
   EnvtChanged[3][0]=true;   // 1000hr fuels

		 return 1;
}




//********************************************************************
bool CI::SetFarsiteEvent(long EventNum, long ThreadNum)
{
	bool ret = 0;
#ifdef WIN32
     switch(EventNum) {
//      case 1: ret=hBurnEvent.SetEvent(ThreadNum); break;
     	case 2:
          ret = hMoistEvent.SetEvent (ThreadNum);
          break;
//     	case 3: ret=hBurnupEvent.SetEvent(ThreadNum); break;
//     	case 4: ret=hIntegEvent.SetEvent(ThreadNum); break;
//     	case 5: ret=hBurnThreadEvent.SetEvent(ThreadNum); break;
     	case 6:
          ret = hMoistThreadEvent.SetEvent(ThreadNum);
          break;
//     	case 7: ret=hBurnupThreadEvent.SetEvent(ThreadNum); break;
//     	case 8: ret=hIntegThreadEvent.SetEvent(ThreadNum); break;
//     	case 9: ret=hCrossEvent.SetEvent(ThreadNum); break;
///     	case 10: ret=hCrossThreadEvent.SetEvent(ThreadNum); break;

      default:
        this->LogicError(" CI::SetFarsiteEvent() ERROR");
        break;

     }
#endif
  return ret;
}

/****************************************************************/
X_HANDLE CI::GetFarsiteEvent(long EventNum, long ThreadNum)
{
X_HANDLE ret = 0;

#ifdef WIN32
   switch (EventNum) {
//      case 1: ret=hBurnEvent.GetEvent(ThreadNum); break;
     	case 2:
        ret = hMoistEvent.GetEvent(ThreadNum);
        break;
//     	case 3: ret=hBurnupEvent.GetEvent(ThreadNum); break;
//     	case 4: ret=hIntegEvent.GetEvent(ThreadNum); break;
//     	case 5: ret=hBurnThreadEvent.GetEvent(ThreadNum); break;
     	case 6:
         ret = hMoistThreadEvent.GetEvent(ThreadNum);
         break;
//     	case 7: ret=hBurnupThreadEvent.GetEvent(ThreadNum); break;
//     	case 8: ret=hIntegThreadEvent.GetEvent(ThreadNum); break;
//     	case 9: ret=hCrossEvent.GetEvent(ThreadNum); break;
//     	case 10: ret=hCrossThreadEvent.GetEvent(ThreadNum); break;
      default:
        this->LogicError(" CI::GetFarsiteEvent() ERROR");
        break;
     }
#endif
     return ret;
}

/**************************************************************************/
bool CI::WaitForFarsiteEvents(long EventNum, long numevents, bool All, unsigned long Wait)
{
bool ret = false;

#ifdef WIN32
   switch (EventNum) {
//      case 1: ret=hBurnEvent.WaitForEvents(numevents, All, Wait); break;
     	case 2:
         ret = hMoistEvent.WaitForEvents (numevents, All, Wait);
         break;
//     	case 3: ret=hBurnupEvent.WaitForEvents(numevents, All, Wait); break;
//     	case 4: ret=hIntegEvent.WaitForEvents(numevents, All, Wait); break;
//     	case 5: ret=hBurnThreadEvent.WaitForEvents(numevents, All, Wait); break;
     	case 6:
         ret = hMoistThreadEvent.WaitForEvents(numevents, All, Wait);
         break;
//     	case 7: ret=hBurnupThreadEvent.WaitForEvents(numevents, All, Wait); break;
//     	case 8: ret=hIntegThreadEvent.WaitForEvents(numevents, All, Wait); break;
//     	case 9: ret=hCrossEvent.WaitForEvents(numevents, All, Wait); break;
//     	case 10: ret=hCrossThreadEvent.WaitForEvents(numevents, All, Wait); break;
      default:
        this->LogicError(" CI::WaitForFarsiteEvents() ERROR");
        break;
     }
#endif
     return ret;
}


//------------------------------------------------------------------------------
//
//  FarsiteEvents
//
//------------------------------------------------------------------------------


/*FarsiteEvent::FarsiteEvent()
{
	 hEvent=0;
  NumEvents=0;
}

FarsiteEvent::~FarsiteEvent()
{
	  FreeEvents();
}

bool FarsiteEvent::AllocEvents(long numevents, char *basename, bool ManReset, bool InitState)
{
#ifdef WIN32
  if (numevents==NumEvents)
    return true;

	 FreeEvents();
  bool NoErr=true;
  long i;
  char Name[128]="";
  hEvent = new HANDLE[numevents];
	 if(hEvent==NULL)
     return false;
	 for ( i=0; i < numevents; i++)	{
	 	 sprintf(Name, "%s_%02ld", basename, i+1);
		  //CA2T str(Name);
		  hEvent[i]=CreateEvent(NULL, ManReset, InitState, TEXT(Name));//str);
		  //LPCTSTR str = A2T(Name);
	  	//hEvent[i]=CreateEvent(NULL, ManReset, InitState, (LPWSTR)Name);
		  if(hEvent[i]==NULL)	{
			   NoErr=true;
		   	break;	}
		 if (GetLastError() == ERROR_ALREADY_EXISTS) 	{
//			  TRACE1("uh oh Event %s already exists!\n", Name);
     printf ("FarsiteEvent::AllocEvents() ERROR\n"); 	}

		NumEvents++;
	 }
  return NoErr;
#else
  return false;
#endif
}


bool FarsiteEvent::FreeEvents()
{
unsigned long i;
bool NoErr=true;
#ifdef WIN32
  for(i=0; i<NumEvents; i++) {
	   if(!CloseHandle(hEvent[i]))
       NoErr=false;
  }
 	if(hEvent)
    delete[] hEvent;
  NumEvents=0;
	 hEvent=0;
#endif
  return NoErr;
}

X_HANDLE FarsiteEvent::GetEvent(long ThreadNum)
{
	 if ( ThreadNum > NumEvents - 1 )
     	return NULL;
  return hEvent[ThreadNum];
}


bool FarsiteEvent::SetEvent(long ThreadNum)
{
	//TRACE1("In FarsiteEvent::SetEvent( ThreadNum = %ld)\n", ThreadNum);
#ifdef WIN32
	if(ThreadNum<NumEvents)
     	return ::SetEvent(hEvent[ThreadNum]);
#endif
     return false;
}


bool FarsiteEvent::ResetEvent(long ThreadNum)
{
#ifdef WIN32
	if(ThreadNum<NumEvents)
     	return ::ResetEvent(hEvent[ThreadNum]);
#endif
     return false;
}


bool FarsiteEvent::WaitForEvents(long numevents, bool All, unsigned long Wait)
{
#ifdef WIN32

	DWORD ret = WaitForMultipleObjects(numevents, hEvent, (BOOL)All, Wait);
#endif
	//WaitForMultipleObjectsEx(numevents, hEvent, (BOOL)All, Wait, FALSE);

     return true;
}


bool FarsiteEvent::WaitForOneEvent(long ThreadNum, unsigned long Wait)
{
#ifdef WIN32
	WaitForSingleObject(hEvent[ThreadNum], Wait);
#endif
     return true;
}

*/
