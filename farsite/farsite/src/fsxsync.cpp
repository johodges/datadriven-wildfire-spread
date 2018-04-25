/*
 * NOTICE OF RELEASE TO THE PUBLIC DOMAIN
 *
 * This work was created using public funds by employees of the
 * USDA Forest Service's Fire Science Lab and Systems for Environmental
 * Management.  It is therefore ineligible for copyright under title 17,
 * section 105 of the United States Code.  You may treat it as you would
 * treat any public domain work: it may be used, changed, copied, or
 * redistributed, with or without permission of the authors, for free or
 * for compensation.  You may not claim exclusive ownership of this code
 * because it is already owned by everyone.  Use this software entirely
 * at your own risk.  No warranty of any kind is given.
 *
 * FARSITE is a trademark owned by Mark Finney.  You may not call derived
 * works by the name FARSITE without explicit written permission.
 *
 * A copy of 17-USC-105 should have accompanied this distribution in the file
 * 17USC105.html.  If not, you may access the law via the US Government's
 * public websites:
 *   - http://www.copyright.gov/title17/92chap1.html#105
 *   - http://www.gpoaccess.gov/uscode/  (enter "17USC105" in the search box.)
 */
//------------------------------------------------------------------------------
//
//  Event Functions
//
//------------------------------------------------------------------------------

//#ifdef WIN32
#include "fsxsync.h"
//#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*FarsiteEvent hBurnEvent;
FarsiteEvent hMoistEvent;
FarsiteEvent hBurnupEvent;
FarsiteEvent hIntegEvent;
FarsiteEvent hCrossEvent;
FarsiteEvent hBurnThreadEvent;
FarsiteEvent hMoistThreadEvent;
FarsiteEvent hBurnupThreadEvent;
FarsiteEvent hIntegThreadEvent;
FarsiteEvent hCrossThreadEvent;*/


bool AllocFarsiteEvents(long EventNum, long numevents, char* basename,
	bool ManReset, bool InitState)
{
/*
	bool ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
	case 2:
		ret = hMoistEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
	case 3:
		ret = hBurnupEvent.AllocEvents(numevents, basename, ManReset,
							InitState); break;
	case 4:
		ret = hIntegEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
	case 5:
		ret = hBurnThreadEvent.AllocEvents(numevents, basename, ManReset,
								InitState); break;
	case 6:
		ret = hMoistThreadEvent.AllocEvents(numevents, basename, ManReset,
									InitState); break;
	case 7:
		ret = hBurnupThreadEvent.AllocEvents(numevents, basename, ManReset,
									InitState); break;
	case 8:
		ret = hIntegThreadEvent.AllocEvents(numevents, basename, ManReset,
									InitState); break;
	case 9:
		ret = hCrossEvent.AllocEvents(numevents, basename, ManReset, InitState); break;
	case 10:
		ret = hCrossThreadEvent.AllocEvents(numevents, basename, ManReset,
									InitState); break;
	}
	return ret;
*/

	return true;
}


bool FreeFarsiteEvents(long EventNum)
{
/*	bool ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.FreeEvents(); break;
	case 2:
		ret = hMoistEvent.FreeEvents(); break;
	case 3:
		ret = hBurnupEvent.FreeEvents(); break;
	case 4:
		ret = hIntegEvent.FreeEvents(); break;
	case 5:
		ret = hBurnThreadEvent.FreeEvents(); break;
	case 6:
		ret = hMoistThreadEvent.FreeEvents(); break;
	case 7:
		ret = hBurnupThreadEvent.FreeEvents(); break;
	case 8:
		ret = hIntegThreadEvent.FreeEvents(); break;
	case 9:
		ret = hCrossEvent.FreeEvents(); break;
	case 10:
		ret = hCrossThreadEvent.FreeEvents(); break;
	}

	return ret;
*/

	return true;
}


bool SetFarsiteEvent(long EventNum, long ThreadNum)
{
     /*
	bool ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.SetEvent(ThreadNum); break;
	case 2:
		ret = hMoistEvent.SetEvent(ThreadNum); break;
	case 3:
		ret = hBurnupEvent.SetEvent(ThreadNum); break;
	case 4:
		ret = hIntegEvent.SetEvent(ThreadNum); break;
	case 5:
		ret = hBurnThreadEvent.SetEvent(ThreadNum); break;
	case 6:
		ret = hMoistThreadEvent.SetEvent(ThreadNum); break;
	case 7:
		ret = hBurnupThreadEvent.SetEvent(ThreadNum); break;
	case 8:
		ret = hIntegThreadEvent.SetEvent(ThreadNum); break;
	case 9:
		ret = hCrossEvent.SetEvent(ThreadNum); break;
	case 10:
		ret = hCrossThreadEvent.SetEvent(ThreadNum); break;
	}

	return ret;
     */

     return true;
}

bool ResetFarsiteEvent(long EventNum, long ThreadNum)
{
	/*
	bool ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.ResetEvent(ThreadNum); break;
	case 2:
		ret = hMoistEvent.ResetEvent(ThreadNum); break;
	case 3:
		ret = hBurnupEvent.ResetEvent(ThreadNum); break;
	case 4:
		ret = hIntegEvent.ResetEvent(ThreadNum); break;
	case 5:
		ret = hBurnThreadEvent.ResetEvent(ThreadNum); break;
	case 6:
		ret = hMoistThreadEvent.ResetEvent(ThreadNum); break;
	case 7:
		ret = hBurnupThreadEvent.ResetEvent(ThreadNum); break;
	case 8:
		ret = hIntegThreadEvent.ResetEvent(ThreadNum); break;
	case 9:
		ret = hCrossEvent.ResetEvent(ThreadNum); break;
	case 10:
		ret = hCrossThreadEvent.ResetEvent(ThreadNum); break;
	}

	return ret;
     */

     return true;
}

bool WaitForFarsiteEvents(long EventNum, long numevents, bool All,
	unsigned long Wait)
{
	/*

	bool ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.WaitForEvents(numevents, All, Wait); break;
	case 2:
		ret = hMoistEvent.WaitForEvents(numevents, All, Wait); break;
	case 3:
		ret = hBurnupEvent.WaitForEvents(numevents, All, Wait); break;
	case 4:
		ret = hIntegEvent.WaitForEvents(numevents, All, Wait); break;
	case 5:
		ret = hBurnThreadEvent.WaitForEvents(numevents, All, Wait); break;
	case 6:
		ret = hMoistThreadEvent.WaitForEvents(numevents, All, Wait); break;
	case 7:
		ret = hBurnupThreadEvent.WaitForEvents(numevents, All, Wait); break;
	case 8:
		ret = hIntegThreadEvent.WaitForEvents(numevents, All, Wait); break;
	case 9:
		ret = hCrossEvent.WaitForEvents(numevents, All, Wait); break;
	case 10:
		ret = hCrossThreadEvent.WaitForEvents(numevents, All, Wait); break;
	}

	return ret;
	*/

     return true;
}


bool WaitForOneFarsiteEvent(long EventNum, long ThreadNum, unsigned long Wait)
{
	/*
	bool ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 2:
		ret = hMoistEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 3:
		ret = hBurnupEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 4:
		ret = hIntegEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 5:
		ret = hBurnThreadEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 6:
		ret = hMoistThreadEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 7:
		ret = hBurnupThreadEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 8:
		ret = hIntegThreadEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 9:
		ret = hCrossEvent.WaitForOneEvent(ThreadNum, Wait); break;
	case 10:
		ret = hCrossThreadEvent.WaitForOneEvent(ThreadNum, Wait); break;
	}

	return ret;
     */

     return true;
}


X_HANDLE GetFarsiteEvent(long EventNum, long ThreadNum)
{
	/*

	X_HANDLE ret;

	switch (EventNum)
	{
	case 1:
		ret = hBurnEvent.GetEvent(ThreadNum); break;
	case 2:
		ret = hMoistEvent.GetEvent(ThreadNum); break;
	case 3:
		ret = hBurnupEvent.GetEvent(ThreadNum); break;
	case 4:
		ret = hIntegEvent.GetEvent(ThreadNum); break;
	case 5:
		ret = hBurnThreadEvent.GetEvent(ThreadNum); break;
	case 6:
		ret = hMoistThreadEvent.GetEvent(ThreadNum); break;
	case 7:
		ret = hBurnupThreadEvent.GetEvent(ThreadNum); break;
	case 8:
		ret = hIntegThreadEvent.GetEvent(ThreadNum); break;
	case 9:
		ret = hCrossEvent.GetEvent(ThreadNum); break;
	case 10:
		ret = hCrossThreadEvent.GetEvent(ThreadNum); break;
	}

	return ret;
     */

     return true;

}


//------------------------------------------------------------------------------
//
//  FarsiteEvents
//
//------------------------------------------------------------------------------


FarsiteEvent::FarsiteEvent()
{
	hEvent = 0;
	NumEvents = 0;
}


FarsiteEvent::~FarsiteEvent()
{
	FreeEvents();
}


bool FarsiteEvent::AllocEvents(long numevents, char* basename, bool ManReset,
	bool InitState)
{
/*
	if (numevents == NumEvents)
		return true;

	FreeEvents();

	bool NoErr = true;
	long i;
	char Name[128] = "";
	char TimeID[128] = "";
	SYSTEMTIME st;
	GetSystemTime(&st);
	sprintf(TimeID, "%ld%ld%ld%ld%ld%ld%ld", st.wYear, st.wMonth, st.wDay,
		st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

	hEvent = new X_HANDLE[numevents];
	if (hEvent == NULL)
		return false;

	for (i = 0; i < numevents; i++)
	{
		sprintf(Name, "%s%s_%02ld", basename, TimeID, i + 1);

		hEvent[i] = CreateEvent(NULL, ManReset, InitState, Name);
		if (hEvent[i] == NULL)
		{
			NoErr = true;
			break;
		}
		NumEvents++;
	}

	return NoErr;
*/

	return false;
}


bool FarsiteEvent::FreeEvents()
{
     /*
	unsigned long i;
	bool NoErr = true;

	for (i = 0; i < NumEvents; i++)
	{
		if (!CloseHandle(hEvent[i]))
			NoErr = false;
	}
	if (hEvent)
		delete[] hEvent;
	NumEvents = 0;
	hEvent = 0;

	return NoErr;
	*/

     return true;
}


X_HANDLE FarsiteEvent::GetEvent(long ThreadNum)
{
// this is really strange to me I think that the
// conversion to long from null is going to be
// zero any way.  So I am jus explictly returning
// zero here.
	if (ThreadNum >(long)( NumEvents - 1))
		return 0;
		//return NULL;


	return hEvent[ThreadNum];
}


bool FarsiteEvent::SetEvent(long ThreadNum)
{
//	if (ThreadNum < NumEvents)
//		return ::SetEvent(hEvent[ThreadNum]);

	return false;
}


bool FarsiteEvent::ResetEvent(long ThreadNum)
{
//	if (ThreadNum < NumEvents)
//		return ::ResetEvent(hEvent[ThreadNum]);

	return false;
}


bool FarsiteEvent::WaitForEvents(long numevents, bool All, unsigned long Wait)
{
//	WaitForMultipleObjects(numevents, hEvent, All, Wait);

	return true;
}


bool FarsiteEvent::WaitForOneEvent(long ThreadNum, unsigned long Wait)
{
//	WaitForSingleObject(hEvent[ThreadNum], Wait);

	return true;
}


//------------------------------------------------------------------------------
//
//    Landscape Semaphore Functions
//
//------------------------------------------------------------------------------

X_HANDLE hLandscapeSemaphore = 0;


X_HANDLE GetLandscapeSemaphore()
{
	return hLandscapeSemaphore;
}


bool CreateLandscapeSemaphore()
{
/*
	CloseLandscapeSemaphore();
	char Name[128] = "";
	char TimeID[128] = "";
	SYSTEMTIME st;
	GetSystemTime(&st);
	sprintf(TimeID, "%ld%ld%ld%ld%ld%ld%ld", st.wYear, st.wMonth, st.wDay,
		st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

	sprintf(Name, "%s%s", "LANDSCAPE_SEMAPHORE", TimeID);
	hLandscapeSemaphore = CreateSemaphore(NULL, 1, 1, Name);
	if (hLandscapeSemaphore == NULL)
		return false;
*/
	return true;
}


void CloseLandscapeSemaphore()
{
//	if (hLandscapeSemaphore)
//		CloseHandle(hLandscapeSemaphore);

	hLandscapeSemaphore = 0;
}
