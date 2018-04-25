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
//  FarsiteEvents
//
//------------------------------------------------------------------------------

#ifndef  FARSITE_SYNCEVENTS
#define  FARSITE_SYNCEVENTS

//#include <process.h>
//#include <windows.h>

#define EVENT_BURN 		1
#define EVENT_MOIST 	2
#define EVENT_BURNUP	3
#define EVENT_INTEG		4
#define EVENT_BURN_THREAD	5
#define EVENT_MOIST_THREAD 	6
#define EVENT_BURNUP_THREAD	7
#define EVENT_INTEG_THREAD	8
#define EVENT_CROSS 		  9
#define EVENT_CROSS_THREAD    10

#ifndef FARSITE_HANDLE
#define FARSITE_HANDLE
typedef long X_HANDLE;
#endif // FARSITE_HANDLE

//#ifdef WIN32
class FarsiteEvent
{
	X_HANDLE* hEvent;
	unsigned long NumEvents;

public:
	FarsiteEvent();
	~FarsiteEvent();
	bool AllocEvents(long numevents, char* basename, bool ManReset,
		bool InitState);
	bool FreeEvents();

	X_HANDLE GetEvent(long ThreadNum);
	bool SetEvent(long ThreadNum);
	bool ResetEvent(long ThreadNum);
	bool WaitForEvents(long numevents, bool All, unsigned long Wait);
	bool WaitForOneEvent(long ThreadNum, unsigned long Wait);
};


bool AllocFarsiteEvents(long EventNum, long numevents, char* basename,
	bool ManReset, bool InitState);
bool FreeFarsiteEvents(long EventNum);
bool SetFarsiteEvent(long EventNum, long ThreadNum);
bool ResetFarsiteEvent(long EventNum, long ThreadNum);
bool WaitForFarsiteEvents(long EventNum, long NumEvents, bool All,
	unsigned long Wait);
bool WaitForOneFarsiteEvent(long EventNum, long ThreadNum, unsigned long Wait);
X_HANDLE GetFarsiteEvent(long EventNum, long ThreadNum);

X_HANDLE GetLandscapeSemaphore();
bool CreateLandscapeSemaphore();
void CloseLandscapeSemaphore();

//#endif //WIN32

#endif // FARSITE_SYNCEVENTS
