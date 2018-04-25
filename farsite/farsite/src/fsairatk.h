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

//******************************************************************************
//   FSAIRATK.H 	 Aerial Suppression Capabilities
//
//
//  				 Copyright 1997
//  				 Mark A. Finney, Systems for Environmental Management
//******************************************************************************



#ifndef AIRATTACK
#define AIRATTACK

#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <mem.h>
#include "fsxwbar.h"
#include "fsxlandt.h"
#include <time.h>
#include <sys/timeb.h>

//------------------------------------------------------------------------------
//
// Bomber Class and Functions
//
//------------------------------------------------------------------------------

struct AirCraft
{
	char AirCraftName[256];
	long Units; 				 // meters by default 0, 1 for feet
	double PatternLength[6];
	double ReturnTime;
	double Cost;

	AirCraft();
};

//------------------------------------------------------------------------------
//
// Air Attack Class and Functions
//
//------------------------------------------------------------------------------

typedef struct
{
	long AirAttackNumber;    // unique index to attack sequence
	long AirCraftNumber;	 // index to AirCraft type
	long CoverageLevel; 	 // retardant coveragelevel
	long PatternNumber; 	 // number of retardant pattern in fire perimeter array
	double ElapsedTime; 	   // time elapsed since attack started
	double EffectiveDuration;  // duration (minutes) of retardant
	void* next;
} AirAttackData;



class AirAttack
{
	AirAttackData* attack;

public:
	long AirAttackNumber;

	AirAttack();
	~AirAttack();

	bool CheckEffectiveness(AirAttackData* atk, double TimeStep);
};


#define GROUPSIZE 20

typedef struct
{
	long Suspended; 		 // flag for suspension of group air attack
	char GroupName[256];	 // name of group air attack
	long GroupAttackNumber;  // unique ID for group attack
	long Direction; 		 // direction around fire front, original or reversed
	double* IndirectLine;      // array of vertices to follow for indirectattack
	long NumPoints; 		 // number of points in indirect line
	long FireNumber;		 // fire number
	long* CoverageLevel;	 // array of coverage levels for indiv aircraft
	long* EffectiveDuration; // array of retardant duraitons for aircraft
	long* AircraftNumber;    // array of aircraft types used here
	double* WaitTime;   	   // indiviual wait times until drop for each aircraft
	long NumCurrentAircraft; // num aircraft in group
	long NumTotalAircraft;   // num currently allocated for group
	void* next; 			 // pointer to next group
} GroupAttackData;


class GroupAirAttack
{
	GroupAttackData* attack;
	//     AirAttackData *airatk;
	Farsite5 *pFarsite;

public:
	long GroupAttackNumber;

	GroupAirAttack(Farsite5 *_pFarsite);
	GroupAirAttack(GroupAttackData* groupattack, Farsite5 *_pFarsite);
	~GroupAirAttack();
	void GetCurrentGroup();
	void SetGroup(GroupAttackData* gatk);
	bool AllocGroup(long NewNumber);
	bool AddGroupMember(long AircraftNumber, long CoverageLevel,
		long EffectiveDuration);
	bool SetGroupAssignment(double* line, long FireNum, bool Reset);
	bool RemoveGroupMember(long membernumber);
	void ReorderGroupList(long Start);
	bool ExecuteAttacks(double TimeIncrement);
	void ExecuteAllIndirectAttacks(double TimeIncrement);
	double GetNextAttackTime(double TimeIncrement);
	void IncrementWaitTimes(double TimeIncrement);
	long CheckSuspendState(long OnOff);
};


//------------------------------------------------------------------------------
//
//   Global Air Attack Support Functions
//
//------------------------------------------------------------------------------

/*long GetNumAirCraft();
void SetNumAirCraft(long NumCraft);
bool LoadAirCraft(char* FileName, bool AppendList);
long SetNewAirCraft();
AirCraft* GetAirCraft(long AirCraftNumber);
long SetAirCraft(long AirCraftNumber);
void FreeAirCraft(long AirCraftNumber);

long SetupAirAttack(long AirCraftNumber, long CoverageLevel, long Duration,
	double* startpoint);
void CancelAirAttack(long AirAttackCounter);
void LoadAirAttack(AirAttackData airattackdata);
void FreeAllAirAttacks();
AirAttackData* GetAirAttack(long AirAttackCounter);
AirAttackData* GetAirAttackByOrder(long OrdinalAttackNumber);
void SetNewFireNumberForAirAttack(long OldFireNum, long NewFireNum);
long GetNumAirAttacks();
void SetNumAirAttacks(long NumAtk);

long SetupGroupAirAttack(long AirCraftNumber, long CoverageLevel,
	long Duration, double* line, long FireNum, char* GroupName);
GroupAttackData* ReassignGroupAirAttack(GroupAttackData* atk);
void CancelGroupAirAttack(long GroupCounter);
void LoadGroupAttack(GroupAttackData groupattackdata);
void FreeAllGroupAirAttacks();
GroupAttackData* GetGroupAirAttack(long GroupAttackCounter);
GroupAttackData* GetGroupAttackByOrder(long OrdinalAttackNumber);
GroupAttackData* GetGroupAttackForFireNumber(long CurrentFire, long StartNum,
	long* LastNum);
void SetNewFireNumberForGroupAttack(long OldFireNum, long NewFireNum);
long GetNumGroupAttacks();
void SetNumGroupAttacks(long NumGroups);


void WriteAirAttackLog(AirAttackData* atk);
*/
#endif







