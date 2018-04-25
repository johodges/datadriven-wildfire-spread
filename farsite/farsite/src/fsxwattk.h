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
// 	FSXWATTK.H	Suppression Capabilities
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environemntal Management
//******************************************************************************

#ifndef ATTACKRESOURCES
#define ATTACKRESOURCES

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
//	Crew Class and Functions
//
//------------------------------------------------------------------------------

struct Crew
{
	char CrewName[256];
	long Compound;
	long Units;
	double FlameLimit;
	double LineProduction[51];
	double Cost;

	Crew();
};


#define COMPOUNDINC 20

typedef struct
{
	long NumCurrentCrews;  // number of crews in compound
	long NumTotalCrews;
	long CompCrew;  	   // index to crew
	long* CrewIndex;	   // array of crew indices
	double* Multiplier;   // multipliers on spread rates
} CompoundCrew;


//------------------------------------------------------------------------
//
//	Attack Class and Functions
//
//------------------------------------------------------------------------

typedef struct
{
	long AttackNumber;  	 // unique attack ID number
	long FireNumber;		// store fire number that is being attacked
	long LineNumber;		// only used for parallel attack
	long CrewNum;   		 // store index to crew number
	long BurnDelay;		// burn delay (m)
	long FireDist;			// distance from fire edge for parallel attack
	long CurrentPoint;  	 // current point on fire perim or indirect segment
	long NextPoint; 		 // next point on fire perim or indirect segment
	long Burnout;   		 // if burnout is desired
	long Suspended; 		 // if suspended 1 else 0
	long Indirect;			// stores tactics for this attack 0, 1, or 2
	long Reverse;   		 // if reverse around fire perimeter
	long NumPoints; 		 // number of points in indirect attack
	long BurnLine[2];		// hold endpts of line segment for burnout
	long BurnDirection; 	 // 0 for left, 1 for right
	double AttackTime;		// total time of attack
	double LineBuilt;		// total length of line built
	double* IndirectLine1;   	// store indirect line route
	double* IndirectLine2;  	// alternative array for indirect line route
	void* next;
} AttackData;


class Attack
{
	AttackData* attack;
	double xpt1, ypt1, ros1, fli1;
	double xpt1n, ypt1n, ros1n, fli1n;
	double xpt2, ypt2, ros2, fli2;
	double xpt2n, ypt2n, ros2n, fli2n;
	double rcx1, rcx1n, rcx2, rcx2n;
	double ChordArcRatio;   	  // Ratio of chord dist to arc dist on circle
	double LineRate;
	double LineOffset;			// random distance from stationary indirect line
	long NumInsertPoints;
	VectorBarrier vectorbarrier;
	LandscapeData ld;
	Farsite5 *pFarsite;

	void BurnOut();
	void BufferBurnout(double xptl, double yptl, double xpt, double ypt,
		double xptn, double yptn, double* x1, double* y1, double* x2,
		double* y2);
	long ProblemQuad();
	bool IndirectSegment();
	long FindCurrentPoint();
	bool Cross(double xpt1, double ypt1, double xpt2, double ypt2,
		double xpt1n, double ypt1n, double xpt2n, double ypt2n, double* newx,
		double* newy);
	bool IterateIntersection(double LineDist, double extendx, double extendy,
		double xstart, double ystart, double xend, double yend, double* newx,
		double* newy);
	void InsertPerimeterPoint(double newxpt, double newypt, double newros,
		double newfli, double newrcx, long InsertType);
	//double pow2(double input);
	void ConvertLandData(short slope, short aspect);
	double CalculateSlopeDist(double x1, double y1, double x2, double y2);
	void GetConvexHull(double* Hull1, long* NumHullPts);
	void FindHullQuadrant(double* Hull1, long* NumHullPts, long StartPt,
		long EndPt, double RefAngle);
	void ExpandConvexHulls(double* Hull1, double* Hull2, long NumHullPts);
	long FindHullPoint(double* Hull2, long NumHullPts, long* HullPt);
	bool AllocParallelLine(double* Hull, long NumHullPts, double TimeStep);
	void HullDensityControl(double* Hull1, double* Hull2, long* NumHullPts);
	void CalcChordArcRatio(double LastX, double LastY);
	//void ReplaceDeadPoints();

public:
	Attack(Farsite5 *_pFarsite);
	~Attack();
	bool DirectAttack(AttackData* attack, double TimeStep);
	bool IndirectAttack(AttackData* attack, double TimeStep);
	bool ParallelAttack(AttackData* atk, double TimeStep);
	void ConductBurnout(AttackData* atk);
	long CheckOverlapAttacks(long* attacks, double TimeStep);
	void BoundingBox();
};


//------------------------------------------------------------------------------
//
// 	Global Suppression Data and Access Functions
//
//------------------------------------------------------------------------------

/*long SetupIndirectAttack(long CrewNum, double* startpt, long numpts);    // Indirect Attack Constructor
long ResetIndirectAttack(AttackData* atk, double* coords, long numpts);
long SetupDirectAttack(long CrewNum, long FireNum, double* coords);	   // DirectAttack Constructor
long ResetDirectAttack(AttackData* atk, long FireNum, double* coords);
long GetNumAttacks();
long GetFireNumberForAttack(long AttackCounter);
AttackData* GetAttackForFireNumber(long NumFire, long start,
	long* LastAttackNumber);
void SetNewFireNumberForAttack(long oldnumfire, long newnumfire);
long GetNumCrews();


AttackData* GetReassignedAttack();
void ReassignAttack(AttackData* atk);
AttackData* GetAttack(long AttackCounter);
AttackData* GetAttackByOrder(long OrdinalAttackNum, bool IndirectOnly);
void LoadAttacks(AttackData attackdata);
void CancelAttack(long AttackCounter);
void FreeAllAttacks();
void WriteAttackLog(AttackData* atk, long Type, long Var1, long Var2);
bool LoadCrews(char* FileName, bool AppendList);
Crew* GetCrew(long CrewNumber);
long SetCrew(long CrewNumber);
long SetNewCrew();
void FreeCrew(long CrewNumber);
void FreeAllCrews();

long GetNumCompoundCrews();
//long SetCompoundCrew(long GroupNumber, char *CrewName);
long SetCompoundCrew(long GroupNumber, char* CrewName, long ExistingCrewNumber);
CompoundCrew* GetCompoundCrew(long CrewNumber);
void FreeCompoundCrew(long CrewNumber);
void RemoveFromCompoundCrew(long CompNumber, long CrewNumber);
void FreeAllCompoundCrews();
bool LoadCompoundCrews(char* FileName);
bool AddToCompoundCrew(long CrewNumber, long NewCrew, double Mult);
void CalculateCompoundRates(long CrewNumber);
*/
#endif
