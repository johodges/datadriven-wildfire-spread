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
//	FSXWBURN.CPP	Burn model functions for FARSITE
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environemntal Management
//******************************************************************************


#include "fsx4.hpp"
#include "fsxwattk.h"
#include "fsairatk.h"
#include "fsxpfront.h"
//#include "fsglbvar.h"
//#ifdef WIN32
#include "fsxsync.h"
//#endif
#include "Farsite5.h"
//#include <process.h>
#include <string.h>

#include <iostream>
using namespace std;

#define BURN_THREAD 0
#define SPOT_THREAD 1


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	BurnThread:: Functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//static X_HANDLE *hSimEvent=0;

BurnThread::BurnThread( Farsite5 *_pFarsite) : MechCalls(_pFarsite), cf(_pFarsite), prod(_pFarsite),
	embers(_pFarsite)
{
	pFarsite = _pFarsite;
	FireIsUnderAttack = false;
	ThreadID = 0;
	Begin = End = -1;
	turn = 0;
	hBurnThread = 0;
	fe = new FELocalSite(pFarsite);
// 	env = Env;
	firering = 0;
	embers.SetFireEnvironmentCalls( fe);
	Started = false;
	DoSpots = false;
	TotalPoints = CuumPoint = 0;
	//hBurnEvent=0;
}


BurnThread::~BurnThread()
{
	//     if(hBurnEvent)
	//     	CloseHandle(hBurnEvent);
	delete fe;
}


void BurnThread::SetRange(long currentfire, double SimTime,
	double cuumtimeincrement, double timeincrement, double timemaxrem,
	long begin, long end, long Turn, bool attack, FireRing* Ring)
{
	if (begin > -1)
	{
		Begin = begin;
		End = end;
	}
	CurrentFire = currentfire;
	SIMTIME = SimTime;
	CuumTimeIncrement = cuumtimeincrement;
	TimeMaxRem = timemaxrem;
	/*if (timemaxrem > 0.001)
	{
		printf("%lf\n", timemaxrem);	}*/
	TimeIncrement = EventTimeStep = timeincrement;
	turn = Turn;
	FireIsUnderAttack = attack;
	firering = Ring;
	CanStillBurn = StillBurning = false;
	SimTimeOffset = CuumTimeIncrement;
	//if(DistanceCheckMethod(GETVAL)==0)
	//     SimTimeOffset-=GetTemporaryTimeStep();
}

/********************************************************************/
X_HANDLE BurnThread::StartBurnThread(long ID)
{
  RunBurnThread(this);
	 return hBurnThread;
}


X_HANDLE BurnThread::GetThreadHandle()
{
	return hBurnThread;
}


unsigned BurnThread::RunBurnThread(void* bt)
{
	static_cast <BurnThread*>(bt)->PerimeterThread();

	return 1;
}

/*****************************************************************************/
void BurnThread::PerimeterThread()
{
	long begin, end;
	long i, TestPointL, TestPointN;
	bool ComputeSpread;				//=true;
	double oldfli;
	double FliL, FliN;
	double mx[20];			// fuel moistures

	begin = Begin;
	end = End;
	Started = true;	// means that the thread has been started
	do	{
	 	if (End < 0)
		  	break;
	 	if (CurrentFire >= 0)	{
		  	if (turn == 0)
			   	distchek(CurrentFire);
			  else if (pFarsite->DistanceCheckMethod(GETVAL) == 0)
			   	distchek(CurrentFire);	}

		 for (CurrentPoint = begin;	CurrentPoint < end;	CurrentPoint++)	{ // FOR ALL POINTS ON EACH FIRE

			  timerem = TimeIncrement;
		  	GetPoints(CurrentFire, CurrentPoint);	   	// Burn::GetPoints();
		  	fe->GetLandscapeData(xpt, ypt, ld); 	 				// Burn::GetLandscapeData();
		  	if (ld.fuel > 0 && ld.elev != -9999) {  // && fli>=0.0)				// if not a rock or lake etc.
				   fli = FliFinal = oldfli = pFarsite->GetPerimeter2Value(CurrentPoint,	FLIVAL);
			 	if (fli < 0.0)	{
					  StillBurning = true;
					  if (FireIsUnderAttack)	{
						   if (turn)	{
							    ComputeSpread = false;
							    for (i = 1; i < 3; i++)	{
								     TestPointL = CurrentPoint - i;
							    	 TestPointN = CurrentPoint + i;
								     if (TestPointL < 0)
									      TestPointL += pFarsite->GetNumPoints(CurrentFire);
								     if (TestPointN >= pFarsite->GetNumPoints(CurrentFire))
								       TestPointN -= pFarsite->GetNumPoints(CurrentFire);
							    	 FliL = pFarsite->GetPerimeter2Value(TestPointL, FLIVAL);
							    	 FliN = pFarsite->GetPerimeter2Value(TestPointN, FLIVAL);
							    	 if (FliL > 0.0 || FliN > 0.0) {
									      ComputeSpread = true;
									      break;}}}
						   else
							   ComputeSpread = true;}
			    else
						   ComputeSpread = false;
				}
				else
					 ComputeSpread = true;
				if (ComputeSpread)	{
					 EmberCoords();  				     // compute ember source coordinates
					 CanStillBurn = true;			  	// has fuel, can still burn

					 fe->GetFireEnvironment ( SIMTIME + SimTimeOffset, false);

				  if (pFarsite->EventMinimumTimeStep(GETVAL) < EventTimeStep &&		pFarsite->EventMinimumTimeStep(GETVAL) > 0.0)
					   	EventTimeStep = pFarsite->EventMinimumTimeStep(GETVAL);

					 NormalizeDirectionWithLocalSlope(); 	// mechcalls function to transfrm orientation of point with slope
				 	RosT1 = RosT = pFarsite->GetPerimeter2Value(CurrentPoint, ROSVAL);	// get forward ROS from last timestep or last substep
				 	SurfaceFire();

// Mark the Burn Spot Grid, that this cell has burned
                    this->pFarsite->m_BSG.Set(xpt,ypt, (char *)"F");


				 	if (oldfli < 0.0)	{
						  fli *= -1.0;
						  FliFinal *= -1.0;
						  react = 0.0;	}
					 if (ld.cover > 0 && ld.height > 0)	 	// if there IS a tree canopy
					  	CrownFire();			                 		// if turn==1 then start try spot fires, else no
					 else
						  cf.CrownLoadingBurned = 0.0;
					 if (oldfli >= 0.0 &&	timerem > 0.001 &&	timerem > TimeMaxRem)	// tolerance for simulation 1/1000 min
					   	TimeMaxRem = timerem;				// LARGEST TIME REMAINING, MEANS FASTEST SPREAD RATE
				}
				else
				 	RosT1 = RosT = pFarsite->GetPerimeter2Value(CurrentPoint, ROSVAL);
			}
			else 	{
				 RosT = RosT1 = 0.0;
				 fli = FliFinal = 0.0;
				 react = 0.0;		}
			if (turn == 1)	{
				 if (FireIsUnderAttack && !ComputeSpread)
					  FliFinal *= -1.0;
			 	prod.cuumslope[0] += (double) ld.slope; 		// CUMULATIVE SLOPE ANGLES
			 	prod.cuumslope[1] += 1;				// CUMULATIVE NUMBER OF PERIMETER POINTS
			 	pFarsite->SetPerimeter1(CurrentFire, CurrentPoint, xpt, ypt);
			 	pFarsite->SetFireChx(CurrentFire, CurrentPoint, RosT1, FliFinal);
				 pFarsite->SetReact(CurrentFire, CurrentPoint, react);

				if (pFarsite->DistanceCheckMethod(GETVAL) == 1 &&	CuumTimeIncrement == 0.0)
				  	pFarsite->SetElev(CurrentPoint, ld.elev);

				if (pFarsite->DistanceCheckMethod(GETVAL) == 0 &&	pFarsite->CheckPostFrontal(GETVAL))   	   	// store current fire perim in ring
				{
					pFarsite->SetElev(CurrentPoint, ld.elev);
					pFarsite->GetCurrentFuelMoistures(ld.fuel, ld.woody,
						(double *) &gmw, mx, 7);
					pFarsite->AddToCurrentFireRing(firering, CurrentPoint, ld.fuel,
						ld.woody, ld.duff, mx, cf.CrownLoadingBurned * 1000.0);
				}
			}
		}
//		SetFarsiteEvent(EVENT_BURN, ThreadOrder);		//hSimEvent[ThreadOrder]


		if (End < 0)
			break;

		prod.cuumslope[0] = 0.0;	// reset after they have been read by burn::
		prod.cuumslope[1] = 0;
		begin = Begin;	// restore local copies from Class data
		end = End;
		if (DoSpots)	{
			 CurrentFire = -1;
			 if (embers.NumEmbers > 0)	{
				  embers.Flight(SIMTIME, SIMTIME + pFarsite->GetActualTimeStep());  	// curent time to Flight is iter*actual, but really (iter+1)*actual
				  if (embers.NumSpots > 0 && pFarsite->EnableSpotFireGrowth(GETVAL))
					   embers.Overlap();		   // check ember resting positions for overlap with burned areas
			}
			begin = 0;	// go back to the begining
			end = 0;
			DoSpots = false;
		}
    		// eliminate if multithreading is restored
          break;
	} while (End > -1);

//	SetFarsiteEvent(EVENT_BURN, ThreadOrder);
}

/***************************************************************************/
void BurnThread::SurfaceFire(void)
{
	// does surface fire calculations of spread rate and intensity
	LoadGlobalFEData(fe);
	fros = spreadrate(ld.slope, m_windspd, ld.fuel);
	if (fros > 0.0)						// if rate of spread is >0
	{
		StillBurning = 1;  			// allows fire to continue burning
		GetAccelConst();			  // get acceleration constants
		VecSurf();		 			// vector wind with slope
		ellipse(vecros, ivecspeed);  	// compute elliptical dimensions
		grow(vecdir);  				// compute time derivatives xt and yt for perimeter point
		AccelSurf1();   			  // compute new equilibrium ROS and Avg. ROS in remaining timestep
		SlopeCorrect(1);
		SubTimeStep = timerem;
		limgrow();   					// limits HORIZONTAL growth to mindist
		AccelSurf2();					// calcl ros & fli for SubTimeStep Kw/m from BTU/ft/s*/
		//	if(SSpotOK && cover==99)
		//	{    cf.FlameLength=0;		// may want to set other crown parameters to 0 here
		//		SpotFire(3);	   	 	// spotting from  winddriven surface fires
		//	}
	}
	else
	{
		timerem = 0;
		RosT = 0;
		fli = 0;
		FliFinal = 0;
	}
}


void BurnThread::CrownFire(void)
{
	// does crown fire calculations of spread rate and intensity (Van Wagner's)
	//double FabsFli=fabs(fli);
	double FabsFli = 3.4613 * (384.0 * (react / 0.189275) * (ExpansionRate /
		.30480060960) /
		(60.0 * savx));	// ending forward fli in timestep
	double ExpRate = ExpansionRate; 															// attempting to account for ws variation

	if (pFarsite->EnableCrowning(GETVAL))			   		   // global flag for disabling crowning
	{
		if (FabsFli > 0.0)  					  // use average fli
			cf.CrownIgnite(ld.height, ld.base, ld.density); 		  // calculate critical intensity from van Wagner
		else
			cf.Io = 1;
		if (FabsFli >= cf.Io)			   // if sufficient energy to ignite crown
		{
			//cf.CrownBurn(avgros, FabsFli, A);
			if(pFarsite->GetCrownFireCalculation() == 0) //Finney
			{
				cf.CrownBurn(ExpRate, FabsFli, A);
				R10 = spreadrate(ld.slope, m_twindspd * 0.4, 10);
			}
			else //Scott/Rhienhart
				R10 = cf.CrownBurn2(ExpRate, FabsFli, A, this);
			if (R10 == 0.0)
				R10 = avgros;
			VecCrown(); 				// get vectored crown spread rate
			ellipse(vecros, ivecspeed); // determine elliptical dims with vectored winds not m_twindspd
			grow(vecdir);    	   	   // compute time derivatives xt & yt for perimeter pt.
			SpreadCorrect();   	   	   // correct vecros for directional spread
			//cros=cf.CrownSpread(avgros, vecros);	// wind-driven crown fire rate of spread Rothermel 1991
			cros = cf.CrownSpread(ExpRate, ExpansionRate);	// wind-driven crown fire rate of spread Rothermel 1991
			if (cros > 0.0)			   // if active crown fire
			{
				timerem = SubTimeStep;   // reset timerem again to before surface spread began in last substep
				A = cf.A;   			 // use crown fire acceleration rate from Crown::CrownBurn()
				AccelCrown1();
				SlopeCorrect(0);
				limgrow();   		   // limits HORIZONTAL growth to mindist
				AccelCrown2();
				SubTimeStep = SubTimeStep - timerem;  // actual subtimestep after crowning
				cf.FlameLength = 0;
				cf.CrownIntensity(cros, &fli);    // calc flamelength and FLI from crownfire
				//	AVGFlameLength=AVGFlameLength+FlameLength/SubTimeStep;  // fl/unittime
				//   CrownDuration=CrownDuration+SubTimeStep;   			 // total time spent crowning
				if (pFarsite->EnableSpotting(GETVAL) && FliFinal > 0.0)    // no spotting from dead perimeter
					SpotFire(0);	  // spotting from active crown fire
			}
			else						// if passive crown fire
			{
				cf.CrownIntensity(avgros, &fli);	// calc flamelength and FLI for passive crown fire
				if (pFarsite->EnableSpotting(GETVAL) && FliFinal > 0.0)    // no spotting from dead perimeter
				{
					cf.FlameLength = 0;
					SpotFire(1);	  // spotting from torching trees
				}
			}
			if (fli > fabs(FliFinal))
			{
				if (FliFinal < 0.0)
					FliFinal = fli * -1.0;
				else
					FliFinal = fli;
			}
		}
		else
			cf.CrownLoadingBurned = 0.0;	// need to set because of post-frontal
	}
	else
		cf.CrownLoadingBurned = 0.0;	// need to set because of post-frontal
}



void BurnThread::SpotFire(int SpotSource)
{
	// calls spot fire functions for lofting embers
	if (turn && fli > 0.0)  								  // if check method 2 and second pass
	{
		if (turn == 1)
			CurrentTime = CuumTimeIncrement + SIMTIME;   	// timerem will be ~zero with method 2
		else
			CurrentTime = (pFarsite->GetActualTimeStep() - timerem) + SIMTIME;		// timerem will represent time remaining in actual TS
		embers.SpotSource = SpotSource;
		embers.Loft(cf.FlameLength, cf.CrownFractionBurned, ld.height,
				cf.CrownLoadingBurned, HorizSpread, SubTimeStep, CurrentTime);
	}
}


void BurnThread::EmberCoords()
{
	// copies coordinates for perimeter segment that generates embers
	embers.Fcoord.x = xpt;
	embers.Fcoord.y = ypt;			// STORE PERIMETER SEGMENT FOR SPOTTING
	embers.Fcoord.xl = xptl;
	embers.Fcoord.xn = xptn;
	embers.Fcoord.yl = yptl;
	embers.Fcoord.yn = yptn;			// natural units (english or metric grid units)
	embers.Fcoord.e = (double) ld.elev; 	// meters
	embers.Fcoord.cover = ld.cover; 		// transfer cover to
}



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	Burn:: Functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


Burn::Burn(Farsite5 *_pFarsite) : Intersections(_pFarsite), prod(_pFarsite)
{
	//InitialAllocFirePerims();
	pFarsite = _pFarsite;
	pFarsite->CreateSpotSemaphore();
	SIMTIME = 0;
	CuumTimeIncrement = 0.0;
	TotalPoints = 0;
	burnthread = 0;
	NumPerimThreads = 0;
	fe = new FELocalSite(pFarsite);
//	env = new FireEnvironment2(pFarsite);
	//embers.SetFireEnvironmentCalls(env, fe);
	NumSpots = 0;
	ExNumPts = 0;
	rast = NULL;
	rast = new Rasterize(_pFarsite);
	firering = NULL;
}


Burn::~Burn()
{
	ResetAllPerimThreads();
	pFarsite->CloseSpotSemaphore();
	pFarsite->FreeAllFirePerims();
	pFarsite->FreeElev();
	delete fe;
//	delete env;
	if(rast)
		delete rast;
}


void Burn::BurnIt(long count)
{
	// controls access to different burn methods and output options
	CurrentFire = count;		  // local copy of currentfire
	if(pFarsite->DistanceCheckMethod(GETVAL)){
        BurnMethod1();  	  // distance checking on individual fire basis
	}
    else{
		BurnMethod2();  	  // distance checking on Simulation basis
    }
}


void Burn::ResetAllPerimThreads()
{
	CloseAllPerimThreads();
}


bool Burn::AllocPerimThreads()
{
	if (NumPerimThreads == pFarsite->GetMaxThreads())
		return true;

	CloseAllPerimThreads();
	burnthread = (BurnThread * *) new BurnThread * [pFarsite->GetMaxThreads()];//GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, GetMaxThreads()*sizeof(BurnThread *));

	long i;
	if (burnthread)
	{
		NumPerimThreads = pFarsite->GetMaxThreads();
		NumSpots = new long[NumPerimThreads];
		for (i = 0; i < NumPerimThreads; i++)
			burnthread[i] = new BurnThread( pFarsite);
		return true;
	}

	return false;
}


void Burn::CloseAllPerimThreads()
{
/*
	long m;

	if (NumPerimThreads == 0)
		return;

	for (m = 0; m < NumPerimThreads; m++)
		burnthread[m]->SetRange(0, 0.0, 0.0, 0.0, 0.0, 0, -1, 0, false, 0);
	for (m = 0; m < NumPerimThreads; m++)
		burnthread[m]->StartBurnThread(m);
	Sleep(50);
	WaitForFarsiteEvents(EVENT_BURN, NumPerimThreads, true, INFINITE);//2000);
*/
	FreePerimThreads();
}

void Burn::FreePerimThreads()
{
	long i;

	if (burnthread)
	{
		for (i = 0; i < NumPerimThreads; i++)
			delete burnthread[i];
		delete[] burnthread;//GlobalFree(burnthread);
		if (NumSpots)
			delete[] NumSpots;
	}

	burnthread = 0;
	NumPerimThreads = 0;
	NumSpots = 0;
}


void Burn::ResumeSpotThreads(long threadct)
{
	long i;

 this->pFarsite->cfmc.CheckMoistureTimes(SIMTIME);
// 	env->CheckMoistureTimes(SIMTIME); //must do here because ember flight uses: GetFireEnvt()


	for (i = 0; i < threadct; i++)
	{
		burnthread[i]->SetRange(0, SIMTIME, CuumTimeIncrement, TimeIncrement,
						TimeMaxRem, 0, 0, 0, FireIsUnderAttack, firering);
		if (!burnthread[i]->Started)
		{
			burnthread[i]->StartBurnThread(i);
			//WaitForOneFarsiteEvent(EVENT_BURN, i, INFINITE);
		}
	}
	for (i = 0; i < threadct; i++)
	{
		burnthread[i]->DoSpots = true;
		burnthread[i]->StartBurnThread(i);
	}
	//Sleep(1);
	//WaitForFarsiteEvents(EVENT_BURN, threadct, TRUE, INFINITE);
}


void Burn::ResumePerimeterThreads(long threadct)
{
	long i;

	for (i = 0; i < threadct; i++)
		burnthread[i]->SetRange(CurrentFire, SIMTIME, CuumTimeIncrement,
						TimeIncrement, TimeMaxRem, -1, 0, turn,
						FireIsUnderAttack, firering);
	for (i = 0; i < threadct; i++)
		burnthread[i]->StartBurnThread(i);
	//WaitForFarsiteEvents(EVENT_BURN, threadct, TRUE, INFINITE);
}

/*************************************************************************/
long Burn::StartPerimeterThreads_Equal()
{
long i;
long begin, end, threadct, range;
double fract, interval, ipart;

 	AllocPerimThreads();

	 interval = ((double) pFarsite->GetNumPoints(CurrentFire)) / ((double) NumPerimThreads);
	 fract = modf(interval, &ipart);
	 range = (long) interval;
	 if (fract > 0.0)
		  range++;
	 begin = threadct = 0;

	 for (i = 0; i < NumPerimThreads; i++)	{
  		end = begin + range;
		  if (begin >= pFarsite->GetNumPoints(CurrentFire))	{
	  		continue;	}
	  	if (end > pFarsite->GetNumPoints(CurrentFire))
			   end = pFarsite->GetNumPoints(CurrentFire);
		  burnthread[i]->SetRange(CurrentFire, SIMTIME, CuumTimeIncrement,
				TimeIncrement, TimeMaxRem, begin, end, turn,
				FireIsUnderAttack, firering);
		  threadct++;
		  begin = end;}

	 for (i = 0; i < threadct; i++)
		   burnthread[i]->StartBurnThread(i);

	 return threadct;
}

/******************************************************************************/

long Burn::StartPerimeterThreads_ActiveOnly()
{
	long i, NumLive;
	long begin, end, liverange, threadct;//, range;
	double fli;

	AllocPerimThreads();
	NumLive = 0;
	if (NumPerimThreads > 1)
	{
		for (CurrentPoint = 0;
			CurrentPoint < pFarsite->GetNumPoints(CurrentFire);
			CurrentPoint++)
		{
			fli = pFarsite->GetPerimeter2Value(CurrentPoint, FLIVAL);
			if (fli >= 0.0)
				NumLive++;
		}
		liverange = NumLive / NumPerimThreads + 1;
		if (liverange > pFarsite->GetNumPoints(CurrentFire))
			liverange = pFarsite->GetNumPoints(CurrentFire);
	}
	begin = threadct = 0;
	for (i = 0; i < NumPerimThreads; i++)
	{
		if (NumPerimThreads > 1)
		{
			NumLive = 0;
			for (CurrentPoint = begin;
				CurrentPoint < pFarsite->GetNumPoints(CurrentFire);
				CurrentPoint++)
			{
				fli = pFarsite->GetPerimeter2Value(CurrentPoint, FLIVAL);
				if (fli >= 0.0)
					NumLive++;
				if (NumLive > liverange)
					break;
			}
			end = CurrentPoint;
			if (begin >= pFarsite->GetNumPoints(CurrentFire))
				continue;
			if (end > pFarsite->GetNumPoints(CurrentFire))
				end = pFarsite->GetNumPoints(CurrentFire);
		}
		else
		{
			begin = 0;
			end = pFarsite->GetNumPoints(CurrentFire);
		}
		burnthread[i]->SetRange(CurrentFire, SIMTIME, CuumTimeIncrement,
						TimeIncrement, TimeMaxRem, begin, end, turn,
						FireIsUnderAttack, firering);
		threadct++;
		begin = end;
	}
	for (i = 0; i < threadct; i++)
		burnthread[i]->StartBurnThread(i);
	//WaitForFarsiteEvents(EVENT_BURN, threadct, true, INFINITE);

	return threadct;
}

extern int perim2Num;

void Burn::BurnMethod1()
{
// In-TimeStep growth of new inward fires 12/27/1994
bool INFIRE = false;
long i, NewFires, NewPts, NewInFires, ThisInFire;
long InFiresBurned = 0; 			 // Number of new inward burning fires
long FireType = pFarsite->GetInout(CurrentFire);    // save fire type in case inward fire burns out in this timestep
long NumStartAttack, NumLastAttack;
bool GroupDirectAttack;	//=false;
double DownTime, xpt, ypt;//, *perim;
long searchDir = 0;

GroupAirAttack* gaat;
AttackData* atk;						// Pointer to AttackData struct
Attack Atk(pFarsite);						// Instance of Attack

 	NewInFires = NewFires = pFarsite->GetNewFires();	// copy of new fire number
	 NumInFires = 0;						// reset BURN::NumInFires;
	 turn = 0;
	 StillBurning = false;   					  // start each fire out as burning (changed)
	 CanStillBurn = false;   					  // must test to see if fire is still active
	 InitRect(CurrentFire);				// resets hi's and lo's for bounding rectangle
	 TimeIncrement = EventTimeStep = pFarsite->GetActualTimeStep();
	 CuumTimeIncrement = 0.0;
	 pFarsite->AllocElev(CurrentFire);				// alloc space for elevations
	 tranz(CurrentFire, 0);				// transfer points to perimeter2 array for next turn
	 TimeMaxRem = 0.0;   					// maximum time remaining
	 if ((atk = pFarsite->GetAttackForFireNumber(CurrentFire, 0, &NumLastAttack)) != 0)
		  FireIsUnderAttack = true;
	 else
		  FireIsUnderAttack = false;

 	if ((pFarsite->GetGroupAttackForFireNumber(CurrentFire, 0, &NumLastAttack)) != 0) 	{
	  	GroupDirectAttack = true;
		  gaat = new GroupAirAttack(pFarsite);}
	 else	{
		  GroupDirectAttack = false;
		  gaat = 0;}
	 //pFarsite->WritePerimeter1Shapefile(perim2Num++, CurrentFire);
  do	{
	  	ExNumPts = 0;
		  if (pFarsite->GetInout(CurrentFire) == 3 && !pFarsite->LEAVEPROCESS)	{
			   CrossFires(0, &CurrentFire);
			   CanStillBurn = true;
			   TimeIncrement = 0.0;
			   CuumTimeIncrement = pFarsite->GetActualTimeStep();
			   break;	}
		  NumStartAttack = 0;
		  if (turn == 0)	{

      this->pFarsite->cfmc.CheckMoistureTimes(SIMTIME + CuumTimeIncrement);

// 			env->CheckMoistureTimes(SIMTIME + CuumTimeIncrement);

  			ThreadCount = StartPerimeterThreads_Equal();	}
		  else	if(!pFarsite->LEAVEPROCESS){
		   	ResumePerimeterThreads(ThreadCount);
			   CanStillBurn = burnthread[0]->CanStillBurn;
			   StillBurning = burnthread[0]->StillBurning;
			   prod.cuumslope[0] += burnthread[0]->prod.cuumslope[0];
			   prod.cuumslope[1] += burnthread[0]->prod.cuumslope[1];
			   for (i = 1; i < ThreadCount; i++)	{
				    if (burnthread[i]->CanStillBurn)
					     CanStillBurn = true;
				    if (burnthread[i]->StillBurning)
					     StillBurning = true;
		    		prod.cuumslope[0] += burnthread[i]->prod.cuumslope[0];
				    prod.cuumslope[1] += burnthread[i]->prod.cuumslope[1];	}
	  	}

		//--------------------------------------------------------------------------
		//--------------------------------------------------------------------------
		//--------------------------------------------------------------------------

		if (turn && !pFarsite->LEAVEPROCESS)
		{
			//pFarsite->WritePerimeter1Shapefile(CurrentFire);
			for (i = 0; i < pFarsite->GetNumPoints(CurrentFire); i++)
			{
				xpt = pFarsite->GetPerimeter1Value(CurrentFire, i, XCOORD);
				ypt = pFarsite->GetPerimeter1Value(CurrentFire, i, YCOORD);
				DetermineHiLo(xpt, ypt);
			}

			if (CuumTimeIncrement == 0.0 && INFIRE == false)	// should only happen first time through
			{
				prod.arp(CurrentFire);				// calc area of previous fire perimeter
				DownTime = pFarsite->GetDownTime();				// time spent out of active burn period
			}
			else
				DownTime = 0.0;
			NewFires = pFarsite->GetNewFires();
			while ((atk = pFarsite->GetAttackForFireNumber(CurrentFire, NumStartAttack, &NumLastAttack)) != 0)
			{
				NumStartAttack = NumLastAttack + 1;
				if (atk->Indirect == 2)
				{
					if (!Atk.ParallelAttack(atk, TimeIncrement + DownTime))
						pFarsite->CancelAttack(atk->AttackNumber);
				}
				else if (!Atk.DirectAttack(atk, TimeIncrement + DownTime))  // perform attack on this fire in timeincrement
					pFarsite->CancelAttack(atk->AttackNumber);
			}
			if (GroupDirectAttack)		// air attack, must be here because it is attacking perim1
			{
				NumStartAttack = 0;
				while ((pFarsite->GetGroupAttackForFireNumber(CurrentFire,
							NumStartAttack, &NumLastAttack)) !=
					0)
				{
					NumStartAttack = NumLastAttack + 1;
					gaat->GetCurrentGroup();
					if (gaat->CheckSuspendState(GETVAL))
						continue;
					gaat->ExecuteAttacks(0.0); //for each aircraft,
				}   									  //execute, but if not==TimeIncrement, just dec waittimes
			}

			WriteHiLo(CurrentFire);
			if (FireIsUnderAttack)
			{
				RestoreDeadPoints(CurrentFire);
				BoundingBox(CurrentFire);
			}
			if (pFarsite->GetInout(CurrentFire) != 0 && StillBurning && !pFarsite->LEAVEPROCESS)	// if inward fire not eliminated && still burning
			{
				if (rastmake)  						// comput raster information
					rast->rasterinit(CurrentFire, 0, SIMTIME, TimeIncrement,
							CuumTimeIncrement + TimeIncrement);
				if (FireIsUnderAttack)
				{
					while (pFarsite->GetNewFires() > NewFires && !pFarsite->LEAVEPROCESS)
					{
						if (pFarsite->GetInout(NewFires) != 1)
						{
							NewFires++;
							continue;
						}
						ReorderPerimeter(NewFires,
							FindExternalPoint(NewFires, searchDir));
						FindOuterFirePerimeter(NewFires);
						NewPts = pFarsite->GetNumPoints(NewFires);
						pFarsite->FreePerimeter1(NewFires);
						if (NewPts > 0)
						{
							pFarsite->AllocPerimeter1(NewFires, NewPts + 1);
							tranz(NewFires, NewPts);
						}
						else
						{
							pFarsite->SetNumPoints(NewFires, 0);
							pFarsite->SetInout(NewFires, 0);
							pFarsite->IncSkipFires(1);
						}
						NewFires++;
					}
				}
				//pFarsite->WritePerimeter1Shapefile(perim2Num++, CurrentFire);
				/*if(CurrentFire == 0)
				{
					pFarsite->WritePerimeter1Shapefile(perim2Num++, CurrentFire);
					pFarsite->WritePerimeter2Shapefile(perim2Num++, CurrentFire);
				}*/
				CrossFires(0, &CurrentFire);   		// clip cross-overs and rediscretize
				/*if(CurrentFire == 0)
				{
					pFarsite->WritePerimeter1Shapefile(perim2Num++, CurrentFire);
					pFarsite->WritePerimeter2Shapefile(perim2Num++, CurrentFire);
				}*/
				if (pFarsite->GetInout(CurrentFire) == 2) 		   // if inward burning fire
				{
					if (arp(1, CurrentFire) >= 0.0)   	// eliminates inward fire if area>0
						EliminateFire(CurrentFire);
				}
				CrossesWithBarriers(CurrentFire);		// merge with barriers if present
				while (pFarsite->GetNewFires() > NewInFires && !pFarsite->LEAVEPROCESS)
				{
					ThisInFire = NewInFires++;  		 // copy before incrementing newinfires
					if (pFarsite->GetInout(ThisInFire) == 2)
					{
				//pFarsite->WritePerimeter1Shapefile(perim2Num++, CurrentFire);
				//pFarsite->WritePerimeter2Shapefile(perim2Num++, CurrentFire);
						tranz(ThisInFire, 0);   	   // transfer points to perimeter2 in case it is removed
						CrossFires(0, &ThisInFire);  	   		// clip cross-overs and rediscretize
				//pFarsite->WritePerimeter2Shapefile(perim2Num++, CurrentFire);
						if (pFarsite->GetInout(ThisInFire) == 2)
						{
							if (arp(1, ThisInFire) >= 0.0)   	// eliminates inward fire if area>0
								EliminateFire(ThisInFire);
							else if ((pFarsite->GetActualTimeStep() -
								CuumTimeIncrement +
								TimeIncrement) >
								0.0)
								AllocNewInFire(ThisInFire,
									pFarsite->GetActualTimeStep() -
									CuumTimeIncrement -
									TimeIncrement);
						}
					}
				}
			}

			CuumTimeIncrement += TimeIncrement;	// accumulate timeincrements

			if (pFarsite->GetInout(CurrentFire) == 0 || !StillBurning)// if inward fire has now been eliminated
			{
				if (FireType == 2 && StillBurning) 		// inward fire was extinguished
				{
					if (rastmake)   					// make rasterizing accurate to the nearest time step
						rast->rasterinit(CurrentFire, ExNumPts, SIMTIME,
								CuumTimeIncrement, pFarsite->GetActualTimeStep());
					// compute raster information
				}
				TimeIncrement = 0.0;					// zero time left in timestep
			}
			else
			{
				TimeIncrement = pFarsite->GetActualTimeStep() - CuumTimeIncrement;
				EventTimeStep = TimeIncrement;  	 // reset Event drivent Time Step
				tranz(CurrentFire, 0);    		// transfer points to perimeter2 array for next turn
				turn = 0;
			}
		}
		else
		{
			TimeMaxRem = burnthread[0]->TimeMaxRem;
			EventTimeStep = burnthread[0]->EventTimeStep;
			for (i = 1; i < ThreadCount; i++)
			{
				if (burnthread[i]->TimeMaxRem > TimeMaxRem)
					TimeMaxRem = burnthread[i]->TimeMaxRem;
				if (burnthread[i]->EventTimeStep < EventTimeStep)
					EventTimeStep = burnthread[i]->EventTimeStep;
			}
			TimeIncrement -= TimeMaxRem;		 // Calculate distance limited TS
			if (GroupDirectAttack)
			{
				while ((pFarsite->GetGroupAttackForFireNumber(CurrentFire,
							NumStartAttack, &NumLastAttack)) !=
					0)
				{
					NumStartAttack = NumLastAttack + 1;
					gaat->GetCurrentGroup();   // get current group attack
					if (gaat->CheckSuspendState(GETVAL))
						continue;
					EventTimeStep = gaat->GetNextAttackTime(EventTimeStep);
				}
			}
			if (EventTimeStep<TimeIncrement && EventTimeStep>0.01)    // Check to see if event limited TS < distance limited TS
				TimeIncrement = EventTimeStep;
			if (CuumTimeIncrement > 0.0 && GroupDirectAttack)
				gaat->IncrementWaitTimes(TimeIncrement);		  // subtract last timestep
			turn = 1;
		}
		TimeMaxRem = 0.0;
		if (TimeIncrement == 0.0 && NumInFires > InFiresBurned)
		{
			do
			{
//				if (InFiresBurned <= NumInFires)  //JAS! was strictly <  JAS!
				if (InFiresBurned < NumInFires)  //BLN original
				{
					GetNewInFire(InFiresBurned);
					InFiresBurned++;
				}
				else
					TimeIncrement = -1.0;
				if (TimeIncrement > 0.0)
				{
					CuumTimeIncrement = pFarsite->GetActualTimeStep() - TimeIncrement;
					EventTimeStep = TimeIncrement; 		// reset EventTimeStep
					pFarsite->AllocElev(CurrentFire);	  // alloc space for elevations
					tranz(CurrentFire, 0);	  // transfer points to perimeter2 array for next turn
					InitRect(CurrentFire);	  // resets hi's and lo's for bounding rectangle
					CanStillBurn = false;		  // reset ability to burn as false
					FireType = 2;   			 // set local var. FireType to inward fire
					turn = 0;   				 // set turn to 1st time through
					INFIRE = true;
				}
			}
			while (TimeIncrement == 0.0 && !pFarsite->LEAVEPROCESS);
		}
//        chint=getch();  // JAS!
	}
	while (TimeIncrement > 0.0 && !pFarsite->LEAVEPROCESS);			  // while time remaining in time step
	  // pFarsite->WritePerimeter2Shapefile(perim2Num++, CurrentFire);
	if (!StillBurning && !CanStillBurn && pFarsite->GetInout(CurrentFire) == 2)	// eliminates fire perimeter around rock or lake islands
	{
		if (pFarsite->PreserveInactiveEnclaves(GETVAL) == false)  			   // but not in fuel
			EliminateFire(CurrentFire);
	}
	FreeNewInFires();
	if (gaat)
		delete gaat;	  // group air attack
}


void Burn::DetermineSimLevelTimeStep()
{
	long i;

	TimeMaxRem = 0.0;
	TotalPoints = 0;
	TimeIncrement = EventTimeStep = pFarsite->GetActualTimeStep() - CuumTimeIncrement;
	for (i = 0; i < pFarsite->GetNumFires(); i++)
		TotalPoints += pFarsite->GetNumPoints(i);
	CuumPoint = 0;
	for (CurrentFire = 0; CurrentFire < pFarsite->GetNumFires(); CurrentFire++)
	{
		if (pFarsite->GetInout(CurrentFire) == 3 || pFarsite->GetInout(CurrentFire) == 0)
			continue;
		PreBurn();
	}
	TimeIncrement -= TimeMaxRem;
	if (EventTimeStep<TimeIncrement && EventTimeStep>0.01)    // Check to see if event limited TS < distance limited TS
		TimeIncrement = EventTimeStep;
	pFarsite->SetTemporaryTimeStep(TimeIncrement);
	CuumPoint = 0;
}


void Burn::PreBurn()
{
	// just finds out what is the fastest spreading point among all fires ==> sets timestep
	bool GroupDirectAttack; //ComputeSpread,
	long i, NumStartAttack = 0, NumLastAttack; //i, TestPointL, TestPointN,
	GroupAirAttack* gaat;

	pFarsite->AllocElev(CurrentFire);				// alloc space for elevations
	tranz(CurrentFire, 0);				// transfer points to perimeter2 array for next turn
	if ((pFarsite->GetAttackForFireNumber(CurrentFire, 0, &NumLastAttack)) != 0)
		FireIsUnderAttack = true;
	else
		FireIsUnderAttack = false;
	if ((pFarsite->GetGroupAttackForFireNumber(CurrentFire, 0, &NumLastAttack)) != 0)
	{
		GroupDirectAttack = true;
		gaat = new GroupAirAttack(pFarsite);
	}
	else
	{
		GroupDirectAttack = false;
		gaat = 0;
	}

	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	turn = 0;
	this->pFarsite->cfmc.CheckMoistureTimes(SIMTIME + CuumTimeIncrement);

//  env->CheckMoistureTimes(SIMTIME + CuumTimeIncrement);
	ThreadCount = StartPerimeterThreads_Equal();

	TimeMaxRem = burnthread[0]->TimeMaxRem;
	EventTimeStep = burnthread[0]->EventTimeStep;
	for (i = 1; i < ThreadCount; i++)
	{
		if (burnthread[i]->TimeMaxRem > TimeMaxRem)
			TimeMaxRem = burnthread[i]->TimeMaxRem;
		if (burnthread[i]->EventTimeStep < EventTimeStep)
			EventTimeStep = burnthread[i]->EventTimeStep;
	}
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------

	if (GroupDirectAttack)
	{
		while ((pFarsite->GetGroupAttackForFireNumber(CurrentFire, NumStartAttack,
					&NumLastAttack)) !=
			0)
		{
			NumStartAttack = NumLastAttack + 1;
			gaat->GetCurrentGroup();   // get current group attack
			if (gaat->CheckSuspendState(GETVAL))
				continue;
			EventTimeStep = gaat->GetNextAttackTime(EventTimeStep);
		}
	}
}




void Burn::BurnMethod2()
{
	// Simulation-level process control, uses time step determind by PreBurn()
	long NewFires;
	long NewPts;
	long FireType = pFarsite->GetInout(CurrentFire);    // save fire type in case inward fire burns out in this timestep
	long NumStartAttack;
	long NumLastAttack;
	bool FireIsUnderAttack;			//=false;
	bool GroupDirectAttack;			//=false;
	long i;
	double xpt, ypt;
	GroupAirAttack* gaat;
	AttackData* atk;						// Pointer to AttackData struct
	Attack Atk(pFarsite);						// Instance of Attack
	// for rasterizing inside of extinct fire
	turn = 1; 								  // always is 'turn' because timestep already determined
	StillBurning = 1;   					  // start each fire out as burning (changed)
	CanStillBurn = false;   					  // must test to see if fire is still active
	InitRect(CurrentFire);				// resets hi's and lo's for bounding rectangle
	TimeIncrement = pFarsite->GetTemporaryTimeStep();
	tranz(CurrentFire, 0);				// transfer points to perimeter2 array for next turn
	if ((atk = pFarsite->GetAttackForFireNumber(CurrentFire, 0, &NumLastAttack)) != 0)
		FireIsUnderAttack = true;
	else
		FireIsUnderAttack = false;
	if ((pFarsite->GetGroupAttackForFireNumber(CurrentFire, 0, &NumLastAttack)) != 0)
	{
		GroupDirectAttack = true;
		gaat = new GroupAirAttack(pFarsite);
	}
	else
	{
		GroupDirectAttack = false;
		gaat = 0;
	}

	if (pFarsite->GetInout(CurrentFire) == 3)
	{
		CrossFires(0, &CurrentFire);
		CanStillBurn = true;
		TimeIncrement = 0.0;

		return;
	}
	pFarsite->AllocElev(CurrentFire);				// alloc space for elevations
	if (pFarsite->CheckPostFrontal(GETVAL))   	   	// store current fire perim in ring
		firering = post.SetupFireRing(CurrentFire,
							SIMTIME + CuumTimeIncrement,
							SIMTIME + CuumTimeIncrement + TimeIncrement);

	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	NumStartAttack = 0;

 this->pFarsite->cfmc.CheckMoistureTimes(SIMTIME + CuumTimeIncrement);

//	env->CheckMoistureTimes(SIMTIME + CuumTimeIncrement);
	ThreadCount = StartPerimeterThreads_Equal();

	CanStillBurn = burnthread[0]->CanStillBurn;
	StillBurning = burnthread[0]->StillBurning;
	prod.cuumslope[0] += burnthread[0]->prod.cuumslope[0];
	prod.cuumslope[1] += burnthread[0]->prod.cuumslope[1];
	for (i = 1; i < ThreadCount; i++)
	{
		if (burnthread[i]->CanStillBurn)
			CanStillBurn = true;
		if (burnthread[i]->StillBurning)
			StillBurning = true;
		prod.cuumslope[0] += burnthread[i]->prod.cuumslope[0];
		prod.cuumslope[1] += burnthread[i]->prod.cuumslope[1];
	}

	for (i = 0; i < pFarsite->GetNumPoints(CurrentFire); i++)
	{
		xpt = pFarsite->GetPerimeter1Value(CurrentFire, i, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(CurrentFire, i, YCOORD);
		DetermineHiLo(xpt, ypt);
	}
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------

	ExNumPts = 0;
	prod.arp(CurrentFire);				// calc area of previous fire perimeter
	NewFires = pFarsite->GetNewFires();
	while ((atk = pFarsite->GetAttackForFireNumber(CurrentFire, NumStartAttack,
					&NumLastAttack)) !=
		0)
	{
		NumStartAttack = NumLastAttack + 1;
		if (atk->Indirect == 2)
		{
			if (!Atk.ParallelAttack(atk, TimeIncrement + pFarsite->GetDownTime()))
				pFarsite->CancelAttack(atk->AttackNumber);
		}
		else if (!Atk.DirectAttack(atk, TimeIncrement + pFarsite->GetDownTime()))  // perform attack on this fire in timeincrement
			pFarsite->CancelAttack(atk->AttackNumber);
	}
	if (GroupDirectAttack)
	{
		NumStartAttack = 0;
		while ((pFarsite->GetGroupAttackForFireNumber(CurrentFire, NumStartAttack,
					&NumLastAttack)) !=
			0)
		{
			NumStartAttack = NumLastAttack + 1;
			gaat->GetCurrentGroup();
			if (gaat->CheckSuspendState(GETVAL))
				continue;
			gaat->ExecuteAttacks(TimeIncrement + pFarsite->GetDownTime()); //for each aircraft,
		}   									  //execute, but if not==TimeIncrement, just dec waittimes
	}
	WriteHiLo(CurrentFire);
	if (FireIsUnderAttack)
	{
		RestoreDeadPoints(CurrentFire);
		BoundingBox(CurrentFire);
	}

	if (FireIsUnderAttack && pFarsite->CheckPostFrontal(GETVAL))
		post.UpdateAttackPoints(firering, CurrentFire);

	if (pFarsite->GetInout(CurrentFire) != 0 && StillBurning)	// if inward fire not eliminated && still burning
	{
		if (rastmake)  						// comput raster information
			rast->rasterinit(CurrentFire, ExNumPts, SIMTIME, TimeIncrement,
					CuumTimeIncrement + TimeIncrement);
		if (FireIsUnderAttack)
		{
			while (pFarsite->GetNewFires() > NewFires)
			{
				if (pFarsite->GetInout(NewFires) != 1)
				{
					NewFires++;
					continue;
				}
				ReorderPerimeter(NewFires, FindExternalPoint(NewFires, 0));
				FindOuterFirePerimeter(NewFires);
				NewPts = pFarsite->GetNumPoints(NewFires);
				pFarsite->FreePerimeter1(NewFires);
				if (NewPts > 0)
				{
					pFarsite->AllocPerimeter1(NewFires, NewPts + 1);
					tranz(NewFires, NewPts);
				}
				else
				{
					//FreePerimeter2();
					pFarsite->SetNumPoints(NewFires, 0);
					pFarsite->SetInout(NewFires, 0);
					pFarsite->IncSkipFires(1);
				}
				NewFires++;
			}
		}
		CrossFires(0, &CurrentFire);   		// clip cross-overs and rediscretize
		if (pFarsite->GetInout(CurrentFire) == 2) 		   // if inward burning fire
		{
			if (arp(1, CurrentFire) >= 0.0)   	// eliminates inward fire if area>0
			{
				EliminateFire(CurrentFire);
				if (pFarsite->CheckPostFrontal(GETVAL))
					pFarsite->FreeFireRing(pFarsite->GetNumRings() - 1);
			}
		}
		CrossesWithBarriers(CurrentFire);		// merge with barriers if present
	}

	if (pFarsite->GetInout(CurrentFire) == 0 || !StillBurning)// if inward fire has now been eliminated
	{
		if (FireType == 2 && StillBurning) 		// inward fire was extinguished
		{
			if (rastmake)   					// make rasterizing accurate to the nearest time step
				rast->rasterinit(CurrentFire, ExNumPts, SIMTIME,
						CuumTimeIncrement, CuumTimeIncrement + TimeIncrement);//GetActualTimeStep());
		}
		TimeIncrement = 0.0;					// zero time left in timestep
	}

	if (!StillBurning && !CanStillBurn && pFarsite->GetInout(CurrentFire) == 2)	// eliminates fire perimeter around rock or lake islands
	{
		if (pFarsite->PreserveInactiveEnclaves(GETVAL) == false)  			   // but not in fuel
			EliminateFire(CurrentFire); 				  		// but not in fuel
		if (pFarsite->CheckPostFrontal(GETVAL))
			pFarsite->FreeFireRing(pFarsite->GetNumRings() - 1);
	}
	if (GroupDirectAttack)
		gaat->IncrementWaitTimes(TimeIncrement + pFarsite->GetDownTime());		  // subtract last timestep
}


void Burn::AllocNewInFire(long NewNum, double TimeInc)
{
	// allocates linked list of new inward fire structures
	if (NumInFires == 0)		// BURN::NumInFires data member
	{
		FirstInFire = (newinfire *) new newinfire;//GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, 1*sizeof(newinfire));
		//TempInNext = (newinfire *)
		//	FirstInFire->next = (newinfire *) new newinfire;//GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, 1*sizeof(newinfire));

		newinfire* t_innext;
		t_innext = new newinfire;
		FirstInFire->next = t_innext;
		TempInNext = t_innext;

		TempInFire = FirstInFire;
	}
	TempInFire->FireNum = NewNum;
	TempInFire->TimeInc = TimeInc;
	TempInFire = TempInNext;
	//TempInNext = (newinfire *) TempInFire->next = (newinfire *) new newinfire;//GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, 1*sizeof(newinfire));

	newinfire* t_innext;
	t_innext = new newinfire;
	TempInFire->next = t_innext;
	TempInNext = t_innext;

	NumInFires++;
}


void Burn::GetNewInFire(long InFire)
{
	// retrieves data from linked list of new inward fires
	if (InFire == 0)
	{
		CurInFire = FirstInFire;
		NextInFire = (newinfire *) FirstInFire->next;
	}
	CurrentFire = CurInFire->FireNum;   	// BURN::CurrentFire
	TimeIncrement = CurInFire->TimeInc;	   // BURN::TimeIncrement
	CurInFire = NextInFire;
	NextInFire = (newinfire *) CurInFire->next;
}


void Burn::FreeNewInFires()
{
	// frees all linked list for new inward fires
	long i;

	if (NumInFires)
	{
		CurInFire = FirstInFire;
		NextInFire = (newinfire *) CurInFire->next;
		for (i = 0; i < NumInFires; i++)
		{
			delete CurInFire;//GlobalFree(CurInFire);
			CurInFire = NextInFire;
			NextInFire = (newinfire *) CurInFire->next;
		}
		delete CurInFire; //GlobalFree(CurInFire);
		delete NextInFire; //GlobalFree(NextInFire);
		NumInFires = 0;
	}
}


void Burn::EliminateFire(long FireNum)
{
	// removes a fire, and cleans up memory for it
	pFarsite->SetNumPoints(FireNum, 0);  		// reset number of points
	pFarsite->SetInout(FireNum, 0);     	 	// reset inward/outward indicator
	pFarsite->FreePerimeter1(FireNum);			// free perimeter array
	pFarsite->IncSkipFires(1);			  	// increment number of extinquished fires
}


void Burn::SetSpotLocation(long loc)
{
	long i;
	if (loc < 0)
	{
		for (i = 0; i < NumPerimThreads; i++)
		{
			if (loc < -1)   					// remove all of them
			{
				burnthread[i]->embers.CarrySpots = 0;
				burnthread[i]->embers.NextSpot = burnthread[i]->embers.FirstSpot;
				if (loc < -2)
					burnthread[i]->embers.EmberReset();
			}
			else if (burnthread[i]->embers.CarrySpots > 0)
				burnthread[i]->embers.NextSpot = (Embers::spotdata *)
					burnthread[i]->embers.CarrySpot->next;
			else
				burnthread[i]->embers.NextSpot = burnthread[i]->embers.FirstSpot;
			if (burnthread[i]->embers.NumSpots > 0)
				burnthread[i]->embers.SpotReset(burnthread[i]->embers.NumSpots,
										burnthread[i]->embers.FirstSpot);
			else if (burnthread[i]->embers.SpotFires > 0)
				burnthread[i]->embers.SpotReset(burnthread[i]->embers.SpotFires -
										burnthread[i]->embers.CarrySpots,
										burnthread[i]->embers.NextSpot);
		}
		SpotCount = 0;

		return;
	}

	CurThread = -1;
	if (pFarsite->EnableSpotFireGrowth(GETVAL))
	{
		for (i = 0; i < NumPerimThreads; i++)
		{
			if (CurThread == -1 && NumSpots[i] > 0)
			{
				CurThread = i;
				burnthread[CurThread]->embers.CurSpot = burnthread[CurThread]->embers.FirstSpot;
				burnthread[CurThread]->embers.CarrySpot = burnthread[CurThread]->embers.FirstSpot;
				break;
			}
		}
	}
	else
	{
		for (i = 0; i < NumPerimThreads; i++)
		{
			if (CurThread == -1 && NumSpots[i] > 0)
			{
				CurThread = i;
				burnthread[CurThread]->embers.CurSpot = burnthread[CurThread]->embers.FirstSpot;
				burnthread[CurThread]->embers.CarrySpot = burnthread[CurThread]->embers.FirstSpot;
				break;
			}
		}
	}
	SpotCount = 0;
}


Embers::spotdata* Burn::GetSpotData(double CurrentTimeStep)
{
	// will add CurrentTimeStep to CurSpot->TimeRem and return 0 if delay is longer than current TimeStep
	if (CurThread >= NumPerimThreads)
		return NULL;

	if (SpotCount >= NumSpots[CurThread])
	{
		SpotCount = 0;
		do
		{
			CurThread++;
			if (CurThread >= NumPerimThreads)
			{
				SpotCount = 0;

				return NULL;
			}
			burnthread[CurThread]->embers.CurSpot = burnthread[CurThread]->embers.FirstSpot;
		}
		while (SpotCount >= NumSpots[CurThread]);
	}
	else if (SpotCount > 0 && SpotCount < NumSpots[CurThread])
	{
		burnthread[CurThread]->embers.NextSpot = (Embers::spotdata *)
			burnthread[CurThread]->embers.CurSpot->next;
		burnthread[CurThread]->embers.CurSpot = burnthread[CurThread]->embers.NextSpot;
	}
	SpotCount++;
	// looks at timeremaining for spot fire and adds time to it for delay if less than 0.0
	// then stores it at the beginning of the linked list
	if (CurrentTimeStep > 0.0)
	{
		if (burnthread[CurThread]->embers.CurSpot->TimeRem <= 0.0)
		{
			if (burnthread[CurThread]->embers.CarrySpots == 0)
				burnthread[CurThread]->embers.CarrySpot = burnthread[CurThread]->embers.FirstSpot;
			else
			{
				burnthread[CurThread]->embers.NextSpot = (Embers::spotdata *)
					burnthread[CurThread]->embers.CarrySpot->next;
				burnthread[CurThread]->embers.CarrySpot = burnthread[CurThread]->embers.NextSpot;
			}
			burnthread[CurThread]->embers.CurSpot->TimeRem += CurrentTimeStep;
			memcpy(burnthread[CurThread]->embers.CarrySpot,
				burnthread[CurThread]->embers.CurSpot, 3 * sizeof(double));
			if (burnthread[CurThread]->embers.CarrySpots >=
				NumSpots[CurThread])
				burnthread[CurThread]->embers.CarrySpots += 1;
			else
				burnthread[CurThread]->embers.CarrySpots++;

			return NULL;
		}
	}

	return burnthread[CurThread]->embers.CurSpot;
}


void Burn::BurnSpotThreads()
{
	long TotalEmbers, dest, * amount, num, excess;
	long i, j, k, m, range;
	double fract, interval, ipart;


	TotalEmbers = 0;
	for (i = 0; i < NumPerimThreads; i++)	// for each thread
		TotalEmbers += burnthread[i]->embers.NumEmbers;

	if (TotalEmbers == 0)
	{
		TotalSpots = SpotFires = 0;

		return;
	}


	interval = ((double) (TotalEmbers) / (double) NumPerimThreads);
	fract = modf(interval, &ipart);
	range = (long) interval;
	if (fract > 0.0)
		range++;

	// this section equalizes the numbers of embers in each thread for flight calculation
	amount = new long[NumPerimThreads];

	for (i = 0; i < NumPerimThreads; i++)
	{
		for (k = 0; k < NumPerimThreads; k++)//NumPerimThreads; i++)
			amount[k] = -1;
		// find where embers are short
		if (burnthread[i]->embers.NumEmbers < range)
		{
			num = range - burnthread[i]->embers.NumEmbers;
			dest = i;
			// find the embers from among all burnthreads[]
			for (j = 0; j < NumPerimThreads; j++)	// for each thread
			{
				if (j == dest)
					continue;
				excess = burnthread[j]->embers.NumEmbers - range;
				if (excess > 0)
				{
					if (excess >= num)
					{
						amount[j] = num;

						break;
					}
					else
					{
						amount[j] = excess;
						num -= excess;
					}
				}
			}
			for (k = 0; k < NumPerimThreads; k++)
			{
				if (amount[k] > 0)
				{
				    emberdata tEmber;
					for (m = 0; m < amount[k]; m++)
					{
					    tEmber = burnthread[k]->embers.ExtractEmber(0);
						burnthread[dest]->embers.AddEmber(&tEmber);
						//burnthread[dest]->embers.AddEmber(&(burnthread[k]->embers.ExtractEmber(0)));
                    }
				}
			}
		}
	}
	delete[] amount;

	//This section starts the spot threads and synchronizes their completion before returning
	SpotCount = 0;
	ResumeSpotThreads(NumPerimThreads);

	TotalSpots = SpotFires = 0;
	CurThread = -1;
	if (pFarsite->EnableSpotFireGrowth(GETVAL))
	{
		for (i = 0; i < NumPerimThreads; i++)
		{
			NumSpots[i] = burnthread[i]->embers.SpotFires;
			TotalSpots += NumSpots[i];
			if (CurThread == -1 && NumSpots[i] > 0)
			{
				CurThread = i;
				burnthread[CurThread]->embers.CurSpot = burnthread[CurThread]->embers.FirstSpot;
			}
			burnthread[i]->embers.CarrySpots = 0;
		}
		SpotFires = TotalSpots;
	}
	else
	{
		for (i = 0; i < NumPerimThreads; i++)
		{
			NumSpots[i] = burnthread[i]->embers.NumSpots;
			TotalSpots += NumSpots[i];
			if (CurThread == -1 && NumSpots[i] > 0)
			{
				CurThread = i;
				burnthread[CurThread]->embers.CurSpot = burnthread[CurThread]->embers.FirstSpot;
			}
			burnthread[i]->embers.CarrySpots = 0;
		}
		SpotFires = 0;
	}
}



