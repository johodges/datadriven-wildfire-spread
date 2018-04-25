/********************************************************************************
* Name: Farsite5.cpp
* Note: --> Now-11-10 This file was update with most recent changes to the
*             original Farsite code.
*
*********************************************************************************/

#include "Farsite5.h"
#include <memory.h>
#include <cstring>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include "barriers.hpp"

#ifndef WIN32
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define strcmpi strcasecmp
#define FALSE 0
#define MAX_PATH 256
void CopyFile(char *in, char *out, bool unused)
{

    int read_fd;
    int write_fd;
    struct stat stat_buf;
    off_t offset = 0;

    read_fd = open(in, O_RDONLY);
    fstat(read_fd, &stat_buf);
    write_fd = open(out, O_WRONLY | O_CREAT, stat_buf.st_mode);
    sendfile(write_fd, read_fd, &offset, stat_buf.st_size);
    close(read_fd);
    close(write_fd);
}
#endif
using namespace std;

const double IGNITON_GRID_LINEDIST_DIVISOR  = 1000.0;

const double PI = acos(-1.0);

extern int CloseAndReturn(FILE *stream, int returnCode);

CSpotData::CSpotData()
{
	launchTime = launchX = launchY = landTime = landX = landY = 0.0;
}

CSpotData::~CSpotData()
{
}

StationGrid::StationGrid()
{
	Grid = 0;
}

StationGrid::~StationGrid()
{
	if (Grid)
		delete[] Grid;
}

CoarseWoody::CoarseWoody()
{
	wd = 0;
	NumClasses = 0;
	Units = -1;
	TotalWeight = 0.0;
	Depth = 0.0;
}

CanopyCharacteristics::CanopyCharacteristics()
{
	DefaultHeight = Height = 15.0;  		 // need default for changing Variables in LCP
	DefaultBase = CrownBase = 4.0;
	DefaultDensity = BulkDensity = 0.20;
	Diameter = 20.0;
	FoliarMC = 100;
	Tolerance = 2;
	Species = 1;
}

FuelConversions::FuelConversions()
{
	long i;

	for (i = 0;
		i < 257;
		i++)   		  		// could also read default file here
		Type[i] = i;
	Type[99] = -1;
	Type[98] = -2;
	Type[97] = -3;
	Type[96] = -4;
	Type[95] = -5;
	Type[94] = -6;
	Type[93] = -7;
	Type[92] = -8;
	Type[91] = -9;
}

bool Farsite5::WriteClocks()
{
	ElTimeConvert();
	CurTimeConvert();
	return true;
}

void Farsite5::ElTimeConvert()
{
	double TEMPTIME = burn.SIMTIME - GetActualTimeStep();
	double Hr, Dy;
	long dy = 0, hr = 0, min;
	char colon[] = ":";

	if (TEMPTIME >= 1440.0)
	{
		Dy = TEMPTIME / 1440.0;
		dy = (long) Dy;
		TEMPTIME -= 1440 * dy;
	}
	if (TEMPTIME >= 60.0)
	{
		Hr = TEMPTIME / 60.0;
		hr = (long) Hr;
		TEMPTIME -= 60 * hr;
	}
	min = (long)TEMPTIME;//*60;
	//TimeKeepElapsed.SetData(dy, hr, min, 0);
	sprintf(ElTime, "%02d %02d%s%02d", (int)dy, (int)hr, colon, (int)min);
}


void Farsite5::CurTimeConvert()
{
	double TEMPTIME = burn.SIMTIME - GetActualTimeStep();
	double Hr, Dy;
	long mo = GetStartMonth(), dy = 0, hr = 0, min;
	char colon[] = ":";
	char slash[] = "/";

	if (TEMPTIME >= 1440.0)
	{
		Dy = TEMPTIME / 1440.0;
		dy = (long) Dy;
		TEMPTIME -= (1440 * dy);					 // truncate to nearest day
	}
	if (TEMPTIME >= 60.0)
	{
		Hr = TEMPTIME / 60.0;
		hr = (long) Hr;
		TEMPTIME -= (60 * hr);					 // truncate to nearesat hour
	}
	min = (long) TEMPTIME;  						// minutes left over
	min += GetStartMin();
	if (min >= 60)
	{
		min -= 60;
		hr++;
	}
	hr += (GetStartHour() / 100);
	if (hr >= 24)
	{
		hr -= 24;
		dy++;
	}
	dy += GetStartDay();
	long days, oldmo;
	do
	{
		oldmo = mo;
		switch (mo)
		{
		case 1:
			days = 31; break;			 // days in each month, ignoring leap year
		case 2:
			days = 28; break;
		case 3:
			days = 31; break;
		case 4:
			days = 30; break;
		case 5:
			days = 31; break;
		case 6:
			days = 30; break;
		case 7:
			days = 31; break;
		case 8:
			days = 31; break;
		case 9:
			days = 30; break;
		case 10:
			days = 31; break;
		case 11:
			days = 30; break;
		case 12:
			days = 31; break;
		}
		if (dy > days)
		{
			dy -= days;
			mo++;
			if (mo > 12)
				mo -= 12;
		}
	}
	while (mo != oldmo);						// allows startup of current clock at any time, will find cur month
	//TimeKeepCurrent.SetData(dy, hr, min, mo);
	sprintf(CurTime, "%02ld%s%02ld  %02ld%s%02ld", mo, slash, dy, hr, colon,
		min);
}




Farsite5::Farsite5(void) : burn(this), Atk(this), Ignition(this), vect(this), shape(this)
{
	timeLaunchFarsite = timeFinish = 0;
	NumCrews = 0;
	OutputVectsAsGrown = 0;
	CrownFireCalculation=0;			// 0=Finney (1998), 1=Scott&Reinhardt (2001)
	ignitionGrid = NULL;
	ignitionCols = 0;
	ignitionRows = 0;
	CantAllocLCP = false;
	landscape = NULL;
	landfile = 0;
	OldFilePosition = 0;
	 NEED_CUST_MODELS = false;	// custom fuel models
	 HAVE_CUST_MODELS = false;
	 NEED_CONV_MODELS = false;     // fuel model conversions
	 HAVE_CONV_MODELS = false;
	SimulationDuration = 0.0;
	CondPeriod = false;

	PerimeterResolution = 0.0;			// maximum tangential perimeter resolution
	DistanceResolution = 0.0; 	   	// minimum radial spread distance resolution
	DynamicDistanceResolution = 0.0;	// minimum run-time radial resolution
	actual = 0.0;					// actual time step
	visible = 0.0;				   	// visible time step
	TemporaryTimestep = 0.0;			// temp timestep for sim-level proc. control
	secondaryvisible = -1.0;  	   	// seconary visible timestep
	EventMinTimeStep = -1.0;			// event driven minimum timestep
	checkexpansions = false;
	checkpostfrontal = false;
	DistanceMethod = 1;				// method for distance checking
	AccelerationState = true;			// Flag for using acceleration constants

	NumPerimAlloc = 0;				// Current number of perimeter arrays allocated with new
	inout = 0;						// fire doesn't exist (0), burning out(2), in(2)
	numpts = 0;				   		// number of points in each fire
	perimeter1 = 0; 		   		// pointers to arrays with perimeter points
	perimeter2 = 0;					// swap array

	PercentageOfEmberIgnitions = 5.0;  	// % embers that start fires
	SpotIgnitionDelay = 0.0;			// delay (minutes) for ignition of spot fires
	CrowningOK = true;  				 // enable crowning
	SpottingOK = true;  				 // enable spotting (just ember generation and flight)
	SpotGrowthOK = false;   			 // enable spot fire growth
	ConstBack = false;  				 // use constant backing spread rate (no wind no slope)

	numfires = 0;					// number of fires
	newfires = 0;   				   	// number of new fires
	numspots = 0;   				   	// number of spot fires
	skipfire = 0;   				   	// number of extinguished fires
	p2numalloc = 0; 				   	// allocated space in perimeter2 array
	GroundElev = 0;					// stores elevation of points
	numelev = 0;
	NumStopLocations = 0;

	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 5; j++)
			EnvtChanged[i][j] = false;
	//1hr
	MoistCalcInterval[0][0] = 60;
	MoistCalcInterval[0][1] = 200;
	MoistCalcInterval[0][2] = 10;
	MoistCalcInterval[0][3] = 45;
	MoistCalcInterval[0][4] = 15;
	//10hr
	MoistCalcInterval[1][0] = 60;
	MoistCalcInterval[1][1] = 200;
	MoistCalcInterval[1][2] = 10;
	MoistCalcInterval[1][3] = 45;
	MoistCalcInterval[1][4] = 15;
	//100hr
	MoistCalcInterval[2][0] = 120;
	MoistCalcInterval[2][1] = 400;
	MoistCalcInterval[2][2] = 10;
	MoistCalcInterval[2][3] = 45;
	MoistCalcInterval[2][4] = 20;
	//1000hr
	MoistCalcInterval[3][0] = 240;
	MoistCalcInterval[3][1] = 400;
	MoistCalcInterval[3][2] = 15;
	MoistCalcInterval[3][3] = 45;
	MoistCalcInterval[3][4] = 30;

	rast_arrivaltime = false;
	rast_fireintensity = false;
	rast_spreadrate = false;
	rast_flamelength = false;
	rast_heatperarea = false;
	rast_crownfire = false;
	rast_firedirection = false;
	rast_reactionintensity = false;
	OutputUnits = 0;
	CanSetRastRes = true;
	AtmGrid = 0;
	NumWeatherStations = 0;
	NumWindStations = 0;
	NorthGridOffset = 0.0;
	EastGridOffset = 0.0;

	for(int i = 0; i < 5; i++)
	{
		wddt[i] = 0;
		wtrdt[i] = 0;
		MaxWeatherObs[i] = MaxWindObs[i] = 0;
	}

	CombineOption = CWD_COMBINE_ABSENT;
	MaxThreads = 1;
	DownTime = 0.0;
	NumAbsoluteData = 0;
	NumRelativeData = 0;
	LastAccess = -1;
	abp = 0;
	rbp = 0;

	InactiveEnclaves = true;

	NumAirCraft = 0;
	NumAirAttacks = 0;
	AirAttackCounter = 0;
	NumGroupAttacks = 0;
	GroupAttackCounter = 0;

	PRECISION = 12;		// number of sampling points
	BURNUP_TIMESTEP = 15;	// seconds

	NumRingStructs = 0;
	NumRings = 0;
	FirstRing = 0;
	NextRing = 0;
	CurRing = 0;

	FuelModelUnits = 0;
	CrownDensityLinkToCC = false;    // crown bulk density is a function of canopy cover

	conditmonth = conditday = 0;
	startmonth = startday = starthour = startmin = startdate = 0;		//INITIAL CONDITIONS AT STARTPOINT
	endmonth = endday = endhour = endmin = enddate = 0;

	NumAttacks = 0;
	AttackCounter = 0;
	FirstAttack = NextAttack = CurAttack = LastAttack = Reassignment = 0;
	LandscapeInputMode = NULLLOCATION;
	ShapeFileName[0] = 0;
	ShapeVisibleStepsOnly = 1;
	ShapePolygonsNotLines = 1;
	ShapeBarriersSeparately = 0;

	lcptheme = 0;

	NextFireAfterInterrupt = 0;
	RastMake = false;

	PrimaryVisEqual = 0;
	PrimaryVisCount = 0;
	SecondaryVisEqual = -1;
	SecondaryVisCount = 0;
	ShowVectsAsGrown = 0;

	InitializeRosRed();			// set rate of spread adjustments to 1.0;
	NumSimThreads = 0;
	TerminateWait = WaitInProgress = false;
	SimRequest = SIMREQ_NULL;
	SIM_SUSPENDED = SIM_COMPLETED = false;
	SIMULATE_GO = false;
	FARSITE_GO = false; LEAVEPROCESS = false;
	NextFireAfterInterrupt = 0;
	OldFireAfterInterrupt = 0;
	VisPerimSize = 0;   			// size of disk copy of vector perimeters
	CanModify = true;
	PreCalcFuelMoistures = true;
        LastFMCalcTime=0;
	VectMake = false;
	ShapeMake = false;

	RastFormat = 5;
	VectFormat = 0;
	gaat = 0;
	VISONLY = false;		// visible time step only for vector file

	F_ON = false;
	IgnitionReset = 0;
	MemCount = 0;
	NumVectorThemes = 0;
	FirstVect = 0;
	CurVect = 0;
	NextVect = 0;
	NumCompoundCrews = 0;
	perimeters = 0;
	lastPerimeter = 0;
	CanModifyInputs(true);
	ResetNewFuels();
	SetModelParams() ;
#ifdef WIN32
	InitializeCriticalSection(&progressCS);
#endif
	InputsFName[0] = 0;
	LandFName[0] = 0;
	SetFarsiteProgress(0.0);

	progress = 0.0;
// SetFarsiteRunStatus (e_StartUp);

  fN_WNToDo = 0;
  fN_WNDone = 0;

	m_nCellsLit = 0;
}

Farsite5::~Farsite5(void)
{
	spotList.clear();
 	if(gaat)
     	delete gaat;
    gaat=0;
	if(ignitionGrid)
	{
		for(long r = 0; r < ignitionRows; r++)
			delete[] ignitionGrid[r];
		delete[] ignitionGrid;
	}
	CleanPerimeters();
	Terminate();
        if (lcptheme) {
          delete lcptheme;
          lcptheme = 0;  }
#ifdef WIN32
	DeleteCriticalSection(&progressCS);
#endif
}

void Farsite5::SetModelParams()
{
	double ActualTimeStep = 30.0;	// minutes
     double VisibleTimeStep=30.0;	// minutes
	//bool secondaryVisibleStep = false;
	bool isEnglishints = false;
	double perimeterResolution = 90.0;     // keep this bigger than distance resolution!
	double distanceResolution = 60.0;

	SetActualTimeStep(ActualTimeStep);
	SetVisibleTimeStep(VisibleTimeStep);
	SetSecondaryVisibleTimeStep(-1.0);

	double MetricConvert = 1.0;
	if (isEnglishints)
		MetricConvert = 0.914634;
	SetPerimRes(double (perimeterResolution) * MetricConvert);  			// actual resolution always in meters ??
	SetDistRes(double (distanceResolution) * MetricConvert);
}


long Farsite5::IgnitionResetAtRestart(long YesNo)
{
	if (YesNo >= 0)
		IgnitionReset = YesNo;

	return IgnitionReset;
}

const char *Farsite5::GetLandFileName()
{
	return LandFName;
}

void Farsite5::SetLandFileName(char* FileName)
{
	//LandFName = bfs::path(FileName, bfs::native) ;
	strcpy(LandFName, FileName);
}

void Farsite5::CloseLandFile()
{
	if (landscape)
	{
		delete[] landscape;//GlobalFree(landscape);//free(landscape);
		landscape = 0;
	}
	CantAllocLCP = false;
	if (landfile)
		fclose(landfile);
	//if (lcptheme)
	//{
	//	delete lcptheme;
	//	lcptheme = 0;
	//}
	LandFName[0] = 0;// = bfs::path() ;
	landfile = 0;
	CanopyChx.Height = CanopyChx.DefaultHeight;
	CanopyChx.CrownBase = CanopyChx.DefaultBase;
	CanopyChx.BulkDensity = CanopyChx.DefaultDensity;
}

celldata Farsite5::CellData(double east, double north, celldata& cell, crowndata& cfuel,
	grounddata& gfuel, long* posit)
{
	long Position;

	if (landscape == 0)
	{
		if (CantAllocLCP == false)
		{
			long i;
			double AllocSize;

			fseek(landfile, headsize, SEEK_SET);
			AllocSize = Header.numnorth * Header.numeast * NumVals;
			if (AllocSize > 2147483647)
			{
				CantAllocLCP = true;
			}
			else
			{
				try
				{
					landscape = new short[Header.numnorth * Header.numeast * NumVals];
				}
				catch (...)
				{
					CantAllocLCP = true;
					landscape = 0;
				}
				if (landscape != NULL)
				{
					//ZeroMemory(landscape,Header.numnorth * Header.numeast * NumVals * sizeof(short));
                                        memset(landscape,0x0,Header.numnorth * Header.numeast * NumVals * sizeof(short));
					for (i = 0; i < Header.numnorth; i++)
						fread(&landscape[i * NumVals * Header.numeast],
							sizeof(short), NumVals * Header.numeast, landfile);
					fseek(landfile, headsize, SEEK_SET);
					//     		 	OldFilePosition=0;     // thread local
					CantAllocLCP = false;
				}
				else
					CantAllocLCP = true;
			}
		}
	}

	Position = GetCellPosition(east, north);
	if (!CantAllocLCP)
	{
		GetCellDataFromMemory(Position, cell, cfuel, gfuel);
          if(posit!=NULL)
          	*posit=Position;

		return cell;
	}

	if (Header.CrownFuels == 20)
	{
		if (Header.GroundFuels == 20)
		{
			fseek(landfile, (Position - OldFilePosition) * sizeof(celldata),
				SEEK_CUR);
			fread(&cell, sizeof(celldata), 1, landfile);
		}
		else
		{
			fseek(landfile,
				(Position - OldFilePosition) * (sizeof(celldata) +
				sizeof(grounddata)),
				SEEK_CUR);
			fread(&cell, sizeof(celldata), 1, landfile);
			fread(&gfuel, sizeof(grounddata), 1, landfile);
		}
	}
	else
	{
		if (Header.GroundFuels == 20)		// none
		{
			fseek(landfile,
				(Position - OldFilePosition) * (sizeof(celldata) +
				sizeof(crowndata)),
				SEEK_CUR);
			fread(&cell, sizeof(celldata), 1, landfile);
			fread(&cfuel, sizeof(crowndata), 1, landfile);
		}
		else
		{
			fseek(landfile,
				(Position - OldFilePosition) * (sizeof(celldata) +
				sizeof(crowndata) +
				sizeof(grounddata)),
				SEEK_CUR);
			fread(&cell, sizeof(celldata), 1, landfile);
			fread(&cfuel, sizeof(crowndata), 1, landfile);
			fread(&gfuel, sizeof(grounddata), 1, landfile);
		}
		if (cfuel.h > 0)
		{
			CanopyChx.Height = (double) cfuel.h / 10.0;
			if (Header.HUnits == 2)
				CanopyChx.Height /= 3.280839;
		}
		else
			CanopyChx.Height = CanopyChx.DefaultHeight;
		if (cfuel.b > 0)
		{
			CanopyChx.CrownBase = (double) cfuel.b / 10.0;
			if (Header.BUnits == 2)
				CanopyChx.CrownBase /= 3.280839;
		}
		else
			CanopyChx.CrownBase = CanopyChx.DefaultBase;
		if (cfuel.p > 0)
		{
			if (Header.PUnits == 1)
				CanopyChx.BulkDensity = ((double) cfuel.p) / 100.0;
			else if (Header.PUnits == 2)
				CanopyChx.BulkDensity = ((double) cfuel.p / 1000.0) * 16.01845;
		}
		else
			CanopyChx.BulkDensity = CanopyChx.DefaultDensity;
	}

	OldFilePosition = Position + 1;
	if (posit != NULL)
		*posit = Position;

	return cell;
}

bool Farsite5::OpenLandFile()
{
	//	SetCanopyChx(15.0, 4.0, 0.2, 30.0, 100, 2, 1);
	if (landfile)
		fclose(landfile);
	if (landscape)
	{
		delete[] landscape;//GlobalFree(landscape);//free(landscape);
		landscape = 0;
		CantAllocLCP = false;
	}
	if ((landfile = fopen(LandFName, "rb")) == NULL)
	{
		landfile = 0;

		return false;
	}

	return true;
}


bool Farsite5::LoadLandscapeFile(char *FileName)
{
     CloseLandFile();

     SetLandFileName(FileName);
     if(!OpenLandFile())
     {	//::MessageBox(HWindow, FileName,
     	//	"Error Loading LCP", MB_OK);
          //lcp->SetCheck(BF_UNCHECKED);
     	//tlcp->SetText("");

     	return false;
     }

     //LandFileOpen=true;
     ReadHeader();
     SetCustFuelModelID(HaveCustomFuelModels());
     SetConvFuelModelID(HaveFuelConversions());
     return true;
}

void Farsite5::ReadHeader()
{
	fseek(landfile, 0, SEEK_SET);

	//     unsigned int sizeh=sizeof(Header);
	//	fread(&Header, sizeh, 1, landfile);

	fread(&Header.CrownFuels, sizeof(int32), 1, landfile);
	fread(&Header.GroundFuels, sizeof(int32), 1, landfile);
	fread(&Header.latitude, sizeof(int32), 1, landfile);
	fread(&Header.loeast, sizeof(double), 1, landfile);
	fread(&Header.hieast, sizeof(double), 1, landfile);
	fread(&Header.lonorth, sizeof(double), 1, landfile);
	fread(&Header.hinorth, sizeof(double), 1, landfile);
	fread(&Header.loelev, sizeof(int32), 1, landfile);
	fread(&Header.hielev, sizeof(int32), 1, landfile);
	fread(&Header.numelev, sizeof(int32), 1, landfile);
	fread(Header.elevs, sizeof(int32), 100, landfile);
	fread(&Header.loslope, sizeof(int32), 1, landfile);
	fread(&Header.hislope, sizeof(int32), 1, landfile);
	fread(&Header.numslope, sizeof(int32), 1, landfile);
	fread(Header.slopes, sizeof(int32), 100, landfile);
	fread(&Header.loaspect, sizeof(int32), 1, landfile);
	fread(&Header.hiaspect, sizeof(int32), 1, landfile);
	fread(&Header.numaspect, sizeof(int32), 1, landfile);
	fread(Header.aspects, sizeof(int32), 100, landfile);
	fread(&Header.lofuel, sizeof(int32), 1, landfile);
	fread(&Header.hifuel, sizeof(int32), 1, landfile);
	fread(&Header.numfuel, sizeof(int32), 1, landfile);
	fread(Header.fuels, sizeof(int32), 100, landfile);
	fread(&Header.locover, sizeof(int32), 1, landfile);
	fread(&Header.hicover, sizeof(int32), 1, landfile);
	fread(&Header.numcover, sizeof(int32), 1, landfile);
	fread(Header.covers, sizeof(int32), 100, landfile);
	fread(&Header.loheight, sizeof(int32), 1, landfile);
	fread(&Header.hiheight, sizeof(int32), 1, landfile);
	fread(&Header.numheight, sizeof(int32), 1, landfile);
	fread(Header.heights, sizeof(int32), 100, landfile);
	fread(&Header.lobase, sizeof(int32), 1, landfile);
	fread(&Header.hibase, sizeof(int32), 1, landfile);
	fread(&Header.numbase, sizeof(int32), 1, landfile);
	fread(Header.bases, sizeof(int32), 100, landfile);
	fread(&Header.lodensity, sizeof(int32), 1, landfile);
	fread(&Header.hidensity, sizeof(int32), 1, landfile);
	fread(&Header.numdensity, sizeof(int32), 1, landfile);
	fread(Header.densities, sizeof(int32), 100, landfile);
	fread(&Header.loduff, sizeof(int32), 1, landfile);
	fread(&Header.hiduff, sizeof(int32), 1, landfile);
	fread(&Header.numduff, sizeof(int32), 1, landfile);
	fread(Header.duffs, sizeof(int32), 100, landfile);
	fread(&Header.lowoody, sizeof(int32), 1, landfile);
	fread(&Header.hiwoody, sizeof(int32), 1, landfile);
	fread(&Header.numwoody, sizeof(int32), 1, landfile);
	fread(Header.woodies, sizeof(int32), 100, landfile);
	fread(&Header.numeast, sizeof(int32), 1, landfile);
	fread(&Header.numnorth, sizeof(int32), 1, landfile);
	fread(&Header.EastUtm, sizeof(double), 1, landfile);
	fread(&Header.WestUtm, sizeof(double), 1, landfile);
	fread(&Header.NorthUtm, sizeof(double), 1, landfile);
	fread(&Header.SouthUtm, sizeof(double), 1, landfile);
	fread(&Header.GridUnits, sizeof(int32), 1, landfile);
	fread(&Header.XResol, sizeof(double), 1, landfile);
	fread(&Header.YResol, sizeof(double), 1, landfile);
	fread(&Header.EUnits, sizeof(short), 1, landfile);
	fread(&Header.SUnits, sizeof(short), 1, landfile);
	fread(&Header.AUnits, sizeof(short), 1, landfile);
	fread(&Header.FOptions, sizeof(short), 1, landfile);
	fread(&Header.CUnits, sizeof(short), 1, landfile);
	fread(&Header.HUnits, sizeof(short), 1, landfile);
	fread(&Header.BUnits, sizeof(short), 1, landfile);
	fread(&Header.PUnits, sizeof(short), 1, landfile);
	fread(&Header.DUnits, sizeof(short), 1, landfile);
	fread(&Header.WOptions, sizeof(short), 1, landfile);
	fread(Header.ElevFile, sizeof(char), 256, landfile);
	fread(Header.SlopeFile, sizeof(char), 256, landfile);
	fread(Header.AspectFile, sizeof(char), 256, landfile);
	fread(Header.FuelFile, sizeof(char), 256, landfile);
	fread(Header.CoverFile, sizeof(char), 256, landfile);
	fread(Header.HeightFile, sizeof(char), 256, landfile);
	fread(Header.BaseFile, sizeof(char), 256, landfile);
	fread(Header.DensityFile, sizeof(char), 256, landfile);
	fread(Header.DuffFile, sizeof(char), 256, landfile);
	fread(Header.WoodyFile, sizeof(char), 256, landfile);
	fread(Header.Description, sizeof(char), 512, landfile);
	// do this in case a version 1.0 file has gotten through
	Header.loeast = ConvertUtmToEastingOffset(Header.WestUtm);
	Header.hieast = ConvertUtmToEastingOffset(Header.EastUtm);
	Header.lonorth = ConvertUtmToNorthingOffset(Header.SouthUtm);
	Header.hinorth = ConvertUtmToNorthingOffset(Header.NorthUtm);

	if (Header.FOptions == 1 || Header.FOptions == 3)
		NEED_CUST_MODELS = true;
	else
		NEED_CUST_MODELS = false;
	if (Header.FOptions == 2 || Header.FOptions == 3)
		NEED_CONV_MODELS = true;
	else
		NEED_CONV_MODELS = false;
	//HAVE_CUST_MODELS=false;
	//HAVE_CONV_MODELS=false;
	// set raster resolution
	RasterCellResolutionX = (Header.EastUtm - Header.WestUtm) /
		(double) Header.numeast;
	RasterCellResolutionY = (Header.NorthUtm - Header.SouthUtm) /
		(double) Header.numnorth;
	/*ViewPortNorth = RasterCellResolutionY * (double) Header.numnorth +
		Header.lonorth;
	ViewPortSouth = Header.lonorth;
	ViewPortEast = RasterCellResolutionX * (double) Header.numeast +
		Header.loeast;
	ViewPortWest = Header.loeast;
	//	NumViewNorth=(ViewPortNorth-ViewPortSouth)/Header.YResol;
	//	NumViewEast=(ViewPortEast-ViewPortWest)/Header.XResol;
	double rows, cols;
	rows = (ViewPortNorth - ViewPortSouth) / Header.YResol;
	NumViewNorth = (long)rows;
	if (modf(rows, &rows) > 0.5)
		NumViewNorth++;
	cols = (ViewPortEast - ViewPortWest) / Header.XResol;
	NumViewEast = (long)cols;
	if (modf(cols, &cols) > 0.5)
		NumViewEast++;
*/
	if (HaveCrownFuels())
	{
		if (HaveGroundFuels())
			NumVals = 10;
		else
			NumVals = 8;
	}
	else
	{
		if (HaveGroundFuels())
			NumVals = 7;
		else
			NumVals = 5;
	}
	CantAllocLCP = false;

	if (lcptheme)
	{
		delete lcptheme;
		lcptheme = 0;
	}
	lcptheme = new LandscapeTheme(false, this);

	if (landscape == 0)
	{
		if (CantAllocLCP == false)
		{
			long i;
			double NumAlloc;

			fseek(landfile, headsize, SEEK_SET);
			//     		if((landscape=(short *) calloc(Header.numnorth*Header.numeast, NumVals*sizeof(short)))!=NULL)
			NumAlloc = (double)
				(Header.numnorth * Header.numeast * NumVals * sizeof(short));
			if (NumAlloc > 2147483647)
			{
				CantAllocLCP = true;

				return;
			}

			try
			{
				landscape = new short[Header.numnorth * Header.numeast * NumVals];
			}
			catch (...)
			{
				landscape = 0;
			}
			if (landscape != NULL)
			{
				//ZeroMemory(landscape,Header.numnorth * Header.numeast * NumVals * sizeof(short));
                                memset(landscape,0x0,Header.numnorth * Header.numeast * NumVals * sizeof(short));
				for (i = 0; i < Header.numnorth; i++)
					fread(&landscape[i * NumVals * Header.numeast],
						sizeof(short), NumVals * Header.numeast, landfile);
				fseek(landfile, headsize, SEEK_SET);
				//     		 	OldFilePosition=0;     // thread local
				CantAllocLCP = false;
			}
			else
				CantAllocLCP = true;
		}
	}
	//	long p;
	//   CellData(Header.loeast, Header.hinorth, &p);
}

long Farsite5::GetCellPosition(double east, double north)
{
	double xpt = (east - Header.loeast) / GetCellResolutionX();
	double ypt = (north - Header.lonorth) / GetCellResolutionY();
	long easti = ((long) xpt);
	long northi = ((long) ypt);
	northi = Header.numnorth - northi - 1;
	if (northi < 0)
		northi = 0;
	long posit = (northi* Header.numeast + easti);

	return posit;
}

void Farsite5::SetCellFuel(long posit, short tfuel)
{
	landscape[posit * NumVals + 3] = tfuel;
}

void Farsite5::GetCellDataFromMemory(long posit, celldata& cell, crowndata& cfuel,
	grounddata& gfuel)
{
	short ldata[10], zero = 0;

	memcpy(ldata, &landscape[posit * NumVals], NumVals * sizeof(short));
	switch (NumVals)
	{
	case 5:
		// only 5 basic themes
		memcpy(&cell, ldata, NumVals * sizeof(short));
		memcpy(&gfuel, &zero, 2 * sizeof(short));
		break;
	case 7:
		// 5 basic and duff and woody
		memcpy(&cell, ldata, 5 * sizeof(short));
		memset(&gfuel, zero, 2 * sizeof(short));
		break;
	case 8:
		// 5 basic and crown fuels
		memcpy(&cell, ldata, 5 * sizeof(short));
		memcpy(&cfuel, &ldata[5], 3 * sizeof(short));
		memset(&gfuel, zero, 2 * sizeof(short));
		break;
	case 10:
		// 5 basic, crown fuels, and duff and woody
		memcpy(&cell, ldata, 5 * sizeof(short));
		memcpy(&cfuel, &ldata[5], 3 * sizeof(short));
		memcpy(&gfuel, &ldata[8], 2 * sizeof(short));
		break;
	}
}

double Farsite5::ConvertEastingOffsetToUtm(double input)
{
	return input;
	double MetersToKm = 1.0;
	double ipart;

	if (Header.GridUnits == 2)
		MetersToKm = 0.001;

	modf(Header.WestUtm / 1000.0, &ipart);

	return (input + ipart * 1000.0) * MetersToKm;
}

double Farsite5::ConvertNorthingOffsetToUtm(double input)
{
	return input;
	double MetersToKm = 1.0;
	double ipart;

	if (Header.GridUnits == 2)
		MetersToKm = 0.001;

	modf(Header.SouthUtm / 1000.0, &ipart);

	return (input + ipart * 1000.0) * MetersToKm;
}

double Farsite5::ConvertUtmToEastingOffset(double input)
{
	return input;
	double KmToMeters = 1.0;
	double ipart;

	if (Header.GridUnits == 2)
		KmToMeters = 1000.0;

	modf(Header.WestUtm / 1000.0, &ipart);

	return input * KmToMeters - ipart * 1000.0;
}

double Farsite5::ConvertUtmToNorthingOffset(double input)
{
	return input;
	double KmToMeters = 1.0;
	double ipart;

	if (Header.GridUnits == 2)
		KmToMeters = 1000.0;

	modf(Header.SouthUtm / 1000.0, &ipart);

	return input * KmToMeters - ipart * 1000.0;
}


char* Farsite5::GetHeaderDescription()
{
	return Header.Description;
}

long Farsite5::HaveCrownFuels()
{
	return Header.CrownFuels - 20;		// subtract 10 to ID file as version 2.x
}

long Farsite5::HaveGroundFuels()
{
	return Header.GroundFuels - 20;
}

double Farsite5::GetCellResolutionX()
{
	if (Header.GridUnits == 2)
		return Header.XResol * 1000.0;   // to kilometers

	return Header.XResol;
}


double Farsite5::GetCellResolutionY()
{
	if (Header.GridUnits == 2)
		return Header.YResol * 1000.0;

	return Header.YResol;
}

bool Farsite5::CreateSpotSemaphore()
{
	CloseSpotSemaphore();
     /*
	char Name[128] = "";
	char TimeID[128] = "";
	SYSTEMTIME st;
	GetSystemTime(&st);
	sprintf(TimeID, "%ld%ld%ld%ld%ld%ld%ld", st.wYear, st.wMonth, st.wDay,
		st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

	sprintf(Name, "%s%s", "SPOTSEMAPHORE", TimeID);
	hNewSpotSemaphore = CreateSemaphore(NULL, 1, 1, Name);

	sprintf(Name, "%s%s", "NEWP1SEMAPHORE", TimeID);
	hNewPerimSemaphore = CreateSemaphore(NULL, 1, 1, Name);

	if (hNewSpotSemaphore == NULL)
		return false;
	if (hNewPerimSemaphore == NULL)
		return false;
     */
	return true;
}

void Farsite5::CloseSpotSemaphore()
{
	/*
	if (hNewSpotSemaphore != NULL)
		CloseHandle(hNewSpotSemaphore);
	hNewSpotSemaphore = 0;
	if (hNewPerimSemaphore != NULL)
		CloseHandle(hNewPerimSemaphore);
	hNewPerimSemaphore = 0;
     */
}

bool Farsite5::AccelerationON()
{
	return AccelerationState;
}

void Farsite5::SetAccelerationON(bool State)
{
	AccelerationState = State;
}


void Farsite5::InitializeRosRed()
{
	for (long i = 0; i < 257; i++)
		redros[i] = 1.0;
}


double Farsite5::GetRosRed(int fuel)
{
	if (redros[fuel] > 0.0)
		return redros[fuel];
	else
		return 1.0;
}

void Farsite5::SetRosRed(int fuel, double rosred)
{
	redros[fuel] = fabs(rosred);
}

long Farsite5::GetInout(long FireNumber)
{
	return inout[FireNumber];
}

void Farsite5::SetInout(long FireNumber, int Inout)
{
	inout[FireNumber] = Inout;
}

/*long Farsite5::GetNumPoints(long FireNumber)
{
	return numpts[FireNumber];
}*/

void Farsite5::SetNumPoints(long FireNumber, long NumPoints)
{
	numpts[FireNumber] = NumPoints;
}

double Farsite5::PercentIgnition(double percent)
{
	/*if (percent >= 0.01 && percent <= 100.0)
		PercentageOfEmberIgnitions = percent;
	else if (percent >= 0.0 && percent < 0.01)
		PercentageOfEmberIgnitions = 0.01;
	else if (percent > 100.0)
		PercentageOfEmberIgnitions = 100.0;*/
	if (percent >= 0.0 && percent <= 100.0)
		PercentageOfEmberIgnitions = percent;
	//else if (percent >= 0.0 && percent < 0.01)
		//PercentageOfEmberIgnitions = 0.01;
	else if (percent > 100.0)
		PercentageOfEmberIgnitions = 100.0;

	return PercentageOfEmberIgnitions;
}


double Farsite5::IgnitionDelay(double delay)
{
	if (delay >= 0.0)
		SpotIgnitionDelay = delay;

	return SpotIgnitionDelay;
}


bool Farsite5::EnableCrowning(long Crowning)
{
	if (Crowning >= 0)
		CrowningOK = (bool) Crowning;

	return CrowningOK;
}

bool Farsite5::EnableSpotting(long Spotting)
{
	if (Spotting >= 0)
		SpottingOK = (bool) Spotting;

	return SpottingOK;
}

bool Farsite5::EnableSpotFireGrowth(long Growth)
{
	if (Growth >= 0)
		SpotGrowthOK = (bool) Growth;

	return SpotGrowthOK;
}

bool Farsite5::ConstantBackingSpreadRate(long Back)
{
	if (Back >= 0)
		ConstBack = (bool) Back;

	return ConstBack;
}



long Farsite5::GetNumFires()
{
	return numfires;
}

void Farsite5::SetNumFires(long input)
{
	numfires = input;
}

void Farsite5::IncNumFires(long MoreFires)
{
	numfires += MoreFires;
	//ATLTRACE2("IncNumFires() numFires = %ld, MoreFires = %ld, ", this->numfires, MoreFires);
	//ATLTRACE("NumAlloc = %ld\n", this->NumPerimAlloc);
}

long Farsite5::GetNewFires()
{
	return newfires;
}

void Farsite5::SetNewFires(long input)
{
//	WaitForSingleObject(hNewSpotSemaphore, INFINITE);
	newfires = input;
	numspots = input;
//	ReleaseSemaphore(hNewSpotSemaphore, 1, &prevct);
}

void Farsite5::IncNewFires(long increment)
{
//	WaitForSingleObject(hNewSpotSemaphore, INFINITE);
	newfires += increment;
	numspots = newfires;
//	ReleaseSemaphore(hNewSpotSemaphore, 1, &prevct);
}

void Farsite5::GetNumSpots(long* num, bool inc)
{
//	WaitForSingleObject(hNewSpotSemaphore, INFINITE);
	*num = numspots;
	if (inc)
	{
		numspots++;
		newfires = numspots;
	}
//	ReleaseSemaphore(hNewSpotSemaphore, 1, &prevct);
}

void Farsite5::SetNumSpots(long input)
{
//	WaitForSingleObject(hNewSpotSemaphore, INFINITE);
	numspots = input;
//	ReleaseSemaphore(hNewSpotSemaphore, 1, &prevct);
}

long Farsite5::GetSkipFires()
{
	return skipfire;
}

void Farsite5::SetSkipFires(long newvalue)
{
//	WaitForSingleObject(hNewPerimSemaphore, INFINITE);
	skipfire = newvalue;
//	ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);
}

void Farsite5::IncSkipFires(long increment)
{
//	WaitForSingleObject(hNewPerimSemaphore, INFINITE);
	skipfire += increment;
//	ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);
}

double Farsite5::GetPerimRes()
{
	return PerimeterResolution;
}

void Farsite5::SetPerimRes(double input)
{
	PerimeterResolution = input;
}


double Farsite5::GetDynamicDistRes()
{
	return DynamicDistanceResolution;
}


void Farsite5::SetDynamicDistRes(double input)
{
	DynamicDistanceResolution = input;
}


double Farsite5::GetDistRes()
{
	return DistanceResolution;
}

void Farsite5::SetDistRes(double input)
{
	DistanceResolution = DynamicDistanceResolution = input; 	 // set both default and dynamic min dist
}


double Farsite5::GetTemporaryTimeStep()
{
	return TemporaryTimestep;
}


void Farsite5::SetTemporaryTimeStep(double value)
{
	TemporaryTimestep = value;
}


double Farsite5::GetActualTimeStep()
{
	return actual;
}

void Farsite5::SetActualTimeStep(double input)
{
	actual = input;
}

double Farsite5::GetVisibleTimeStep()
{
	return visible;
}

void Farsite5::SetVisibleTimeStep(double input)
{
	long nuvis = (long)(input / actual);
	if (nuvis < 1)
		nuvis = 1;
	visible = nuvis * actual;
}


void Farsite5::SetSecondaryVisibleTimeStep(double input)
{
	secondaryvisible = input;
}


double Farsite5::GetSecondaryVisibleTimeStep()
{
	return secondaryvisible;
}


double Farsite5::EventMinimumTimeStep(double time)
{
	// returns EventMinTimeStep if "time" is negative
	if (time >= 0)
		EventMinTimeStep = time;

	return EventMinTimeStep;
}


bool Farsite5::CheckExpansions(long YesNo)
{
	if (YesNo > -1)
		checkexpansions = YesNo;

	return checkexpansions;
}


bool Farsite5::CheckPostFrontal(long YesNo)
{
	if (YesNo > -1)
		checkpostfrontal = YesNo;

	return checkpostfrontal;
}


long Farsite5::DistanceCheckMethod(long Method)
{
	if (Method >= 0)
		DistanceMethod = Method;

	return DistanceMethod;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	Fire perimeter2, swap space for fire growth calculations
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void Farsite5::GetPerimeter2(long coord, double* xpt, double* ypt, double* ros,
	double* fli, double* rct)
{
	if (coord < p2numalloc)
	{
		coord *= NUMDATA;
		*xpt = perimeter2[coord];
		*ypt = perimeter2[++coord];
		*ros = perimeter2[++coord];
		*fli = perimeter2[++coord];
		*rct = perimeter2[++coord];
	}
}

double Farsite5::GetPerimeter2Value(long coord, long value)
{
	if (coord < 0 || value < 0)
		return (double) p2numalloc;
	else if (perimeter2 && coord < p2numalloc)
		return perimeter2[coord * NUMDATA + value];

	return 0.0; //(double) p2numalloc
}

void Farsite5::SetPerimeter2(long coord, double xpt, double ypt, double ros, double fli,
	double rct)
{
	if (coord < p2numalloc)
	{
		coord *= NUMDATA;
		perimeter2[coord] = xpt;
		perimeter2[++coord] = ypt;
		perimeter2[++coord] = ros;
		perimeter2[++coord] = fli;
		perimeter2[++coord] = rct;
	}
}


double* Farsite5::AllocPerimeter2(long NumPoints)
{
	if (NumPoints)
	{
		if (NumPoints >= p2numalloc)
		{
			FreePerimeter2();
			nmemb = NumPoints * NUMDATA;
			perimeter2 = new double[nmemb];
			if (perimeter2 != NULL)
				p2numalloc = NumPoints;
		}
	}
	else
		return NULL;

	return perimeter2;
}


void Farsite5::FreePerimeter2()
{
	if (perimeter2)
		delete[] perimeter2;//GlobalFree(perimeter2);//free(perimeter2);
	perimeter2 = 0;
	p2numalloc = 0;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	Must Call at begining of the program
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


void Farsite5::FreeAllFirePerims()
{
	if (perimeter1)
		delete[] perimeter1;
	if (numpts)
		delete[] numpts;
	if (inout)
		delete[] inout;
	perimeter1 = 0;
	numpts = 0;
	inout = 0;
	NumPerimAlloc = 0;
}


long Farsite5::GetNumPerimAlloc()
{
	return NumPerimAlloc;
}


bool Farsite5::AllocFirePerims(long num)
{
	FreeAllFirePerims();
	perimeter1 = new double * [num];
	if (perimeter1 == NULL)
		return false;
	numpts = new long[num];
	if (numpts == NULL)
		return false;
	inout = new long[num];
	if (inout == NULL)
		return false;
	NumPerimAlloc = num;
	memset(perimeter1, 0x0, num * sizeof(double *));
	memset(numpts, 0x0, num * sizeof(long));
	memset(inout, 0x0, num * sizeof(long));

	return true;
}


bool Farsite5::ReAllocFirePerims()
{
	long i, OldNumAlloc;
	long* newinout, * newnumpts;
	double* temp1;
	double** newperim1;

	newperim1 = perimeter1;
	newinout = inout;
	newnumpts = numpts;

	perimeter1 = 0;
	inout = 0;
	numpts = 0;
	OldNumAlloc = NumPerimAlloc;

	if (!AllocFirePerims(NumPerimAlloc + SIZE_FIREPERIM_BLOCK))
		return false;

	if (newinout)
	{
		memcpy(inout, newinout, OldNumAlloc * sizeof(long));
		delete[] newinout;
	}
	if (newnumpts)
	{
		memcpy(numpts, newnumpts, OldNumAlloc * sizeof(long));
		delete[] newnumpts;
	}

	if (newperim1)
	{
		for (i = 0; i < OldNumAlloc; i++)
		{
			temp1 = perimeter1[i];
			perimeter1[i] = newperim1[i];
			if (numpts[i] > 0)
				delete[] temp1;
		}
		delete[] newperim1;
	}


	return true;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	Fire Perimeter1, main perimeter storage and retrieval functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

double* Farsite5::AllocPerimeter1(long NumFire, long NumPoints)
{
	//WaitForSingleObject(hNewPerimSemaphore, INFINITE);
	if (NumPoints)
	{
		if (NumFire >= NumPerimAlloc)
		{
			if (ReAllocFirePerims() == false)
			{
				//ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);

				return NULL;
			}
		}
		nmemb = (NumPoints) * NUMDATA;			// add 1 to make room for bounding rectangle
                if (perimeter1[NumFire] && perimeter1[NumFire] > 0)
			FreePerimeter1(NumFire);
		perimeter1[NumFire] = new double[nmemb];

		if (perimeter1[NumFire] == NULL)
		{
			NumFire = -1;		// debugging
			perimeter1[NumFire] = 0;
			//ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);

			return NULL;
		}
	}
	//ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);

	return perimeter1[NumFire];
}

void Farsite5::FreePerimeter1(long NumFire)
{
	if (perimeter1[NumFire])
	{
		delete[] perimeter1[NumFire];
		perimeter1[NumFire] = 0;
	}
}

/*double Farsite5::GetPerimeter1Value(long NumFire, long NumPoint, int coord)
{
	if (perimeter1[NumFire])
		return perimeter1[NumFire][NumPoint * NUMDATA + coord];

	return 0.0;
}*/


double* Farsite5::GetPerimeter1Address(long NumFire, long NumPoint)
{
	return &perimeter1[NumFire][NumPoint * NUMDATA];
}


void Farsite5::SetPerimeter1(long NumFire, long NumPoint, double xpt, double ypt)
{
	//	WaitForSingleObject(hNewPerimSemaphore, INFINITE);
	NumPoint *= NUMDATA;
	perimeter1[NumFire][NumPoint] = xpt;
	perimeter1[NumFire][NumPoint + 1] = ypt;
	//	ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);
}

void Farsite5::SetFireChx(long NumFire, long NumPoint, double ros, double fli)
{
	//	WaitForSingleObject(hNewPerimSemaphore, INFINITE);
	NumPoint *= NUMDATA;
	perimeter1[NumFire][NumPoint + 2] = ros;
	perimeter1[NumFire][NumPoint + 3] = fli;
	//	ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);
}

void Farsite5::SetReact(long NumFire, long NumPoint, double react)
{
	//	WaitForSingleObject(hNewPerimSemaphore, INFINITE);
	perimeter1[NumFire][NumPoint * NUMDATA + 4] = react;
	//	ReleaseSemaphore(hNewPerimSemaphore, 1, &prevct);
}

long Farsite5::SwapFirePerims(long NumFire1, long NumFire2)
{
	double* TempFire;
	long TempInout, TempNum;

	if (NumFire1 >= 0 && NumFire2 >= 0) 		// two fires in perim1
	{
		TempFire = perimeter1[NumFire2];
		perimeter1[NumFire2] = perimeter1[NumFire1];
		perimeter1[NumFire1] = TempFire;
		TempInout = inout[NumFire2];
		inout[NumFire2] = inout[NumFire1];
		inout[NumFire1] = TempInout;
		TempNum = numpts[NumFire2];
		numpts[NumFire2] = numpts[NumFire1];
		numpts[NumFire1] = TempNum;

		return 1;
	}
	else if (NumFire1 < 0 && NumFire2 >= 0)
	{
		AllocPerimeter2(numpts[NumFire2]);
		if (std::memmove(perimeter2, perimeter1[NumFire2],
				numpts[NumFire2] * NUMDATA * sizeof(double)))
			return 1;
		else
			return 0;
	}
	else if (NumFire1 >= 0 && NumFire2 < 0)
	{
		if (perimeter1[NumFire1])
		{
			if (std::memmove(perimeter1[NumFire1], perimeter2,
					(NumFire2 * -1) * NUMDATA * sizeof(double)))
				return 1;
		}
		return 0;
	}

	return 0;
}


void Farsite5::AllocElev(long CurrentFire)
{
	nmemb = numpts[CurrentFire];
	if (nmemb >= (unsigned long) numelev)
	{
		FreeElev();
		GroundElev = new long[nmemb];
		numelev = nmemb;
	}
}


void Farsite5::SetElev(long Num, long elev)
{
	if (Num >= numelev)
	{
		long* tempelev;

		tempelev = new long[numelev];
		memcpy(tempelev, GroundElev, numelev * sizeof(long));
		delete[] GroundElev;
		GroundElev = new long[numelev * 2];
		memcpy(GroundElev, tempelev, numelev * sizeof(long));
		numelev *= 2;
		delete[] tempelev;
	}
	GroundElev[Num] = elev;
}


long Farsite5::GetElev(long Num)
{
	if (!GroundElev)
		return (long) NULL;

	return GroundElev[Num];
}

long* Farsite5::GetElevAddress(long Num)
{
	if (!GroundElev)
		return (long) NULL;

	return &GroundElev[Num];
}

void Farsite5::FreeElev()
{
	if (GroundElev)
		delete[] GroundElev;//GlobalFree(GroundElev);//free(GroundElev);
	GroundElev = 0;
	numelev = 0;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	Stop Location Functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


long Farsite5::SetStopLocation(double xcoord, double ycoord)
{
	if (NumStopLocations < MAXNUM_STOPLOCATIONS)
	{
		StopLocation[NumStopLocations * 2] = xcoord;
		StopLocation[NumStopLocations * 2 + 1] = ycoord;
		StopEnabled[NumStopLocations] = true;
	}
	else
		return 0;

	return ++NumStopLocations;
}

bool Farsite5::GetStopLocation(long StopNum, double* xcoord, double* ycoord)
{
	if (StopNum < MAXNUM_STOPLOCATIONS)
	{
		*xcoord = StopLocation[StopNum * 2];
		*ycoord = StopLocation[StopNum * 2 + 1];
	}
	if (StopEnabled[StopNum])
		return true;

	return false;
}


void Farsite5::ResetStopLocation(long StopNum)
{
	if (StopNum < NumStopLocations)
	{
		memcpy(&StopLocation[StopNum * 2],
			&StopLocation[(StopNum + 1) * 2],
			(NumStopLocations - StopNum - 1) * sizeof(double) * 2);
		memcpy(&StopEnabled[StopNum], &StopEnabled[StopNum + 1],
			(NumStopLocations - StopNum - 1) * sizeof(bool));
		NumStopLocations--;
	}
}


bool Farsite5::EnableStopLocation(long StopNum, long Action)
{
	if (Action >= 0 && StopNum < NumStopLocations)
		StopEnabled[StopNum] = (bool) Action;

	return (bool) StopEnabled[StopNum];
}


void Farsite5::ResetAllStopLocations()
{
	NumStopLocations = 0;
	memset(StopEnabled, 0x0, MAXNUM_STOPLOCATIONS * sizeof(bool));
}

long Farsite5::GetNumStopLocations()
{
	return NumStopLocations;
}


void Farsite5::SetCustFuelModelID(bool True_False)
{
	HAVE_CUST_MODELS = True_False;
}


void Farsite5::SetConvFuelModelID(bool True_False)
{
	HAVE_CONV_MODELS = True_False;
}


bool Farsite5::NeedCustFuelModels()
{
	return NEED_CUST_MODELS;
}


bool Farsite5::NeedConvFuelModels()
{
	return NEED_CONV_MODELS;
}


bool Farsite5::HaveCustomFuelModels()
{
	return HAVE_CUST_MODELS;
}


bool Farsite5::HaveFuelConversions()
{
	return HAVE_CONV_MODELS;
}


bool Farsite5::LoadFuelMoistureFile(char *FileName)
{

	long ModNum, F1, F10, F100, FLW, FLH;
     FILE *CurrentFile;

     if((CurrentFile=fopen(FileName, "r"))==NULL)
     	return false;

	while(!feof(CurrentFile))
     {    fscanf(CurrentFile,"%ld", &ModNum);
          if(feof(CurrentFile))
               break;
          fscanf(CurrentFile, "%ld %ld %ld %ld %ld", &F1, &F10, &F100, &FLH, &FLW);
		if(ModNum>256)
		{	//::MessageBox(HWindow, "Correct Fuel Moisture File Before Proceeding",
			//		"Error: Model Number>50", MB_OK);
               //fms->SetCheck(BF_UNCHECKED);
               //tfms->SetText("");

			return false;
		}
		if(F1<=0 || F10<=0 || F100<=0 || FLW<=0 || FLH<=0)
		{	//::MessageBox(HWindow, "Correct Before Proceeding",
			//		"Error: Fuel Moisture <=0", MB_OK);
               //fms->SetCheck(BF_UNCHECKED);
               //tfms->SetText("");

			return false;
		}
          SetInitialFuelMoistures(ModNum, F1, F10, F100, FLH, FLW);
	}
     fclose(CurrentFile);

     return true;
}


void Farsite5::LoadFuelMoist ()
{
int i;

/* Load Fuel Moist File if we have a file name                           */
   if ( strcmp (icf.cr_FMF,"" )) {
     LoadFuelMoistureFile (icf.cr_FMF);
     return;  }

/* Look for the Default Model and load it to all Models .................... */
   for ( i = 0; i < icf.iN_FMD; i++ ) {
     if ( icf.a_FMD[i].i_Model != e_DefFulMod )
       continue;
     SetAllMoistures(icf.a_FMD[i].i_TL1,icf.a_FMD[i].i_TL10,
           icf.a_FMD[i].i_TL100, icf.a_FMD[i].i_TLLH,icf.a_FMD[i].i_TLLW);
   }

/* Load all other Models                                                     */
   for ( i = 0; i < icf.iN_FMD; i++ ) {
     if ( icf.a_FMD[i].i_Model == e_DefFulMod )  /* Skip the Default Model            */
       continue;
     SetInitialFuelMoistures(icf.a_FMD[i].i_Model,
                icf.a_FMD[i].i_TL1,icf.a_FMD[i].i_TL10,
                icf.a_FMD[i].i_TL100, icf.a_FMD[i].i_TLLH,
                icf.a_FMD[i].i_TLLW);
   }
}

void Farsite5::SetAllMoistures(int _fm1, int _fm10, int _fm100,
		int _fmHerb, int _fmWoody)
{
	long i, j, k;
	for(i = 0; i <= 256; i++)
	{
                fm[i].TL1   = max(_fm1, 2);
                fm[i].TL10  = max(_fm10, 2);
                fm[i].TL100 = max(_fm100, 2);
                fm[i].TLLH  = max(_fmHerb, 2);
                fm[i].TLLW  = max(_fmWoody, 2);
		if(fm[i].TL1>1 && fm[i].TL10>1)
			fm[i].FuelMoistureIsHere=true;
		else
			fm[i].FuelMoistureIsHere=false;

		for(k=0; k<4; k++)       // only up to 1000 hr fuels [3]
		{
			for(j=0; j<5; j++)
				EnvtChanged[k][j]=true;
		}
	}
}


int Farsite5::LoadInputsFile(char *FileName)
{
int i;

   strcpy(InputsFName, FileName);

   i = icf.InputFarsite(FileName);          /* Load info from commnad file       */
   if ( i != 1 )                        /*  some basic checking done         */
    return i;                           /* Error - return error number       */

   i = icf.ValidateFarsite();           /* Validate inputs, detailed err chk */
   if ( i != 1 )                        /* Return err num                    */
     return i;
   i = LoadCustomFuelsData();
   if ( i != 1 )                        /* Return err num                    */
     return i;

   i = CheckCustomFuelsCoverage();
   if(i != 1)
	   return i;

   SetStartMonth (icf.i_FarsiteStartMth);      /* Farsite simulation start date */
   SetStartDay   (icf.i_FarsiteStartDay);
   SetStartHour  (icf.i_FarsiteStartHour);

   SetEndMonth   (icf.i_FarsiteEndMth);        /* Farsite simulation end date */
   SetEndDay     (icf.i_FarsiteEndDay);
   SetEndHour    (icf.i_FarsiteEndHour);

   LoadFuelMoist();                        // Fuel Moist, File or Embeded data  /
   LoadFoliarMoist(icf.i_FolMoi);          // Foliar Moisture      /
   LoadWeatherStream (&this->icf);
   LoadCrownFireMethod(icf.cr_CroFirMet);  /* Crown Fire Method, Finney,Reih */

   LoadWind_Table (&this->icf);

   if(strlen(icf.cr_FarsiteIgnition) > 0)  {
     strcpy(Ignition.ifile, icf.cr_FarsiteIgnition);
     CreateIgnitionGrid(); }

   SetSimulationDuration (ConvertActualTimeToSimtime(GetEndMonth(),
                          GetEndDay(), GetEndHour(), GetEndMin(), false));

   SetActualTimeStep(icf.f_FarsiteActualTimeStep);
   this->SetDistRes(icf.f_FarsiteDistanceRes);
   this->SetPerimRes(icf.f_FarsitePerimeterRes);
   this->IgnitionDelay(icf.f_FarsiteSpotIgnitionDelay);
   SetRastMake(true);
   this->SetAccelerationON((icf.i_FarsiteAccelerationOn == 0) ? false : true);
   this->EnableCrowning(1);
   if (icf.f_FarsiteSpotProb <= 0.0) {
      this->EnableSpotting(0);
      this->EnableSpotFireGrowth(0);
      this->PercentIgnition(icf.f_FarsiteSpotProb * 100.0); }
   else  {
      this->EnableSpotting(1);
      this->EnableSpotFireGrowth(1);
      this->PercentIgnition(icf.f_FarsiteSpotProb * 100.0);
	  if(icf.i_SpottingSeed >= 0)
		Rand.SetFixedSeed(icf.i_SpottingSeed);
	  else
		  Rand.ResetRandomSeed();
   }

 /* If the Spot Grid Resolution input not found, use landscape resolution */
  if ( icf.f_FarsiteSpotGridResolution == ei_ICFInit )
     icf.f_FarsiteSpotGridResolution = 0.0;//this->GetCellResolutionX()/2;


   LoadBurnPeriods();
	LoadROSAdjustFile();
   UseConditioningPeriod(1);              /* Have Condition Weather data */
   return 1;
}

char  *Farsite5::LoadInputError (int i_Num)
{
  return icf.ErrorMessage(i_Num);
}

bool Farsite5::SetInitialFuelMoistures(long Model, long t1, long t10, long t100,
	long tlh, long tlw)
{
	if (Model > 256 || Model < 1)
		return false;

        fm[Model].TL1   = max(t1, 2L);
        fm[Model].TL10  = max(t10, 2L);
        fm[Model].TL100 = max(t100, 2L);
        fm[Model].TLLH  = max(tlh, 2L);
        fm[Model].TLLW  = max(tlw, 2L);

	if (t1 > 1 && t10 > 1)
		fm[Model].FuelMoistureIsHere = true;
	else
		fm[Model].FuelMoistureIsHere = false;

	long i, j;
	for (i = 0; i < 3; i++) 	  // only up to 100 hr fuels [3]
	{
		for (j = 0; j < 5; j++)
			EnvtChanged[i][j] = true;
	}

	return fm[Model].FuelMoistureIsHere;
}

bool Farsite5::GetInitialFuelMoistures(long Model, long* t1, long* t10, long* t100,
	long* tlh, long* tlw)
{
	if (Model > 256 || Model < 1)
		return false;

	if (fm[Model].FuelMoistureIsHere)
	{
		*t1 = fm[Model].TL1;
		*t10 = fm[Model].TL10;
		*t100 = fm[Model].TL100;
		*tlh = fm[Model].TLLH;
		*tlw = fm[Model].TLLW;

		return true;
	}

	return false;
}

long Farsite5::GetInitialFuelMoisture(long Model, long FuelClass)
{
	if (Model > 256 || Model < 1)
		return 2;

	long mx;

	switch (FuelClass)
	{
	case 0:
		mx = fm[Model].TL1; break;
	case 1:
		mx = fm[Model].TL10; break;
	case 2:
		mx = fm[Model].TL100; break;
	case 3:
		mx = fm[Model].TLLH; break;
	case 4:
		mx = fm[Model].TLLW; break;
	}

	return mx;
}

bool Farsite5::InitialFuelMoistureIsHere(long Model)
{
	if (Model > 256 || Model < 1)
		return false;

	return fm[Model].FuelMoistureIsHere;
}

/***********************************************************
* Name: SetSpreadDirection
* Desc: set the spread direction from inputs file.
*       value comes in from input file switch
*       SPREAD_DIRECTION_FROM_ (MAX or NORTH)
*       Inputs should have been validate when read in.
*   In: North >= 0 set Absolute. of if Max >= 0 set Relative
*       Set both params to negative to do nothing and set to
*       defualt - see below
***********************************************************/
void Farsite5::SetSpreadDirection (double North, double Max)
{
	/*
double d;

  if ( North >= 0 ) {
    this->SetOutputDirection ( ABSOLUTEDIR );
    d = North;
    if ( d == 0 )
      d = 360.0;
    this->SetOffsetFromMax(d); }

  else if ( Max >= 0 ) {
    this->SetOutputDirection (RELATIVEDIR );
    this->SetOffsetFromMax(Max); }

 else {
    this->SetOutputDirection (RELATIVEDIR );   // Default //
    this->SetOffsetFromMax(0); }
*/
}

void Farsite5::SetFileOutputOptions(long FileType, bool YesNo)
{
	switch (FileType)
	{
	case RAST_ARRIVALTIME:
		rast_arrivaltime = YesNo;
		break;
	case RAST_FIREINTENSITY:
		rast_fireintensity = YesNo;
		break;
	case RAST_SPREADRATE:
		rast_spreadrate = YesNo;
		break;
	case RAST_FLAMELENGTH:
		rast_flamelength = YesNo;
		break;
	case RAST_HEATPERAREA:
		rast_heatperarea = YesNo;
		break;
	case RAST_CROWNFIRE:
		rast_crownfire = YesNo;
		break;
	case RAST_FIREDIRECTION:
		rast_firedirection = YesNo;
		break;
	case RAST_REACTIONINTENSITY:
		rast_reactionintensity = YesNo;
		break;
	}
}


bool Farsite5::GetFileOutputOptions(long FileType)
{
	bool YesNo = false;

	switch (FileType)
	{
	case RAST_ARRIVALTIME:
		YesNo = rast_arrivaltime;
		break;
	case RAST_FIREINTENSITY:
		YesNo = rast_fireintensity;
		break;
	case RAST_SPREADRATE:
		YesNo = rast_spreadrate;
		break;
	case RAST_FLAMELENGTH:
		YesNo = rast_flamelength;
		break;
	case RAST_HEATPERAREA:
		YesNo = rast_heatperarea;
		break;
	case RAST_CROWNFIRE:
		YesNo = rast_crownfire;
		break;
	case RAST_FIREDIRECTION:
		YesNo = rast_firedirection;
		break;
	case RAST_REACTIONINTENSITY:
		YesNo = rast_reactionintensity;
		break;
	}

	return YesNo;
}

void Farsite5::SetFoliarMoistureContent(long Percent)
{
     if(Percent<1)
     	Percent=100;
     if(Percent>300)
     	Percent=300;
	CanopyChx.FoliarMC=Percent;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: LoadFoliarMoist
* Desc: Load the Foliar Moisture found in the command file, if none
*       then set the default value.
*   In:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void Farsite5::LoadFoliarMoist(int i_FolMoi)
{
  if ( i_FolMoi == ei_ICFInit )         /* none was set in cmd file */
    SetFoliarMoistureContent (e_FM_Def);
  else
    SetFoliarMoistureContent (i_FolMoi);
}

void Farsite5::SetRasterFileName(const char* FileName)
{
	memset(RasterArrivalTime, 0x0, sizeof(RasterArrivalTime));
	memset(RasterFireIntensity, 0x0, sizeof(RasterFireIntensity));
	memset(RasterFlameLength, 0x0, sizeof(RasterFlameLength));
	memset(RasterSpreadRate, 0x0, sizeof(RasterSpreadRate));
	memset(RasterHeatPerArea, 0x0, sizeof(RasterHeatPerArea));
	memset(RasterCrownFire, 0x0, sizeof(RasterCrownFire));
	memset(RasterFireDirection, 0x0, sizeof(RasterFireDirection));
	memset(RasterReactionIntensity, 0x0, sizeof(RasterReactionIntensity));
	memset(RastFName, 0x0, sizeof(RastFName));
	sprintf(RasterArrivalTime, "%s%s", FileName, ".toa");
	sprintf(RasterFireIntensity, "%s%s", FileName, ".fli");
	sprintf(RasterFlameLength, "%s%s", FileName, ".fml");
	sprintf(RasterSpreadRate, "%s%s", FileName, ".ros");
	sprintf(RasterHeatPerArea, "%s%s", FileName, ".hpa");
	sprintf(RasterCrownFire, "%s%s", FileName, ".cfr");
	sprintf(RasterFireDirection, "%s%s", FileName, ".sdr");
	sprintf(RasterReactionIntensity, "%s%s", FileName, ".rci");
	sprintf(RastFName, "%s%s", FileName, ".rst");
}


char* Farsite5::GetRasterFileName(long Type)
{
	if (Type == 0)
		return RastFName;
	if (Type == RAST_ARRIVALTIME)
		return RasterArrivalTime;
	if (Type == RAST_FIREINTENSITY)
		return RasterFireIntensity;
	if (Type == RAST_FLAMELENGTH)
		return RasterFlameLength;
	if (Type == RAST_SPREADRATE)
		return RasterSpreadRate;
	if (Type == RAST_HEATPERAREA)
		return RasterHeatPerArea;
	if (Type == RAST_CROWNFIRE)
		return RasterCrownFire;
	if (Type == RAST_FIREDIRECTION)
		return RasterFireDirection;
	if (Type == RAST_REACTIONINTENSITY)
		return RasterReactionIntensity;

	return NULL;
}

void Farsite5::SetVectorFileName(const char* FileName)
{
	memset(VectFName, 0x0, sizeof(VectFName));
	sprintf(VectFName, "%s", FileName);
	FILE* newfile;

	newfile = fopen(VectFName, "w");
	if (newfile == NULL)
	{
		newfile = fopen(VectFName, "w");
	}
	fclose(newfile);
}


char* Farsite5::GetVectorFileName()
{
	return VectFName;
}

double Farsite5::GetFoliarMC()
{
	return CanopyChx.FoliarMC;
}

long Farsite5::AccessOutputUnits(long val)
{
	if (val >= 0)
		OutputUnits = val;

	return OutputUnits;
}

long Farsite5::CanSetRasterResolution(long YesNo)
{
	if (YesNo > -1)
		CanSetRastRes = YesNo;

	return CanSetRastRes;
}


void Farsite5::SetRastRes(double XResol, double YResol)
{
	// raster output resolution
	RasterCellResolutionX = XResol;
	RasterCellResolutionY = YResol;
}


void Farsite5::GetRastRes(double* XResol, double* YResol)
{
	// raster output resolution
	*XResol = RasterCellResolutionX;
	*YResol = RasterCellResolutionY;
}

int Farsite5::CheckCellResUnits()
{
	return Header.GridUnits;
}

long Farsite5::GetNumEast()
{
	return Header.numeast;
}


long Farsite5::GetNumNorth()
{
	return Header.numnorth;
}


double Farsite5::GetWestUtm()
{
	return Header.WestUtm;
}

double Farsite5::GetEastUtm()
{
	return Header.EastUtm;
}

double Farsite5::GetSouthUtm()
{
	return Header.SouthUtm;
}

double Farsite5::GetNorthUtm()
{
	return Header.NorthUtm;
}

double Farsite5::GetLoEast()
{
	return Header.loeast;
}
double Farsite5::GetHiEast()
{
	return Header.hieast;
}

double Farsite5::GetLoNorth()
{
	return Header.lonorth;
}

double Farsite5::GetHiNorth()
{
	return Header.hinorth;
}

long Farsite5::GetLoElev()
{
	return Header.loelev;
}

long Farsite5::GetHiElev()
{
	return Header.hielev;
}

//--------------------------------------------------------------------------------
//
//	manage weather/wind stream data
//
//--------------------------------------------------------------------------------

bool Farsite5::CalcFirstLastStreamData()
{
	bool BadRange = false;
	long sn;
	unsigned long  FirstWtr, FirstWnd, LastWtr, LastWnd;
	double ipart, fract;

	for (sn = 0; sn < GetNumStations(); sn++)
	{
		if (AtmosphereGridExists() < 2)
		{
			if (!wtrdt[sn] || (!wddt[sn] && AtmosphereGridExists() == 0))
			{
				continue;
			}
		}

		FirstWtr = (unsigned long)((GetJulianDays(FirstMonth[sn].wtr) + FirstDay[sn].wtr) * 1440.0);
		fract = modf((double) FirstHour[sn].wnd / 100.0, &ipart);
		FirstWnd = (unsigned long)((GetJulianDays(FirstMonth[sn].wnd) + FirstDay[sn].wnd) * 1440.0 +
			ipart * 60.0 +
			fract);
		LastWtr = (unsigned long)((GetJulianDays(LastMonth[sn].wtr) + LastDay[sn].wtr) * 1440.0);
		fract = modf((double) LastHour[sn].wnd / 100.0, &ipart);
		LastWnd = (unsigned long)((GetJulianDays(LastMonth[sn].wnd) + LastDay[sn].wnd) * 1440.0 +
			ipart * 60.0 +
			fract);

		if (FirstWtr > LastWtr)
			LastWtr += (unsigned long)(365.0 * 1440.0);
		if (FirstWnd > LastWnd)
			LastWnd += (unsigned long)(365.0 * 1440.0);

		if (FirstWnd <= FirstWtr)
		{
			if (LastWnd <= FirstWtr)
				BadRange = true;
		}
		else if (FirstWnd > FirstWtr)
		{
			if (FirstWnd > LastWtr)
				BadRange = true;
		}
		if (BadRange)
		{
			FirstMonth[sn].all = 0;
			FirstDay[sn].all = 0;
			FirstHour[sn].all = 0;

			return false;
		}

		if (FirstMonth[sn].wtr > FirstMonth[sn].wnd)
		{
			FirstMonth[sn].all = FirstMonth[sn].wtr;
			FirstDay[sn].all = FirstDay[sn].wtr;
			FirstHour[sn].all = FirstHour[sn].wtr;
		}
		else if (FirstMonth[sn].wtr < FirstMonth[sn].wnd)
		{
			FirstMonth[sn].all = FirstMonth[sn].wnd;
			FirstDay[sn].all = FirstDay[sn].wnd;
			FirstHour[sn].all = FirstHour[sn].wnd;
		}
		else
		{
			FirstMonth[sn].all = FirstMonth[sn].wtr;
			if (FirstDay[sn].wtr > FirstDay[sn].wnd)
			{
				FirstDay[sn].all = FirstDay[sn].wtr;
				FirstHour[sn].all = FirstHour[sn].wtr;
			}
			else if (FirstDay[sn].wtr < FirstDay[sn].wnd)
			{
				FirstDay[sn].all = FirstDay[sn].wnd;
				FirstHour[sn].all = FirstHour[sn].wnd;
			}
			else
			{
				FirstDay[sn].all = FirstDay[sn].wnd;
				if (FirstHour[sn].wtr > FirstHour[sn].wnd)
					FirstHour[sn].all = FirstHour[sn].wtr;
				else
					FirstHour[sn].all = FirstHour[sn].wnd;
			}
		}
		if (LastMonth[sn].wtr < LastMonth[sn].wnd)
		{
			LastMonth[sn].all = LastMonth[sn].wtr;
			LastDay[sn].all = LastDay[sn].wtr;
			LastHour[sn].all = LastHour[sn].wtr;
		}
		else if (LastMonth[sn].wtr > LastMonth[sn].wnd)
		{
			LastMonth[sn].all = LastMonth[sn].wnd;
			LastDay[sn].all = LastDay[sn].wnd;
			LastHour[sn].all = LastHour[sn].wnd;
		}
		else
		{
			LastMonth[sn].all = LastMonth[sn].wtr;
			if (LastDay[sn].wtr < LastDay[sn].wnd)
			{
				LastDay[sn].all = LastDay[sn].wtr;
				LastHour[sn].all = LastHour[sn].wtr;
			}
			else if (LastDay[sn].wtr > LastDay[sn].wnd)
			{
				LastDay[sn].all = LastDay[sn].wnd;
				LastHour[sn].all = LastHour[sn].wnd;
			}
			else
			{
				LastDay[sn].all = LastDay[sn].wnd;
				if (LastHour[sn].wtr < LastHour[sn].wnd)
					LastHour[sn].all = LastHour[sn].wtr;
				else
					LastHour[sn].all = LastHour[sn].wnd;
			}
		}
	}

	return true;
}

long Farsite5::GetNumStations()
{
	if (NumWeatherStations > NumWindStations)
		return NumWindStations;

	return NumWeatherStations;
}

long Farsite5::AtmosphereGridExists()
{
	if (AtmGrid)
	{
		if (AtmGrid->AtmGridWTR)
			return 2;
		else if (AtmGrid->AtmGridWND)
			return 1;
	}

	return 0;
}

long Farsite5::GetJulianDays(long Month)
{
	long days;

	switch (Month)
	{
	case 1:
		days = 0; break;			// cumulative days to begin of month
	case 2:
		days = 31; break;   		// except ignores leapyear, but who cares anyway,
	case 3:
		days = 59; break;
	case 4:
		days = 90; break;
	case 5:
		days = 120; break;
	case 6:
		days = 151; break;
	case 7:
		days = 181; break;
	case 8:
		days = 212; break;
	case 9:
		days = 243; break;
	case 10:
		days = 273; break;
	case 11:
		days = 304; break;
	case 12:
		days = 334; break;
	default:
		days = 0; break;
	}

	return days;
}

//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//
//	Atmosphere Grid Global Functions
//
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------


bool Farsite5::SetAtmosphereGrid(long NumGrids)
{
	if (NumGrids == 0)
	{
		/*
		if(AtmGrid->AtmGridWTR)
		{	NumWeatherStations=0;    // reset number of weather stations
			NumWindStations=0;  	 // reset number of wind stations
		}
		else if(AtmGrid->AtmGridWND)
			NumWindStations=0;  	 // reset number of wind stations
		*/
		if (AtmGrid)
			delete AtmGrid;
		AtmGrid = 0;
		NorthGridOffset = 0.0;
		EastGridOffset = 0.0;

		return false;
	}
	if (AtmGrid == 0)
		AtmGrid = new AtmosphereGrid(NumGrids, this);
	else
	{
		delete AtmGrid;
		AtmGrid = new AtmosphereGrid(NumGrids, this);
	}

	return true;
}


AtmosphereGrid* Farsite5::GetAtmosphereGrid()
{
	return AtmGrid;
}

void Farsite5::SetGridNorthOffset(double offset)
{
	NorthGridOffset = offset;
}

void Farsite5::SetGridEastOffset(double offset)
{
	EastGridOffset = offset;
}

double Farsite5::GetGridNorthOffset()
{
	return NorthGridOffset;
}

double Farsite5::GetGridEastOffset()
{
	return EastGridOffset;
}

void Farsite5::SetGridEastDimension(long XDim)
{
	grid.XDim = XDim;
}


void Farsite5::SetGridNorthDimension(long YDim)
{
	grid.YDim = YDim;
}

long Farsite5::GetGridEastDimension()
{
	return grid.XDim;
}

long Farsite5::GetGridNorthDimension()
{
	return grid.YDim;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: LoadWeatherStream
* Desc: Load into FlamMap the weather stream data that was previously
*        read into the ICF class, the data could have come from a file(.wtr)
*        or embedded directly in the cmd file.
* Ret: 0 = OK
*      else error number, see below
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int  Farsite5::LoadWeatherStream (d_ICF *a_ICF)
{
int i, i_Mth, iX;
long  StationNumber;
d_Wtr *a;

  if ( icf.iN_Wtr <= 0 )                 /* No Weather Data switches found    */
    return 0;                           /*  That's ok                        */

  FreeWeatherData(0);          /* Chk and free any previous 'new'   */

  StationNumber = AllocWeatherData (0, icf.iN_Wtr);
  if ( StationNumber < 0 )
    return e_EMS_Aloc;

  a = icf.a_Wtr;
  iX = -1;

/* Load data for each observation PLUS and then load an extra 13 mth using   */
/*  the last observation's data                                              */
  for ( i = 0; i <= icf.iN_Wtr; i++ ) {
    if ( i == icf.iN_Wtr )               /* done with obsrvation data         */
      i_Mth = 13;                       /* Set month to 13                   */
    else {
      iX++;                             /* each stored struct of obsvrd data */
      i_Mth = a[iX].i_Mth;
   }

    SetWeatherData(   /* Load into FlamMap class           */
               StationNumber,  /*        long StationNumber         */
               i,              /*        long NumObs,               */
               i_Mth,          /* month, long month,       Mth = month,                                     */
               a[iX].i_Day,  /* day,   long day,         Day = day,                                       */
               a[iX].i_Year,
      (double) a[iX].f_Per,  /* ppt,   double rain,      Per = precip in 100th of inch (ex 10 = 0.1 inches*/
               a[iX].i_mTH,  /* hmorn, long time1,       mTH = min_temp_hour 0-2400,                      */
               a[iX].i_xTH,  /* haft,  long time2,       xTH = max_temp_hour 0 - 2400,                    */
      (double) a[iX].f_mT,   /* Tmin,  double temp1,     mT  = min_temp,                                  */
      (double) a[iX].f_xT,   /* Tmax,  double temp2,     xT  = max_temp,                                  */
               a[iX].i_xH,   /* Hmax,  long humid1,      mH  = max_humidity,                              */
               a[iX].i_mH,   /* Hmin,  long humid2,      xH  = min_humidity,                              */
      (double) a[iX].i_Elv,  /* elref, double elevation  Elv = elevation,                                 */
               a[iX].i_PST,  /* 0,     long tr1,         PST = precip_start_time 0-2400,                  */
               a[iX].i_PET); /* 0);    long tr2)         PET = precip_end_time 0-2400)                    */

  } /* for i */

  return 0;
}

void Farsite5::LoadBurnPeriods()
{
	//int i, i_Mth, iX;
	//long  StationNumber;
//	d_WDS *aWDS;
	FreeBurnPeriod();
	if(!CalcFirstLastStreamData())
		return;
    if ( icf.iN_BurnPeriods <= 0 )                /* No Burn Period switches found    */
		return;                           /*  That's ok                        */

	AllocBurnPeriod (icf.iN_BurnPeriods);
	for(long i = 0; i < icf.iN_BurnPeriods; i++)
	{
		SetBurnPeriod(i, icf.a_BurnPeriods[i].i_Month,
                         icf.a_BurnPeriods[i].i_Day,
                         icf.a_BurnPeriods[i].i_Year,
                         icf.a_BurnPeriods[i].i_HourStart,
                         icf.a_BurnPeriods[i].i_HourEnd);
	}
}

int Farsite5::LoadROSAdjustFile()
{
	int ret = 0;
	if(strlen(icf.cr_ROSAdjustFile) > 0)
	{
		FILE *in = fopen(icf.cr_ROSAdjustFile, "rt");
		if(!in)
		{
			printf("Error opening ROS Adjust file: %s\nContinuing without ROS Adjustments\n", icf.cr_ROSAdjustFile);
			return ret;
		}
		long fnum;
		double adj;

		do
		{	int nRead;
			nRead = fscanf(in, "%ld %lf", &fnum, &adj);
			if(nRead == 2)
			{
				if(fnum>=0 && fnum<257 && adj >= 0)
					SetRosRed(fnum, adj);
			}
		}while(!feof(in));

		fclose(in);

	}
	return ret;
}

int Farsite5::LoadCustomFuelFile(char *FileName)
{
	int ret = 1;
	SetCustFuelModelID(false);
	char *ptr, dynamic[32]="", code[32]="";
     char Line[256]="", head[64]="", ErrMsg[256]="", BackupFile[256]="", comment[256]="";
     bool Metric=false, BadFile=false;
	long num, count, ModNum, FileFormat=0;
     double s1, slh, slw;
     NewFuel newfuel, newfuelm;
     FILE *fout;
     FILE *CurrentFile;

     CurrentFile=fopen(FileName, "rt");
     memset(&newfuel, 0x0, sizeof(NewFuel));
     memset(&newfuelm, 0x0, sizeof(NewFuel));
     do
     {    rewind(CurrentFile);
     	memset(head, 0x0, 64*sizeof(char));
     	memset(Line, 0x0, sizeof(Line));
	     fgets(Line, 255, CurrentFile);
     	sscanf(Line, "%s", head);
	    	AccessFuelModelUnits(0);
     	if(!strcmp(head, "metric"))
	     {    Metric=true;
     	     AccessFuelModelUnits(1);
	     }
     	else if(strcmpi(head, "english"))   // no header in file
	     {    if(atol(head)>256)
     	     {     //::MessageBox(Client->HWindow, "Bad Header in File", "Custom Fuel Model File Error", MB_OK);
          	     fclose(CurrentFile);

	               return false;
     	     }
               memset(head, 0x0, 64*sizeof(char));
     		rewind(CurrentFile);
	     }
          if(FileFormat>0)
          {    if(FileFormat==1)
          	{  	strcat(BackupFile, FileName);
	               strcat(BackupFile, ".old");
     	          CopyFile( FileName,  BackupFile, false);
          	     fclose(CurrentFile);
               	CurrentFile=fopen(BackupFile, "r");
#ifdef WIN32
			     if((access(FileName, 02))==-1)
			     {	SetFileAttributes( FileName, FILE_ATTRIBUTE_NORMAL);
				     DeleteFile( FileName);
			     }
#endif
                    fout=fopen(FileName, "w");
                    if(Metric)
                    	fprintf(fout, "METRIC\n");
                    else
                    	fprintf(fout, "ENGLISH\n");
               }
               else
               	fout=NULL;
          	break;
          }

	     memset(Line, 0x0, 256*sizeof(char));
	     fgets(Line, 255, CurrentFile);
	     if(feof(CurrentFile))
	         	break;
	     if(strlen(Line)==0 || !strncmp(Line, "\n", 1))
	         	continue;
	     num=sscanf(Line, "%ld %s %lf %lf %lf %lf %lf %s",
	               &ModNum, head, &newfuel.h10, &newfuel.h100,
	               &newfuel.lh, &newfuel.lw, &s1, comment);
          //if(!strcmp(_strlwr(comment), "dynamic"))
          //  	FileFormat=2;
          //else if(!strcmp(_strlwr(comment), "static"))
          //	FileFormat=2;
          //else
          if(strstr(comment, "d") || strstr(comment, "D"))
          	FileFormat=2;
          else if(strstr(comment, "s") || strstr(comment,"S"))
          	FileFormat=2;
          else
          	FileFormat=1;
     } while(FileFormat>0);

     rewind(CurrentFile);
     memset(head, 0x0, 64*sizeof(char));
     fgets(Line, 255, CurrentFile);
    	sscanf(Line, "%s", head);
    	if(strcmpi(head, "metric") && strcmp(head, "english"))
     	rewind(CurrentFile);

     count=0;
	while(!feof(CurrentFile))
	{    memset(Line, 0x0, 256*sizeof(char));
     	fgets(Line, 255, CurrentFile);
          if(strlen(Line)==0 || !strncmp(Line, "\n", 1))
          	continue;
          //if(feof(CurrentFile))
          //	break;
          memset(comment, 0x0, 256*sizeof(char));
          if(FileFormat==1)
          {	num=sscanf(Line, "%ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %s",
               	&ModNum, &newfuel.h1, &newfuel.h10, &newfuel.h100,
	               &newfuel.lh, &newfuel.lw, &s1, &slh, &slw,
     	          &newfuel.depth, &newfuel.xmext, &newfuel.heatd, &newfuel.heatl, comment);
          	if(num<13)
          		break;
               sprintf(newfuel.code, "FM%ld", ModNum);
               newfuel.dynamic=0;
          }
          else
          {    memset(code, 0x0, 32*sizeof(char));
          	memset(dynamic, 0x0, 32*sizeof(char));
          	num=sscanf(Line, "%ld %s %lf %lf %lf %lf %lf %s %lf %lf %lf %lf %lf %lf %lf %s",
               	&ModNum, code, &newfuel.h1, &newfuel.h10, &newfuel.h100,
	               &newfuel.lh, &newfuel.lw, dynamic, &s1, &slh, &slw,
     	          &newfuel.depth, &newfuel.xmext, &newfuel.heatd, &newfuel.heatl, comment);
          	if(num<15)
          		break;
               //if(!strcmp(_strupr(dynamic), "DYNAMIC"))
               //	newfuel.dynamic=1;
          	if(strstr(dynamic, "d") || strstr(dynamic, "D"))
               	newfuel.dynamic=1;
               else
               	newfuel.dynamic=0;
               strncpy(newfuel.code, code, 7);
		}

          newfuel.sav1=(long) s1;
          newfuel.savlh=(long) slh;
          newfuel.savlw=(long) slw;
          memset(newfuel.desc, 0x0, 256*sizeof(char));
          if(strlen(comment)>0)
	     {    ptr=strstr(Line, comment);
          	strncpy(newfuel.desc, ptr, 64);
          }
          if(Metric)     // convert to english
          {    memcpy(&newfuelm, &newfuel, sizeof(NewFuel));
          	newfuel.h1/=2.2417;
               newfuel.h10/=2.2417;
               newfuel.h100/=2.2417;
               newfuel.lh/=2.2417;
               newfuel.lw/=2.2417;
               newfuel.sav1=s1*30.480060960;
               newfuel.savlh=slh*30.480060960;
               newfuel.savlw=slw*30.480060960;
               newfuel.depth/=30.480060960;
               newfuel.heatd/=2.324375;
               newfuel.heatl/=2.324375;
          }
          count++;
		if(ModNum<14 || ModNum>256)
		{    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "Fuel Model Number >256 or <14", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_FM;
		}
		else if(newfuel.xmext<=0.0)
		{    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "Extinction Moisture = 0", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_EM;
		}
          else if(newfuel.h1+newfuel.h10+newfuel.h100<=0.0)
          {    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "Fuel Model Has No Dead Fuel", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_DF;
          }
		else if(newfuel.depth<=0.0)
		{    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "Depth = 0.0", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_DEPTH;
		}
		else if(newfuel.heatl<6000)
		{    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "Live Heat Content Too Low", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_LH;
		}
		else if(newfuel.heatd<4000)
		{    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "Dead Heat Content Too Low", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_DH;
		}
		else if(newfuel.sav1>4000 || newfuel.savlh>4000 || newfuel.savlw>4000)
		{    sprintf(ErrMsg, "Model Data Error Line => %ld", count);
          	//::MessageBox(Client->HWindow, "SAV Ratios Out of Range", ErrMsg, MB_OK);
               BadFile=true;
			   ret = e_EMS_CF_SAV;
		}

          if(BadFile)
          {    fclose(CurrentFile);
			return ret;
          }
		newfuel.xmext/=100.0;
          newfuelm.xmext=newfuel.xmext;
          newfuel.number=newfuelm.number=ModNum;
          SetNewFuel(&newfuel);
	     if(FileFormat==1 && fout!=NULL)
          {    if(newfuel.dynamic==0)
               	sprintf(dynamic, "static");
          	else
               	sprintf(dynamic, "dynamic");
          	if(Metric)
	     		fprintf(fout, "%ld %s %lf %lf %lf %lf %lf %s %ld %ld %ld %lf %ld %lf %lf %s\n",
     	          	newfuelm.number, newfuelm.code, newfuelm.h1, newfuelm.h10, newfuelm.h100,
	     	          newfuelm.lh, newfuelm.lw, dynamic, newfuelm.sav1, newfuelm.savlh, newfuelm.savlw,
     	     	     newfuelm.depth, (long) (newfuelm.xmext*100.0), newfuelm.heatd, newfuelm.heatl, newfuelm.desc);
               else
	     		fprintf(fout, "%ld %s %lf %lf %lf %lf %lf %s %ld %ld %ld %lf %ld %lf %lf %s\n",
     	          	newfuel.number, newfuel.code, newfuel.h1, newfuel.h10, newfuel.h100,
	     	          newfuel.lh, newfuel.lw, dynamic, newfuel.sav1, newfuel.savlh, newfuel.savlw,
     	     	     newfuel.depth, (long) (newfuelm.xmext*100.0), newfuel.heatd, newfuel.heatl, newfuel.desc);
          }
	}
	fclose(CurrentFile);
     if(fout)
     {	fclose(fout);
   		//::MessageBox(Client->HWindow, "Existing File Renamed with .OLD extension",
		//	"Custom Fuel Model File Converted to new Format", MB_OK);
     }

	SetCustFuelModelID(true);

	return 1;//success
}

int Farsite5::LoadCustomFuelsData()
{
/* Load custom fuel File if we have a file name                           */
	int ret = 1;
   if ( strcmp (icf.cr_CFF,"" ))
   {
	   ret = LoadCustomFuelFile(icf.cr_CFF);
   }
   //maybe it was embedded in inputs file...
   else if(icf.iN_CustomFuels > 0 && icf.a_customFuels != NULL)
   {
	   for(int f = 0; f < icf.iN_CustomFuels; f++)
	   {
		   NewFuel newfuel;
		   newfuel.number = icf.a_customFuels[f].i_Model;
		   newfuel.h1 = icf.a_customFuels[f].f_h1;
		   newfuel.h10 = icf.a_customFuels[f].f_h10;
		   newfuel.h100 = icf.a_customFuels[f].f_h100;
		   newfuel.lh = icf.a_customFuels[f].f_lh;
		   newfuel.lw = icf.a_customFuels[f].f_lw;
		   newfuel.depth = icf.a_customFuels[f].f_depth;
		   newfuel.xmext = icf.a_customFuels[f].f_xmext;
		   newfuel.heatd = icf.a_customFuels[f].f_heatd;
		   newfuel.heatl = icf.a_customFuels[f].f_heatl;
		   newfuel.dynamic = 1;
		   if(icf.a_customFuels[f].dynamic[0] != 'd' || icf.a_customFuels[f].dynamic[0] != 'D' )
			   newfuel.dynamic = 0;
		   strncpy(newfuel.code, icf.a_customFuels[f].cr_code, 7);
		   newfuel.sav1 = icf.a_customFuels[f].f_sl;
		   newfuel.savlh = icf.a_customFuels[f].f_slh;
		   newfuel.savlw = icf.a_customFuels[f].f_slw;
		   strncpy(newfuel.desc, icf.a_customFuels[f].cr_comment, 255);
			newfuel.xmext/=100.0;
		   if(icf.cr_CustFuelUnits[0] == 'M' || icf.cr_CustFuelUnits[0] == 'm')//inputs were metric
		   {
         		newfuel.h1/=2.2417;
               newfuel.h10/=2.2417;
               newfuel.h100/=2.2417;
               newfuel.lh/=2.2417;
               newfuel.lw/=2.2417;
               newfuel.sav1*=30.480060960;
               newfuel.savlh*=30.480060960;
               newfuel.savlw*=30.480060960;
               newfuel.depth/=30.480060960;
               newfuel.heatd/=2.324375;
               newfuel.heatl/=2.324375;
		   }
          SetNewFuel(&newfuel);

	   }
   }
   //if(ret == 1)
	//return CheckCustomFuelsCoverage();
   return ret;
}

int Farsite5::CheckCustomFuelsCoverage()
{	//check header.fuels to make sure all burnable fuels are defined
	int ret = 1;
	for(int f = 0; f < Header.numfuel; f++)
	{
		int fuel = Header.fuels[f];
		if(fuel > 0 && fuel <= 256)
		{
			if(fuel < 90 || fuel > 99)
			{
				if (!IsNewFuelReserved(fuel))// check to see if custom model required
				{
					if (newfuels[fuel].number == 0)   // check to see if cust mod here
					{
						sprintf(icf.cr_ErrExt, "Fuel %d undefined", fuel);
						/*printf("Header Fuels: %ld\n", Header.numfuel);
						for(int i = 0; i < Header.numfuel; i++)
						{
                                printf("%ld\n", Header.fuels[i]);
                                getchar();

						}*/
						return e_EMS_FUEL_UNDEFINED;
					}
				}
			}
		}
	}
	return ret;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: LoadWind_Table
* Desc: Load the wind table with either the original type wind data
*        or the new RAWS data.
*       There should always be one or the other type of wind data, which
*        should have been previously checked.
*   In: a_ICF....inputs file data
*  Ret: < 0 error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*/
int  Farsite5::LoadWind_Table (d_ICF *a_ICF)
{
int i;
  if ( icf.iN_RAWS != 0 ) {
    i = this->LoadWind_RAWS(a_ICF);
    return i; }

  if ( icf.iN_Wnd != 0 ) {
    i = this->LoadWind_Regular(a_ICF);
    return i; }

/* Should always be doing one of the above */
  return 0;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: LoadWind_Regular
* Desc: Load the wind table with the original wind data from the inputs file.
* NOTE: see the LoadWind_RAWS() function, it loads the RAWS wind data.
*  Ret: see below
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int Farsite5::LoadWind_Regular (d_ICF *a_ICF)
{
int i, i_Mth, iX;
long  StationNumber;
d_Wnd *aWDS;

  if ( icf.iN_Wnd <= 0 )
    return 0;

  FreeWindData(0);

  StationNumber = AllocWindData (0, icf.iN_Wnd+1);
  if ( StationNumber < 0 )
    return e_EMS_Aloc;

  aWDS = icf.a_Wnd;
  iX = -1;

/* Load data for each observation PLUS and then load an extra 13 mth using   */
/*  the last observation's data                                              */
  for ( i = 0; i <= icf.iN_Wnd; i++ ) {
    if ( i == icf.iN_Wnd )              /* done with obsrvation data         */
      i_Mth = 13;                       /* Set month to 13                   */
    else {
      iX++;                             /* each stored struct of obsvrd data */
      i_Mth = aWDS[iX].i_Mth;
     _WDSConver(&aWDS[iX],icf.cr_WiDU);}/* Do any needed unit conversion     */

    SetWindData(               /* Load into FlamMap class           */
               StationNumber,           /*        long StationNumber         */
               i,                       /*        long NumObs,               */
               i_Mth,                   /* month, long month,                */
               aWDS[iX].i_Day,          /*  (long) day,                      */
               aWDS[iX].i_Year,
               aWDS[iX].i_Hr,           /*  (long) hhour,                    */
               aWDS[iX].f_Spd,          /*  (long) wss,                      */
               aWDS[iX].i_Dir,          /*  (long) wwwinddir,                */
               aWDS[iX].i_CloCov);      /*  (long) cloudcover)               */

  } /* for i */
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: LoadWind_RAWS
* Desc: Load the wind table with the RAWS data from inputs file
*  Ret: see below
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*/
int Farsite5::LoadWind_RAWS (d_ICF *a_ICF)
{
int i, i_Mth, iX;
long  StationNumber;
d_RAWS *a;

  if ( icf.iN_RAWS <= 0 )
    return 0;

  FreeWindData(0);

  StationNumber = AllocWindData (0, icf.iN_RAWS+1);
  if ( StationNumber < 0 )
    return e_EMS_Aloc;

/* Farsite will use this even though we don't load RAWS weather data into Farsite because */
/*  all it realy needs is the wind/cloud data */
  this->NumWeatherStations = 1;

  a = icf.a_RAWS;
  iX = -1;

/* Load data for each observation PLUS and then load an extra 13 mth using   */
/*  the last observation's data                                              */
  for ( i = 0; i <= icf.iN_RAWS; i++ ) {
    if ( i == icf.iN_RAWS )              /* done with obsrvation data         */
      i_Mth = 13;                       /* Set month to 13                   */
    else {
      iX++;                             /* each stored struct of obsvrd data */
      i_Mth = a[iX].i_Mth; }

   SetWindData(               /* Load into FlamMap class           */
               StationNumber,           /*        long StationNumber         */
               i,                       /*        long NumObs,               */
               i_Mth,                   /* month, long month,                */
               a[iX].i_Day,            /*  (long) day,                      */
               a[iX].i_Yr,
               a[iX].i_Time,           /*  (long) hhour,                    */
               (long) a[iX].f_WinSpd,          /*  (long) wss,                      */
               (long) a[iX].f_WinDir,          /*  (long) wwwinddir,                */
               (long) a[iX].f_CloCov);      /*  (long) cloudcover)               */

  } /* for i */
  return 1;
}




/********************************************************************************************/
void Farsite5::FreeWeatherData(long StationNumber)
{
	if(wtrdt[StationNumber])
	{    delete[] wtrdt[StationNumber];
          MaxWeatherObs[StationNumber]=0;
		NumWeatherStations--;
	}
	wtrdt[StationNumber]=0;
     FirstMonth[StationNumber].wtr=0;
	LastMonth[StationNumber].wtr=0;
	FirstDay[StationNumber].wtr=0;
	LastDay[StationNumber].wtr=0;
	FirstHour[StationNumber].wtr=0;
	LastHour[StationNumber].wtr=0;
}

//----------------------------------------------------
void Farsite5::FreeWindData(long StationNumber)
{
//int i;

	if (wddt[StationNumber]) {
// `est
   /*for ( i = 0; i < MaxWindObs[StationNumber]; i++ )
   {
	   if(wddt[StationNumber][i].a_FWN != NULL)
			DeleteWindGrids (wddt[StationNumber][i].a_FWN, wddt[StationNumber][i].iS_WN);
   }*/

   delete[] wddt[StationNumber];
	 	NumWindStations--;
   MaxWindObs[StationNumber]=0;	}

	wddt[StationNumber]=0;
 FirstMonth[StationNumber].wnd=0;
	LastMonth[StationNumber].wnd=0;
	FirstDay[StationNumber].wnd=0;
	LastDay[StationNumber].wnd=0;
	FirstHour[StationNumber].wnd=0;
	LastHour[StationNumber].wnd=0;
}
//---------------------------------------------------------



long Farsite5::AllocWeatherData(long StatNum, long NumObs)
{
	long StationNumber = StatNum;

	if (wtrdt[StationNumber])
	{
		delete[] wtrdt[StationNumber];//free(wtrdt[StationNumber]);
		wtrdt[StationNumber] = 0;
		MaxWeatherObs[StatNum] = 0;
	}
	else
	{
		StationNumber = GetOpenWeatherStation();
		if (StationNumber < 5)
			NumWeatherStations = StationNumber + 1;
	}
	if (StationNumber<5 && NumObs>0)
	{
		size_t nmemb = MaxWeatherObs[StationNumber] = NumObs + 20;		// alloc 20 more than needed
		if ((wtrdt[StationNumber] = new WeatherData[nmemb]) == NULL)
			StationNumber = -1;
	}
	else
		StationNumber = -1;

	return StationNumber;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: _WSConver
* Desc: Do any needed conversion to english if the data comes in as
*       metric...conversions were take directly from the existing
*       FlamMap::LoadWeatherdFile()
*   In: aWS.....weather stream struct
*       cr......Unit
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void  Farsite5::_WSConver (d_Wtr *a, char cr[])
{
  if ( !strcmpi (cr,"Metric")){
    a->f_Per *= 3.93;                 /*  ppt  *= 3.93;     */
    a->f_mT  *= 1.8;                  /*  Tmin *= 1.8;      */
    a->f_mT  += 32;                   /*  Tmin += 32;       */
    a->f_xT  *= 1.8;                  /*  Tmax *= 1.8;      */
    a->f_xT  += 32;                   /*  Tmax += 32;       */
    a->i_Elv *= 3.28; }               /*  elref *= 3.28;    */

/* Not sure about this but it was done in FlamMap code                       */
  if ( a->i_mH > 99 )                  /*  if ( Hmax > 99 )  */
    a->i_mH = 99;                      /*    Hmax = 99;      */
  if ( a->i_xH > 99 )                  /*  if ( Hmin > 99)   */
   a->i_xH = 99;                       /*    Hmin = 99;      */

}
/********************************************************************
* Note-1, added year 4-11-12,
*         this might come in handy at some point i put it in when
*         I was added year to the wind and burnperiod recs.
********************************************************************/
long Farsite5::SetWeatherData(long StationNumber, long NumObs, long month, long day,
	long year, double rain, long time1, long time2, double temp1, double temp2,
	long humid1, long humid2, double elevation, long tr1, long tr2)
{
	if (NumObs < MaxWeatherObs[StationNumber])
	{
		wtrdt[StationNumber][NumObs].mo = month;
		wtrdt[StationNumber][NumObs].dy = day;
        wtrdt[StationNumber][NumObs].yr = year;  /* See note-1 above */
		wtrdt[StationNumber][NumObs].rn = rain;
		wtrdt[StationNumber][NumObs].t1 = time1;
		wtrdt[StationNumber][NumObs].t2 = time2;
		wtrdt[StationNumber][NumObs].T1 = temp1;
		wtrdt[StationNumber][NumObs].T2 = temp2;
		wtrdt[StationNumber][NumObs].H1 = humid1;
		wtrdt[StationNumber][NumObs].H2 = humid2;
		wtrdt[StationNumber][NumObs].el = elevation;
		wtrdt[StationNumber][NumObs].tr1 = tr1;
		wtrdt[StationNumber][NumObs].tr2 = tr2;

		if (month == 13)
		{
			FirstMonth[StationNumber].wtr = wtrdt[StationNumber][0].mo;
			LastMonth[StationNumber].wtr = wtrdt[StationNumber][NumObs - 1].mo;
			FirstDay[StationNumber].wtr = wtrdt[StationNumber][0].dy;
			LastDay[StationNumber].wtr = wtrdt[StationNumber][NumObs - 1].dy;
			FirstHour[StationNumber].wtr = wtrdt[StationNumber][0].t1;
			LastHour[StationNumber].wtr = wtrdt[StationNumber][NumObs - 1].t2;
		}

		EnvtChanged[0][StationNumber] = true;   // 1hr fuels
		EnvtChanged[1][StationNumber] = true;   // 10hr fuels
		EnvtChanged[2][StationNumber] = true;   // 100hr fuels
		EnvtChanged[3][StationNumber] = true;   // 1000hr fuels
		return 1;
	}

	return 0;
}

/***************************************************************************
* Name: AllocWindData
* Desc: Allocate wind table records.
*
****************************************************************************/
long Farsite5::AllocWindData(long StatNum, long NumObs)
{
int i;
	long StationNumber = StatNum;

	if (wddt[StationNumber])	{
	 	delete[] wddt[StationNumber];
		 wddt[StationNumber] = 0;
		 MaxWindObs[StatNum] = 0;}
	else	{
		 StationNumber = GetOpenWindStation();
		 if (StationNumber < 5)
			  NumWindStations = StationNumber + 1;	}

	if (StationNumber < 5 && NumObs > 0)	{

	 	 size_t nmemb = MaxWindObs[StationNumber] = NumObs;
// Original code, Not sure why that times 2 was there ?
//	 	size_t nmemb = MaxWindObs[StationNumber] = NumObs * 2;			// alloc 2* number needed

		  if ((wddt[StationNumber] = new WindData[nmemb]) == NULL)
			   StationNumber = -1;

// WN-Test
    for ( i = 0; i < nmemb; i++ )
       wddt[StationNumber][i].a_FWN = NULL;

  }

	else
		 StationNumber = -1;

	return StationNumber;
}
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: _WDSConver
* Desc: do wind speed conversion if data comes in as metric
*
*   In:
*  Ret:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void Farsite5::_WDSConver (d_Wnd *a, char cr[])
{
  if ( !strcmpi (cr,"Metric")) /* if Metric                                  */
    a->f_Spd *= 0.5402;     /* (0.62125/1.15) 10m wind kph to 20ft wind mph*/
}


long Farsite5::SetWindData(long StationNumber, long NumObs, long month, long day,
	long year, long hour, double windspd, long winddir, long cloudy)
{
	if (NumObs < MaxWindObs[StationNumber])
	{
		wddt[StationNumber][NumObs].mo = month;
		wddt[StationNumber][NumObs].dy = day;
        wddt[StationNumber][NumObs].yr = year;
		wddt[StationNumber][NumObs].hr = hour;
		wddt[StationNumber][NumObs].ws = windspd;
		wddt[StationNumber][NumObs].wd = winddir;
		wddt[StationNumber][NumObs].cl = cloudy;

		if (month == 13)
		{
			FirstMonth[StationNumber].wnd = wddt[StationNumber][0].mo;
			LastMonth[StationNumber].wnd = wddt[StationNumber][NumObs - 1].mo;
			FirstDay[StationNumber].wnd = wddt[StationNumber][0].dy;
			LastDay[StationNumber].wnd = wddt[StationNumber][NumObs - 1].dy;
			FirstHour[StationNumber].wnd = wddt[StationNumber][0].hr;
			LastHour[StationNumber].wnd = wddt[StationNumber][NumObs - 1].hr;
		}

		return 1;
	}

	return 0;
}

long Farsite5::GetOpenWeatherStation()
{
	long i;

	for (i = 0; i < 5; i++)
	{
		if (!wtrdt[i])
			break;
	}

	return i;
}


long Farsite5::GetOpenWindStation()
{
	long i;

	for (i = 0; i < 5; i++)
	{
		if (!wddt[i])
			break;
	}

	return i;
}


double Farsite5::MetricResolutionConvert()
{
	if (Header.GridUnits == 1)
		return 3.280839895; 	// metric conversion to meters
	else
		return 1.0;
}

void Farsite5::ResetNewFuels()
{
	memset(newfuels, 0x0, 257 * sizeof(NewFuel));

	InitializeNewFuel();

	long i;
	for (i = 0; i < 257; i++)
	{
		if (newfuels[i].number)
			newfuels[i].number *= -1;	// indicate permanence
	}

	HAVE_CUST_MODELS = false;
}


bool Farsite5::SetNewFuel(NewFuel* newfuel)
{
	long FuelNum;

	if (newfuel == 0)
		return false;

	FuelNum = newfuel->number;
	if (FuelNum > 256 || FuelNum < 0)
		return false;

	newfuel->number = 0;
	if (newfuel->h1 > 0.0)
		newfuel->number = FuelNum;
	else if (newfuel->h10 > 0.0)
		newfuel->number = FuelNum;
	else if (newfuel->h100 > 0.0)
		newfuel->number = FuelNum;
	else if (newfuel->lh > 0.0)
		newfuel->number = FuelNum;
	else if (newfuel->lw > 0.0)
		newfuel->number = FuelNum;

	memcpy(&newfuels[FuelNum], newfuel, sizeof(NewFuel));

	return true;
}


bool Farsite5::GetNewFuel(long number, NewFuel* newfuel)
{
	if (number < 0)
		return false;

	if (newfuels[number].number == 0)
		return false;

	if (newfuel != 0)
	{
		memcpy(newfuel, &newfuels[number], sizeof(NewFuel));
		newfuel->number = labs(newfuel->number);	// return absolute value of number
	}

	return true;
}


bool Farsite5::IsNewFuelReserved(long number)
{
	if (number < 0)
		return false;

	if (newfuels[number].number < 0)
		return true;

	return false;
}


void Farsite5::GetCurrentFuelMoistures(long fuelmod, long woodymod, double* mxin,
	double* mxout, long NumMx)
{
	bool Combine;
	long i, NumClasses = 0;

//	ZeroMemory(mxout, 20 * sizeof(double));
	memset(mxout,0x0, 20 * sizeof(double));

	switch (WoodyCombineOptions(GETVAL))
	{
	case CWD_COMBINE_NEVER:
		Combine = false; break;
	case CWD_COMBINE_ALWAYS:
		Combine = true; break;
	case CWD_COMBINE_ABSENT:
		if (woodymod < 1)
			Combine = true;
		else if (coarsewoody[woodymod - 1].wd == 0)
			Combine = true;
		else
			Combine = false;
		break;
	}

	if (Combine)
	{
		if (fuelmod < 14)
		{
			switch (fuelmod)
			{
			case 0:
				break;
			case 1:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 2:
				memcpy(mxout, mxin, 3 * sizeof(double));
				mxout[3] = mxin[5];
				NumClasses = 4;
				break;
			case 3:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 4:
				memcpy(mxout, mxin, 3 * sizeof(double));
				mxout[3] = mxin[5];
				NumClasses = 4;
				break;
			case 5:
				memcpy(mxout, mxin, 3 * sizeof(double));
				mxout[3] = mxin[5];
				NumClasses = 4;
				break;
			case 6:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 7:
				memcpy(mxout, mxin, 4 * sizeof(double));
				NumClasses = 4;
				break;
			case 8:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 9:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 10:
				memcpy(mxout, mxin, 3 * sizeof(double));
				mxout[3] = mxin[5];
				NumClasses = 4;
				break;
			case 11:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 12:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			case 13:
				memcpy(mxout, mxin, 3 * sizeof(double));
				NumClasses = 3;
				break;
			}
		}
		else
		{
			//if(FuelMod.TL1[fuelmod-14]>0.0)   	 // always have 1, 10, 100 hr fuels
			mxout[NumClasses++] = mxin[0];
			//if(FuelMod.TL10[fuelmod-14]>0.0)
			mxout[NumClasses++] = mxin[1];
			//if(FuelMod.TL100[fuelmod-14]>0.0)
			mxout[NumClasses++] = mxin[2];
			if (newfuels[fuelmod].lh > 0.0)//FuelMod.TLLiveH[fuelmod-14]>0.0)
				mxout[NumClasses++] = mxin[6];
			if (newfuels[fuelmod].lw > 0.0)//FuelMod.TLLiveW[fuelmod-14]>0.0)
				mxout[NumClasses++] = mxin[5];
		}
	}
	if (woodymod > 0)
	{
		for (i = 0; i < coarsewoody[woodymod - 1].NumClasses; i++)
			mxout[NumClasses + i] = coarsewoody[woodymod - 1].wd[i].FuelMoisture;
		memcpy(&mxout[NumClasses], mxin, 3 * sizeof(double));
		mxout[3] = mxin[5];
	}
}

long Farsite5::WoodyCombineOptions(long Option)
{
	if (Option > 0)
		CombineOption = Option;

	return CombineOption;
}
long Farsite5::GetMaxThreads()
{
	return MaxThreads;
}


void Farsite5::SetMaxThreads(long numthreads)
{
	if (numthreads > 0 && numthreads < 65)
		MaxThreads = numthreads;
}

void Farsite5::AddDownTime(double time)
{
	if (time < 0)
		DownTime = 0.0;
	else
		DownTime += time;
}

double Farsite5::GetDownTime()
{
	return DownTime;
}

bool Farsite5::PreserveInactiveEnclaves(long YesNo)
{
	if (YesNo >= 0)
		InactiveEnclaves = (bool) YesNo;

	return InactiveEnclaves;
}

long Farsite5::GetNumAirCraft()
{
	return NumAirCraft;
}

void Farsite5::SetNumAirCraft(long NumCraft)
{
	NumAirCraft = NumCraft;
}


bool Farsite5::LoadAirCraft(char* FileName, bool AppendList)
{
/*	FILE* AirAttackFile;
	char garbage[256], str[256];
	char ch[2] = "";
	long craftnumber, covlev;
	double patlen;

	if ((AirAttackFile = fopen(FileName, "r")) != NULL)
	{
		long i, j;
		j = GetNumAirCraft();
		if (!AppendList)
		{
			for (i = 0; i < j; i++)
				FreeAirCraft(i);
		}
		ch[0] = getc(AirAttackFile);
		do
		{
			memset(garbage, 0x0, sizeof(garbage));
			fgets(garbage, 255, AirAttackFile);
			if (feof(AirAttackFile))
				break;
			craftnumber = SetNewAirCraft();
			if (craftnumber == -1)
				return false;
			aircraft[craftnumber] = GetAirCraft(craftnumber);
			memset(aircraft[craftnumber]->AirCraftName, 0x0,
				sizeof(aircraft[craftnumber]->AirCraftName));
			strncpy(aircraft[craftnumber]->AirCraftName, garbage,
				strlen(garbage) - 2);

			fgets(garbage, 255, AirAttackFile);
			sscanf(garbage, "%s", str);
			std::transform(str, str+strlen(str), str, toupper);
			if (!strcmp(str, "METERS"))
				aircraft[craftnumber]->Units = 0;
			else if (!strcmp(str, "FEET"))
				aircraft[craftnumber]->Units = 1;
			else
			{
				//::MessageBox(HWindow, "Units Not 'Feet' or 'Meters'", "Error, Unrecognized Units", MB_OK);
				aircraft[craftnumber]->Units = 0;    // default to meters
			}
			for (i = 0; i < 6; i++)
			{
				fgets(garbage, 255, AirAttackFile);
				sscanf(garbage, "%ld %lf", &covlev, &patlen);
				if (covlev < 99)
				{
					if (covlev > 0 && covlev < 5)
						aircraft[craftnumber]->PatternLength[covlev - 1] = (long)
							patlen;
					else if (covlev == 6)
						aircraft[craftnumber]->PatternLength[4] = (long)
							patlen;
					else if (covlev == 8)
						aircraft[craftnumber]->PatternLength[5] = (long)
							patlen;
				}
				else
					break;
			}
			ch[0] = getc(AirAttackFile);
			if ((toupper(ch[0])=='R') || (toupper(ch[0])=='C'))
			{
				fgets(garbage, 255, AirAttackFile);
				sscanf(garbage, "%s %lf", str, &patlen);
				std::transform(str, str+strlen(str), str, toupper) ;
				if (!strcmp(str, "ETURN_TIME"))
					aircraft[craftnumber]->ReturnTime = patlen;
				else if (!strcmp(str, "OST"))
					aircraft[craftnumber]->Cost = patlen;
				ch[0] = getc(AirAttackFile);
				if ((toupper(ch[0])=='R') || (toupper(ch[0])=='C'))
				{
					fgets(garbage, 255, AirAttackFile);
					sscanf(garbage, "%s %lf", str, &patlen);
					std::transform(str, str+strlen(str), str, toupper) ;
					if (!strcmp(str, "ETURN_TIME"))
						aircraft[craftnumber]->ReturnTime = patlen;
					else if (!strcmp(str, "OST"))
						aircraft[craftnumber]->Cost = patlen;
					ch[0] = getc(AirAttackFile);
				}
			}
			else // set return time and cost to unknown values
			{
				aircraft[craftnumber]->ReturnTime = 60.0;
				aircraft[craftnumber]->Cost = 1000.0;
			}
		}
		while (!feof(AirAttackFile));//strncmp(ch, "#", 1));  //(!feof(AirAttackFile));
		fclose(AirAttackFile);

		return true;
	}*/

	return false;
}

long Farsite5::SetNewAirCraft()
{
	if ((aircraft[NumAirCraft] = (AirCraft *) calloc(1, sizeof(AirCraft))) !=
		NULL)
		return NumAirCraft++;

	return -1;
}

AirCraft* Farsite5::GetAirCraft(long AirCraftNumber)
{
	if (aircraft[AirCraftNumber])
		return aircraft[AirCraftNumber];

	return 0;
}

long Farsite5::SetAirCraft(long AirCraftNumber)
{
	if ((aircraft[AirCraftNumber] = (AirCraft *) calloc(1, sizeof(AirCraft))) !=
		NULL)
		return NumAirCraft++;

	return -1;
}

void Farsite5::FreeAirCraft(long AirCraftNumber)
{
	if (aircraft[AirCraftNumber])
	{
		free(aircraft[AirCraftNumber]);
		NumAirCraft--;
		if (NumAirCraft < 0)
			NumAirCraft = 0;
	}

	aircraft[AirCraftNumber] = 0;
}

long Farsite5::GetNumAirAttacks()
{
	return NumAirAttacks;
}


void Farsite5::SetNumAirAttacks(long NumAtk)
{
	NumAirAttacks = NumAtk;
}

long Farsite5::SetupAirAttack(long AirCraftNumber, long CoverageLevel, long Duration,
	double* startpoint)
{
	long i;

	for (i = 0; i <= NumAirAttacks; i++)
	{
		if (NumAirAttacks == 0)
		{
			if ((FirstAirAttack = (AirAttackData *) calloc(1,
														sizeof(AirAttackData))) !=
				NULL)
			{
				CurAirAttack = FirstAirAttack;
				if (AirAttackCounter == 0)
				{
					memset(AirAttackLog, 0x0, sizeof(AirAttackLog));
					//getcwd(AirAttackLog, 255);
					strcat(AirAttackLog, "airattk.log");
					//DeleteFile(AirAttackLog);
					remove(AirAttackLog);
				}
			}
			else
				return 0;
		}
		else if (i == 0)
			CurAirAttack = FirstAirAttack;
		else
			CurAirAttack = NextAirAttack;
		if (i < NumAirAttacks)
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
	}

	if ((NextAirAttack = (AirAttackData *) calloc(1, sizeof(AirAttackData))) !=
		NULL)
	{
		NumAirAttacks++;
		CurAirAttack->next = (AirAttackData *) NextAirAttack;
		CurAirAttack->AirAttackNumber = ++AirAttackCounter;
		CurAirAttack->EffectiveDuration = Duration;
		CurAirAttack->CoverageLevel = CoverageLevel;
		CurAirAttack->AirCraftNumber = AirCraftNumber;
		CurAirAttack->ElapsedTime = 0.0;

		if (startpoint)
		{
			double InitDist, MinDist, Mult = 1.0, Ratio;
			double NewEndPtX, NewEndPtY, XDist, YDist;
			double xpt, ypt, xmax, ymax, xmin, ymin;
			double DistRes, PerimRes, PtX, PtY;

			DistRes = GetDistRes() / 1.4 * MetricResolutionConvert();
			PerimRes = GetDistRes() / 2.0 * MetricResolutionConvert();//GetPerimRes()/2.0;
			if (aircraft[AirCraftNumber]->Units == 1)
				Mult = 0.3048;  	  // feet to meters
			XDist = startpoint[0] - startpoint[2];
			YDist = startpoint[1] - startpoint[3];
			InitDist = sqrt(pow2(XDist) + pow2(YDist));
			MinDist = aircraft[AirCraftNumber]->PatternLength[CoverageLevel] * Mult * MetricResolutionConvert();
			Ratio = InitDist / MinDist;
			NewEndPtX = startpoint[0] - (XDist) / Ratio;
			NewEndPtY = startpoint[1] - (YDist) / Ratio;

			Ratio = InitDist / (MinDist + PerimRes);
			PtX = startpoint[0] - (XDist) / Ratio;
			PtY = startpoint[1] - (YDist) / Ratio;

			XDist = startpoint[0] - NewEndPtX;
			YDist = startpoint[1] - NewEndPtY;

			CurAirAttack->PatternNumber = GetNewFires();
			AllocPerimeter1(CurAirAttack->PatternNumber, 6);
			SetInout(CurAirAttack->PatternNumber, 3);
			SetNumPoints(CurAirAttack->PatternNumber, 5);

			xpt = xmin = xmax = startpoint[0] + DistRes / MinDist * YDist;
			ypt = ymin = ymax = startpoint[1] - DistRes / MinDist * XDist;
			SetPerimeter1(GetNewFires(), 0, xpt, ypt);
			SetFireChx(GetNewFires(), 0, 0, -1);

			xpt = startpoint[0] - DistRes / MinDist * YDist;
			ypt = startpoint[1] + DistRes / MinDist * XDist;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 1, xpt, ypt);
			SetFireChx(GetNewFires(), 1, 0, -1);

			xpt = NewEndPtX - DistRes / MinDist * YDist;
			ypt = NewEndPtY + DistRes / MinDist * XDist;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 2, xpt, ypt);
			SetFireChx(GetNewFires(), 2, 0, -1);

			xpt = PtX;
			ypt = PtY;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 3, xpt, ypt);
			SetFireChx(GetNewFires(), 3, 0, -1);

			xpt = NewEndPtX + DistRes / MinDist * YDist;
			ypt = NewEndPtY - DistRes / MinDist * XDist;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 4, xpt, ypt);
			SetFireChx(GetNewFires(), 4, 0, -1);

			SetPerimeter1(GetNewFires(), 5, xmin, xmax);
			SetFireChx(GetNewFires(), 5, ymin, ymax);
			IncNewFires(1);
			IncNumFires(1);
		}
	}
	else
		return 0;

	return AirAttackCounter;
}


void Farsite5::CancelAirAttack(long AirAttackCounter)
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		if (CurAirAttack->AirAttackNumber == AirAttackCounter)
		{
			if (i == 0)
				FirstAirAttack = (AirAttackData *) CurAirAttack->next;
			else
				LastAirAttack->next = (AirAttackData *) NextAirAttack;
			WriteAirAttackLog(CurAirAttack);
			free(CurAirAttack);
			NumAirAttacks--;
			if (NumAirAttacks == 0)
				free(NextAirAttack);
			break;
		}
		else
		{
			LastAirAttack = CurAirAttack;
			CurAirAttack = NextAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
	}
}

void Farsite5::LoadAirAttack(AirAttackData airattackdata)
{
	// function only for loading air attacks from bookmark
	if (NumAirAttacks == 0)
	{
		FirstAirAttack = (AirAttackData *) calloc(1, sizeof(AirAttackData));
		CurAirAttack = FirstAirAttack;
		memcpy(FirstAirAttack, &airattackdata, sizeof(AirAttackData));
	}
	memcpy(CurAirAttack, &airattackdata, sizeof(AirAttackData));
	NextAirAttack = (AirAttackData *) calloc(1, sizeof(AirAttackData));
	CurAirAttack->next = (AirAttackData *) NextAirAttack;
	if (NumAirAttacks == 0)
		FirstAirAttack->next = (AirAttackData *) NextAirAttack;
	NumAirAttacks++;
	CurAirAttack = NextAirAttack;
}

void Farsite5::FreeAllAirAttacks()
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		WriteAirAttackLog(CurAirAttack);
		free(CurAirAttack);
		CurAirAttack = NextAirAttack;
		NextAirAttack = (AirAttackData *) CurAirAttack->next;
	}
	if (NumAirAttacks > 0)
	{
		free(CurAirAttack);
		NumAirAttacks = 0;
	}
	AirAttackCounter = 0;
}

AirAttackData* Farsite5::GetAirAttack(long AirAttackCounter)
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		if (CurAirAttack->AirAttackNumber == AirAttackCounter)
			return CurAirAttack;
		else
		{
			CurAirAttack = NextAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
	}

	return NULL;
}

AirAttackData* Farsite5::GetAirAttackByOrder(long OrdinalAttackNum)
{
	// retrieves indirect attack in order
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
			CurAirAttack = FirstAirAttack;
		else
			CurAirAttack = NextAirAttack;
		NextAirAttack = (AirAttackData *) CurAirAttack->next;
		if (i == OrdinalAttackNum)
			return CurAirAttack;
	}

	return 0;
}


void Farsite5::SetNewFireNumberForAirAttack(long oldnumfire, long newnumfire)
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		if (CurAirAttack->PatternNumber == oldnumfire)
		{
			CurAirAttack->PatternNumber = newnumfire;

			break;
		}
		else
		{
			CurAirAttack = NextAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
	}
}


void Farsite5::WriteAirAttackLog(AirAttackData* atk)
{
	FILE* airatklog;
	char units[24];
	long covlevel;

	if ((airatklog = fopen(AirAttackLog, "a")) != NULL)
	{
		if (aircraft[atk->AirCraftNumber]->Units == 0)
			sprintf(units, "meters");
		else
			sprintf(units, "feet");

		if (atk->CoverageLevel == 4)
			covlevel = 6;
		else if (atk->CoverageLevel == 5)
			covlevel = 8;
		else
			covlevel = atk->CoverageLevel + 1;

		if (atk->EffectiveDuration > 0.0)
			fprintf(airatklog, "%s, %s %ld, %s %lf %s, %s %ld %s\n",
				aircraft[atk->AirCraftNumber]->AirCraftName,
				"Coverage Level:", covlevel, "Line Length:",
				aircraft[atk->AirCraftNumber]->PatternLength[atk->CoverageLevel],
				units, "Duration:", (long)
				atk->EffectiveDuration, "mins");
		else
			fprintf(airatklog, "%s, %s %ld, %s %lf %s, %s\n",
				aircraft[atk->AirCraftNumber]->AirCraftName,
				"Coverage Level:", covlevel, "Line Length:",
				aircraft[atk->AirCraftNumber]->PatternLength[atk->CoverageLevel],
				units, "Duration: Unlimited");

		fclose(airatklog);
	}
}
long Farsite5::SetupGroupAirAttack(long AircraftNumber, long CoverageLevel,
	long Duration, double* line, long FireNum, char* GroupName)
{
	long i;
	GroupAirAttack* gp;

	for (i = 0; i <= NumGroupAttacks; i++)
	{
		if (NumGroupAttacks == 0)
		{
			if ((FirstGroupAttack = (GroupAttackData *)
				calloc(1,
					sizeof(GroupAttackData))) !=
				NULL)
				CurGroupAttack = FirstGroupAttack;
			else
				return 0;
		}
		else if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		else
			CurGroupAttack = NextGroupAttack;
		if (i < NumGroupAttacks)
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
	}

	if ((NextGroupAttack = (GroupAttackData *) calloc(1,
												sizeof(GroupAttackData))) !=
		NULL)
	{
		NumGroupAttacks++;
		CurGroupAttack->next = (GroupAttackData *) NextGroupAttack;
		CurGroupAttack->GroupAttackNumber = ++GroupAttackCounter;
		strcpy(CurGroupAttack->GroupName, GroupName);
		CurGroupAttack->Suspended = 0;	// false
		gp = new GroupAirAttack(CurGroupAttack, this);
		gp->AllocGroup(GROUPSIZE);
		gp->AddGroupMember(AircraftNumber, CoverageLevel, Duration);
		gp->SetGroupAssignment(line, FireNum, false);
		//gp->ExecuteAttacks(0.0);
		delete gp;
	}

	return GroupAttackCounter;
}


GroupAttackData* Farsite5::ReassignGroupAirAttack(GroupAttackData* atk)
{
	if (atk != 0)
		ReAtk = atk;

	return ReAtk;
}


void Farsite5::CancelGroupAirAttack(long GroupCounter)
{
	long i;

	for (i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
		{
			CurGroupAttack = FirstGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
		if (CurGroupAttack->GroupAttackNumber == GroupCounter)
		{
			if (i == 0)
				FirstGroupAttack = (GroupAttackData *) CurGroupAttack->next;
			else
				LastGroupAttack->next = (GroupAttackData *) NextGroupAttack;
			if (CurGroupAttack->IndirectLine)
				free(CurGroupAttack->IndirectLine);
			free(CurGroupAttack->WaitTime);
			free(CurGroupAttack->CoverageLevel);
			free(CurGroupAttack->EffectiveDuration);
			free(CurGroupAttack->AircraftNumber);
			free(CurGroupAttack);
			NumGroupAttacks--;
			if (NumGroupAttacks == 0)
				free(NextGroupAttack);
			break;
		}
		else
		{
			LastGroupAttack = CurGroupAttack;
			CurGroupAttack = NextGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
	}
}


void Farsite5::LoadGroupAttack(GroupAttackData gatk)
{
	if (gatk.NumCurrentAircraft <= 0)
		return;

	long i;
	for (i = 0; i <= NumGroupAttacks; i++)
	{
		if (NumGroupAttacks == 0)
		{
			if ((FirstGroupAttack = (GroupAttackData *)
				calloc(1,
					sizeof(GroupAttackData))) !=
				NULL)
				CurGroupAttack = FirstGroupAttack;
			else
				return;
		}
		else if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		else
			CurGroupAttack = NextGroupAttack;
		if (i < NumGroupAttacks)
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
	}

	if ((NextGroupAttack = (GroupAttackData *) calloc(1,
												sizeof(GroupAttackData))) !=
		NULL)
	{
		NumGroupAttacks++;
		CurGroupAttack->next = (GroupAttackData *) NextGroupAttack;
		CurGroupAttack->GroupAttackNumber = ++GroupAttackCounter;
		strcpy(CurGroupAttack->GroupName, gatk.GroupName);
		CurGroupAttack->Suspended = 0;	// false
		if (gatk.NumPoints < 0)
		{
			CurGroupAttack->IndirectLine = (double *)
				calloc(4, sizeof(double));
			memcpy(CurGroupAttack->IndirectLine, gatk.IndirectLine,
				4 * sizeof(double));
		}
		else
		{
			CurGroupAttack->IndirectLine = (double *)
				calloc(gatk.NumPoints * 2,
														sizeof(double));
			memcpy(CurGroupAttack->IndirectLine, gatk.IndirectLine,
				gatk.NumPoints * 2 * sizeof(double));
		}
		CurGroupAttack->CoverageLevel = (long *) calloc(gatk.NumTotalAircraft,
													sizeof(long));
		CurGroupAttack->EffectiveDuration = (long *)
			calloc(gatk.NumTotalAircraft,
														sizeof(long));
		CurGroupAttack->AircraftNumber = (long *)
			calloc(gatk.NumTotalAircraft,
													sizeof(long));
		CurGroupAttack->WaitTime = (double *) calloc(gatk.NumTotalAircraft,
												sizeof(double));
		memcpy(CurGroupAttack->CoverageLevel, gatk.CoverageLevel,
			gatk.NumCurrentAircraft * sizeof(long));
		memcpy(CurGroupAttack->EffectiveDuration, gatk.EffectiveDuration,
			gatk.NumCurrentAircraft * sizeof(long));
		memcpy(CurGroupAttack->AircraftNumber, gatk.AircraftNumber,
			gatk.NumCurrentAircraft * sizeof(long));
		memcpy(CurGroupAttack->WaitTime, gatk.WaitTime,
			gatk.NumCurrentAircraft * sizeof(double));
		CurGroupAttack->NumTotalAircraft = gatk.NumTotalAircraft;
		CurGroupAttack->NumCurrentAircraft = gatk.NumCurrentAircraft;
		CurGroupAttack->Suspended = gatk.Suspended;
		memcpy(CurGroupAttack->GroupName, gatk.GroupName,
			256 * sizeof(char));
		CurGroupAttack->NumPoints = gatk.NumPoints;
		CurGroupAttack->FireNumber = gatk.FireNumber;
	}
}


void Farsite5::FreeAllGroupAirAttacks()
{
	long i;

	for (i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
		{
			CurGroupAttack = FirstGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
		if (CurGroupAttack->IndirectLine)
			free(CurGroupAttack->IndirectLine);
		free(CurGroupAttack->CoverageLevel);
		free(CurGroupAttack->EffectiveDuration);
		free(CurGroupAttack->AircraftNumber);
		free(CurGroupAttack->WaitTime);
		free(CurGroupAttack);
		CurGroupAttack = NextGroupAttack;
		NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
	}
	if (NumGroupAttacks > 0)
		free(CurGroupAttack);

	NumGroupAttacks = 0;
	GroupAttackCounter = 0;
}

GroupAttackData* Farsite5::GetGroupAirAttack(long GroupAttackCounter)
{
	for (long i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
		{
			CurGroupAttack = FirstGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
		if (CurGroupAttack->GroupAttackNumber == GroupAttackCounter)
			return CurGroupAttack;
		else
		{
			CurGroupAttack = NextGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
	}

	return NULL;
}

GroupAttackData* Farsite5::GetGroupAttackByOrder(long OrdinalAttackNumber)
{
	for (long i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		else
			CurGroupAttack = NextGroupAttack;
		NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		if (i == OrdinalAttackNumber)
			return CurGroupAttack;
	}

	return 0;
}


GroupAttackData* Farsite5::GetGroupAttackForFireNumber(long NumFire,
	long StartAttackNum, long* LastAttackNumber)
{
	for (long i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		if (i >= StartAttackNum)
		{
			if (CurGroupAttack->FireNumber == NumFire)// && CurAttack->Suspended==0)
			{
				*LastAttackNumber = i;

				return CurGroupAttack;
			}
		}
		CurGroupAttack = NextGroupAttack;
	}

	return 0;
}

void Farsite5::SetNewFireNumberForGroupAttack(long OldFireNum, long NewFireNum)
{
	long ThisGroup, NextGroup;

	ThisGroup = 0;
	while (GetGroupAttackForFireNumber(OldFireNum, ThisGroup, &NextGroup))
	{
		ThisGroup = NextGroup + 1;
		CurGroupAttack->FireNumber = NewFireNum;
	}
}

long Farsite5::GetNumGroupAttacks()
{
	return NumGroupAttacks;
}

void Farsite5::SetNumGroupAttacks(long NumGroups)
{
	NumGroupAttacks = NumGroups;
}

long Farsite5::GetCanopySpecies()
{
	return CanopyChx.Species;
}

double Farsite5::GetAverageDBH()
{
	return CanopyChx.Diameter;
}

FireRing* Farsite5::AllocFireRing(long NumPoints, double start, double end)
{
	long curplace, i, ThisRing;
	double Ring;

	modf(((double) NumRings / (double) RINGS_PER_STRUCT), &Ring);
	ThisRing = (long) Ring;
	curplace = NumRings - ThisRing * RINGS_PER_STRUCT;

	if (FirstRing == 0)
	{
		if ((FirstRing = new RingStruct) == NULL)
			return 0;
		CurRing = FirstRing;
		CurRing->NumFireRings = 0;
		CurRing->StructNum = 0;
		NumRingStructs++;
		for (i = 0; i < RINGS_PER_STRUCT; i++)
		{
			CurRing->firering[i].perimpoints = 0;
			CurRing->firering[i].mergepoints = 0;
			CurRing->firering[i].NumPoints = 0;
		}
	}
	else if (ThisRing >= NumRingStructs)
	{
		if (CurRing->StructNum != NumRingStructs - 1)
			GetLastRingStruct();
		//NextRing = (RingStruct *) CurRing->next = new RingStruct;
		RingStruct* t_next;
		t_next = new RingStruct;
		CurRing->next=t_next;
		//NextRing=CurRing->next;
		NextRing = t_next;
		CurRing = NextRing;
		CurRing->NumFireRings = 0;
		CurRing->StructNum = NumRingStructs;
		NumRingStructs++;
		for (i = 0; i < RINGS_PER_STRUCT; i++)
		{
			CurRing->firering[i].perimpoints = 0;
			CurRing->firering[i].mergepoints = 0;
			CurRing->firering[i].NumPoints = 0;
		}
	}
	else if (ThisRing != CurRing->StructNum)
		GetRing(NumRings);
	CurRing->firering[curplace].perimpoints = new PerimPoints[NumPoints];
	for (i = 0; i < NumPoints; i++)
//		ZeroMemory(&CurRing->firering[curplace].perimpoints[i],	sizeof(PerimPoints));
		memset(&CurRing->firering[curplace].perimpoints[i],0x0,	sizeof(PerimPoints));
	CurRing->firering[curplace].NumPoints = new long[2];
	CurRing->firering[curplace].NumPoints[0] = NumPoints;
	CurRing->firering[curplace].NumFires = 1;
	CurRing->firering[curplace].StartTime = start;
	CurRing->firering[curplace].ElapsedTime = end - start;
	CurRing->firering[curplace].mergepoints = 0;
	CurRing->firering[curplace].NumMergePoints[0] = 0;
	CurRing->firering[curplace].NumMergePoints[1] = 0;

	CurRing->NumFireRings++;
	NumRings++;

	return &CurRing->firering[curplace];
}


void Farsite5::GetLastRingStruct()
{
	long i;

	CurRing = FirstRing;
	for (i = 0; i < NumRingStructs - 1; i++)
	{
		NextRing = (RingStruct *) CurRing->next;
		CurRing = NextRing;
	}
}


void Farsite5::FreeAllFireRings()
{
	long i, j;

	CurRing = FirstRing;
	for (i = 0; i < NumRingStructs; i++)
	{
		if (CurRing != NULL)
		{
			for (j = 0; j < RINGS_PER_STRUCT; j++)
			{
				if (CurRing->firering[j].perimpoints)
				{
					delete[] CurRing->firering[j].perimpoints;
					if (CurRing->firering[j].NumPoints)
						delete[] CurRing->firering[j].NumPoints;
					if (CurRing->firering[j].mergepoints)
						delete[] CurRing->firering[j].mergepoints;
					CurRing->firering[j].perimpoints = 0;
					CurRing->firering[j].NumPoints = 0;
					CurRing->firering[j].mergepoints = 0;
				}
			}
		}
		NextRing = (RingStruct *) CurRing->next;
		delete CurRing;
		CurRing = NextRing;
	}
	NumRings = 0;
	NumRingStructs = 0;
	FirstRing = 0;
	NextRing = 0;
	CurRing = 0;
}


void Farsite5::FreeFireRing(long RingNum)
{
	long i, curplace, ThisRing;
	double Ring;

	modf(((double) RingNum / (double) RINGS_PER_STRUCT), &Ring);
	ThisRing = (long) Ring;
	curplace = RingNum - ThisRing * RINGS_PER_STRUCT;

	if (CurRing->StructNum != ThisRing)
	{
		CurRing = FirstRing;
		for (i = 0; i < ThisRing; i++)
		{
			NextRing = (RingStruct *) CurRing->next;
			CurRing = NextRing;
		}
	}
	if (CurRing->firering[curplace].perimpoints)
	{
		try
		{
			delete[] CurRing->firering[curplace].perimpoints;
			if (CurRing->firering[curplace].NumPoints)
				delete[] CurRing->firering[curplace].NumPoints;
			CurRing->firering[curplace].perimpoints = 0;
			CurRing->firering[curplace].NumPoints = 0;
			if (CurRing->firering[curplace].mergepoints)
				delete[] CurRing->firering[curplace].mergepoints;
			CurRing->firering[curplace].mergepoints = 0;
			CurRing->firering[curplace].OriginalFireNumber = -(CurRing->firering[curplace].OriginalFireNumber +
				1);
		}
		catch (...)
		{
		}
	}
}


FireRing* Farsite5::GetRing(long RingNum)
{
	long i, curplace, ThisRing;
	double Ring;

	if (RingNum < 0) //original JAS!
	//if (RingNum <= 0)   //Modified JAS!
		return (FireRing *) NULL;

	modf(((double) RingNum / (double) RINGS_PER_STRUCT), &Ring);
	ThisRing = (long) Ring;
	curplace = RingNum - ThisRing * RINGS_PER_STRUCT;//-1;
        CurRing->StructNum = 0;  //added JAS!

        if (CurRing->StructNum != ThisRing)
	{
		CurRing = FirstRing;
		for (i = 0; i < ThisRing; i++)
		{
			NextRing = (RingStruct *) CurRing->next;
			CurRing = NextRing;
		}
	}
	if (CurRing == NULL)
		return NULL;

	return &CurRing->firering[curplace];
}


FireRing* Farsite5::GetSpecificRing(long FireNumber, double StartTime)
{
	long i, j;

	CurRing = FirstRing;
	j = 0;
	for (i = 0; i < NumRings; i++)
	{
		if (fabs(CurRing->firering[j].StartTime - StartTime) < 1e-9)
		{
			if (CurRing->firering[j].OriginalFireNumber == FireNumber)
				break;
		}
		else
			continue;
		j++;
		if (j == RINGS_PER_STRUCT)
		{
			NextRing = (RingStruct *) CurRing->next;
			CurRing = NextRing;
			j = 0;
		}
	}

	return &CurRing->firering[j];
}


void Farsite5::CondenseRings(long RingNum)
{
	// searches starting with RingNum for FireRings with no points.  Then it shifts
	// the pointers down to the earliest vacant slot.  This function is called after
	// all mergers are completed.

	if (RingNum < 0)
		return;
	else if (RingNum == GetNumRings() - 1)
		return;

	FireRing* ring1, * ring2;
	long i, j, TotalPts;
	long NewRingNum;//=GetNumRings();
	long NewStructNum;

	for (i = RingNum; i < GetNumRings(); i++)
	{
		ring1 = GetRing(i);
		if (!ring1)
			continue;
		if (ring1->mergepoints)
		{
			delete[] ring1->mergepoints;
			ring1->mergepoints = 0;
			ring1->NumMergePoints[0] = 0;
		}
		if (ring1->perimpoints == NULL)
		{
			FreeFireRing(i);
			TotalPts = 0;
			for (j = i + 1; j < GetNumRings(); j++)
			{
				ring2 = GetRing(j);
				if (!ring2)
					continue;
				if (ring2->perimpoints)
				{
					TotalPts = ring2->NumPoints[ring2->NumFires - 1];
					ring1->perimpoints = ring2->perimpoints;
					ring1->NumPoints = ring2->NumPoints;
					ring2->perimpoints = 0;
					ring2->NumPoints = 0;
					ring1->NumFires = ring2->NumFires;
					ring1->OriginalFireNumber = ring2->OriginalFireNumber;
					ring1->StartTime = ring2->StartTime;
					ring2->OriginalFireNumber = (long)0;
					ring2->StartTime = 0.0;
					ring1->ElapsedTime = ring2->ElapsedTime;
					ring2->NumFires = 0;
					ring2->OriginalFireNumber = -1;

					break;
				}
			}
		}
	}

	// free up CurRing if no points left!!!!!
	RingStruct* LastRing;
	CurRing = LastRing = FirstRing;
	NewStructNum = 0;
	for (i = 0; i < NumRingStructs; i++)
	{
		if (CurRing != NULL)
		{
			NewRingNum = 0;
			for (j = 0; j < RINGS_PER_STRUCT; j++)
			{
				if (CurRing->firering[j].perimpoints != NULL)
					NewRingNum++;
			}
			CurRing->NumFireRings = NewRingNum;
			if (NewRingNum > 0)
				NewStructNum++;
			else
			{
				NextRing = (RingStruct *) CurRing->next;
				LastRing->next = NextRing;
				if (CurRing == LastRing)		 // will only occur if there are no rings now
				{
					delete FirstRing;
					FirstRing = LastRing = 0;
				}
				else
					delete CurRing;
				CurRing = LastRing;
			}
		}
		if (CurRing == NULL)
			break;
		LastRing = CurRing;
		NextRing = (RingStruct *) CurRing->next;
		CurRing = NextRing;
	}
	CurRing = FirstRing;
	NewRingNum = 0;
	for (i = 0; i < NewStructNum; i++)
	{
		NewRingNum += CurRing->NumFireRings;
		NextRing = (RingStruct *) CurRing->next;
		CurRing = NextRing;
	}
	SetNumRings(NewRingNum);//(NewStructNum-1)*RINGS_PER_STRUCT+NewRingNum);
	NumRingStructs = NewStructNum;
	CurRing = FirstRing;
}


void Farsite5::SetNewFireNumber(long OldNum, long NewNum, long RefNum)
{
	bool found = false;
	long i;
	FireRing* ring;

	for (i = RefNum; i < GetNumRings(); i++)
	{
		ring = GetRing(i);
		if (OldNum == ring->OriginalFireNumber)
		{
			ring->OriginalFireNumber = NewNum;
			found = true;
			break;
		}
	}
	if (!found) 	   // for debugging
		found = true;
}


void Farsite5::SetNumRings(long NewNumRings)
{
	NumRings = NewNumRings;
}

long Farsite5::GetNumRings()
{
	return NumRings;
}

bool Farsite5::AddToCurrentFireRing(FireRing* firering, long PointNum,
	long SurfFuelType, long WoodyModel, double DuffLoad, double* mx,
	double CrownLoadingBurned)
{
	if (!firering)
		return false;

	long i, NumFire;
	double x, y, r, f, c;

	NumFire = firering->OriginalFireNumber;
	f = GetPerimeter2Value(PointNum, FLIVAL);
	if (f < 0.0)
	{
		x = GetPerimeter2Value(PointNum, XCOORD);
		y = GetPerimeter2Value(PointNum, YCOORD);
		r = fabs(GetPerimeter2Value(PointNum, ROSVAL));
		f = fabs(f);
		c = GetPerimeter2Value(PointNum, RCXVAL);
	}
	else
	{
		x = GetPerimeter1Value(NumFire, PointNum, XCOORD);
		y = GetPerimeter1Value(NumFire, PointNum, YCOORD);
		r = fabs(GetPerimeter1Value(NumFire, PointNum, ROSVAL));
		f = fabs(GetPerimeter1Value(NumFire, PointNum, FLIVAL));
		c = GetPerimeter1Value(NumFire, PointNum, RCXVAL);
	}
	firering->perimpoints[PointNum].x2 = x;
	firering->perimpoints[PointNum].y2 = y;
	if (c < 1e-2)
		c = 0.0;
	firering->perimpoints[PointNum].hist.ReactionIntensity[1] = (float) c;
	if (fabs(r) > 0.0 && c > 0.0)
		firering->perimpoints[PointNum].hist.FlameResidenceTime[1] = (float)
			((f * 60.0) / (r * c)); // seconds
	else
		firering->perimpoints[PointNum].hist.FlameResidenceTime[1] = 0.0;

	firering->perimpoints[PointNum].hist.WoodyFuelType = (unsigned char)
		WoodyModel;
	firering->perimpoints[PointNum].hist.SurfaceFuelModel = (unsigned char)
		SurfFuelType;
	if (DuffLoad < 0.1)
		DuffLoad = 0.1;
	firering->perimpoints[PointNum].hist.DuffLoad = (short) (DuffLoad * 10.0);
	//mx=GetAllCurrentMoistures(&NumMx);

	if (mx)
	{
		for (i = 0; i < MAXNO; i++)
			firering->perimpoints[PointNum].hist.Moistures[i] = (unsigned char)
				(mx[i] * 100.0);
	}
	//for(i=NumMx-1; i<20; i++)
	//	firering->perimpoints[PointNum].hist.Moistures[i]=0;
	firering->perimpoints[PointNum].hist.LastWtRemoved = 0.0;	// initialize last values of integration
	firering->perimpoints[PointNum].hist.WeightPolyNum = 0;
	firering->perimpoints[PointNum].hist.FlamePolyNum = 0;
	firering->perimpoints[PointNum].hist.CrownLoadingBurned = (float)
		CrownLoadingBurned;

	return true;
}



bool Farsite5::AllocCoarseWoody(long ModelNumber, long NumClasses)
{
	if (coarsewoody[ModelNumber - 1].wd)
		FreeCoarseWoody(ModelNumber);
	if ((coarsewoody[ModelNumber - 1].wd = new WoodyData[NumClasses]) == NULL) //(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NumClasses*sizeof(WoodyData)))==NULL)
		return false;
//	ZeroMemory(coarsewoody[ModelNumber - 1].wd, NumClasses * sizeof(WoodyData));
	memset(coarsewoody[ModelNumber - 1].wd,0x0, NumClasses * sizeof(WoodyData));
	coarsewoody[ModelNumber - 1].NumClasses = NumClasses;
	coarsewoody[ModelNumber - 1].TotalWeight = 0.0;
//	ZeroMemory(coarsewoody[ModelNumber - 1].Description, 64 * sizeof(char));
	memset(coarsewoody[ModelNumber - 1].Description,0x0, 64 * sizeof(char));

	return true;
}


void Farsite5::FreeCoarseWoody(long ModelNumber)
{
	if (coarsewoody[ModelNumber - 1].wd)
		delete[] coarsewoody[ModelNumber - 1].wd;//GlobalFree(coarsewoody[ModelNumber-1].wd);//free(coarsewoody[ModelNumber-1].wd);
	coarsewoody[ModelNumber - 1].NumClasses = 0;
	coarsewoody[ModelNumber - 1].wd = 0;
	coarsewoody[ModelNumber - 1].Units = -1;
	coarsewoody[ModelNumber - 1].TotalWeight = 0.0;
	coarsewoody[ModelNumber - 1].Depth = 0.0;
}

void Farsite5::FreeAllCoarseWoody()
{
	long i;

	for (i = 1; i < MAXNUM_COARSEWOODY_MODELS + 1; i++)
		FreeCoarseWoody(i);
	for (i = 0; i < 256; i++)
	{
		if (NFFLWoody[i].wd)
			delete[] NFFLWoody[i].wd;//GlobalFree(NFFLWoody[i].wd);//free(NFFLWoody[i].wd);
		NFFLWoody[i].NumClasses = 0;
		NFFLWoody[i].wd = 0;
	}
	if (tempwoody.wd)
		delete[] tempwoody.wd;//GlobalFree(tempwoody.wd);//free(tempwoody.wd);
}

bool Farsite5::SetWoodyData(long ModelNumber, long ClassNumber, WoodyData* wd,
	long units)
{
	if (ModelNumber > MAXNUM_COARSEWOODY_MODELS - 1)
		return false;

	if (coarsewoody[ModelNumber - 1].NumClasses <= ClassNumber)
		return false;

	//long i;
	double Total = 0.0;

	//coarsewoody[ModelNumber-1].TotalWeight=0.0;
	memcpy(&coarsewoody[ModelNumber - 1].wd[ClassNumber], wd,
		sizeof(WoodyData));
	coarsewoody[ModelNumber - 1].Units = units;
	//for(i=0; i<=ClassNumber; i++)
	Total = coarsewoody[ModelNumber - 1].TotalWeight + wd[0].Load;//coarsewoody[ModelNumber-1].wd[i];
	coarsewoody[ModelNumber - 1].TotalWeight = Total;
	SetNFFLWoody();
	EnvtChanged[3][0] = true;
	EnvtChanged[3][1] = true;
	EnvtChanged[3][2] = true;
	EnvtChanged[3][3] = true;
	EnvtChanged[3][4] = true;

	return true;
}

bool Farsite5::SetWoodyDataDepth(long ModelNumber, double depth, const char* Description)
{
	if (ModelNumber > MAXNUM_COARSEWOODY_MODELS - 1)
		return false;

	coarsewoody[ModelNumber - 1].Depth = depth;
	memcpy(coarsewoody[ModelNumber - 1].Description, Description,
		64 * sizeof(char));

	return true;
}


bool Farsite5::SetNFFLWoody()
{
	//long i, j, k;
	long i, k;

	if (NFFLWoody[0].wd)
		return true;

	for (i = 0; i < 256; i++)
	{
		if (NFFLWoody[i].wd)
			delete[] NFFLWoody[i].wd;
		NFFLWoody[i].NumClasses = 0;
		NFFLWoody[i].wd = 0;
	}

	for (i = 0; i < 256; i++)
	{
		if (newfuels[i + 1].number > 0)
		{
			NFFLWoody[i].NumClasses = 0;
			if (newfuels[i + 1].h1 > 0.0)
				NFFLWoody[i].NumClasses++;
			if (newfuels[i + 1].h10 > 0.0)
				NFFLWoody[i].NumClasses++;
			if (newfuels[i + 1].h100 > 0.0)
				NFFLWoody[i].NumClasses++;
			if (newfuels[i + 1].lh > 0.0)
				NFFLWoody[i].NumClasses++;
			if (newfuels[i + 1].lw > 0.0)
				NFFLWoody[i].NumClasses++;
			if ((NFFLWoody[i].wd = new WoodyData[NFFLWoody[i].NumClasses]) ==
				NULL)
				return false;
			//ZeroMemory(NFFLWoody[i].wd,NFFLWoody[i].NumClasses * sizeof(WoodyData));
                       	memset(NFFLWoody[i].wd,0x0,NFFLWoody[i].NumClasses * sizeof(WoodyData));
			k = 0;
			if (newfuels[i + 1].h1 > 0.0)
			{
				NFFLWoody[i].wd[k].Load = newfuels[i + 1].h1;
				NFFLWoody[i].wd[k++].Load = newfuels[i + 1].sav1;
			}
			if (newfuels[i + 1].h10 > 0.0)
			{
				NFFLWoody[i].wd[k].Load = newfuels[i + 1].h10;
				NFFLWoody[i].wd[k++].Load = 109.0;
			}
			if (newfuels[i + 1].h100 > 0.0)
			{
				NFFLWoody[i].wd[k].Load = newfuels[i + 1].h100;
				NFFLWoody[i].wd[k++].Load = 30.0;
			}
			if (newfuels[i + 1].lh > 0.0)
			{
				NFFLWoody[i].wd[k].Load = newfuels[i + 1].lh;
				NFFLWoody[i].wd[k++].Load = newfuels[i + 1].savlh;
			}
			if (newfuels[i + 1].lw > 0.0)
			{
				NFFLWoody[i].wd[k].Load = newfuels[i + 1].lw;
				NFFLWoody[i].wd[k++].Load = newfuels[i + 1].savlw;
			}
		}
		else
		{
			NFFLWoody[i].NumClasses = 0;
			NFFLWoody[i].Depth = 0.0;
			NFFLWoody[i].TotalWeight = 0.0;
			NFFLWoody[i].Units = 1;
			NFFLWoody[i].wd = 0;
		}
	}


	//	if((tempwoody.wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, 20*sizeof(WoodyData)))==NULL)
	//     	return false;

	/*
	// fuel model 1
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=1.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[i].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
		 if((NFFLWoody[i].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[i].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[i].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=3500.0;
		 NFFLWoody[i].wd[0].Load=0.74;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=0.0;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=0.0;
	// fuel model 2
		NFFLWoody[i].NumClasses=4;
		 NFFLWoody[i].Depth=1.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[1].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
		 if((NFFLWoody[1].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
	//     if((NFFLWoody[1].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
//		 ZeroMemory(NFFLWoody[1].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[1].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=3000.0;
		 NFFLWoody[i].wd[0].Load=2.00;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=1.00;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i].wd[2].Load=0.50;
		 NFFLWoody[i].wd[3].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i++].wd[3].Load=0.50;
	// fuel model 3
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=2.5;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[2].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[2].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
		 if((NFFLWoody[2].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[2].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[2].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i].wd[0].Load=3.01;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=0.0;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=0.0;
	// fuel model 4
		NFFLWoody[i].NumClasses=4;
		 NFFLWoody[i].Depth=6.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[3].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[3].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[3].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[3].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[3].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=2000.0;
		 NFFLWoody[i].wd[0].Load=5.01;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=4.01;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i].wd[2].Load=2.00;
		 NFFLWoody[i].wd[3].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i++].wd[3].Load=5.01;
	// fuel model 5
		NFFLWoody[i].NumClasses=4;
		 NFFLWoody[i].Depth=2.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[4].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[4].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[4].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[4].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[4].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=2000.0;
		 NFFLWoody[i].wd[0].Load=1.00;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=0.5;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i].wd[2].Load=0.00;
		 NFFLWoody[i].wd[3].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i++].wd[3].Load=2.00;
	// fuel model 6
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=2.5;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[5].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[5].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[5].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[5].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[5].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=1750.0;
		 NFFLWoody[i].wd[0].Load=1.50;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=2.5;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=2.0;
	// fuel model 7
		NFFLWoody[i].NumClasses=4;
		 NFFLWoody[i].Depth=2.5;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[6].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[6].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[6].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[6].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[6].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=1750.0;
		 NFFLWoody[i].wd[0].Load=1.13;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=1.87;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i].wd[2].Load=1.5;
		 NFFLWoody[i].wd[3].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i++].wd[3].Load=0.37;
	// fuel model 8
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=0.2;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[7].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[7].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[7].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[7].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[7].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=2000.0;
		 NFFLWoody[i].wd[0].Load=1.50;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=1.00;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=2.5;
	// fuel model 9
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=0.2;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[8].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[8].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[8].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[8].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[8].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=2500.0;
		 NFFLWoody[i].wd[0].Load=2.92;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=0.41;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=0.15;
	// fuel model 10
		NFFLWoody[i].NumClasses=4;
		 NFFLWoody[i].Depth=1.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[9].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[9].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[9].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[9].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[9].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=2000.0;
		 NFFLWoody[i].wd[0].Load=3.01;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=2.0;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i].wd[2].Load=5.01;
		 NFFLWoody[i].wd[3].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i++].wd[3].Load=2.00;
	// fuel model 11
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=1.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[10].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[10].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[10].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[10].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[10].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i].wd[0].Load=1.50;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=4.51;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=5.51;
	// fuel model 12
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=2.3;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[11].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[11].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[11].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[11].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[11].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i].wd[0].Load=4.01;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=14.03;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i++].wd[2].Load=16.53;
	// fuel model 13
		NFFLWoody[i].NumClasses=3;
		 NFFLWoody[i].Depth=3.0;
		 NFFLWoody[i].Units=1;
	//     if((NFFLWoody[12].wd=(WoodyData *) calloc(NFFLWoody[i].NumClasses, sizeof(WoodyData)))==NULL)
	//     if((NFFLWoody[12].wd=(WoodyData *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, NFFLWoody[i].NumClasses*sizeof(WoodyData)))==NULL)
	//     	return false;
		 if((NFFLWoody[12].wd=new WoodyData[NFFLWoody[i].NumClasses])==NULL)
		 	return false;
//		 ZeroMemory(NFFLWoody[12].wd, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 memset(NFFLWoody[12].wd,0x0, NFFLWoody[i].NumClasses*sizeof(WoodyData));
		 NFFLWoody[i].wd[0].SurfaceAreaToVolume=1500.0;
		 NFFLWoody[i].wd[0].Load=7.01;
		 NFFLWoody[i].wd[1].SurfaceAreaToVolume=109.0;
		 NFFLWoody[i].wd[1].Load=23.04;
		 NFFLWoody[i].wd[2].SurfaceAreaToVolume=30.0;
		 NFFLWoody[i].wd[2].Load=28.05;
		 long j;
		 // convert all to metric
		 for(i=0; i<13; i++)
		 {	NFFLWoody[i].Depth/=3.280839895;
		 	for(j=0; j<NFFLWoody[i].NumClasses; j++)
			  {	NFFLWoody[i].wd[j].SurfaceAreaToVolume*=3.280839895;
			  	NFFLWoody[i].wd[j].Load*=0.224169061;
				   NFFLWoody[i].wd[j].HeatContent=8000.0*2.32599;
				   NFFLWoody[i].wd[j].Density=513.0;
				   NFFLWoody[i].TotalWeight+=NFFLWoody[i].wd[j].Load;
			  }
		 }
	*/
	return true;
}


double Farsite5::GetWoodyFuelMoisture(long ModelNumber, long SizeClass)
{
	if (ModelNumber > MAXNUM_COARSEWOODY_MODELS)
		return 0.0;

	if (coarsewoody[ModelNumber - 1].NumClasses < SizeClass)
		return 0.0;

	return coarsewoody[ModelNumber - 1].wd[SizeClass].FuelMoisture;
}


long Farsite5::GetWoodyDataUnits(long ModelNumber, char* Description)
{
	memcpy(Description, coarsewoody[ModelNumber - 1].Description,
		64 * sizeof(char));

	return coarsewoody[ModelNumber - 1].Units;
}


void Farsite5::GetWoodyData(long WoodyModelNumber, long SurfModelNumber,
	long* NumClasses, WoodyData* woody, double* depth, double* load)
{
	bool Combine = false;

	*NumClasses = 0;
	*load = 0.0;


	switch (WoodyCombineOptions(GETVAL))
	{
	case CWD_COMBINE_NEVER:
		Combine = false; break;
	case CWD_COMBINE_ALWAYS:
		Combine = true; break;
	case CWD_COMBINE_ABSENT:
		if (WoodyModelNumber < 1)
			Combine = true;
		else if (coarsewoody[WoodyModelNumber - 1].wd == 0)
			Combine = true;
		else
			Combine = false;
		break;
	}

	if (Combine)
	{
		// gather surface fuel model data
		if (SurfModelNumber > 0 && SurfModelNumber < 14)														 // alloc the max
		{
			*NumClasses = NFFLWoody[SurfModelNumber - 1].NumClasses;//tempwoody.NumClasses=
			*depth = NFFLWoody[SurfModelNumber - 1].Depth; //tempwoody.Depth=
			*load = NFFLWoody[SurfModelNumber - 1].TotalWeight;//tempwoody.TotalWeight=
			memcpy(woody, NFFLWoody[SurfModelNumber - 1].wd,
				NFFLWoody[SurfModelNumber - 1].NumClasses * sizeof(WoodyData));
		}
		else if (SurfModelNumber > 0)
		{
			//double t1, t10, t100, tLH, tLW, s1, sLH, sLW, hd, hl, d, xm;
			double  hd, hl;
			long i = 0, j;
			NewFuel nf;
			memset(&nf, 0x0, sizeof(NewFuel));

			GetNewFuel(SurfModelNumber, &nf);
			*depth = nf.depth;
			*load = 0.0;
			hd = nf.heatd;
			hl = nf.heatl;
			if (nf.h1 > 0.0)
			{
				woody[i].Load = nf.h1;//*0.224169061;
				woody[i].SurfaceAreaToVolume = nf.sav1;//s1;//*3.280839895;
				woody[i++].HeatContent = hd;//*2.32599;
			}
			if (nf.h10 > 0.0)
			{
				woody[i].Load = nf.h10;
				woody[i].SurfaceAreaToVolume = 109.0;//*3.280839895;
				woody[i++].HeatContent = hd;//*2.32599;
			}
			if (nf.h100 > 0.0)
			{
				woody[i].Load = nf.h100;
				woody[i].SurfaceAreaToVolume = 30;//*3.280839895;
				woody[i++].HeatContent = hd;//*2.32599;
			}
			if (nf.lw > 0.0)
			{
				woody[i].Load = nf.lw;
				woody[i].SurfaceAreaToVolume = nf.savlw;//*3.280839895;
				woody[i++].HeatContent = hl;//*2.32599;
			}
			if (nf.lh > 0.0)
			{
				woody[i].Load = nf.lh;
				woody[i].SurfaceAreaToVolume = nf.savlh;//*3.280839895;
				woody[i++].HeatContent = hl;//*2.32599;
			}
			for (j = 0; j < i; j++)
			{
				woody[j].Density = 513.0;
				*load += woody[j].Load;
			}
			*NumClasses = tempwoody.NumClasses = i;
		}
	}
	// patch into coarsewoody model data if present
	if (WoodyModelNumber > 0)
	{
		if (coarsewoody[WoodyModelNumber - 1].wd)
		{
			memcpy(&woody[*NumClasses],
				coarsewoody[WoodyModelNumber - 1].wd,
				coarsewoody[WoodyModelNumber - 1].NumClasses * sizeof(WoodyData));
			*NumClasses += coarsewoody[WoodyModelNumber - 1].NumClasses;
			*depth = coarsewoody[WoodyModelNumber - 1].Depth;
			*load += coarsewoody[WoodyModelNumber - 1].TotalWeight;
		}
	}
}

double Farsite5::WeightLossErrorTolerance(double value)
{
	if (value > 0.0)
		WeightLossErrorTol = value;

	return WeightLossErrorTol;
}

void Farsite5::AllocStationGrid(long XDim, long YDim)
{
	long x, y, Num;

	FreeStationGrid();
	size_t nmemb = 2 * XDim* YDim;
	grid.XDim = XDim;
	grid.YDim = YDim;
	grid.Width = ((double) (GetHiEast() - GetLoEast())) / ((double) XDim);
	grid.Height = ((double) (GetHiNorth() - GetLoNorth())) / ((double) YDim);
	grid.Grid = new long[nmemb];
	for (y = 0; y < YDim; y++)
	{
		for (x = 0; x < XDim; x++)  // initialize grid to station 1
		{
			Num = x + y * grid.XDim;
			grid.Grid[Num] = 1;
		}
	}
}


long Farsite5::FreeStationGrid()
{
	if (grid.Grid)
	{
		delete[] grid.Grid;
		grid.Grid = 0;

		return 1;
	}
	grid.Grid = 0;

	return 0;
}


long Farsite5::GetStationNumber(double xpt, double ypt)
{
	long XCell = (long)((xpt - GetLoEast()) / grid.Width);
	long YCell = (long)((ypt - GetLoNorth()) / grid.Height);
	long CellNum = XCell + YCell* grid.XDim;

	if (grid.Grid[CellNum] > NumWeatherStations)   // check to see if data exist
		return 0;   		  // for a given weather station
	if (grid.Grid[CellNum] > NumWindStations)
		return 0;

	return grid.Grid[CellNum];
}


void Farsite5::ResetFuelConversions()
{
	long i;

	for (i = 0;
		i < 257;
		i++)   		  		// could also read default file here
		fuelconversion.Type[i] = i;
	fuelconversion.Type[0] = -1;
	for (i = 91; i < 100; i++)
		fuelconversion.Type[i] -= 100;
	HAVE_CONV_MODELS = false;
}


void Farsite5::ResetCustomFuelModels()
{
	/*
		long Changes;
		for(Changes=0; Changes<9; Changes++)
			FuelMod.OrderToFuel[Changes]=0;
		for(Changes=0; Changes<37; Changes++)
		{	SetFuel(Changes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "");
			FuelMod.ModelIsHere[Changes]=false;
		}
		FuelMod.NumChanges=0;
		HAVE_CUST_MODELS=false;
	*/
}
int Farsite5::GetFuelConversion(int fuel)
{
	// retrieve fuel model conversions
	int cnv = -1;

	if (fuel >= 0 && fuel < 257)	// check fuel for valid array range
	{
		cnv = fuelconversion.Type[fuel];   // get new fuel
		if (cnv < 1)  // if not
		{
			if (cnv < -9)
				cnv = -1;
			//if(cnv!=0 && cnv!=-1 && cnv!=-2)
			//	cnv=-1;    // return rock of other than -1 or -2
		}
		else if (!IsNewFuelReserved(cnv)) //NewFuel>13 && NewFuel<51)    // check to see if custom model required
		{
			if (newfuels[cnv].number == 0)   // check to see if cust mod here
				cnv = -1;     // return rock if no fuel model
		}
		else if (cnv > 256)  // if new fuel too high
			cnv = -1;
	}

	return cnv;
}


int Farsite5::SetFuelConversion(int From, int To)
{
	// set fuel model conversions
	if (From >= 0 && From < 257 && To < 257 && To >= 0)
	{
		if (To > 90 && To < 100)
			To = To - 100;		// make all negative for the 90's, indicate unburnable
		else if (To == 0)
			To = -1;
		fuelconversion.Type[From] = To;
	}
	else
		return false;

	return true;
}

double Farsite5::GetFuelDepth(int Number)
{
	if (Number < 0)
		return 0.0;

	if (newfuels[Number].number == 0)
		return 0.0;

	return newfuels[Number].depth;
}

long Farsite5::AccessFuelModelUnits(long Val)
{
	if (Val > -1)
		FuelModelUnits = Val;

	return FuelModelUnits;
}

short Farsite5::GetTheme_Units(short DataTheme)
{
	short units;

	switch (DataTheme)
	{
	case 0:
		units = Header.EUnits; break;
	case 1:
		units = Header.SUnits; break;
	case 2:
		units = Header.AUnits; break;
	case 3:
		units = Header.FOptions; break;
	case 4:
		units = Header.CUnits; break;
	case 5:
		units = Header.HUnits; break;
	case 6:
		units = Header.BUnits; break;
	case 7:
		units = Header.PUnits; break;
	case 8:
		units = Header.DUnits; break;
	case 9:
		units = Header.WOptions; break;
	}

	return units;
}

double Farsite5::GetDefaultCrownHeight()
{
	return CanopyChx.Height;
}


double Farsite5::GetDefaultCrownBase()
{
	return CanopyChx.CrownBase;
}


double Farsite5::GetDefaultCrownBD(short cover)
{
	if (CrownDensityLinkToCC)
		return CanopyChx.BulkDensity * ((double) cover) / 100.0;

	return CanopyChx.BulkDensity;
}

bool Farsite5::LinkDensityWithCrownCover(long TrueFalse)
{
	if (TrueFalse >= 0)
		CrownDensityLinkToCC = TrueFalse;

	return CrownDensityLinkToCC;
};



long Farsite5::GetConditMonth()
{
	return conditmonth;
}


long Farsite5::GetConditDay()
{
	return conditday;
}


long Farsite5::GetConditMinDeficit()
{
	if (CondPeriod == false)
		return 0;

	long CondMin;
	long StartMin;

	CondMin = (GetJulianDays(conditmonth) + conditday) * 1440;// 00:01 on this day
	StartMin = (GetJulianDays(startmonth) + startday) * 1440 +
		(starthour / 100 * 60) +
		startmin;
	if (StartMin < CondMin)
		StartMin += 365 * 1440;

	return StartMin - CondMin;
}


void Farsite5::SetConditMonth(long input)
{
	conditmonth = input;
}


void Farsite5::SetConditDay(long input)
{
	conditday = input;
}
long Farsite5::GetWeatherMonth(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].mo;
}

long Farsite5::GetWeatherDay(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].dy;
}

double Farsite5::GetWeatherRain(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].rn;
}

long Farsite5::GetWeatherTime1(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].t1;
}

long Farsite5::GetWeatherTime2(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].t2;
}

double Farsite5::GetWeatherTemp1(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].T1;
}

double Farsite5::GetWeatherTemp2(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].T2;
}

long Farsite5::GetWeatherHumid1(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].H1;
}

long Farsite5::GetWeatherHumid2(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].H2;
}

double Farsite5::GetWeatherElev(long StationNumber, long NumObs)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	return wtrdt[StationNumber][NumObs].el;
}


void Farsite5::GetWeatherRainTimes(long StationNumber, long NumObs, long* tr1, long* tr2)
{
	if (NumObs > MaxWeatherObs[StationNumber] - 1)
		NumObs = MaxWeatherObs[StationNumber] - 1;

	*tr1 = wtrdt[StationNumber][NumObs].tr1;
	*tr2 = wtrdt[StationNumber][NumObs].tr2;
}


long Farsite5::GetMaxWeatherObs(long StationNumber)
{
	if (StationNumber > 4)
		return -1;

	return MaxWeatherObs[StationNumber] - 1;
}

long Farsite5::GetTolerance()
{
	return CanopyChx.Tolerance;
}

void Farsite5::LoadAttacks(AttackData attackdata)
{
	// function only for loading attacks from bookmark
	if (NumAttacks == 0)
	{
		FirstAttack = new AttackData;//(AttackData *) calloc(1, sizeof(AttackData));
		CurAttack = FirstAttack;
		memcpy(FirstAttack, &attackdata, sizeof(AttackData));
	}
	memcpy(CurAttack, &attackdata, sizeof(AttackData));
	NextAttack = new AttackData;//(AttackData *) calloc(1, sizeof(AttackData));
	CurAttack->next = (AttackData *) NextAttack;
	if (NumAttacks == 0)
		FirstAttack->next = (AttackData *) NextAttack;
	NumAttacks++;
	CurAttack = NextAttack;
}


long Farsite5::SetupIndirectAttack(long crewnum, double* startpt, long numpts)
{
	// indirect attack constructor

	long i;

	for (i = 0; i <= NumAttacks; i++)
	{
		if (NumAttacks == 0)
		{
			if ((FirstAttack = new AttackData) != NULL)//(AttackData *) calloc(1, sizeof(AttackData)))!=NULL)
			{
				CurAttack = FirstAttack;
				if (AttackCounter == 0)
				{
					memset(AttackLog, 0x0, sizeof(AttackLog));
					//getcwd(AttackLog, 255);
					strcat(AttackLog, "grndattk.log");
					remove(AttackLog);
				}
			}
			else
				return 0;
		}
		else if (i == 0)
			CurAttack = FirstAttack;
		else
			CurAttack = NextAttack;
		if (i < NumAttacks)
			NextAttack = (AttackData *) CurAttack->next;
	}

	if ((NextAttack = new AttackData) != NULL)//(AttackData *) calloc(1, sizeof(AttackData)))!=NULL)
	{
		CurAttack->next = (AttackData *) NextAttack;
		CurAttack->IndirectLine1 = 0;
		CurAttack->CrewNum = crewnum;
		CurAttack->Suspended = 0;
		CurAttack->AttackNumber = ++AttackCounter;
		CurAttack->Burnout = 0;
		CurAttack->AttackTime = 0.0;
		CurAttack->LineBuilt = 0.0;
		if (ResetIndirectAttack(CurAttack, startpt, numpts))
			++NumAttacks;
		else
		{
			delete NextAttack;//free(NextAttack);
			AttackCounter--;

			return 0;
		}
	}
	else
		return 0;

	return AttackCounter;
}


long Farsite5::ResetIndirectAttack(AttackData* atk, double* coords, long numpts)
{
	if (atk->AttackTime > 0.0)
	{
		WriteAttackLog(atk, 0, 0, 0);
		atk->AttackTime = 0.0;
		atk->LineBuilt = 0.0;
	}
	atk->NumPoints = numpts;
	atk->Indirect = 1;
	if (atk->IndirectLine1)
		delete[] atk->IndirectLine1;//free(atk->IndirectLine1);
	if ((atk->IndirectLine1 = new double[numpts * 2]) != NULL) //(double *) calloc(numpts*2, sizeof(double)))!=NULL)
	{
		//NumInsertPoints=numpts;
		for (long i = 0; i < numpts; i++)
		{
			atk->IndirectLine1[i * 2] = coords[i * 2];
			atk->IndirectLine1[i * 2 + 1] = coords[i * 2 + 1];
		}
		atk->CurrentPoint = 0;
		atk->FireNumber = -1;
		atk->BurnLine[0] = -1;		// use -1 as flag for reset
	}
	else
	{
		atk->IndirectLine1 = 0;

		return 0;
	}

	return 1;
}


long Farsite5::SetupDirectAttack(long crewnum, long FireNum, double* coords)
{
	// direct attack constructor
	// find nearst vertex on all fires, && establish direction for line building
	long i;

	for (i = 0; i <= NumAttacks; i++)
	{
		if (NumAttacks == 0)
		{
			if ((FirstAttack = new AttackData) != NULL) //(AttackData *) calloc(1, sizeof(AttackData)))!=NULL)
			{
				CurAttack = FirstAttack;
				if (AttackCounter == 0)
				{
					memset(AttackLog, 0x0, sizeof(AttackLog));
					//getcwd(AttackLog, 255);
					strcat(AttackLog, "grndattk.log");
					//DeleteFile(AttackLog);
					remove(AttackLog);
				}
			}
			else
				return 0;
		}
		else if (i == 0)
			CurAttack = FirstAttack;
		else
			CurAttack = NextAttack;
		if (i < NumAttacks)
			NextAttack = (AttackData *) CurAttack->next;
	}

	if ((NextAttack = new AttackData) != NULL)//(AttackData *) calloc(1, sizeof(AttackData)))!=NULL)
	{
		CurAttack->next = (AttackData *) NextAttack;
		CurAttack->AttackNumber = ++AttackCounter;
		CurAttack->CrewNum = crewnum;
		CurAttack->Suspended = 0;
		CurAttack->AttackTime = 0.0;
		CurAttack->LineBuilt = 0.0;
		CurAttack->IndirectLine1 = 0;
		CurAttack->IndirectLine2 = 0;
		CurAttack->NumPoints = 0;
		CurAttack->Reverse = 0;
		CurAttack->Burnout = 0;
		if (ResetDirectAttack(CurAttack, FireNum, coords))
			++NumAttacks;
		else
		{
			AttackCounter--;
			delete[] NextAttack;//free(NextAttack);

			return 0;
		}
	}
	else
		return 0;

	return AttackCounter;
}


long Farsite5::ResetDirectAttack(AttackData* atk, long FireNum, double* coords)
{
	long i;
	double xpt1, ypt1;
	double xdist, ydist, hdist, mindist = 0;
	double startpointx, startpointy, endpointx, endpointy;
	bool first = true;

	if (atk->AttackTime > 0.0)
	{
		WriteAttackLog(atk, 0, 0, 0);
		atk->AttackTime = 0.0;
		atk->LineBuilt = 0.0;
	}
	if (GetInputMode() == PARALLELLOCATION ||
		GetInputMode() == RELOCATEPARALLEL)
		atk->Indirect = 2;
	else
		atk->Indirect = 0;
	atk->Burnout = 0;
	atk->BurnLine[0] = -1;		// use -1 as flag for reset
	atk->CurrentPoint = -1;		// number of vertex on Indirect Line, not used here
	atk->LineNumber = -1;
	atk->FireNumber = FireNum;
	startpointx = coords[0];   	// transfer coords to local vars
	startpointy = coords[1];
	endpointx = coords[2];
	endpointy = coords[3];
	for (i = 0; i < GetNumPoints(CurAttack->FireNumber); i++)
	{
		xpt1 = GetPerimeter1Value(CurAttack->FireNumber, i, XCOORD);
		ypt1 = GetPerimeter1Value(CurAttack->FireNumber, i, YCOORD);
		xdist = pow2(xpt1 - startpointx);
		ydist = pow2(ypt1 - startpointy);
		hdist = xdist + ydist;
		if (first)
		{
			mindist = hdist;
			atk->CurrentPoint = i;
			first = false;
		}
		else if (hdist < mindist)
		{
			mindist = hdist;
			atk->CurrentPoint = i;
		}
	}
	first = true;
	for (i = 0; i < GetNumPoints(atk->FireNumber); i++)
	{
		xpt1 = GetPerimeter1Value(CurAttack->FireNumber, i, XCOORD);
		ypt1 = GetPerimeter1Value(CurAttack->FireNumber, i, YCOORD);
		xdist = pow2(xpt1 - endpointx);
		ydist = pow2(ypt1 - endpointy);
		hdist = xdist + ydist;
		if (first && i != CurAttack->CurrentPoint)
		{
			mindist = hdist;
			atk->NextPoint = i;
			first = 0;
		}
		else if (hdist < mindist && i != CurAttack->CurrentPoint)
		{
			mindist = hdist;
			atk->NextPoint = i;
		}
	}
	atk->Reverse = false;
	if (GetInout(atk->FireNumber) == 1 || GetInout(atk->FireNumber) == 2)
	{
		if (atk->NextPoint > atk->CurrentPoint)
		{
			if (atk->NextPoint -
				atk->CurrentPoint >
				(double) GetNumPoints(atk->FireNumber) /
				2.0)
			{
				atk->Reverse = true;
				//			else
				//  				atk->Reverse=false;
			}
		}
		else if (atk->CurrentPoint -
			atk->NextPoint <
			(double) GetNumPoints(atk->FireNumber) /
			2.0)
		{
			atk->Reverse = true;
			//  		  	else
			//  					atk->Reverse=false;
		}
	}
	else
		return 0;

	// 	allocate 2 points for xpt2 & ypt2 for determining start point later

	if (atk->IndirectLine1)
		delete[] atk->IndirectLine1;//free(atk->IndirectLine1);
	atk->IndirectLine1 = 0;
	if (atk->IndirectLine2)
		delete[] atk->IndirectLine2;//free(atk->IndirectLine2);
	atk->IndirectLine2 = 0;
	if (atk->Indirect == 0) 		  // direct attack
	{
		if ((atk->IndirectLine1 = new double[3]) != NULL)//(double *) calloc(3, sizeof(double)))!=NULL)
		{
			atk->IndirectLine1[0] = GetPerimeter1Value(atk->FireNumber,
										atk->CurrentPoint, XCOORD);
			atk->IndirectLine1[1] = GetPerimeter1Value(atk->FireNumber,
										atk->CurrentPoint, YCOORD);
			atk->IndirectLine1[2] = -1.0;
		}
		else
		{
			atk->IndirectLine1 = 0;

			return 0;
		}
	}
	else						  // parallel attack
	{
		if ((atk->IndirectLine2 = new double[3]) != NULL)//(double *) calloc(3, sizeof(double)))!=NULL)
		{
			atk->IndirectLine2[0] = GetPerimeter1Value(atk->FireNumber,
										atk->CurrentPoint, XCOORD);
			atk->IndirectLine2[1] = GetPerimeter1Value(atk->FireNumber,
										atk->CurrentPoint, YCOORD);
			atk->IndirectLine2[2] = -1.0;
		}
		else
		{
			atk->IndirectLine1 = 0;
			atk->IndirectLine2 = 0;
			return 0;
		}
	}

	return 1;
}

long Farsite5::GetNumAttacks()
{
	// returns the number of active attacks
	return NumAttacks;
}

long Farsite5::GetFireNumberForAttack(long AttackCounter)
{
	// returns fire number associated with a given attack
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
			CurAttack = FirstAttack;
		NextAttack = (AttackData *) CurAttack->next;
		if (CurAttack->AttackNumber == AttackCounter)
			return CurAttack->FireNumber;
		CurAttack = NextAttack;
	}

	return -1;
}


AttackData* Farsite5::GetAttackForFireNumber(long NumFire, long StartAttackNum,
	long* LastAttackNumber)
{
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
			CurAttack = FirstAttack;
		NextAttack = (AttackData *) CurAttack->next;
		if (i >= StartAttackNum)
		{
			if (CurAttack->FireNumber == NumFire && CurAttack->Suspended == 0)
			{
				*LastAttackNumber = i;

				return CurAttack;
			}
		}
		CurAttack = NextAttack;
	}

	return 0;
}

void Farsite5::SetNewFireNumberForAttack(long oldnumfire, long newnumfire)
{
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
		{
			CurAttack = FirstAttack;
			NextAttack = (AttackData *) CurAttack->next;
		}
		if (CurAttack->FireNumber == oldnumfire)
		{
			if (GetInout(newnumfire) == 3)
			{
				if (CurAttack->Indirect == 1)
					CurAttack->FireNumber = newnumfire;
			}
			else if (CurAttack->Indirect != 1)
				CurAttack->FireNumber = newnumfire;
		}
		else if (CurAttack->Indirect == 2)  		   // if parallel attack only
		{
			if (CurAttack->LineNumber == oldnumfire)
				CurAttack->LineNumber = newnumfire;
		}
		CurAttack = NextAttack;
		NextAttack = (AttackData *) CurAttack->next;
	}
}

long Farsite5::GetNumCrews()
{
	return NumCrews;
}

AttackData* Farsite5::GetAttackByOrder(long OrdinalAttackNum, bool IndirectOnly)
{
	// retrieves indirect attack in order
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
			CurAttack = FirstAttack;
		else
			CurAttack = NextAttack;
		NextAttack = (AttackData *) CurAttack->next;
		if (i == OrdinalAttackNum)
		{
			if (IndirectOnly)
			{
				if (CurAttack->Indirect > 0 && CurAttack->Suspended == 0)
					return CurAttack;
				else
					return 0;
			}
			else
				return CurAttack;
		}
	}

	return 0;
}


AttackData* Farsite5::GetAttack(long AttackCounter)
{
	// Get Attack Instance, Attack Number is sequential based on
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
		{
			CurAttack = FirstAttack;
			NextAttack = (AttackData *) CurAttack->next;
		}
		if (CurAttack->AttackNumber == AttackCounter)
			return CurAttack;
		else
		{
			CurAttack = NextAttack;
			NextAttack = (AttackData *) CurAttack->next;
		}
	}

	return 0;
}


void Farsite5::CancelAttack(long AttackCounter)
{
	//make sure Attack Number is "1-based" NOT "0-based"
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
		{
			CurAttack = FirstAttack;
			NextAttack = (AttackData *) CurAttack->next;
		}
		if (CurAttack->AttackNumber == AttackCounter)
		{
			if (i == 0)
				FirstAttack = (AttackData *) CurAttack->next;
			else
				LastAttack->next = (AttackData *) NextAttack;
			WriteAttackLog(CurAttack, 0, 0, 0);
			if (CurAttack->IndirectLine1)
				delete[] CurAttack->IndirectLine1;//free(CurAttack->IndirectLine1);
			CurAttack->IndirectLine1 = 0;
			if (CurAttack->Indirect == 2 && CurAttack->IndirectLine2)
				delete[] CurAttack->IndirectLine2;//free(CurAttack->IndirectLine2);
			CurAttack->IndirectLine2 = 0;
			delete CurAttack;//free(CurAttack);
			NumAttacks--;
			if (NumAttacks == 0)
				delete NextAttack;//free(NextAttack);
			break;
		}
		else
		{
			LastAttack = CurAttack;
			CurAttack = NextAttack;
			NextAttack = (AttackData *) CurAttack->next;
		}
	}
}


AttackData* Farsite5::GetReassignedAttack()
{
	return Reassignment;
}

void Farsite5::ReassignAttack(AttackData* atk)
{
	Reassignment = atk;
}


void Farsite5::FreeAllAttacks()
{
	for (long i = 0; i < NumAttacks; i++)
	{
		if (i == 0)
		{
			CurAttack = FirstAttack;
			NextAttack = (AttackData *) CurAttack->next;
		}
		WriteAttackLog(CurAttack, 0, 0, 0);
		if (CurAttack->IndirectLine1)
			delete[] CurAttack->IndirectLine1;//free(CurAttack->IndirectLine1);
		CurAttack->IndirectLine1 = 0;
		if (CurAttack->Indirect == 2 && CurAttack->IndirectLine2)	// if parallel attack
			delete[] CurAttack->IndirectLine2;//free(CurAttack->IndirectLine2);
		CurAttack->IndirectLine2 = 0;
		delete CurAttack;//free(CurAttack);
		CurAttack = NextAttack;
		NextAttack = (AttackData *) CurAttack->next;
	}
	if (NumAttacks > 0)
	{
		delete CurAttack;//free(CurAttack);
		NumAttacks = 0;
	}
	AttackCounter = 0;
}


void Farsite5::WriteAttackLog(AttackData* atk, long type, long var1, long var2)
{
	FILE* atklog;
	char AttackType[32];

	memset(AttackType, 0x0, sizeof(AttackType));
	if ((atklog = fopen(AttackLog, "a")) != NULL)
	{
		if (atk)
		{
			switch (atk->Indirect)
			{
			case 0:
				sprintf(AttackType, "%s", "Direct");
				break;
			case 1:
				sprintf(AttackType, "%s", "Indirect");
				break;
			case 2:
				sprintf(AttackType, "%s", "Parallel");
				break;
			}
		}

		switch (type)
		{
		case 0:
			// end of attack
			fprintf(atklog,
				"%s, %s, %ld %s, %ld %s, $%ld since last instruction\n",
				GetCrew(atk->CrewNum)->CrewName, AttackType,
				(long) atk->AttackTime, "mins", (long) atk->LineBuilt, "m",
				(long) (GetCrew(atk->CrewNum)->Cost * atk->AttackTime / 60.0));
			break;
		case 1:
			// reassign attack crew
			fprintf(atklog, "%s reassigned %s\n", crew[var1]->CrewName,
				AttackType);
			break;
		case 2:
			// crew moved to different compound crew
			fprintf(atklog, "%s moved to %s\n", crew[var1]->CrewName,
				crew[var2]->CrewName);
			break;
		case 3:
			fprintf(atklog, "%s exceeded flamelength limit, direct attack\n",
				GetCrew(atk->CrewNum)->CrewName);
			break;
		case 4:
			// crew removed from group
			fprintf(atklog, "%s removed from group %s\n",
				crew[var1]->CrewName, crew[var2]->CrewName);
			break;
		case 5:
			// end of attack
			fprintf(atklog,
				"%s, %s, %s, %ld %s, %ld %s, $%ld since last instruction\n",
				GetCrew(atk->CrewNum)->CrewName, "Crew Added", AttackType,
				(long) atk->AttackTime, "mins", (long) atk->LineBuilt, "m",
				(long) (GetCrew(atk->CrewNum)->Cost * atk->AttackTime / 60.0));
			break;
		}
		fclose(atklog);
	}
}


bool Farsite5::LoadCrews(char* FileName, bool AppendList)
{
/*	FILE* crewfile;
	char garbage[256] = "", data[256] = "";
	char ch[2] = "";
	double RateMult, rate, test;
	long fuelnumber, crewnumber;

	if ((crewfile = fopen(FileName, "r")) != NULL)
	{
		long i, j;
		j = GetNumCrews();
		if (!AppendList)
		{
			for (i = 0; i < j; i++)
				FreeCrew(0);
		}

		fgets(garbage, 255, crewfile);
		do
		{
			if (feof(crewfile))
				break;
			strncpy(ch, garbage, 1);
			if (strcmp(ch, "#"))
				break;
			crewnumber = SetNewCrew();
			if (crewnumber == -1)
				return false;
			crew[crewnumber] = GetCrew(crewnumber);
			crew[crewnumber]->Compound = -1;
			memset(crew[crewnumber]->CrewName, 0x0,
				sizeof(crew[crewnumber]->CrewName));
			for (i = 1; i < (int)strlen(garbage) - 2; i++)
				strncat(crew[crewnumber]->CrewName, &garbage[i], 1);
			fgets(garbage, 255, crewfile);
			sscanf(garbage, "%s", data);
			std::transform(data, data+strlen(data), data, toupper) ;
			if (!strcmp(data, "CHAINS_PER_HOUR"))
			{
				crew[crewnumber]->Units = 3;
				RateMult = 0.3048 * 1.1;
			}
			else if (!strcmp(data, "METERS_PER_MINUTE"))
			{
				crew[crewnumber]->Units = 1;
				RateMult = 1.0;
			}
			else if (!strcmp(data, "FEET_PER_MINUTE"))
			{
				crew[crewnumber]->Units = 2;
				RateMult = 0.3048;
			}
			fgets(garbage, 255, crewfile);
			sscanf(garbage, "%s %lf", data, &test);
			std::transform(data, data+strlen(data), data, toupper) ;
			if (!strcmp(data, "FLAME_LIMIT"))
				crew[crewnumber]->FlameLimit = test;
			if (crew[crewnumber]->Units == 2 || crew[crewnumber]->Units == 3)
				crew[crewnumber]->FlameLimit *= 0.3048;
			do
			{
				fgets(garbage, 255, crewfile);
				sscanf(garbage, "%ld %lf", &fuelnumber, &rate);
				if (fuelnumber < 51)
					crew[crewnumber]->LineProduction[fuelnumber - 1] = rate * RateMult;
				else if (fuelnumber == 99)
					break;
			}
			while (!feof(crewfile));
			fgets(garbage, 255, crewfile);
			sscanf(garbage, "%s", data);
			std::transform(data, data+strlen(data), data, toupper) ;
			if (!strcmp(data, "COST_PER_HOUR"))
			{
				sscanf(garbage, "%s %lf", data, &test);
				crew[crewnumber]->Cost = test;
				fgets(garbage, 255, crewfile);
			}
			else
				crew[crewnumber]->Cost = -1.0;
		}
		while (!feof(crewfile));
		fclose(crewfile);
	}
	else
		return false;
*/
	return true;
}

long Farsite5::GetInputMode()
{
	// gets input mode to the landscape window

	return LandscapeInputMode;
}


void Farsite5::SetInputMode(long ModeCode, bool Pending)
{
	// sets input model to the landscape window
	LandscapeInputMode = ModeCode;
	if (Pending)
		LandscapeInputMode += 100;
}

void Farsite5::SetShapeFileChx(const char* ShapeName, long VisOnly, long Polygons,
	long BarriersSep)
{
	memset(ShapeFileName, 0x0, sizeof(ShapeFileName));
	strcpy(ShapeFileName, ShapeName);
	ShapeVisibleStepsOnly = VisOnly;
	ShapePolygonsNotLines = Polygons;
	ShapeBarriersSeparately = BarriersSep;
}


char* Farsite5::GetShapeFileChx(long* VisOnly, long* Polygons, long* BarriersSep)
{
	*VisOnly = ShapeVisibleStepsOnly;
	*Polygons = ShapePolygonsNotLines;
	*BarriersSep = ShapeBarriersSeparately;

	return ShapeFileName;
}

/**********************************************************************************/
// Determines actual date and time from SIMTIME variable
long Farsite5::Chrono(double SIMTIME, long* hour, double* hours, bool RelCondit)
{
long date, month, day, min;
double mins, days;

 	*hours = SIMTIME / 60.0;
 	*hour = (long) * hours; 					  // truncates time to nearest hour
 	min = (long) ((*hours - *hour) * 60);
        mins = (*hours - *hour) * 60.0;
        if (RelCondit && UseConditioningPeriod(GETVAL))
	  	*hours = ((double) * hour) * 100.0;   			  // elapsed time in hours
        else
	  	*hours = ((double) * hour) * 100.0 + (double) GetStartHour() +	GetStartMin();	// elapsed time in hours
        days = *hours / 2400.0;
        day = (long) days;
        if (RelCondit && UseConditioningPeriod(GETVAL)){
		  date = day + GetConditDay();
	  	month = GetJulianDays(GetConditMonth());}
	 else {
		  date = day + GetStartDay();
	  	month = GetJulianDays(GetStartMonth());}

 	date = date + month;
 	if (date > 365)
		  date -= 365;
 	days = day;
 	*hour = (long) (*hours - days * 2400) + min;		 // integer hour
 	*hours = (*hours - days * 2400.0) + mins;   		 // double precision hours
 	return date;
}

/******************************************************************************/
bool Farsite5::UseConditioningPeriod(long YesNo)
{
	if (YesNo > -1)
		CondPeriod = (bool) YesNo;

	return CondPeriod;
}

bool Farsite5::EnvironmentChanged(long YesNo, long StationNumber, long FuelSize)
{
	if (FuelSize > 3)
		FuelSize = 3;
	if (YesNo > -1)
		EnvtChanged[FuelSize][StationNumber] = (bool) YesNo;

	return EnvtChanged[FuelSize][StationNumber];
}

long Farsite5::GetMoistCalcInterval(long FM_SIZECLASS, long CATEGORY)
{
	return MoistCalcInterval[FM_SIZECLASS][CATEGORY];
}


void Farsite5::SetMoistCalcInterval(long FM_SIZECLASS, long CATEGORY, long Val)
{
	switch (CATEGORY)
	{
	case FM_INTERVAL_TIME:
		switch (FM_SIZECLASS)
		{
		case 1:
			if (Val < 60)
				Val = 60;
			if (Val > 240)
				Val = 240;
			break;
		case 2:
			if (Val < 60)
				Val = 600;
			if (Val > 240)
				Val = 240;
			break;
		case 3:
			if (Val < 60)
				Val = 6000;
			if (Val > 240)
				Val = 240;
			break;
		}
		break;
	case FM_INTERVAL_ELEV:
		if (Val < 10)
			Val = 10;
		break;
	case FM_INTERVAL_SLOPE:
		if (Val < 5)
			Val = 5;
		break;
	case FM_INTERVAL_ASP:
		if (Val < 5)
			Val = 5;
		break;
	case FM_INTERVAL_COV:
		if (Val < 5)
			Val = 5;
		break;
	}
	MoistCalcInterval[FM_SIZECLASS][CATEGORY] = Val;
}


/***************************************************************************
* Name: ConvertActualTimeToSimTime
* Desc: Return the time in minutes of the difference between the send in
*        date/times and the start/cond date/time
*  Ret: number of minutes
***************************************************************************/
double Farsite5::ConvertActualTimeToSimtime(long mo, long dy, long hr, long mn,	bool FromCondit)
{
	double SimTime, RefStart;

	if (!FromCondit)
	{
		if (GetStartDate() == 0)
		{
			SimTime = -1.0;
			return SimTime;
		}
		RefStart = (GetJulianDays(startmonth) + startday) * 1440.0 +
			starthour /
			100.0 * 60.0 +
			startmin;
	}
	else
	{
		if (conditmonth == 0 && conditday == 0)
		{
			SimTime = -1.0;
			return SimTime;
		}
		RefStart = (GetJulianDays(conditmonth) + conditday) * 1440.0;
	}

	if (!FromCondit)
	{
		if (mo >= startmonth)
			SimTime = (GetJulianDays(mo) + dy) * 1440.0 +
				hr /
				100.0 * 60.0 +
				mn;
		else
			SimTime = (GetJulianDays(mo) + dy + 365.0) * 1440.0 +
				hr /
				100.0 * 60.0 +
				mn;
	}
	else
	{
		if (mo >= conditmonth)
			SimTime = (GetJulianDays(mo) + dy) * 1440.0 +
				hr /
				100.0 * 60.0 +
				mn;
		else
			SimTime = (GetJulianDays(mo) + dy + 365.0) * 1440.0 +
				hr /
				100.0 * 60.0 +
				mn;
	}
	SimTime -= RefStart;

	return SimTime;
}


void Farsite5::ConvertSimtimeToActualTime(double SimTime, long* mo, long* dy, long* hr,
	long* mn, bool FromCondit)
{
	long months, days, hours, mins, ndays;
	double fday, fhour;
	long i;
	fday = (SimTime / 1440.0);	// days from minutes
	days = (long) fday;
	fhour = ((fday - days) * 24.0);  // hours from days
	hours = (long) fhour;
	mins = (long)((fhour - hours) * 60);	// minutes from hours

	if (!FromCondit)
	{
		hours += GetStartHour() / 100;
		mins += GetStartMin();
	}
	if (mins > 60)
	{
		mins -= 60;
		hours++;
	}
	if (hours >= 24)
	{
		hours -= 24;
		days++;
	}
        if (hours < 0)
	{
		hours += 24;
		days--;
	}
	hours *= 100;
	if (!FromCondit)
	{
		if (GetStartDate() + days > 365)
		{
			days = GetStartDate() + days - 365;
			for (i = 1; i < 12; i++)
			{
				if (days < GetJulianDays(i))
					break;
			}
			days -= GetJulianDays(i - 1);
			months = i - 1;
		}
		else
		{
			for (i = 1; i < 12; i++)
			{
				if (GetStartMonth() + i > 12)
					ndays = 365 + GetJulianDays(i);
				else
					ndays = GetJulianDays(GetStartMonth() + i);
				if (days + GetStartDate() <= ndays)
					break;
			}
			if (GetStartMonth() + i - 1 > 12)
				ndays = 365 + GetJulianDays(i - 1);
			else
				ndays = GetJulianDays(GetStartMonth() + i - 1);
			days -= (ndays - GetStartDate());//(GetJulianDays(GetStartMonth()+i-1)-GetStartDate());
			months = GetStartMonth() + (i - 1);
			if (months > 12)
				months -= 12;
		}
	}
	else
	{
		long ConDate = GetJulianDays(conditmonth) + conditday;

		if (ConDate + days > 365)
		{
			days = ConDate + days - 365;
			for (i = 1; i < 12; i++)
			{
				if (days < GetJulianDays(i))
					break;
			}
			days -= GetJulianDays(i - 1);
			months = i - 1;
		}
		else
		{
			for (i = 1; i < 12; i++)
			{
				if (conditmonth + i > 12)
					ndays = 365 + GetJulianDays(i);
				else
					ndays = GetJulianDays(conditmonth + i);
				if (days + ConDate < ndays)//GetJulianDays(conditmonth+i))
					break;
			}
			if (conditmonth + i > 12)
				ndays = 365 + GetJulianDays(i - 1);
			else
				ndays = GetJulianDays(conditmonth + i - 1);
			days -= (ndays - ConDate);//(GetJulianDays(conditmonth+i-1)-ConDate);
			months = conditmonth + (i - 1);
			if (months > 12)
				months -= 12;
		}
	}

	*mo = months;
	*dy = days;
	*hr = hours;
	*mn = mins;
}

LandscapeTheme* Farsite5::GetLandscapeTheme()
{
	return lcptheme;
}

long Farsite5::GetTheme_HiValue(short DataTheme)
{
	long hi;

	switch (DataTheme)
	{
	case 0:
		hi = Header.hielev; break;
	case 1:
		hi = Header.hislope; break;
	case 2:
		hi = Header.hiaspect; break;
	case 3:
		hi = Header.hifuel; break;
	case 4:
		hi = Header.hicover; break;
	case 5:
		hi = Header.hiheight; break;
	case 6:
		hi = Header.hibase; break;
	case 7:
		hi = Header.hidensity; break;
	case 8:
		hi = Header.hiduff; break;
	case 9:
		hi = Header.hiwoody; break;
	}

	return hi;
}

long Farsite5::GetTheme_LoValue(short DataTheme)
{
	long lo;

	switch (DataTheme)
	{
	case 0:
		lo = Header.loelev; break;
	case 1:
		lo = Header.loslope; break;
	case 2:
		lo = Header.loaspect; break;
	case 3:
		lo = Header.lofuel; break;
	case 4:
		lo = Header.locover; break;
	case 5:
		lo = Header.loheight; break;
	case 6:
		lo = Header.lobase; break;
	case 7:
		lo = Header.lodensity; break;
	case 8:
		lo = Header.loduff; break;
	case 9:
		lo = Header.lowoody; break;
	}

	return lo;
}

void Farsite5::InitializeNewFuel()
{
	newfuels[1].number = 1;
	strcpy(newfuels[1].code, "FM1");
	newfuels[1].h1 = 0.740000;
	newfuels[1].h10 = 0.000000;
	newfuels[1].h100 = 0.000000;
	newfuels[1].lh = 0.000000;
	newfuels[1].lw = 0.000000;
	newfuels[1].dynamic = 0;
	newfuels[1].sav1 = 3500;
	newfuels[1].savlh = 1800;
	newfuels[1].savlw = 1500;
	newfuels[1].depth = 1.000000;
	newfuels[1].xmext = 0.120000;
	newfuels[1].heatd = 8000.000000;
	newfuels[1].heatl = 8000.000000;
	strcpy(newfuels[1].desc, "Short Grass");

	newfuels[2].number = 2;
	strcpy(newfuels[2].code, "FM2");
	newfuels[2].h1 = 2.000000;
	newfuels[2].h10 = 1.000000;
	newfuels[2].h100 = 0.500000;
	newfuels[2].lh = 0.000000;
	newfuels[2].lw = 0.500000;
	newfuels[2].dynamic = 0;
	newfuels[2].sav1 = 3000;
	newfuels[2].savlh = 1800;
	newfuels[2].savlw = 1500;
	newfuels[2].depth = 1.000000;
	newfuels[2].xmext = 0.150000;
	newfuels[2].heatd = 8000.000000;
	newfuels[2].heatl = 8000.000000;
	strcpy(newfuels[2].desc, "Timber Grass/Understory");

	newfuels[3].number = 3;
	strcpy(newfuels[3].code, "FM3");
	newfuels[3].h1 = 3.010000;
	newfuels[3].h10 = 0.000000;
	newfuels[3].h100 = 0.000000;
	newfuels[3].lh = 0.000000;
	newfuels[3].lw = 0.000000;
	newfuels[3].dynamic = 0;
	newfuels[3].sav1 = 1500;
	newfuels[3].savlh = 1800;
	newfuels[3].savlw = 1500;
	newfuels[3].depth = 2.500000;
	newfuels[3].xmext = 0.250000;
	newfuels[3].heatd = 8000.000000;
	newfuels[3].heatl = 8000.000000;
	strcpy(newfuels[3].desc, "Tall Grass");

	newfuels[4].number = 4;
	strcpy(newfuels[4].code, "FM4");
	newfuels[4].h1 = 5.010000;
	newfuels[4].h10 = 4.010000;
	newfuels[4].h100 = 2.000000;
	newfuels[4].lh = 0.000000;
	newfuels[4].lw = 5.010000;
	newfuels[4].dynamic = 0;
	newfuels[4].sav1 = 2000;
	newfuels[4].savlh = 1800;
	newfuels[4].savlw = 1500;
	newfuels[4].depth = 6.000000;
	newfuels[4].xmext = 0.200000;
	newfuels[4].heatd = 8000.000000;
	newfuels[4].heatl = 8000.000000;
	strcpy(newfuels[4].desc, "Chaparral");

	newfuels[5].number = 5;
	strcpy(newfuels[5].code, "FM5");
	newfuels[5].h1 = 1.000000;
	newfuels[5].h10 = 0.500000;
	newfuels[5].h100 = 0.000000;
	newfuels[5].lh = 0.000000;
	newfuels[5].lw = 2.000000;
	newfuels[5].dynamic = 0;
	newfuels[5].sav1 = 2000;
	newfuels[5].savlh = 1800;
	newfuels[5].savlw = 1500;
	newfuels[5].depth = 2.000000;
	newfuels[5].xmext = 0.200000;
	newfuels[5].heatd = 8000.000000;
	newfuels[5].heatl = 8000.000000;
	strcpy(newfuels[5].desc, "Short Brush");

	newfuels[6].number = 6;
	strcpy(newfuels[6].code, "FM6");
	newfuels[6].h1 = 1.500000;
	newfuels[6].h10 = 2.500000;
	newfuels[6].h100 = 2.000000;
	newfuels[6].lh = 0.000000;
	newfuels[6].lw = 0.000000;
	newfuels[6].dynamic = 0;
	newfuels[6].sav1 = 1750;
	newfuels[6].savlh = 1800;
	newfuels[6].savlw = 1500;
	newfuels[6].depth = 2.500000;
	newfuels[6].xmext = 0.250000;
	newfuels[6].heatd = 8000.000000;
	newfuels[6].heatl = 8000.000000;
	strcpy(newfuels[6].desc, "Dormant Brush");

	newfuels[7].number = 7;
	strcpy(newfuels[7].code, "FM7");
	newfuels[7].h1 = 1.130000;
	newfuels[7].h10 = 1.870000;
	newfuels[7].h100 = 1.500000;
	newfuels[7].lh = 0.000000;
	newfuels[7].lw = 0.370000;
	newfuels[7].dynamic = 0;
	newfuels[7].sav1 = 1550;
	newfuels[7].savlh = 1800;
	newfuels[7].savlw = 1500;
	newfuels[7].depth = 2.500000;
	newfuels[7].xmext = 0.400000;
	newfuels[7].heatd = 8000.000000;
	newfuels[7].heatl = 8000.000000;
	strcpy(newfuels[7].desc, "Southern Rough");

	newfuels[8].number = 8;
	strcpy(newfuels[8].code, "FM8");
	newfuels[8].h1 = 1.500000;
	newfuels[8].h10 = 1.000000;
	newfuels[8].h100 = 2.500000;
	newfuels[8].lh = 0.000000;
	newfuels[8].lw = 0.000000;
	newfuels[8].dynamic = 0;
	newfuels[8].sav1 = 2000;
	newfuels[8].savlh = 1800;
	newfuels[8].savlw = 1500;
	newfuels[8].depth = 0.200000;
	newfuels[8].xmext = 0.300000;
	newfuels[8].heatd = 8000.000000;
	newfuels[8].heatl = 8000.000000;
	strcpy(newfuels[8].desc, "Closed Timber Litter");

	newfuels[9].number = 9;
	strcpy(newfuels[9].code, "FM9");
	newfuels[9].h1 = 2.920000;
	newfuels[9].h10 = 0.410000;
	newfuels[9].h100 = 0.150000;
	newfuels[9].lh = 0.000000;
	newfuels[9].lw = 0.000000;
	newfuels[9].dynamic = 0;
	newfuels[9].sav1 = 2500;
	newfuels[9].savlh = 1800;
	newfuels[9].savlw = 1500;
	newfuels[9].depth = 0.200000;
	newfuels[9].xmext = 0.250000;
	newfuels[9].heatd = 8000.000000;
	newfuels[9].heatl = 8000.000000;
	strcpy(newfuels[9].desc, "Hardwood Litter");

	newfuels[10].number = 10;
	strcpy(newfuels[10].code, "FM10");
	newfuels[10].h1 = 3.010000;
	newfuels[10].h10 = 2.000000;
	newfuels[10].h100 = 5.010000;
	newfuels[10].lh = 0.000000;
	newfuels[10].lw = 2.000000;
	newfuels[10].dynamic = 0;
	newfuels[10].sav1 = 2000;
	newfuels[10].savlh = 1800;
	newfuels[10].savlw = 1500;
	newfuels[10].depth = 1.000000;
	newfuels[10].xmext = 0.250000;
	newfuels[10].heatd = 8000.000000;
	newfuels[10].heatl = 8000.000000;
	strcpy(newfuels[10].desc, "Timber Litter/Understory");

	newfuels[11].number = 11;
	strcpy(newfuels[11].code, "FM11");
	newfuels[11].h1 = 1.500000;
	newfuels[11].h10 = 4.510000;
	newfuels[11].h100 = 5.510000;
	newfuels[11].lh = 0.000000;
	newfuels[11].lw = 0.000000;
	newfuels[11].dynamic = 0;
	newfuels[11].sav1 = 1500;
	newfuels[11].savlh = 1800;
	newfuels[11].savlw = 1500;
	newfuels[11].depth = 1.000000;
	newfuels[11].xmext = 0.150000;
	newfuels[11].heatd = 8000.000000;
	newfuels[11].heatl = 8000.000000;
	strcpy(newfuels[11].desc, "Light Slash");

	newfuels[12].number = 12;
	strcpy(newfuels[12].code, "FM12");
	newfuels[12].h1 = 4.010000;
	newfuels[12].h10 = 14.030000;
	newfuels[12].h100 = 16.530000;
	newfuels[12].lh = 0.000000;
	newfuels[12].lw = 0.000000;
	newfuels[12].dynamic = 0;
	newfuels[12].sav1 = 1500;
	newfuels[12].savlh = 1800;
	newfuels[12].savlw = 1500;
	newfuels[12].depth = 2.300000;
	newfuels[12].xmext = 0.200000;
	newfuels[12].heatd = 8000.000000;
	newfuels[12].heatl = 8000.000000;
	strcpy(newfuels[12].desc, "Medium Slash");

	newfuels[13].number = 13;
	strcpy(newfuels[13].code, "FM13");
	newfuels[13].h1 = 7.010000;
	newfuels[13].h10 = 23.040000;
	newfuels[13].h100 = 28.050000;
	newfuels[13].lh = 0.000000;
	newfuels[13].lw = 0.000000;
	newfuels[13].dynamic = 0;
	newfuels[13].sav1 = 1500;
	newfuels[13].savlh = 1800;
	newfuels[13].savlw = 1500;
	newfuels[13].depth = 3.000000;
	newfuels[13].xmext = 0.250000;
	newfuels[13].heatd = 8000.000000;
	newfuels[13].heatl = 8000.000000;
	strcpy(newfuels[13].desc, "Heavy Slash");

	newfuels[99].number = 99;
	strcpy(newfuels[99].code, "NB1");
	strcpy(newfuels[99].desc, "Barren");

	newfuels[98].number = 98;
	strcpy(newfuels[98].code, "NB2");
	strcpy(newfuels[98].desc, "Water");

	newfuels[97].number = 97;
	strcpy(newfuels[97].code, "NB3");
	strcpy(newfuels[97].desc, "Snow or Ice");

	newfuels[96].number = 96;
	strcpy(newfuels[96].code, "NB4");
	strcpy(newfuels[96].desc, "Agricultural or Cropland");

	newfuels[95].number = 95;
	strcpy(newfuels[95].code, "NB5");
	strcpy(newfuels[95].desc, "Urban or Developed");

	newfuels[101].number = 101;
	strcpy(newfuels[101].code, "GR1");
	newfuels[101].h1 = 0.100000;
	newfuels[101].h10 = 0.000000;
	newfuels[101].h100 = 0.000000;
	newfuels[101].lh = 0.300000;
	newfuels[101].lw = 0.000000;
	newfuels[101].dynamic = 1;
	newfuels[101].sav1 = 2200;
	newfuels[101].savlh = 2000;
	newfuels[101].savlw = 1500;
	newfuels[101].depth = 0.400000;
	newfuels[101].xmext = 0.150000;
	newfuels[101].heatd = 8000.000000;
	newfuels[101].heatl = 8000.000000;
	strcpy(newfuels[101].desc, "Short, sparse, dry climate grass");

	newfuels[102].number = 102;
	strcpy(newfuels[102].code, "GR2");
	newfuels[102].h1 = 0.100000;
	newfuels[102].h10 = 0.000000;
	newfuels[102].h100 = 0.000000;
	newfuels[102].lh = 1.000000;
	newfuels[102].lw = 0.000000;
	newfuels[102].dynamic = 1;
	newfuels[102].sav1 = 2000;
	newfuels[102].savlh = 1800;
	newfuels[102].savlw = 1500;
	newfuels[102].depth = 1.000000;
	newfuels[102].xmext = 0.150000;
	newfuels[102].heatd = 8000.000000;
	newfuels[102].heatl = 8000.000000;
	strcpy(newfuels[102].desc, "Low load, dry climate grass");

	newfuels[103].number = 103;
	strcpy(newfuels[103].code, "GR3");
	newfuels[103].h1 = 0.100000;
	newfuels[103].h10 = 0.400000;
	newfuels[103].h100 = 0.000000;
	newfuels[103].lh = 1.500000;
	newfuels[103].lw = 0.000000;
	newfuels[103].dynamic = 1;
	newfuels[103].sav1 = 1500;
	newfuels[103].savlh = 1300;
	newfuels[103].savlw = 1500;
	newfuels[103].depth = 2.000000;
	newfuels[103].xmext = 0.300000;
	newfuels[103].heatd = 8000.000000;
	newfuels[103].heatl = 8000.000000;
	strcpy(newfuels[103].desc, "Low load, very coarse, humid climate grass");

	newfuels[104].number = 104;
	strcpy(newfuels[104].code, "GR4");
	newfuels[104].h1 = 0.250000;
	newfuels[104].h10 = 0.000000;
	newfuels[104].h100 = 0.000000;
	newfuels[104].lh = 1.900000;
	newfuels[104].lw = 0.000000;
	newfuels[104].dynamic = 1;
	newfuels[104].sav1 = 2000;
	newfuels[104].savlh = 1800;
	newfuels[104].savlw = 1500;
	newfuels[104].depth = 2.000000;
	newfuels[104].xmext = 0.150000;
	newfuels[104].heatd = 8000.000000;
	newfuels[104].heatl = 8000.000000;
	strcpy(newfuels[104].desc, "Moderate load, dry climate grass");

	newfuels[105].number = 105;
	strcpy(newfuels[105].code, "GR5");
	newfuels[105].h1 = 0.400000;
	newfuels[105].h10 = 0.000000;
	newfuels[105].h100 = 0.000000;
	newfuels[105].lh = 2.500000;
	newfuels[105].lw = 0.000000;
	newfuels[105].dynamic = 1;
	newfuels[105].sav1 = 1800;
	newfuels[105].savlh = 1600;
	newfuels[105].savlw = 1500;
	newfuels[105].depth = 1.500000;
	newfuels[105].xmext = 0.400000;
	newfuels[105].heatd = 8000.000000;
	newfuels[105].heatl = 8000.000000;
	strcpy(newfuels[105].desc, "Low load, humid climate grass");

	newfuels[106].number = 106;
	strcpy(newfuels[106].code, "GR6");
	newfuels[106].h1 = 0.100000;
	newfuels[106].h10 = 0.000000;
	newfuels[106].h100 = 0.000000;
	newfuels[106].lh = 3.400000;
	newfuels[106].lw = 0.000000;
	newfuels[106].dynamic = 1;
	newfuels[106].sav1 = 2200;
	newfuels[106].savlh = 2000;
	newfuels[106].savlw = 1500;
	newfuels[106].depth = 1.500000;
	newfuels[106].xmext = 0.400000;
	newfuels[106].heatd = 9000.000000;
	newfuels[106].heatl = 9000.000000;
	strcpy(newfuels[106].desc, "Moderate load, humid climate grass");

	newfuels[107].number = 107;
	strcpy(newfuels[107].code, "GR7");
	newfuels[107].h1 = 1.000000;
	newfuels[107].h10 = 0.000000;
	newfuels[107].h100 = 0.000000;
	newfuels[107].lh = 5.400000;
	newfuels[107].lw = 0.000000;
	newfuels[107].dynamic = 1;
	newfuels[107].sav1 = 2000;
	newfuels[107].savlh = 1800;
	newfuels[107].savlw = 1500;
	newfuels[107].depth = 3.000000;
	newfuels[107].xmext = 0.150000;
	newfuels[107].heatd = 8000.000000;
	newfuels[107].heatl = 8000.000000;
	strcpy(newfuels[107].desc, "High load, dry climate grass");

	newfuels[108].number = 108;
	strcpy(newfuels[108].code, "GR8");
	newfuels[108].h1 = 0.500000;
	newfuels[108].h10 = 1.000000;
	newfuels[108].h100 = 0.000000;
	newfuels[108].lh = 7.300000;
	newfuels[108].lw = 0.000000;
	newfuels[108].dynamic = 1;
	newfuels[108].sav1 = 1500;
	newfuels[108].savlh = 1300;
	newfuels[108].savlw = 1500;
	newfuels[108].depth = 4.000000;
	newfuels[108].xmext = 0.300000;
	newfuels[108].heatd = 8000.000000;
	newfuels[108].heatl = 8000.000000;
	strcpy(newfuels[108].desc, "High load, very coarse, humid climate grass");

	newfuels[109].number = 109;
	strcpy(newfuels[109].code, "GR9");
	newfuels[109].h1 = 1.000000;
	newfuels[109].h10 = 1.000000;
	newfuels[109].h100 = 0.000000;
	newfuels[109].lh = 9.000000;
	newfuels[109].lw = 0.000000;
	newfuels[109].dynamic = 1;
	newfuels[109].sav1 = 1800;
	newfuels[109].savlh = 1600;
	newfuels[109].savlw = 1500;
	newfuels[109].depth = 5.000000;
	newfuels[109].xmext = 0.400000;
	newfuels[109].heatd = 8000.000000;
	newfuels[109].heatl = 8000.000000;
	strcpy(newfuels[109].desc, "Very high load, humid climate grass");

	newfuels[121].number = 121;
	strcpy(newfuels[121].code, "GS1");
	newfuels[121].h1 = 0.200000;
	newfuels[121].h10 = 0.000000;
	newfuels[121].h100 = 0.000000;
	newfuels[121].lh = 0.500000;
	newfuels[121].lw = 0.650000;
	newfuels[121].dynamic = 1;
	newfuels[121].sav1 = 2000;
	newfuels[121].savlh = 1800;
	newfuels[121].savlw = 1800;
	newfuels[121].depth = 0.900000;
	newfuels[121].xmext = 0.150000;
	newfuels[121].heatd = 8000.000000;
	newfuels[121].heatl = 8000.000000;
	strcpy(newfuels[121].desc, "Low load, dry climate grass-shrub");

	newfuels[122].number = 122;
	strcpy(newfuels[122].code, "GS2");
	newfuels[122].h1 = 0.500000;
	newfuels[122].h10 = 0.500000;
	newfuels[122].h100 = 0.000000;
	newfuels[122].lh = 0.600000;
	newfuels[122].lw = 1.000000;
	newfuels[122].dynamic = 1;
	newfuels[122].sav1 = 2000;
	newfuels[122].savlh = 1800;
	newfuels[122].savlw = 1800;
	newfuels[122].depth = 1.500000;
	newfuels[122].xmext = 0.150000;
	newfuels[122].heatd = 8000.000000;
	newfuels[122].heatl = 8000.000000;
	strcpy(newfuels[122].desc, "Moderate load, dry climate grass-shrub");

	newfuels[123].number = 123;
	strcpy(newfuels[123].code, "GS3");
	newfuels[123].h1 = 0.300000;
	newfuels[123].h10 = 0.250000;
	newfuels[123].h100 = 0.000000;
	newfuels[123].lh = 1.450000;
	newfuels[123].lw = 1.250000;
	newfuels[123].dynamic = 1;
	newfuels[123].sav1 = 1800;
	newfuels[123].savlh = 1600;
	newfuels[123].savlw = 1600;
	newfuels[123].depth = 1.800000;
	newfuels[123].xmext = 0.400000;
	newfuels[123].heatd = 8000.000000;
	newfuels[123].heatl = 8000.000000;
	strcpy(newfuels[123].desc, "Moderate load, humid climate grass-shrub");

	newfuels[124].number = 124;
	strcpy(newfuels[124].code, "GS4");
	newfuels[124].h1 = 1.900000;
	newfuels[124].h10 = 0.300000;
	newfuels[124].h100 = 0.100000;
	newfuels[124].lh = 3.400000;
	newfuels[124].lw = 7.100000;
	newfuels[124].dynamic = 1;
	newfuels[124].sav1 = 1800;
	newfuels[124].savlh = 1600;
	newfuels[124].savlw = 1600;
	newfuels[124].depth = 2.100000;
	newfuels[124].xmext = 0.400000;
	newfuels[124].heatd = 8000.000000;
	newfuels[124].heatl = 8000.000000;
	strcpy(newfuels[124].desc, "High load, humid climate grass-shrub");

	newfuels[141].number = 141;
	strcpy(newfuels[141].code, "SH1");
	newfuels[141].h1 = 0.250000;
	newfuels[141].h10 = 0.250000;
	newfuels[141].h100 = 0.000000;
	newfuels[141].lh = 0.150000;
	newfuels[141].lw = 1.300000;
	newfuels[141].dynamic = 1;
	newfuels[141].sav1 = 2000;
	newfuels[141].savlh = 1800;
	newfuels[141].savlw = 1600;
	newfuels[141].depth = 1.000000;
	newfuels[141].xmext = 0.150000;
	newfuels[141].heatd = 8000.000000;
	newfuels[141].heatl = 8000.000000;
	strcpy(newfuels[141].desc, "Low load, dry climate shrub");

	newfuels[142].number = 142;
	strcpy(newfuels[142].code, "SH2");
	newfuels[142].h1 = 1.350000;
	newfuels[142].h10 = 2.400000;
	newfuels[142].h100 = 0.750000;
	newfuels[142].lh = 0.000000;
	newfuels[142].lw = 3.850000;
	newfuels[142].dynamic = 0;
	newfuels[142].sav1 = 2000;
	newfuels[142].savlh = 1800;
	newfuels[142].savlw = 1600;
	newfuels[142].depth = 1.000000;
	newfuels[142].xmext = 0.150000;
	newfuels[142].heatd = 8000.000000;
	newfuels[142].heatl = 8000.000000;
	strcpy(newfuels[142].desc, "Moderate load, dry climate shrub");

	newfuels[143].number = 143;
	strcpy(newfuels[143].code, "SH3");
	newfuels[143].h1 = 0.450000;
	newfuels[143].h10 = 3.000000;
	newfuels[143].h100 = 0.000000;
	newfuels[143].lh = 0.000000;
	newfuels[143].lw = 6.200000;
	newfuels[143].dynamic = 0;
	newfuels[143].sav1 = 1600;
	newfuels[143].savlh = 1800;
	newfuels[143].savlw = 1400;
	newfuels[143].depth = 2.400000;
	newfuels[143].xmext = 0.400000;
	newfuels[143].heatd = 8000.000000;
	newfuels[143].heatl = 8000.000000;
	strcpy(newfuels[143].desc, "Moderate load, humid climate shrub");

	newfuels[144].number = 144;
	strcpy(newfuels[144].code, "SH4");
	newfuels[144].h1 = 0.850000;
	newfuels[144].h10 = 1.150000;
	newfuels[144].h100 = 0.200000;
	newfuels[144].lh = 0.000000;
	newfuels[144].lw = 2.550000;
	newfuels[144].dynamic = 0;
	newfuels[144].sav1 = 2000;
	newfuels[144].savlh = 1800;
	newfuels[144].savlw = 1600;
	newfuels[144].depth = 3.000000;
	newfuels[144].xmext = 0.300000;
	newfuels[144].heatd = 8000.000000;
	newfuels[144].heatl = 8000.000000;
	strcpy(newfuels[144].desc, "Low load, humid climate timber-shrub");

	newfuels[145].number = 145;
	strcpy(newfuels[145].code, "SH5");
	newfuels[145].h1 = 3.600000;
	newfuels[145].h10 = 2.100000;
	newfuels[145].h100 = 0.000000;
	newfuels[145].lh = 0.000000;
	newfuels[145].lw = 2.900000;
	newfuels[145].dynamic = 0;
	newfuels[145].sav1 = 750;
	newfuels[145].savlh = 1800;
	newfuels[145].savlw = 1600;
	newfuels[145].depth = 6.000000;
	newfuels[145].xmext = 0.150000;
	newfuels[145].heatd = 8000.000000;
	newfuels[145].heatl = 8000.000000;
	strcpy(newfuels[145].desc, "High load, dry climate shrub");

	newfuels[146].number = 146;
	strcpy(newfuels[146].code, "SH6");
	newfuels[146].h1 = 2.900000;
	newfuels[146].h10 = 1.450000;
	newfuels[146].h100 = 0.000000;
	newfuels[146].lh = 0.000000;
	newfuels[146].lw = 1.400000;
	newfuels[146].dynamic = 0;
	newfuels[146].sav1 = 750;
	newfuels[146].savlh = 1800;
	newfuels[146].savlw = 1600;
	newfuels[146].depth = 2.000000;
	newfuels[146].xmext = 0.300000;
	newfuels[146].heatd = 8000.000000;
	newfuels[146].heatl = 8000.000000;
	strcpy(newfuels[146].desc, "Low load, humid climate shrub");

	newfuels[147].number = 147;
	strcpy(newfuels[147].code, "SH7");
	newfuels[147].h1 = 3.500000;
	newfuels[147].h10 = 5.300000;
	newfuels[147].h100 = 2.200000;
	newfuels[147].lh = 0.000000;
	newfuels[147].lw = 3.400000;
	newfuels[147].dynamic = 0;
	newfuels[147].sav1 = 750;
	newfuels[147].savlh = 1800;
	newfuels[147].savlw = 1600;
	newfuels[147].depth = 6.000000;
	newfuels[147].xmext = 0.150000;
	newfuels[147].heatd = 8000.000000;
	newfuels[147].heatl = 8000.000000;
	strcpy(newfuels[147].desc, "Very high load, dry climate shrub");

	newfuels[148].number = 148;
	strcpy(newfuels[148].code, "SH8");
	newfuels[148].h1 = 2.050000;
	newfuels[148].h10 = 3.400000;
	newfuels[148].h100 = 0.850000;
	newfuels[148].lh = 0.000000;
	newfuels[148].lw = 4.350000;
	newfuels[148].dynamic = 0;
	newfuels[148].sav1 = 750;
	newfuels[148].savlh = 1800;
	newfuels[148].savlw = 1600;
	newfuels[148].depth = 3.000000;
	newfuels[148].xmext = 0.400000;
	newfuels[148].heatd = 8000.000000;
	newfuels[148].heatl = 8000.000000;
	strcpy(newfuels[148].desc, "High load, humid climate shrub");

	newfuels[149].number = 149;
	strcpy(newfuels[149].code, "SH9");
	newfuels[149].h1 = 4.500000;
	newfuels[149].h10 = 2.450000;
	newfuels[149].h100 = 0.000000;
	newfuels[149].lh = 1.550000;
	newfuels[149].lw = 7.000000;
	newfuels[149].dynamic = 1;
	newfuels[149].sav1 = 750;
	newfuels[149].savlh = 1800;
	newfuels[149].savlw = 1500;
	newfuels[149].depth = 4.400000;
	newfuels[149].xmext = 0.400000;
	newfuels[149].heatd = 8000.000000;
	newfuels[149].heatl = 8000.000000;
	strcpy(newfuels[149].desc, "Very high load, humid climate shrub");

	newfuels[161].number = 161;
	strcpy(newfuels[161].code, "TU1");
	newfuels[161].h1 = 0.200000;
	newfuels[161].h10 = 0.900000;
	newfuels[161].h100 = 1.500000;
	newfuels[161].lh = 0.200000;
	newfuels[161].lw = 0.900000;
	newfuels[161].dynamic = 1;
	newfuels[161].sav1 = 2000;
	newfuels[161].savlh = 1800;
	newfuels[161].savlw = 1600;
	newfuels[161].depth = 0.600000;
	newfuels[161].xmext = 0.200000;
	newfuels[161].heatd = 8000.000000;
	newfuels[161].heatl = 8000.000000;
	strcpy(newfuels[161].desc, "Light load, dry climate timber-grass-shrub");

	newfuels[162].number = 162;
	strcpy(newfuels[162].code, "TU2");
	newfuels[162].h1 = 0.950000;
	newfuels[162].h10 = 1.800000;
	newfuels[162].h100 = 1.250000;
	newfuels[162].lh = 0.000000;
	newfuels[162].lw = 0.200000;
	newfuels[162].dynamic = 0;
	newfuels[162].sav1 = 2000;
	newfuels[162].savlh = 1800;
	newfuels[162].savlw = 1600;
	newfuels[162].depth = 1.000000;
	newfuels[162].xmext = 0.300000;
	newfuels[162].heatd = 8000.000000;
	newfuels[162].heatl = 8000.000000;
	strcpy(newfuels[162].desc, "Moderate load, humid climate timber-shrub");

	newfuels[163].number = 163;
	strcpy(newfuels[163].code, "TU3");
	newfuels[163].h1 = 1.100000;
	newfuels[163].h10 = 0.150000;
	newfuels[163].h100 = 0.250000;
	newfuels[163].lh = 0.650000;
	newfuels[163].lw = 1.100000;
	newfuels[163].dynamic = 1;
	newfuels[163].sav1 = 1800;
	newfuels[163].savlh = 1600;
	newfuels[163].savlw = 1400;
	newfuels[163].depth = 1.300000;
	newfuels[163].xmext = 0.300000;
	newfuels[163].heatd = 8000.000000;
	newfuels[163].heatl = 8000.000000;
	strcpy(newfuels[163].desc,
		"Moderate load, humid climate timber-grass-shrub");

	newfuels[164].number = 164;
	strcpy(newfuels[164].code, "TU4");
	newfuels[164].h1 = 4.500000;
	newfuels[164].h10 = 0.000000;
	newfuels[164].h100 = 0.000000;
	newfuels[164].lh = 0.000000;
	newfuels[164].lw = 2.000000;
	newfuels[164].dynamic = 0;
	newfuels[164].sav1 = 2300;
	newfuels[164].savlh = 1800;
	newfuels[164].savlw = 2000;
	newfuels[164].depth = 0.500000;
	newfuels[164].xmext = 0.120000;
	newfuels[164].heatd = 8000.000000;
	newfuels[164].heatl = 8000.000000;
	strcpy(newfuels[164].desc, "Dwarf conifer with understory");

	newfuels[165].number = 165;
	strcpy(newfuels[165].code, "TU5");
	newfuels[165].h1 = 4.000000;
	newfuels[165].h10 = 4.000000;
	newfuels[165].h100 = 3.000000;
	newfuels[165].lh = 0.000000;
	newfuels[165].lw = 3.000000;
	newfuels[165].dynamic = 0;
	newfuels[165].sav1 = 1500;
	newfuels[165].savlh = 1800;
	newfuels[165].savlw = 750;
	newfuels[165].depth = 1.000000;
	newfuels[165].xmext = 0.250000;
	newfuels[165].heatd = 8000.000000;
	newfuels[165].heatl = 8000.000000;
	strcpy(newfuels[165].desc, "Very high load, dry climate timber-shrub");

	newfuels[181].number = 181;
	strcpy(newfuels[181].code, "TL1");
	newfuels[181].h1 = 1.000000;
	newfuels[181].h10 = 2.200000;
	newfuels[181].h100 = 3.600000;
	newfuels[181].lh = 0.000000;
	newfuels[181].lw = 0.000000;
	newfuels[181].dynamic = 0;
	newfuels[181].sav1 = 2000;
	newfuels[181].savlh = 1800;
	newfuels[181].savlw = 1600;
	newfuels[181].depth = 0.200000;
	newfuels[181].xmext = 0.300000;
	newfuels[181].heatd = 8000.000000;
	newfuels[181].heatl = 8000.000000;
	strcpy(newfuels[181].desc, "Low load, compact conifer litter");

	newfuels[182].number = 182;
	strcpy(newfuels[182].code, "TL2");
	newfuels[182].h1 = 1.400000;
	newfuels[182].h10 = 2.300000;
	newfuels[182].h100 = 2.200000;
	newfuels[182].lh = 0.000000;
	newfuels[182].lw = 0.000000;
	newfuels[182].dynamic = 0;
	newfuels[182].sav1 = 2000;
	newfuels[182].savlh = 1800;
	newfuels[182].savlw = 1600;
	newfuels[182].depth = 0.200000;
	newfuels[182].xmext = 0.250000;
	newfuels[182].heatd = 8000.000000;
	newfuels[182].heatl = 8000.000000;
	strcpy(newfuels[182].desc, "Low load broadleaf litter");

	newfuels[183].number = 183;
	strcpy(newfuels[183].code, "TL3");
	newfuels[183].h1 = 0.500000;
	newfuels[183].h10 = 2.200000;
	newfuels[183].h100 = 2.800000;
	newfuels[183].lh = 0.000000;
	newfuels[183].lw = 0.000000;
	newfuels[183].dynamic = 0;
	newfuels[183].sav1 = 2000;
	newfuels[183].savlh = 1800;
	newfuels[183].savlw = 1600;
	newfuels[183].depth = 0.300000;
	newfuels[183].xmext = 0.200000;
	newfuels[183].heatd = 8000.000000;
	newfuels[183].heatl = 8000.000000;
	strcpy(newfuels[183].desc, "Moderate load confier litter");

	newfuels[184].number = 184;
	strcpy(newfuels[184].code, "TL4");
	newfuels[184].h1 = 0.500000;
	newfuels[184].h10 = 1.500000;
	newfuels[184].h100 = 4.200000;
	newfuels[184].lh = 0.000000;
	newfuels[184].lw = 0.000000;
	newfuels[184].dynamic = 0;
	newfuels[184].sav1 = 2000;
	newfuels[184].savlh = 1800;
	newfuels[184].savlw = 1600;
	newfuels[184].depth = 0.400000;
	newfuels[184].xmext = 0.250000;
	newfuels[184].heatd = 8000.000000;
	newfuels[184].heatl = 8000.000000;
	strcpy(newfuels[184].desc, "Small downed logs");

	newfuels[185].number = 185;
	strcpy(newfuels[185].code, "TL5");
	newfuels[185].h1 = 1.150000;
	newfuels[185].h10 = 2.500000;
	newfuels[185].h100 = 4.400000;
	newfuels[185].lh = 0.000000;
	newfuels[185].lw = 0.000000;
	newfuels[185].dynamic = 0;
	newfuels[185].sav1 = 2000;
	newfuels[185].savlh = 1800;
	newfuels[185].savlw = 1600;
	newfuels[185].depth = 0.600000;
	newfuels[185].xmext = 0.250000;
	newfuels[185].heatd = 8000.000000;
	newfuels[185].heatl = 8000.000000;
	strcpy(newfuels[185].desc, "High load conifer litter");

	newfuels[186].number = 186;
	strcpy(newfuels[186].code, "TL6");
	newfuels[186].h1 = 2.400000;
	newfuels[186].h10 = 1.200000;
	newfuels[186].h100 = 1.200000;
	newfuels[186].lh = 0.000000;
	newfuels[186].lw = 0.000000;
	newfuels[186].dynamic = 0;
	newfuels[186].sav1 = 2000;
	newfuels[186].savlh = 1800;
	newfuels[186].savlw = 1600;
	newfuels[186].depth = 0.300000;
	newfuels[186].xmext = 0.250000;
	newfuels[186].heatd = 8000.000000;
	newfuels[186].heatl = 8000.000000;
	strcpy(newfuels[186].desc, "High load broadleaf litter");

	newfuels[187].number = 187;
	strcpy(newfuels[187].code, "TL7");
	newfuels[187].h1 = 0.300000;
	newfuels[187].h10 = 1.400000;
	newfuels[187].h100 = 8.100000;
	newfuels[187].lh = 0.000000;
	newfuels[187].lw = 0.000000;
	newfuels[187].dynamic = 0;
	newfuels[187].sav1 = 2000;
	newfuels[187].savlh = 1800;
	newfuels[187].savlw = 1600;
	newfuels[187].depth = 0.400000;
	newfuels[187].xmext = 0.250000;
	newfuels[187].heatd = 8000.000000;
	newfuels[187].heatl = 8000.000000;
	strcpy(newfuels[187].desc, "Large downed logs");

	newfuels[188].number = 188;
	strcpy(newfuels[188].code, "TL8");
	newfuels[188].h1 = 5.800000;
	newfuels[188].h10 = 1.400000;
	newfuels[188].h100 = 1.100000;
	newfuels[188].lh = 0.000000;
	newfuels[188].lw = 0.000000;
	newfuels[188].dynamic = 0;
	newfuels[188].sav1 = 1800;
	newfuels[188].savlh = 1800;
	newfuels[188].savlw = 1600;
	newfuels[188].depth = 0.300000;
	newfuels[188].xmext = 0.350000;
	newfuels[188].heatd = 8000.000000;
	newfuels[188].heatl = 8000.000000;
	strcpy(newfuels[188].desc, "Long-needle litter");

	newfuels[189].number = 189;
	strcpy(newfuels[189].code, "TL9");
	newfuels[189].h1 = 6.650000;
	newfuels[189].h10 = 3.300000;
	newfuels[189].h100 = 4.150000;
	newfuels[189].lh = 0.000000;
	newfuels[189].lw = 0.000000;
	newfuels[189].dynamic = 0;
	newfuels[189].sav1 = 1800;
	newfuels[189].savlh = 1800;
	newfuels[189].savlw = 1600;
	newfuels[189].depth = 0.600000;
	newfuels[189].xmext = 0.350000;
	newfuels[189].heatd = 8000.000000;
	newfuels[189].heatl = 8000.000000;
	strcpy(newfuels[189].desc, "Very high load broadleaf litter");

	newfuels[201].number = 201;
	strcpy(newfuels[201].code, "SB1");
	newfuels[201].h1 = 1.500000;
	newfuels[201].h10 = 3.000000;
	newfuels[201].h100 = 11.000000;
	newfuels[201].lh = 0.000000;
	newfuels[201].lw = 0.000000;
	newfuels[201].dynamic = 0;
	newfuels[201].sav1 = 2000;
	newfuels[201].savlh = 1800;
	newfuels[201].savlw = 1600;
	newfuels[201].depth = 1.000000;
	newfuels[201].xmext = 0.250000;
	newfuels[201].heatd = 8000.000000;
	newfuels[201].heatl = 8000.000000;
	strcpy(newfuels[201].desc, "Low load activity fuel");

	newfuels[202].number = 202;
	strcpy(newfuels[202].code, "SB2");
	newfuels[202].h1 = 4.500000;
	newfuels[202].h10 = 4.250000;
	newfuels[202].h100 = 4.000000;
	newfuels[202].lh = 0.000000;
	newfuels[202].lw = 0.000000;
	newfuels[202].dynamic = 0;
	newfuels[202].sav1 = 2000;
	newfuels[202].savlh = 1800;
	newfuels[202].savlw = 1600;
	newfuels[202].depth = 1.000000;
	newfuels[202].xmext = 0.250000;
	newfuels[202].heatd = 8000.000000;
	newfuels[202].heatl = 8000.000000;
	strcpy(newfuels[202].desc, "Moderate load activity or low load blowdown");

	newfuels[203].number = 203;
	strcpy(newfuels[203].code, "SB3");
	newfuels[203].h1 = 5.500000;
	newfuels[203].h10 = 2.750000;
	newfuels[203].h100 = 3.000000;
	newfuels[203].lh = 0.000000;
	newfuels[203].lw = 0.000000;
	newfuels[203].dynamic = 0;
	newfuels[203].sav1 = 2000;
	newfuels[203].savlh = 1800;
	newfuels[203].savlw = 1600;
	newfuels[203].depth = 1.200000;
	newfuels[203].xmext = 0.250000;
	newfuels[203].heatd = 8000.000000;
	newfuels[203].heatl = 8000.000000;
	strcpy(newfuels[203].desc,
		"High load activity fuel or moderate load blowdown");

	newfuels[204].number = 204;
	strcpy(newfuels[204].code, "SB4");
	newfuels[204].h1 = 5.250000;
	newfuels[204].h10 = 3.500000;
	newfuels[204].h100 = 5.250000;
	newfuels[204].lh = 0.000000;
	newfuels[204].lw = 0.000000;
	newfuels[204].dynamic = 0;
	newfuels[204].sav1 = 2000;
	newfuels[204].savlh = 1800;
	newfuels[204].savlw = 1600;
	newfuels[204].depth = 2.700000;
	newfuels[204].xmext = 0.250000;
	newfuels[204].heatd = 8000.000000;
	newfuels[204].heatl = 8000.000000;
	strcpy(newfuels[204].desc, "High load blowdown");
}

Crew* Farsite5::GetCrew(long CrewNumber)
{
	if (crew[CrewNumber])
		return crew[CrewNumber];

	return 0;
}

long Farsite5::SetNewCrew()
{
	// doesn't require crew number
	if ((crew[NumCrews] = new Crew) != NULL)//(Crew *) calloc(1, sizeof(Crew)))!=NULL)
		return NumCrews++;

	return -1;
}

long Farsite5::SetCrew(long CrewNumber)
{
	// requires crew number for allocation
	if ((crew[CrewNumber] = new Crew) != NULL)//(Crew *) calloc(1, sizeof(Crew)))!=NULL)
		return NumCrews++;

	return -1;
}


void Farsite5::FreeAllCrews()
{
	long i;

	for (i = 0; i < NumCrews; i++)
		FreeCrew(0);
	NumCrews = 0;
}


void Farsite5::FreeCrew(long CrewNumber)
{
	long i;
	Crew* crw;

	if (crew[CrewNumber])
	{
		crw = crew[CrewNumber];
		for (i = CrewNumber; i < NumCrews - 1; i++)
			crew[i] = crew[i + 1];
		NumCrews--;
		crew[NumCrews] = crw;
		delete crew[NumCrews];//free(crew[NumCrews]);
		crew[NumCrews] = 0;
	}
}

long Farsite5::GetStartMonth()
{
	return startmonth;
}

long Farsite5::GetStartDay()
{
	return startday;
}

long Farsite5::GetStartHour()
{
	return starthour;
}


long Farsite5::GetStartMin()
{
	return startmin;
}

long Farsite5::GetStartDate()
{
	return startdate;
}

long Farsite5::GetLatitude()
{
	return Header.latitude;
}

long Farsite5::GetWindMonth(long StationNumber, long NumObs)
{
	if (NumObs > MaxWindObs[StationNumber] - 1)
		NumObs = MaxWindObs[StationNumber] - 1;

	return wddt[StationNumber][NumObs].mo;
}

long Farsite5::GetWindDay(long StationNumber, long NumObs)
{
	if (NumObs > MaxWindObs[StationNumber] - 1)
		NumObs = MaxWindObs[StationNumber] - 1;

	return wddt[StationNumber][NumObs].dy;
}

long Farsite5::GetWindHour(long StationNumber, long NumObs)
{
	if (NumObs > MaxWindObs[StationNumber] - 1)
		NumObs = MaxWindObs[StationNumber] - 1;

	return wddt[StationNumber][NumObs].hr;
}

double Farsite5::GetWindSpeed(long StationNumber, long NumObs)
{
	if (NumObs > MaxWindObs[StationNumber] - 1)
		NumObs = MaxWindObs[StationNumber] - 1;

	return wddt[StationNumber][NumObs].ws;
}

long Farsite5::GetWindDir(long StationNumber, long NumObs)
{
	if (NumObs > MaxWindObs[StationNumber] - 1)
		NumObs = MaxWindObs[StationNumber] - 1;

	return wddt[StationNumber][NumObs].wd;
}

long Farsite5::GetWindCloud(long StationNumber, long NumObs)
{
	if (NumObs > MaxWindObs[StationNumber] - 1)
		NumObs = MaxWindObs[StationNumber] - 1;

	return wddt[StationNumber][NumObs].cl;
}
void Farsite5::SetStartMonth(long input)
{
	startmonth = input;
}

void Farsite5::SetStartDay(long input)
{
	startday = input;
}

void Farsite5::SetStartHour(long input)
{
	starthour = input;
}

void Farsite5::SetStartMin(long input)
{
	startmin = input;
}

void Farsite5::SetStartDate(long input)
{
	startdate = input;
}

void Farsite5::SetEndMonth(long input)
{
	endmonth = input;
}

void Farsite5::SetEndDay(long input)
{
	endday = input;
}

void Farsite5::SetEndHour(long input)
{
	endhour = input;
}

void Farsite5::SetEndMin(long input)
{
	endmin = input;
}

void Farsite5::SetEndDate(long input)
{
	enddate = input;
}

void Farsite5::SetRastMake(bool YesNo)
{
	RastMake = YesNo;
}


bool Farsite5::GetRastMake()
{
	return RastMake;
}

void Farsite5::SetVectVisOnly(bool YesNo)
{
	VISONLY = YesNo;
}


bool Farsite5::GetVectVisOnly()
{
	return VISONLY;
}


bool Farsite5::CanModifyInputs(long YesNo)
{
	if (YesNo >= 0)
		CanModify = (bool) YesNo;

	return CanModify;
}

void Farsite5::SetVectFormat(long Type)
{
	VectFormat = Type;
}


long Farsite5::GetVectFormat()
{
	return VectFormat;
}


void Farsite5::SetRastFormat(long Type)
{
	RastFormat = Type;
}


long Farsite5::GetRastFormat()
{
	return RastFormat;
}

bool Farsite5::PreCalcMoistures(long YesNo)
{
	if (YesNo > -1)
		PreCalcFuelMoistures = (bool) YesNo;

	return PreCalcFuelMoistures;
}

void Farsite5::CountFires()
{
	VectorBarriers = 0;
	CountInwardFires = 0;
	CountTotalFires = GetNewFires();//GetNumFires();
	for (long i = 0; i < CountTotalFires; i++)
	{
		if (GetInout(i) == 2)
			CountInwardFires++;
		if (GetInout(i) == 3)
			VectorBarriers++;
	}
	CountTotalFires -= VectorBarriers;
	/* DDE.NumFires=CountTotalFires;
	 DDE.NumBarriers=VectorBarriers;
	 DDE.StartStatus=(long) SIMULATE_GO;
	 DDE.ResumeStatus=(long) FARSITE_GO;
	 */
}




double Farsite5::GetSimulationDuration()
{
	return SimulationDuration;
}

void Farsite5::SetSimulationDuration(double simdur)
{
	SimulationDuration = simdur;
}

long Farsite5::ShowFiresAsGrown(long YesNo)
{
	if (YesNo >= 0)
		ShowVectsAsGrown = YesNo;

	return ShowVectsAsGrown;
}

long Farsite5::ExcludeBarriersFromVectorFiles(long YesNo)
{
	if (YesNo >= 0)
		ShapeBarriersSeparately = YesNo;

	return ShapeBarriersSeparately;
}




/****************************************************************************/
/****************************************************************************/
/****************************************************************************/




void Farsite5::CheckSteps()
{
	if (NextFireAfterInterrupt == 0)
	{
		burn.rastmake = GetRastMake();					// update decision to make raster file
		SetNumFires(GetNewFires());				// newfires set from mouse or from file
		/*long oldequal = PrimaryVisEqual; 			// recalc PrimaryVisEqual from new visual and actual ts
		if (oldequal !=
			(PrimaryVisEqual = (long)
			(GetVisibleTimeStep() / GetActualTimeStep())))
		{
			PrimaryVisCount = 0;					// reset visible interval to 0;
			SecondaryVisCount = 0;  				  // reset secondary visible interval to 0;
		}
		if (GetSecondaryVisibleTimeStep() != -1)
			SecondaryVisEqual = (long)
				(GetSecondaryVisibleTimeStep() / GetActualTimeStep());
		else
			SecondaryVisCount = SecondaryVisEqual - 1;*/
		//if (!VISDIM)
			maximum = (long)GetSimulationDuration();//numaximum;
		//else						  			 // maximum time is determined by visual time steps
		//	maximum = GetSimulationDuration() * GetVisibleTimeStep();
		/*if (DistanceCheckMethod(GETVAL) == DISTCHECK_FIRELEVEL)
			OutputVectsAsGrown = ShowFiresAsGrown(GETVAL);
		else
			OutputVectsAsGrown = 0;*/
		CheckThreadNumbers();
	}
	// DDE.StartStatus=(long) SIMULATE_GO;
	// DDE.ResumeStatus=(long) FARSITE_GO;
	SIM_COMPLETED = false;
}

bool Farsite5::CheckThreadNumbers()
{
	if (NumSimThreads != GetMaxThreads())
	{
		burn.post.ResetAllThreads();
		burn.ResetAllPerimThreads();
//		burn.env->ResetAllThreads();
		burn.CloseCrossThreads();
		//Sleep(100);
		//add EB
		ResetThreads();
		//  ::SendMessage(Client->HWindow, WM_COMMAND, CM_RESETTHREADS, NULL);
	}

	return true;
}

void Farsite5::SetVectMake(bool YesNo)
{
	VectMake = YesNo;
}


bool Farsite5::GetVectMake()
{
	return VectMake;
}

void Farsite5::SetShapeMake(bool YesNo)
{
	ShapeMake = YesNo;
}


bool Farsite5::GetShapeMake()
{
	return ShapeMake;
}

/*****************************************************************************************
* THIS FUNCTION ACTIVATES THE FARSITE LOOP AND CHECKS FOR MESSAGES BETWEEN ITERATIONS
*
* Ret: 1 = ok
*      < 0 Error Number, use icf.ErrorMessage()
******************************************************************************************/
int  Farsite5::FarsiteSimulationLoop()
{
int i_Ret,  CondSw = 0;
//bool b;
//char cr_ErrMes[1000];

    i_Ret = 1;

// Init the Burn Spot Grid Class, this keeps track of burned cells and spot hits
	if(icf.f_FarsiteSpotGridResolution > 0.0)
	{
	    //char nullStr[16] = "";
		m_BSG.Init( GetLoEast(), GetHiEast(),
			GetLoNorth(), GetHiNorth(),
			icf.f_FarsiteSpotGridResolution); // A file name here creates a test output file
		AddIgnitionToSpotGrid();
	}

     m_FPC.SetTimeSlice (0,this);

    m_FPC.Set_CondRunning(); /* Progress Class Conditioning will start */


	CheckSteps();						// check visual and actual timestep for changes
	systime1 = clock();
	systime2 = systime1 + 1;					// keep systime2-systime1>0.0

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-..-*/
	 while (burn.SIMTIME <= maximum)	{

		  if (maximum > 0)
			   SetFarsiteProgress (burn.SIMTIME / (double)maximum);

		  if (LEAVEPROCESS)				// allows exit from FARSITE process control
		   	break;

		  systime = clock();

		  if (SimRequest != SIMREQ_NULL) 	{
		   	CanModifyInputs(true);
			   break ;	}

		  CanModifyInputs(false);
		  CheckSteps();

		  if (FARSITE_GO)		{
	  		 if (PreCalcMoistures(GETVAL)) {
        if ( CondSw == 0 ) {             /* Do once to do all conditioning */
//          SetFarsiteRunStatus (e_Condition);
          i_Ret = this->Run_CondDLL();       /* Run Moisture Cond, load/check inputs, run */

          m_FPC.Set_FarsiteRunning();  /* Notify Process Class, Farsite is runnin */

//          SetFarsiteRunStatus (e_Farsite);
          if ( i_Ret < 0 )              /* Error or terminate requested */
            break;

          CondSw++;                     /* Set one time only switch */
//        this->cfmc.ExportMoistureDataText("C:/larryf/stutest/Test.txt", "");
        }

// Original...............
//	 PreCalculateFuelMoisturesNoUI();
//----------------------------------------------------------

			  	  if (!FARSITE_GO)	{  // can happen if moistures are canceled
				   	  FARSITE_GO = true;
				   	  SimRequest = SIMREQ_RESUMESUSPEND;
					     continue;	}
			   }

			else	{
// This original code never gets called so - ok to comment out
//			 	burn.env->CalcMapFuelMoistures(burn.SIMTIME +	GetActualTimeStep());
				 LastFMCalcTime = burn.SIMTIME + GetActualTimeStep();	}

			if (ProcNum < 3) 				// if all inputs ready for FARSITE model
			{
				if (NextFireAfterInterrupt == 0)
					CountFires();
				if (ProcNum == 1)
				{
					if (FarsiteProcess1())    // do another iteration of FARSITE process
					{
						while (FarsiteProcess2())
						{
							if (!FarsiteProcessSpots())
								break;
						}
					}
					else
						break;
				}
				else if (ProcNum == 2)
				{
					if (FarsiteProcessSpots())
					{
						while (FarsiteProcess2())
						{
							if (!FarsiteProcessSpots())
								break;
						}
					}
				}
			}
			//}
			//else
			//	break;
			//CheckMessageLoop();				  // allow messages to be received
			if (LEAVEPROCESS)				// allows exit from FARSITE process control
				break;
			//CheckSteps();						// check visual and actual timesteps for changes
			//if(FARSITE_GO)
			//{
			if (ProcNum == 3)
			{
				//if(burn.SIMTIME==0)
				//	WriteOutputs(1);
				FarsiteProcess3();  		 	// mergers between fires and increment iteration
				ProcNum = 1;
				if (!OutputVectsAsGrown && NextFireAfterInterrupt == 0)
				{
					for (CurrentFire = 0;
						CurrentFire < GetNumFires();
						CurrentFire++)
					{
						if (GetInout(CurrentFire) == 0)
							continue;
						WriteVectorOutputs();   	  // output perimeters to screen and/or vector file
					}
				}
				if (NextFireAfterInterrupt == 0)
				{
					if (burn.SIMTIME > 0.0 &&
						PrimaryVisCount == PrimaryVisEqual)
					{
						if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)
						{
							if (burn.CuumTimeIncrement == 0.0)//GetActualTimeStep())
							{
								if (CheckPostFrontal(GETVAL))
								{
									//DrawPostFrontal();
								}
								WriteOutputs(-1);   		  // write areas and perimeters to data structures and screen
								PrimaryVisCount = 1;
							}
						}
						else
						{
							WriteOutputs(-1);   			   // write areas and perimeters to data structures and screen
							PrimaryVisCount = 1;
						}
					}
					else
					{
						if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)
						{
							if (burn.CuumTimeIncrement == 0.0)//GetActualTimeStep())
							{
								//if(CheckPostFrontal(GETVAL))
								//	DrawPostFrontal();
								WriteOutputs(1);
								PrimaryVisCount++;
							}
						}
						else
						{
							WriteOutputs(1);
							PrimaryVisCount++;
						}
					}
					if (burn.SIMTIME > 0.0)
					{
						if (SecondaryVisCount >= SecondaryVisEqual)
							SecondaryVisCount = 1;
						else
						{
							if (DistanceCheckMethod(GETVAL) ==
								DISTCHECK_SIMLEVEL)
							{
								if (burn.CuumTimeIncrement == 0.0)//GetActualTimeStep())
									SecondaryVisCount++;
							}
							else
								SecondaryVisCount++;
						}
					}
					else
						SecondaryVisCount = 1;
				}
			}
		}

		else
			break;  						   // check visual and actual timesteps for changes
		systime2 = clock();
		systime1 = systime;


		if (burn.SIMTIME > maximum)
		{
			FARSITE_GO = false;
			/*if (GetRastMake()) {
				WriteGISLogFile(0);
			}
			if (GetVectMake())
				WriteGISLogFile(1);
			if (GetShapeMake())
				WriteGISLogFile(2);*/
			//EB commentsout 1 line
			//printf("  End of Run    Simulation Completed\n");
			sprintf(MBStatus, "    %s", "SIMULATION COMPLETED");
			//EB 1 line
			//WriteMessageBar(0);
			SimRequest = SIMREQ_NULL ;
			//SetXORArrows(false);
			SIM_SUSPENDED = true;
			StepThrough = false;
			SIM_COMPLETED = true;
			//SetEvent(hWaitEvent);
			CanModifyInputs(true);
			//SuspendThread(hSimThread);
			//WaitForSingleObject(hFarSimEvent, INFINITE);
			//ResetEvent(hWaitEvent);
			CheckSteps();
		}

	}  /* 	while (burn.SIMTIME <= maximum)	*/
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-..-*/


	ProcessSimRequest();
	SetFarsiteProgress(1.0);
	//progress = 1.0;
	// write raster files at end of timestep...
	/*if (GetRastMake()) {
		burn.rast->SelectMemOutputs(GetRastFormat());	// write to raster file
	}	*/

	//std::cerr << "In FarsiteBurnLoop()" << burn.SIMTIME << std::endl ;
/*	if (burn.SIMTIME > maximum)
	{
		FARSITE_GO = false;
		if (GetRastMake())
			WriteGISLogFile(0);
		if (GetVectMake())
			WriteGISLogFile(1);
		if (GetShapeMake())
			WriteGISLogFile(2);
		//printf("End of Run Simulation Completed\n");
	}*/
   return i_Ret;
}

/**********************************************************************************/
void Farsite5::ResetThreads()
{
#ifdef WIN32
	FreeFarsiteEvents(EVENT_BURN);
	FreeFarsiteEvents(EVENT_MOIST);
	FreeFarsiteEvents(EVENT_BURNUP);
	FreeFarsiteEvents(EVENT_INTEG);
	FreeFarsiteEvents(EVENT_CROSS);
	FreeFarsiteEvents(EVENT_BURN_THREAD);
	FreeFarsiteEvents(EVENT_MOIST_THREAD);
	FreeFarsiteEvents(EVENT_BURNUP_THREAD);
	FreeFarsiteEvents(EVENT_INTEG_THREAD);
	FreeFarsiteEvents(EVENT_CROSS_THREAD);
	AllocFarsiteEvents(EVENT_BURN, GetMaxThreads(), "SyncEventBurn", false,
		false);
	AllocFarsiteEvents(EVENT_MOIST, GetMaxThreads(), "SyncEventMoist", false,
		false);
	AllocFarsiteEvents(EVENT_BURNUP, GetMaxThreads(), "SyncEventBurnup",
		false, false);
	AllocFarsiteEvents(EVENT_INTEG, GetMaxThreads(), "SyncEventInteg", false,
		false);
	AllocFarsiteEvents(EVENT_CROSS, GetMaxThreads(), "SyncEventCross", false,
		false);
	AllocFarsiteEvents(EVENT_BURN_THREAD, GetMaxThreads(),
		"SyncEventBurnThread", true, false);
	AllocFarsiteEvents(EVENT_MOIST_THREAD, GetMaxThreads(),
		"SyncEventMoistThread", true, false);
	AllocFarsiteEvents(EVENT_BURNUP_THREAD, GetMaxThreads(),
		"SyncEventBurnupThread", true, false);
	AllocFarsiteEvents(EVENT_INTEG_THREAD, GetMaxThreads(),
		"SyncEventIntegThread", true, false);
	AllocFarsiteEvents(EVENT_CROSS_THREAD, GetMaxThreads(),
		"SyncEventCrossThread", true, false);
#endif
	NumSimThreads = GetMaxThreads();
}



bool Farsite5::FarsiteProcess1()
	// THIS FUNCTION CONTAINS THE FARSITE PROCESS CONTROL
{
	long InOut, FireNum;

	IN_BURNPERIOD = true;
	if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)  	  // if simlevel timestep and/or burnout enabled
	{
		if (NextFireAfterInterrupt == 0)
			burn.DetermineSimLevelTimeStep();
		if (!InquireInBurnPeriod(burn.SIMTIME+GetTemporaryTimeStep()))//GetTemporaryTimeStep()))
		{
			if (burn.prod.cuumslope[1] > 1e-9)
				sarea = (a1 - firesizeh) /
					(cos(atan((PI * burn.prod.cuumslope[0] / 180.0) /
							burn.prod.cuumslope[1])));
			else
				sarea = 0.0;
			sarea = firesizes + sarea;
			firesizeh = a1;
			firesizes = sarea;
			AddDownTime(GetTemporaryTimeStep());
			burn.CuumTimeIncrement += GetTemporaryTimeStep();
			IN_BURNPERIOD = false;

			return true;
		}
		else if (NextFireAfterInterrupt == 0 && burn.CuumTimeIncrement == 0.0)
		{
			a1 = 0.0;
			p1 = 0.0;
			p2 = 0.0;
		}
		for (CurrentFire = NextFireAfterInterrupt; CurrentFire < GetNumFires(); CurrentFire++)			// for all fires
		{	if ((InOut = GetInout(CurrentFire)) == 0)
				continue;
			burn.BurnIt(CurrentFire);
			if (InOut < 3 && burn.CuumTimeIncrement == 0.0)//GetTemporaryTimeStep()==burn.CuumTimeIncrement)
			{
				a1 += burn.prod.area;   			// +area of all fires (&& -inward fire area)
				p1 += burn.prod.perimeter;  		// +perimeter of all fires)
				p2 += burn.prod.sperimeter; 		// +slope perimeter of all fires
			}
			//if(CurrentFire<GetNumFires()-1)
			//{	NextFireAfterInterrupt=CurrentFire+1;
			//if(SimRequest==SIMREQ_RESUMESUSPEND && !StepThrough) // Allow interruptions by user
			//{	Execute_ResumeSuspend();
			//	SuspendThread(hSimThread);
			//}
			//if(!FARSITE_GO)
			//	break;
			//}
		}
		//if(SimRequest==SIMREQ_RESUMESUSPEND && !StepThrough)
		//{	Execute_ResumeSuspend();
		//   	SuspendThread(hSimThread);
		//}
		if (!FARSITE_GO)
			return false;
		NextFireAfterInterrupt = 0;
		AddDownTime(-1.0);
	}
	else
	{
		if (!InquireInBurnPeriod(burn.SIMTIME+GetActualTimeStep()))
		{
			if (burn.prod.cuumslope[1] > 1e-9)
				sarea = (a1 - firesizeh) /
					(cos(atan((PI * burn.prod.cuumslope[0] / 180.0) /
							burn.prod.cuumslope[1])));
			else
				sarea = 0.0;
			sarea = firesizes + sarea;
			firesizeh = a1;
			firesizes = sarea;
			AddDownTime(GetActualTimeStep());
			for (CurrentFire = NextFireAfterInterrupt; CurrentFire < GetNumFires(); CurrentFire++)			// for all fires
			{	if (OutputVectsAsGrown)
					WriteVectorOutputs();   		   // output perimeters to screen and/or vector file
			}
			IN_BURNPERIOD = false;

			return true;
		}
		else if (NextFireAfterInterrupt == 0)
		{
			a1 = 0.0;
			p1 = 0.0;
			p2 = 0.0;
		}
		for (CurrentFire = NextFireAfterInterrupt;
			CurrentFire < GetNumFires();
			CurrentFire++)			// for all fires
		{
			if ((InOut = GetInout(CurrentFire)) == 0)
				continue;
			//if(GetInout(CurrentFire)<3)   		// no outputs of vector barriers
			if (OutputVectsAsGrown) {
				WriteVectorOutputs();   		   // output perimeters to screen and/or vector file
			}
			burn.BurnIt(CurrentFire);   			// burn the fire
			if (InOut < 3)
			{
				a1 += burn.prod.area;   			// +area of all fires (&& -inward fire area)
				p1 += burn.prod.perimeter;  		// +perimeter of all fires)
				p2 += burn.prod.sperimeter; 		// +slope perimeter of all fires
			}
			//if(CurrentFire<GetNumFires()-1)
			//{	NextFireAfterInterrupt=CurrentFire+1;
			//if(SimRequest==SIMREQ_RESUMESUSPEND && !StepThrough)
			//{	Execute_ResumeSuspend();
			//	SuspendThread(hSimThread);
			//}
			//if(!FARSITE_GO)
			//	break;
			//}
    	}
		//if(SimRequest==SIMREQ_RESUMESUSPEND && !StepThrough)
		//{	Execute_ResumeSuspend();
		//    	SuspendThread(hSimThread);
		//}
		if(!FARSITE_GO)
			return false;
		NextFireAfterInterrupt = 0;
		AddDownTime(-1.0);
     }

	while (GetNumFires() < GetNewFires())   		// condense array of fire perimeter pointers
	{
		CurrentFire = GetNumFires();
		if (GetInout(CurrentFire) != 0)
		{
			FireNum = 0;
			while (GetInout(FireNum) > 0)
			{
				FireNum++;
			}
			if (FireNum < CurrentFire)
			{
				FireNum = CurrentFire;
				burn.RePositionFire(&CurrentFire);
				if (CheckPostFrontal(GETVAL))
					SetNewFireNumber(FireNum, CurrentFire,
						burn.post.AccessReferenceRingNum(1, GETVAL));
			}
		}
		IncNumFires(1);				   // increment number of active fires
	}
	IncNumFires(-GetSkipFires());			// subtract merged or extinguished fires
	SetNewFires(GetNumFires());
	SetSkipFires(0);						// reset skipfire to 0
	if (burn.prod.cuumslope[1] > 1e-9)
		sarea = (a1 - firesizeh) /
			(cos(atan((PI * burn.prod.cuumslope[0] / 180.0) /
					burn.prod.cuumslope[1])));
	else
		sarea = 0.0;
	sarea = firesizes + sarea;
	firesizeh = a1;
	firesizes = sarea;

	return true;
}


bool Farsite5::FarsiteProcessSpots()
	// THIS FUNCTION CONTAINS THE FARSITE PROCESS CONTROL
	{
		 if(!IN_BURNPERIOD)
			  return true;

	double Actual;
	double simtime = burn.SIMTIME;
	double cuumtime = burn.CuumTimeIncrement;
	double TempStep = 0.0;
	bool SpotActivityChanged = false;
	long InOut, TempTime, CurrentSpot;
	Embers::spotdata* curspot;

	if (EnableSpotting(GETVAL))
	{
		EnableSpotting(false);
		SpotActivityChanged = true;
	}
	if (OldFireAfterInterrupt == 0)
		burn.SetSpotLocation(0);//burn.embers.CurSpot=burn.embers.FirstSpot;
	Actual = GetActualTimeStep();
	if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)
		TempStep = GetTemporaryTimeStep();

	for (CurrentFire = 0;
		CurrentFire < burn.TotalSpots;
		CurrentFire++)			// for all fires
	{
		curspot = burn.GetSpotData(Actual);
		if (curspot == NULL)
			continue;
		if (curspot->TimeRem < 0.10)
			curspot->TimeRem = 0.10;

		CurrentSpot = InsertSpotFire(curspot->x, curspot->y);
		if (CurrentSpot < 0)
			continue;

		//CurrentSpot=(long) curspot->vecdir;  // this variable is used to hold the position
		// in the perimeter1[] array
		if ((InOut = GetInout(CurrentSpot)) == 0)
			continue;
		// truncate to nearest 0.1 minutes for easier precision in Burn::
		curspot->TimeRem *= 10.0;
		TempTime = (long)curspot->TimeRem;
		curspot->TimeRem = (double) TempTime / 10.0;

		burn.SIMTIME += (Actual - curspot->TimeRem);//burn.embers.CurSpot->TimeRem;
		if (burn.SIMTIME < 0.0)
			burn.SIMTIME = 0.0;

		if (DistanceCheckMethod(GETVAL) == DISTCHECK_FIRELEVEL)
		{
			burn.CuumTimeIncrement = 0.0;
			SetActualTimeStep(curspot->TimeRem);//burn.embers.CurSpot->TimeRem);
		}
		else
		{
			burn.CuumTimeIncrement = Actual - curspot->TimeRem;
			SetTemporaryTimeStep(curspot->TimeRem);
		}

		if (!InquireInBurnPeriod(burn.SIMTIME+GetActualTimeStep()))
		{
			burn.SIMTIME = simtime;

			continue;
		}

		burn.BurnIt(CurrentSpot);   	   	// burn the fire
		if (InOut < 3)
		{
			a1 += burn.prod.area;   			// +area of all fires (&& -inward fire area)
			p1 += burn.prod.perimeter;  		// +perimeter of all fires)
			p2 += burn.prod.sperimeter; 		// +slope perimeter of all fires
		}
		burn.SIMTIME = simtime;
		burn.CuumTimeIncrement = cuumtime;
	}
	NextFireAfterInterrupt = OldFireAfterInterrupt = 0;
	long FireNum;
	while (GetNumFires() < GetNewFires())      // condense array of fire perimeter pointers
	{
		CurrentFire = GetNumFires();
		if (GetInout(CurrentFire) != 0)
		{
			FireNum = 0;
			while (GetInout(FireNum) > 0)
			{
				FireNum++;
			}
			if (FireNum < CurrentFire)
			{
				FireNum = CurrentFire;
				burn.RePositionFire(&CurrentFire);
				if (CheckPostFrontal(GETVAL))
					SetNewFireNumber(FireNum, CurrentFire,
						burn.post.AccessReferenceRingNum(1, GETVAL));
			}
		}
		IncNumFires(1);				   // increment number of active fires
	}
	IncNumFires(-GetSkipFires());			// subtract merged or extinguished fires
	SetNewFires(GetNumFires());
	SetSkipFires(0);						// reset skipfire to 0
	if (burn.prod.cuumslope[1] > 1e-9)
		sarea = (a1 - firesizeh) /
			(cos(atan((PI * burn.prod.cuumslope[0] / 180.0) /
					burn.prod.cuumslope[1])));
	else
		sarea = 0.0;
	sarea = firesizes + sarea;
	firesizeh = a1;
	firesizes = sarea;
	if (DistanceCheckMethod(GETVAL) == DISTCHECK_FIRELEVEL)
		SetActualTimeStep(Actual);
	else
		SetTemporaryTimeStep(TempStep);
	if (SpotActivityChanged)
		EnableSpotting(true);
	burn.SetSpotLocation(-1);     // elimate all spots

	return true;
}

bool Farsite5::FarsiteProcess2()
{
	if (!IN_BURNPERIOD)
	{
		ProcNum = 3;

		return false;
	}

	if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)
	{
		if (ProcNum == 1)	// only first time through
			burn.CuumTimeIncrement += GetTemporaryTimeStep();
		if (burn.CuumTimeIncrement < GetActualTimeStep())
		{
			ProcNum = 3;

			return false;
		}
		else
			burn.CuumTimeIncrement = GetActualTimeStep();
	}

	long OldNumSpots, Numspots;

	OldNumSpots = NumLocalSpots;//numspots; 			 	// save old number of spots/embers
	burn.BurnSpotThreads();				// ember flight and ignition, after all fires have burned this step
	//DrawSpots();
	//ATLTRACE("OldNumSpots = %ld, numspots = %ld\n", OldNumSpots, numspots);
	if(this->EnableSpotFireGrowth(GETVAL))
		NumLocalSpots = OldNumSpots + burn.SpotFires;				  // sum OldNumSpots and new numspots, after DrawSpots()
		//numspots = OldNumSpots + burn.SpotFires;				  // sum OldNumSpots and new numspots, after DrawSpots()
	else
		NumLocalSpots = OldNumSpots + burn.TotalSpots;				  // sum OldNumSpots and new numspots, after DrawSpots()
		//numspots = OldNumSpots + burn.TotalSpots;				  // sum OldNumSpots and new numspots, after DrawSpots()

	//numspots += OldNumSpots;				  // sum OldNumSpots and new numspots, after DrawSpots()
	//IncNewFires(burn.TotalSpots);//embers.SpotFires); 	// add spotfires to total number to check for mergers
	//SetNumFires(GetNewFires());   		  // update the number of new fires
	GetNumSpots(&Numspots, false);
	SetNumFires(Numspots);
	if (burn.SpotFires) //burn.embers.SpotFires)
	{
		NextFireAfterInterrupt = GetNumFires() - (burn.SpotFires);//+burn.FailedSpots);//burn.embers.SpotFires;
		burn.SpotFires = 0;//burn.embers.SpotFires=0;
		ProcNum = 2;

		return true;
	}
	ProcNum = 3;
	burn.SetSpotLocation(-2);

	return false;
}


void Farsite5::FarsiteProcess3()
	// THIS FUNCTION PROCESSES FIRE MERGERS
{
	long i, j, NewFires, NewPts;
	double mx[10] =
	{
		90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0
	};
	AttackData* atk;
	AirAttackData* aatk;
	FireRing* firering;


	for (i = 0;
		i < GetNumAttacks();
		i++)   	 // perform indirect attack after all fires have burned
	{
		if ((atk = GetAttackByOrder(i, true)) != 0)
		{
			NewFires = GetNewFires();		// save value before attack
			if (atk->FireNumber == -1)
				NewFires++;
			if (atk->Indirect == 2)
				Atk.ConductBurnout(atk);
			else if (!Atk.IndirectAttack(atk, GetActualTimeStep()))
			{
				CancelAttack(atk->AttackNumber);
				i--;	  // need to decrement counter because numattacks has changed
			}   		   // and order in GetAttackByOrder() has also changed
			/*else
									{	//burn.ReorderPerimeter(atk->FireNumber, burn.FindExternalPoint(atk->FireNumber, 0));
									burn.FindOuterFirePerimeter(atk->FireNumber);
									NewPts=GetNumPoints(atk->FireNumber);
									FreePerimeter1(atk->FireNumber);
									AllocPerimeter1(atk->FireNumber, NewPts+1);
									burn.tranz(atk->FireNumber, NewPts);
										 Atk.BoundingBox();
									}*/
			if (NewFires < GetNewFires())
			{
				burn.ReorderPerimeter(NewFires,
						burn.FindExternalPoint(NewFires, 0));
				burn.FindOuterFirePerimeter(NewFires);
				NewPts = GetNumPoints(NewFires);
				FreePerimeter1(NewFires);
				if (NewPts > 0)
				{
					AllocPerimeter1(NewFires, NewPts + 1);
					burn.tranz(NewFires, NewPts);
					Ignition.BoundingBox(NewFires);
					if (CheckPostFrontal(GETVAL))
					{
						firering = burn.post.SetupFireRing(NewFires,
												burn.SIMTIME +
												burn.CuumTimeIncrement,
												burn.SIMTIME +
												burn.CuumTimeIncrement);
						for (j = 0; j < NewPts; j++)
							AddToCurrentFireRing(firering, j, 0, 0, 0, mx, 0.0);
					}
				}
				else
				{
					//FreePerimeter2();
					SetNumPoints(NewFires, 0);
					SetInout(NewFires, 0);
					IncSkipFires(1);
				}
			}
			SetNumFires(GetNewFires());
		}
	}
	for (i = 0;
		i < GetNumAirAttacks();
		i++)		  // check air attack effectiveness
	{
		aatk = GetAirAttackByOrder(i);
		if (!AAtk.CheckEffectiveness(aatk, GetActualTimeStep()))
		{
			//DrawFire(-aatk->PatternNumber); 		// redraw elapsed pattern
			--aatk->PatternNumber;
			FreePerimeter1(aatk->PatternNumber);
			SetNumPoints(aatk->PatternNumber, 0);
			SetInout(aatk->PatternNumber, 0);
			IncSkipFires(1);
			CancelAirAttack(aatk->AirAttackNumber);
			i--;	  // decrement i becuase airattack number has changed.
		}
	}
	if (gaat)    // if object for handling group air attacks exists
	{
		if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)
			gaat->ExecuteAllIndirectAttacks(GetTemporaryTimeStep());
		else
			gaat->ExecuteAllIndirectAttacks(GetActualTimeStep());
	}

	if (IN_BURNPERIOD == true && !LEAVEPROCESS)
	{
		if (GetNumFires() > 1)
		{
			CurrentFire = 0;
			burn.CrossFires(1, &CurrentFire);  	// check for crosses between fires outward fires
			SetNumFires(GetNewFires());
		}
		CheckStopLocations();
	}

	if (CheckPostFrontal(GETVAL))   		  // reset to flag first num
	{
		CondenseRings(burn.post.AccessReferenceRingNum(1, GETVAL));
		burn.post.bup.BurnFireRings(burn.post.AccessReferenceRingNum(1, GETVAL),
						GetNumRings());
		burn.post.ComputePostFrontal(burn.SIMTIME + burn.CuumTimeIncrement,
					&smolder, &flaming);
		//CondenseRings(0);	// done in burn.post.ComputePostFrontal()
		burn.post.AccessReferenceRingNum(1, GetNumRings()); // could set to -1 for automatic
		//DrawPostFrontal();
	}
	burn.ResetIntersectionArrays();		// free allocations for intersections

	if (StepThrough)
	{
		if (PrimaryVisCount == PrimaryVisEqual)
		{
			if (DistanceCheckMethod(GETVAL) == DISTCHECK_SIMLEVEL)
			{
				if (burn.CuumTimeIncrement == GetActualTimeStep())
				{
					SimRequest = SIMREQ_RESUMESUSPEND;//FARSITE_GO=false;
					StepThrough = false;
					sprintf(MBStatus, "    %s", "SIMULATION SUSPENDED");
					//printf(MBStatus);
					//Beep(24000, 150);
					//if(DDE.AdviseLoopActive)	// DDE every fire as it is written
					//	DDE.PostAdvise(DDE_STEPFINISHED, "Finished");
				}
			}
			else
			{
				SimRequest = SIMREQ_RESUMESUSPEND;//FARSITE_GO=false;
				StepThrough = false;
				sprintf(MBStatus, "    %s", "SIMULATION SUSPENDED");
				//printf(MBStatus);
			}
		}
	}
	if (DistanceCheckMethod(GETVAL) == DISTCHECK_FIRELEVEL)
	{
		burn.SIMTIME += GetActualTimeStep();		// update simulation time
		burn.CuumTimeIncrement = 0.0;
	}
	else
	{
		if (GetActualTimeStep() - burn.CuumTimeIncrement <= 0.0)  // end of timestep
		{
			burn.SIMTIME += GetActualTimeStep();	// update simulation time
			burn.CuumTimeIncrement = 0.0;

			pfdata.SetData(burn.SIMTIME, flaming / GetActualTimeStep() * 60.0,
					smolder / GetActualTimeStep() * 60.0);
			smolder = 0.0;
			flaming = 0.0;
		}
	}
	//   FreeElev();
}
void Farsite5::WriteVectorOutputs()
{
	bool WriteVect = false;


	if(burn.SIMTIME==0.0 || PrimaryVisCount==PrimaryVisEqual)   // if visible time step
    {
     	if(DistanceCheckMethod(GETVAL)==DISTCHECK_SIMLEVEL)
        {
        	if(burn.CuumTimeIncrement!=0.0)
               	return;
        }
    }
	if (GetVectMake())
	{
		if (!GetVectVisOnly())						  // if write all perimeters
			WriteVect = true;
		else if (burn.SIMTIME == 0.0 || PrimaryVisCount == PrimaryVisEqual)
			WriteVect = true;
		if (ExcludeBarriersFromVectorFiles(GETVAL) &&
			GetInout(CurrentFire) == 3)
			WriteVect = false;
		if (WriteVect)
			vect.VectData(CurrentFire, burn.SIMTIME);
	}
	WriteVect = false;
	if (GetShapeMake())    // if shapefileoutput
	{
		if (!shape.VisOnly)
			WriteVect = true;
		else if (burn.SIMTIME == 0 || PrimaryVisCount == PrimaryVisEqual)
			WriteVect = true;
		if (ExcludeBarriersFromVectorFiles(GETVAL) &&
			GetInout(CurrentFire) == 3)
			WriteVect = false;
		if (WriteVect)
			shape.ShapeData(CurrentFire, burn.SIMTIME);	// write fires
		//(WriteVect)
	}
	AddCurrPerimeter();
}


/*******************************************************************************************
*
*  NOTE as of Nov 11 2010 - This functions DOESN'T REALLY do anything yet.
*
*******************************************************************************************/
void Farsite5::WriteOutputs(int type)
{
	// if area and perimeter calculations are desired
	//	if (WStation.Number > 0)
	//		CurrentWeather();
	WriteClocks();						// update the clocks if they're visible
 		if (GetNumStations() > 1 || AtmosphereGridExists() > 0)
		{
			//			CmWindGaugeControl();
		}
		else
		{
			EnvironmentData env;
			//headdata hd=GetHeaderInformation();

			burn.fe->GetLandscapeData(GetLoEast() + GetNumEast() / 2,
						GetLoNorth() + GetNumNorth() / 2);
			burn.fe->ld.elev = (short)
				(GetLoElev() + (GetHiElev() - GetLoElev()) / 2);

			burn.fe->ld.slope = 0;
			burn.fe->ld.aspect = 0;
			burn.fe->ld.fuel = -1;//(short) GetFuelConversion(hd.fuels[1]);   // make sure that there is fuel there
			burn.fe->ld.cover = 0;   // make sure that there is fuel there

//		burn.fe->GetFireEnvironment(burn.env, burn.SIMTIME, false);
		burn.fe->GetFireEnvironment( burn.SIMTIME, false);

			burn.fe->GetEnvironmentData(&env);


			//WindGauge->tws = env.tws;
			//WindGauge->winddir = env.winddir;
			//	InvalidateRect(WindGauge->HWindow, NULL, true);
			//	UpdateWindow(WindGauge->HWindow);
		}
     /*
	adata.SetData(a1 * type, sarea * type);    	// type determines the color of the output
	pdata.SetData(p1 * type, p2 * type);			// if visible timestep, then red else blue
	fdata.count = CountInwardFires;
	fdata.count = 0;
	for (int i = 0;
		i < GetNumFires();
		i++)  	// counting is now done in ::CountFires
	{
		if (GetInout(i) == 2)
			fdata.count++;
	}
	fdata.SetData(GetNumFires() * type, numspots * type, fdata.count * type);
     */
	//if (burn.SIMTIME > 0)
	//	fdata.SetData((CountTotalFires + numspots) * type, numspots * type,
	//			fdata.count * type);
	//	else
	//		fdata.SetData(CountTotalFires*type, 0, fdata.count*type);
	//	if(p2>pdata.HIY)						// keep record of largest Y-value
	//		pdata.HIY=p2;
	WriteArea();
	WritePerim();
	WriteFireData();
	WritePFData();
	//if(InquireInBurnPeriod)
	//{	a1=0.0;
	//	p1=0.0;
	//	p2=0.0;					// reset area and perimeter accumulators
	//}
	NumLocalSpots=0;//numspots = 0;

	//
	//UpdateWindow(FireChxChart->HWindow);
}

void Farsite5::WriteGISLogFile(long LogType)
{
		char LogFile[256];
		memset(LogFile, 0x0, sizeof(LogFile));
		long len;
		//SYSTEMTIME SysTime;

		//GetLocalTime(&SysTime);
		if (LogType == 0)
		{
			len = strlen(GetRasterFileName(0));
			strncpy(LogFile, GetRasterFileName(0), len - 3);
			strcat(LogFile, "LGR");
		}
		else if (LogType == 1)
		{
			len = strlen(GetVectorFileName());
			strncpy(LogFile, GetVectorFileName(), len - 3);
			strcat(LogFile, "LGV");
		}
		else if (LogType == 2)
		{
			if (!shape.GetShapeFileName())
				return;
			len = strlen(shape.GetShapeFileName());
			strncpy(LogFile, shape.GetShapeFileName(), len - 3);
			strcat(LogFile, "LGS");
		}
		else
			return;

		FILE* logfile = fopen(LogFile, "w");
#if 0
		if (logfile == NULL)
		{
			//SetFileAttributes(LogFile, FILE_ATTRIBUTE_NORMAL);
			chmod(LogFile,S_IRWXO);
			logfile = fopen(LogFile, "w");
		}
#endif
		fprintf(logfile, "%s %s\n", "Log File: ", LogFile);
		//fprintf(logfile, "%s %02u%s%02u%s%d\n", "Date File Created:",
		//	SysTime.wMonth, "\\", SysTime.wDay, "\\", SysTime.wYear);
		//fprintf(logfile, "%s %02u%s%02u\n\n", "Time File Created:",
		//	SysTime.wHour, ":", SysTime.wMinute);
		if (LogType < 2)
		{
			if (AccessOutputUnits(GETVAL) == 0)
				fprintf(logfile, "File units: Metric\n");
			else
				fprintf(logfile, "File units: English\n");
		}
		fprintf(logfile, "%s %s\n", "Landscape File: ",
			LandFName);
		for (len = 0; len < GetNumStations(); len++)
		{
			fprintf(logfile, "%s %ld: %s\n", "Weather File", len + 1,
				"KRAP");//Inputs.WeatherFile[len]);
			fprintf(logfile, "%s %ld: %s\n", "Wind File", len + 1,
				"KRAP");//Inputs.WindFile[len]);
			if (len > 4)
				break;
		}
		fprintf(logfile, "%s %s\n", "Adjustment File: ", "KRAP");//Inputs.AdjustmentFile);
		fprintf(logfile, "%s %s\n", "Fuel Moisture File: ",
			"KRAP");//Inputs.FuelMoistureFile);
		if (HaveFuelConversions())
			fprintf(logfile, "%s %s\n", "Conversion File: ",
				"KRAP");//Inputs.ConvertFile);
		else
			fprintf(logfile, "%s %s\n", "Conversion File: ", "None");
		if (HaveCustomFuelModels())
			fprintf(logfile, "%s %s\n", "Custom Fuel Model File: ",
				"KRAP");//Inputs.FuelModelFile);
		else
			fprintf(logfile, "%s %s\n", "Custom Fuel Model File: ", "None");
		if (EnableCrowning(GETVAL))
		{
			fprintf(logfile, "%s\n", "Crown Fire: Enabled");
			if (LinkDensityWithCrownCover(GETVAL))
				fprintf(logfile, "%s\n", "Crown Density LINKED to Crown Cover");
			else
				fprintf(logfile, "%s\n",
					"Crown Density NOT LINKED to Crown Cover");
		}
		else
			fprintf(logfile, "%s\n", "Crown Fire: Disabled");
		if (EnableSpotting(GETVAL))
		{
			fprintf(logfile, "%s\n", "Ember Generation: Enabled");
			if (EnableSpotFireGrowth(GETVAL))
				fprintf(logfile, "%s\n", "Spot Growth: Enabled");
			else
				fprintf(logfile, "%s\n", "Spot Growth: Disabled");
		}
		else
			fprintf(logfile, "%s\n", "Ember Generation: Disabled");
		if (ConstantBackingSpreadRate(GETVAL))
			fprintf(logfile, "%s\n",
				"Backing Spread: Calculated from No Wind/No Slope");
		else
			fprintf(logfile, "%s\n",
				"Backing Spread: Calculated from Elliptical Dimensions");
		if (AccelerationON())
		{
			/*	if (strlen(TransAccelData.Dat.AccelLoad) > 0)
			{
				if (strlen(TransAccelData.Dat.AccelSave) > 0)
					fprintf(logfile, "%s %s\n", "Acceleration File Used: ",
						TransAccelData.Dat.AccelSave);
				else
					fprintf(logfile, "%s %s\n", "Acceleration File Used: ",
						TransAccelData.Dat.AccelLoad);
			}
			else
				fprintf(logfile, "%s %s\n", "Acceleration File Used: ",
					"Default Values");
						  */
		}
		else
			fprintf(logfile, "%s %s\n", "Acceleration File Used: ",
				"None (feature toggled off)");
		fprintf(logfile, "\n");
		fprintf(logfile, "%s %ld%s%ld %ld:00\n",
			"Simulation Started (Day Hour:Min):", GetStartMonth(), "/",
			GetStartDay(), GetStartHour() / 100);
		fprintf(logfile, "%s %s\n", "Simulation Ended (Day Hour:Min):",
			CurTime);
		fprintf(logfile, "%s %s\n\n", "Elapsed Time (Days Hours:Mins):",
			ElTime);
		fprintf(logfile, "%s %lf\n", "Actual Time Step (min):",
			GetActualTimeStep());
		fprintf(logfile, "%s %lf\n", "Visible Time Step (min):",
			GetVisibleTimeStep());
		fprintf(logfile, "%s %lf\n", "Perimeter Resolution (m):",
			GetPerimRes());
		fprintf(logfile, "%s %lf\n", "Distance Resolution (m):", GetDistRes());
		fclose(logfile);
		//free(LogFile);
}
void Farsite5::ProcessSimRequest()
{
	switch (SimRequest)
	{
			case SIMREQ_NULL:
			break;
			case SIMREQ_INITIATETERMINATE:
			//FARSITE_GO=true;
			SimRequest = SIMREQ_NULL;
			ResetFarsite();
			Execute_InitiateTerminate();  	// doesn't work with multithreading ?? who knows why
			//::SendMessage(Client->HWindow, WM_COMMAND, CM_EXECUTETERMINATE, NULL);
			break;
			case SIMREQ_STARTRESTART:
			FARSITE_GO = true;
			SimRequest = SIMREQ_NULL;
			Execute_StartRestart();
			//::SendMessage(Client->HWindow, WM_COMMAND, CM_EXECUTERESTART, NULL);
			break;
			case SIMREQ_RESUMESUSPEND:
			//FARSITE_GO=true;
			SimRequest=SIMREQ_NULL;
			Execute_ResumeSuspend();
			break;
			case SIMREQ_RESET:
			FARSITE_GO = true;
			SimRequest = SIMREQ_NULL;
			Execute_Reset();
			//::SendMessage(Client->HWindow, WM_COMMAND, CM_EXECUTERESET, NULL);
			break;
	}
	//SIM_SUSPENDED = false;
	//if (hWaitEvent != NULL)
		//SetEvent(hWaitEvent);
	CanModifyInputs(true);
}


long Farsite5::InsertSpotFire(double xpt, double ypt)
{
	long k, newspots;
	long NumVertex;
	double Sxpt, Sypt, NewAngle;
	double Xlo, Xhi, Ylo, Yhi;
	double OutsideX, OutsideY;
	double AngleIncrement, AngleOffset;

	NumVertex = 6;
	GetNumSpots(&newspots, true);	// reserve an address

	Xlo = Xhi = xpt;
	Ylo = Yhi = ypt;	 							// initialize bounding rectangle
	AngleOffset = ((double) (Rand.rand3() * 1000)) / 999.0 * PI / 2.0;	// randomize orientation of fire polygon
	AngleIncrement = PI / ((double) NumVertex / 2.0);				// to avoid coincident sides
	OutsideX = -1.0;	// no outside points
	OutsideY = -1.0;
	AllocPerimeter1(newspots, NumVertex + 1);
	for (k = 0; k < NumVertex; k++)
	{
		NewAngle = ((double) k) * AngleIncrement + AngleOffset;
		Sxpt = xpt + cos(NewAngle);   	//Stime is the time step for growing spot fires
		Sypt = ypt + sin(NewAngle);
		SetPerimeter1(newspots, k, Sxpt, Sypt); 	 // save head fire ros at time step
		SetFireChx(newspots, k, 0.0, 0.0);
		SetReact(newspots, k, 0.0);
		if (Sxpt < Xlo)
			Xlo = Sxpt;
		else if (Sxpt > Xhi)
			Xhi = Sxpt;
		if (Sypt < Ylo)
			Ylo = Sypt;
		else if (Sypt > Yhi)
			Yhi = Sypt;
	}
	SetPerimeter1(newspots, k, Xlo, Xhi);    // save bounding rectangle
	SetFireChx(newspots, k, Ylo, Yhi);
	for (k = 0; k < NumVertex; k++) 	// check to see if
	{
		Sxpt = GetPerimeter1Value(newspots, k, 0);    // points outside
		Sypt = GetPerimeter1Value(newspots, k, 1);
		if (Sxpt <= GetLoEast())
			OutsideX = Xhi;
		else if (Sxpt >= GetHiEast())
			OutsideX = Xlo;
		if (Sypt >= GetHiNorth())
			OutsideY = Ylo;
		else if (Sypt <= GetLoNorth())
			OutsideX = Yhi;
	}
	if (OutsideX != -1.0 || OutsideY != -1.0)
	{
		FreePerimeter1(newspots);
		newspots = -1;
	}
	else
	{
		SetInout(newspots, 1);
		SetNumPoints(newspots, NumVertex);
	}

	return newspots;
}


void Farsite5::CheckStopLocations()
{
	bool found = false;
	//int response;
	long i, j;
	double xpt, ypt;

	for (j = 0; j < GetNumStopLocations(); j++)
	{
		if (!GetStopLocation(j, &xpt, &ypt))
			continue;
		for (i = 0; i < GetNumFires(); i++)
		{
			if (GetInout(i) != 1)
				continue;
			burn.startx = xpt;
			burn.starty = ypt;
			if (burn.Overlap(i))
			{
				EnableStopLocation(j, false);
				if (!found)
				{
					//response = IDYES;
					//printf("Do you want to suspend the simulation?? Fire Encountered Stop Location\n");
					found = true;
					//if (response == IDYES)
						//SimRequest = SIMREQ_RESUMESUSPEND;
				}
			}
		}
	}
}

bool Farsite5::WriteArea()
{
	return true;
}
bool Farsite5::WritePerim()
{
	return true;
}
bool Farsite5::WriteFireData()
{
	return true;
}
bool Farsite5::WritePFData()
{
	return true;
}

void Farsite5::ResetFarsite()
	// resets data structures for model
{
	/*
	long i;
	a1 = p1 = p2 = 0.0;
	sarea = firesizeh = firesizes = 0.0;
	burn.prod.cuumslope[0] = burn.prod.cuumslope[1] = 0.0;
	burn.SetSpotLocation(-3);			// destroy all remaining spot fires
	for (i = 0; i < NUM_FUEL_SIZES; i++)
	burn.env->ResetData(i);
	memset(FMMFileName, 0x0, 256 * sizeof(char));
	memset(FMMDescription, 0x0, 512 * sizeof(char));
	burn.ResetIntersectionArrays();
	//burn.post.ResetAllThreads();
	if (WARROWS_ON)
	{
	//WARROWS_ON=false;
	if (WData)
	delete[] WData;					 // reset stored wind data
	WData = 0;
	NumWData = 0;
	}
	if (GetRastMake())
	{
	WriteGISLogFile(0);
	burn.rast.RasterReset();
	}
	if (GetVectMake())
	{
	if (GetVectVisOnly())
	{
	if (PrimaryVisCount == PrimaryVisEqual)    // visible timestep
	{
		for (i = 0; i < GetNumFires(); i++)
		{
			if (GetInout(i) == 0)   		  // if fire exists
				continue;
			if (ExcludeBarriersFromVectorFiles(GETVAL) &&
				GetInout(i) == 3)
				continue;
			vect.VectData(i, burn.SIMTIME);
		}   							   // write last perim array that was not shown
	}
	}
	else
	{
	for (i = 0; i < GetNumFires(); i++)
	{
		if (GetInout(i) == 0)   		  // if fire exists
			continue;
		if (ExcludeBarriersFromVectorFiles(GETVAL) && GetInout(i) == 3)
			continue;
		vect.VectData(i, burn.SIMTIME);
	}
	}
	vect.VectData(-1, burn.SIMTIME);	 // terminate vector file if present
	SetVectMake(false); 				// reset vector output
	WriteGISLogFile(1);
	}
	if (GetShapeMake())
	{
	bool WriteVect = false;
	if (!shape.VisOnly)
	WriteVect = true;
	else if (PrimaryVisCount == PrimaryVisEqual)
	WriteVect = true;
	if (WriteVect)
	{
	for (i = 0; i < GetNumFires(); i++)
	{
		if (GetInout(i) == 0)
			continue;
		if (ExcludeBarriersFromVectorFiles(GETVAL) && GetInout(i) == 3)
			continue;
		shape.ShapeData(i, burn.SIMTIME);	// write fires
	}
	 if(DDE.AdviseLoopActive)	// DDE every fire as it is written
						{    if(GetInout(CurrentFire)<3)
							{    if(!DDE.PostAdvise(DDE_FIREFILE, shape.GetShapeFileName()))
																	printf(" No Data Sent  DDE connection Error\n");						}
							 else
							{    if(!DDE.PostAdvise(DDE_BARRIERFILE, ""));
							 		printf(" No Data Sent  DDE connection Error\n");						}
						}


			}
			WriteGISLogFile(2);
		}
		shape.ResetShape();
		SetShapeMake(false);
		PrimaryVisEqual = 0;
		PrimaryVisCount = 0;
		SecondaryVisEqual = -1;
		SecondaryVisCount = 0;
		SetRastMake(false); 					 // reset raster output
		TransOutputData.ResetData(false);
		if (VisPerimSize)
			SaveVisPerimeter(); 		  	// save diskcopy of visible fire perimeters
		ResetVisPerimFile();
		//-----------------------------
		// NOT NEEDED IN VERSION 4.0
		//if(ProcNum==2)
		//	burn.embers.SpotReset(GetNumFires()-NextFireAfterInterrupt, burn.embers.CurSpot);
		//-----------------------------
		ClearAllFires();
		if (CheckPostFrontal(GETVAL))
		{
			// finish outputs for post frontal, then...
			FreeAllFireRings();
		}
		SetNumFires(0);
		SetNewFires(0);
		SetSkipFires(0);
		CurrentFire = 0;
		//-----------------------------
		//burn.embers.CarryOverEmbers=0;		 // no carry over embers
		//burn.embers.EmberReset();
		//-----------------------------
		//pdata.ReSet();
		//adata.ReSet();
		//fdata.ReSet();
		//pfdata.ResetData();
		burn.SIMTIME = 0.0;
		SetTemporaryTimeStep(0.0);
		burn.CuumTimeIncrement = 0.0;
		numspots = 0;
		if (DurationResetAtRestart(GETVAL))
		{
			maximum = 0; // numaximum=
			ResetDuration();
		}
		TimeKeepCurrent.ReSet();		   	  	// global time keeping structures
		TimeKeepElapsed.ReSet();
		sprintf(ElTime, "%02d %02d%s%02d", 0, 0, ":", 0);
		sprintf(CurTime, "%02d%s%02d  %02d%s%02d", GetStartMonth(), "/",
			GetStartDay(), GetStartHour() / 100, ":", GetStartMin());
		for (i = 0; i < 5; i++) 	// reset weather stats for wind gauges
			WStat[i].tws = -1.0;
		WStation.ReSet();
		sprintf(MBStatus, "    %s", "SIMULATION RESET");
		//WriteMessageBar(0);
		ResetBitmap(2);
		ResetBitmap(3);
		SetFuelsAltered(0);
		LEAVEPROCESS = true;
		FARSITE_GO = false;
		ProcNum = 1;							   // process number 1 in simulation
		NextFireAfterInterrupt = 0; 			   // reset beginning of burn-loop
		OldFireAfterInterrupt = 0;
		FreeAllAttacks();					 // eliminate all attacks on fires
		FreeAllAirAttacks();
		FreeAllGroupAirAttacks();
		//FreeAllCompoundCrews();			// can't do it here, because may need them
		StepThrough = false;
		//DDE.DDE_SIMCOMMAND=false;
		FreeElev();
		CountFires();
		CanSetRasterResolution(1);  	  // false
		SIM_SUSPENDED = false;
		EndSimThread();
		CanModifyInputs(true);
		NumSimThreads = 0;
		for (i = 0; i < GetNumStopLocations(); i++)
			EnableStopLocation(i, true);
	*/
}
void Farsite5::Execute_InitiateTerminate()
	// DISPLAYS LANDSCAPE WINDOW AND ENABLES/DISABLES MENU SELECTIONS
{
	//bool CanGo = true;
	if (F_ON)
	{
	      // 	Beep(24000, 150);
		//int Response = IDYES;
	}
	else
	{	/*	if (!Inputs.LandID)
			{
				printf("Load Landscape File (.lcp) Data Input Incomplete\n");				CanGo = false;
			}
			else if (!Inputs.WtrID)
			{
				printf("Load Weather File (.wtr) Data Input Incomplete\n");				CanGo = false;
			}
			else if (!Inputs.WndID)
			{
				printf("Load Wind File (.wnd) Data Input Incomplete\n");				CanGo = false;
			}
			else if (!Inputs.FuelMoisID)
			{
				printf("Load Initial Fuel Moistures (.fms) Data Input Incomplete\n");				 	CanGo = false;
			}
			else if (!Inputs.AdjID)
			{
				printf("Load Adjustment File (.adj) Data Input Incomplete\n");				CanGo = false;
			}
			else if (NeedCustFuelModels() && !HaveCustomFuelModels())
			{
				printf("Load Custom Fuel Models (.fmd) Data Input Incomplete\n");				 	CanGo = false;
			}
			else if (NeedConvFuelModels() && !HaveFuelConversions())
			{
				printf("Load Fuel Conversions (.cnv) Data Input Incomplete\n");				CanGo = false;
			}
			if (CanGo)
			{*/
				Initiate() ;
			//}
		}
		StepThrough = false;
	//SaveSettingsToFile("./Settings.txt");    //JAS  added 9/30/05
}

/**********************************************************************************
* THIS FUNCTION MAKES STARTS SIMULATION PROCESS LOOP
* Ret: 1 ok
*      < 0 Error,  use icf.ErrorMessage(errnum)
***********************************************************************************/
int Farsite5::Execute_StartRestart()
{
int i_Ret;

	/*     if(DDE.DDE_SIMCOMMAND)   	// delay response for dde return
				 {    Sleep(500);
					  DDE.DDE_SIMCOMMAND=false;
				 }
			*/
	if (SIMULATE_GO)
	{
		//Beep(24000, 150);

		//int Response = IDYES;

		//if (!FARSITE_GO)
		//	Response = IDYES;
		//printf("? Are You Sure You Want To Restart The Simulation ? RESTART FARSITE SIMULATION ?\n");		if (Response == IDYES)
		//{
			Execute_Reset();
			//  							  //CmSimulateRestart();
			// reset the simulation to beginning
			if (IgnitionResetAtRestart(GETVAL))
				LoadIgnitions();
			CountFires();
		//}
	}
	else
	{
		if (F_ON)
		{
			long i, RealFires = 0;
			for (i = 0; i < GetNewFires(); i++)
			{
				if (GetInout(i) > 0 && GetInout(i) < 3)
				{
					RealFires = 1;
					break;
				}
			}
			if (RealFires > 0)	//GetNewFires()>0
			{
 				if (GetSimulationDuration() <= 0.0)//numaximum==0)
				{
					StepThrough = false;
					printf("Set Duration (Simulate | Duration) No Duration For Simulation\n");				}
				else if (GetActualTimeStep() <= 0)
				{
					StepThrough = false;
					printf("Set Model | Parameters before Starting the Simulation Time- and Space-resolutions not set\n");				}
				else
				{
					long i;
					bool NEEDMX = false;
					char msg[128] = "", msg2[128] = "";
					LandscapeTheme* grid;

					/*
															for(i=14;  i<257; i++)
															{	if(!IsNewFuelReserved(i) && GetNewFuel(i, 0))
																 {	if(!InitialFuelMoistureIsHere(i))
																	{	NEEDMX=true;
																		sprintf(msg, "Custom Fuel Model %ld Has No Initial Fuel Moisture", i);
																		   sprintf(msg2, "Update File %s before continuing", Inputs.MBMois);
																		   break;
																	}
															   }
															}
															*/

					if (!NEEDMX)
					{
						grid = GetLandscapeTheme();
						for (i = 1; i <= grid->NumAllCats[3]; i++)
						{
							if (GetFuelConversion(grid->AllCats[3][i]) > 0 &&
								GetFuelConversion(grid->AllCats[3][i]) < 257)
							{
								if (!InitialFuelMoistureIsHere(GetFuelConversion(grid->AllCats[3][i])))
								{
									NEEDMX = true;
									//sprintf(msg, "Fuel Model %ld Has No Initial Fuel Moisture",(long) GetFuelConversion(grid->AllCats[3][i]));
									//sprintf(msg2,
									//	"Update File %s before continuing",
									//	Inputs.MBMois.c_str());
									break;
								}
							}
						}
					}

					if (!NEEDMX)
					{
						if (HaveGroundFuels())
						{
							if (GetTheme_Units(W_DATA) != 0 &&
								//Inputs.CwdID == false &&
								CheckPostFrontal(GETVAL))
							{
								sprintf(msg,
									"Post-frontal combustion enabled, but missing CWD file");
								sprintf(msg2,
									"Add .CWD File to Project before continuing");
								NEEDMX = true;
							}
						}
					}

					if (NEEDMX)
					{
						printf(" %s   %s\n",msg2,msg);  //Pretty sure this was the intended use of msg2 and msg.  So I've inserted it to replace next line down!  JAS 10/17/05
						return -1;
					}

					//CopyDDEFileInfo();
					//SaveIgnitions();
					////mb->NULLHintTextPointer();
					SIMULATE_GO = true;
					FARSITE_GO = true;
					sprintf(MBStatus, "    %s", "SIMULATION RUNNING");
					//NULLINPUTS();
					//WriteMessageBar(0);
					//mb->NULLHintTextPointer();
					LEAVEPROCESS = false;
					SetFarsiteProgress(0.0);
					burn.SIMTIME = 0.0;				   // reset start of FARSITE iterations
					burn.CuumTimeIncrement = 0.0;
					maximum = 0;
					NumLocalSpots=0;//numspots = 0;
					//pdata.ReSet();
					//adata.ReSet();
					ProcNum = 1;					   // first process can begin
					burn.DistMethod = DistanceCheckMethod(GETVAL);  		 // set distance check method
					AddDownTime(-1.0);	// set down time to 0.0;
					ConvertAbsoluteToRelativeBurnPeriod();
					NextFireAfterInterrupt = 0;
					OldFireAfterInterrupt = 0;
					smolder = flaming = 0.0;
					//EB next line only
					//StartSimThread();

					//uncommented EB

					i_Ret = FarsiteSimulationLoop();
					if ( i_Ret < 0 )  /* See notes in function heading */
                      return i_Ret ;

                    // start FARSITE process and check message loop between iteratations
				}
			}
			else
			{
				StepThrough = false;
				printf("Input Ignition(s) on Landscape Ignition(s) Required\n");			}
		}
		else
			printf("Must Initiate Simulation First (Display Landscape) Can't Start Simulation\n");	}
    return 1;
}





void Farsite5::Execute_ResumeSuspend()
{
	// resume/suspend simulation
	if (F_ON)
	{
		if (SIMULATE_GO)
		{
			if (FARSITE_GO && SIM_SUSPENDED == false)
			{
				StepThrough = false;
				//FARSITE_GO=false;		   // don't do it with mt version
				sprintf(MBStatus, "    %s", "SIMULATION SUSPENDED");
				SIM_SUSPENDED = true;
			}
			else
			{
				if (GetSimulationDuration() == 0.0)//numaximum==0)
				{
					StepThrough = false;
					printf("Verify or Reset Duration Weather/Wind Streams Changed\n");				}
				else
				{
					SIM_SUSPENDED = false;
					SimRequest = SIMREQ_NULL;
					//mb->NULLHintTextPointer();
					FARSITE_GO = true;
					//mb->NULLHintTextPointer();
					//if (!StepThrough)
						//Beep(24000, 150);
					//if(burn.SIMTIME>=maximum)	 // if duration has been extended
					//CopyDDEFileInfo();
					ConvertAbsoluteToRelativeBurnPeriod();
					FarsiteSimulationLoop() ; // for single thread. [BLN]
					//SetEvent(hFarSimEvent);//StartSimThread();//FarsiteSimulationLoop();	   // reenter FarsiteSimulationLoop()
				}
			}
		}
		else
			printf(" START Simulation First Can't Resume Simulation\n");	}
	else
		printf("Can't Run Simulation\n");
}






void Farsite5::Execute_Reset()
{
	ResetFarsite();
	SIMULATE_GO = false;
}

void Farsite5::Initiate()
{
	long i;

	//CopyDDEFileInfo();
	//mb->NULLHintTextPointer();
	a1 = 0;
	p1 = 0;
	p2 = 0;
	sarea = 0;
	firesizeh = 0;
	firesizes = 0;			// for area and perimeter calculations
	PrimaryVisEqual = 0;
	PrimaryVisCount = 0;
	SecondaryVisEqual = -1;
	SecondaryVisCount = 0;
	CurrentFire = 0;
	//added EB  for override
	F_ON = true;

	ResetVisPerimFile();		 // reset visperim file to 0
	for (i = 0; i < 5; i++)
		WStat[i].tws = -1.0;
	LastFMCalcTime = 0;
	//SYSTEMTIME tb;
	//GetSystemTime(&tb);
	//srand(tb.wMilliseconds);
	idum = -((rand() % 20) + 1);				// initialize random num gen
	StepThrough = false;
}

void Farsite5::Terminate()
{
	long i, j ;

	FARSITE_GO=false;
	SIMULATE_GO=false;
    ResetFarsite();
	CloseLandFile();
	if(F_ON==true)
	{
	//	Inputs.LandID=false;
	//	Inputs.WtrID=false;
	//	Inputs.WndID=false;
	//	Inputs.AdjID=false;
	//	Inputs.FuelMoisID=false;
		for(i=0; i<5; i++)    // initialize strings for fileanmes
		{
			//Inputs.WeatherFile[i][0] = 0;//bfs::path() ;
			//Inputs.WindFile[i][0]    = 0;//bfs::path() ;
			//Inputs.MBWtr[i]="Load Weather";
			//Inputs.MBWnd[i]="Load Winds";
			FreeWeatherData(i);
			FreeWindData(i);
		}
        SetAtmosphereGrid(0);
	}
	//Inputs.LandscapeFile[0] = 0;//bfs::path() ;
	FreeStationGrid();								// reset data structures for model
	//Inputs.MBLand="Load Landscape";
	sprintf(MBStatus, "%s", "SIMULATION TERMINATED");
	ResetCustomFuelModels();
	ResetFuelConversions();
	SetActualTimeStep(0.0);    // reset actual timestep, must be reset for new sim
	SetDistRes(0.0);           // reset distance check, must be reset for new simulation
    SetPerimRes(0.0);
	ResetDuration();
    SetStartDate(0);
    SetStartMonth(0);
    SetStartDay(0);
    SetStartHour(0);
    SetStartMin(0);
	SetEndDate(0);
    SetEndMonth(0);
    SetEndDay(0);
    SetEndHour(0);
    SetEndMin(0);
    SetConditMonth(0);
    SetConditDay(0);
	//Inputs.ResetData();
    FreeAllVectorThemes();	//StoredVectorFiles();
    FreeAllCompoundCrews();
    j=GetNumCrews();           /// must use constant j because NumCrews changes in FreeCrew();
    for(i=j-1; i>-1; i--)
     	FreeCrew(i);
    j=GetNumAirCraft();        /// must use constant j because NumCrews changes in FreeCrew();
    for(i=0; i<j; i++)
     	FreeAirCraft(i);
    memset(CrewFileName, 0x0, sizeof(CrewFileName));//TransGroundData.ResetData();
 	if(gaat)
     	delete gaat;
    gaat=0;
    memset(CrewFileName, 0x0, sizeof(CrewFileName));
    FreeBurnPeriod();

    FreeAllCoarseWoody();
    LastFMCalcTime=0;
    burn.post.ResetAllThreads();
   	burn.ResetAllPerimThreads();
//    burn.env->ResetAllThreads();
//    burn.env->TerminateHistory();
    burn.CloseCrossThreads();
    ResetAllStopLocations();
    CheckPostFrontal(false);
    ResetNewFuels();
}

void Farsite5::LoadIgnitions()
{
	// for reloading ignitions after restarting
	long i, j, k, l, m, n;
	double xpt, ypt, ros, fli;
	FILE* IgFile;

	if ((IgFile = fopen(Ignition.ifile, "r")) != NULL)
	{
		fscanf(IgFile, "%ld", &m);
		//SetNewFires(m);
		n = 0;
		for (i = 0; i < m; i++)
		{
			fscanf(IgFile, "%ld %ld %ld", &j, &k, &l);
			if (l > 0)
			{
				SetNumPoints(j, k - 1);
				SetInout(j, l);
				AllocPerimeter1(j, k);
				n++;
			}
			for (j = 0; j < k; j++)
			{
				fscanf(IgFile, "%lf %lf %lf %lf", &xpt, &ypt, &ros, &fli);
				if (l < 1)
					continue;
				SetPerimeter1(i, j, xpt, ypt);
				SetFireChx(i, j, ros, fli);
			}
		}
		SetNewFires(n);
		fclose(IgFile);
		//		for (i = 0; i < GetNewFires(); i++)
		//			DrawFire(i);
	}
}

void Farsite5::SaveIgnitions()
{
	// for saving ignitions before restarting
	long i, j, k;
	double xpt, ypt, ros, fli;
	FILE* IgFile;

	//memset(IgFileName, 0x0, sizeof(IgFileName));
	//strcpy(IgFileName, FarsiteDirectory);
	//strcat(IgFileName, "\\ignition.ign");
	IgFile = fopen(IgFileName, "w") ;
#if 0
	if (IgFile == NULL)
	{
		//SetFileAttributes(IgFileName, FILE_ATTRIBUTE_NORMAL);
		chmod(IgFileName,S_IRWXO);
		IgFile = fopen(IgFileName, "w");
	}
#endif

	if (IgFile)
	{
		fprintf(IgFile, "%ld\n", GetNewFires());
		for (i = 0; i < GetNewFires(); i++)
		{
			k = GetNumPoints(i) + 1;
			if (k > 0)
			{
				fprintf(IgFile, "%ld %ld %ld\n", i, k, GetInout(i));
				for (j = 0; j < k; j++)
				{
					xpt = GetPerimeter1Value(i, j, XCOORD);
					ypt = GetPerimeter1Value(i, j, YCOORD);
					ros = GetPerimeter1Value(i, j, ROSVAL);
					fli = GetPerimeter1Value(i, j, FLIVAL);
					fprintf(IgFile, "%lf %lf %lf %lf\n", xpt, ypt, ros, fli);
				}
			}
		}
		fclose(IgFile);
	}
	else
		printf("Check File Attributes and Change From 'READ ONLY' Could Not Write File\n");
}
void Farsite5::ResetVisPerimFile()
{
	//	system("del c:\visperim.vsp");
	//WaitForSingleObject(hVisperimSemaphore, INFINITE);
	//bfs::path visPerimPath = bfs::path(VisPerim) ;
	//if (bfs::exists(visPerimPath))
	//{
		//SetFileAttributes(VisPerim, FILE_ATTRIBUTE_NORMAL);
		//bfs::remove(visPerimPath) ;
	//}

	// compose a randomly generated filename
	char RNum[16] = "";
	sprintf(RNum, "%d", rand() % 1000);
	//std::string fname = "visperim" + std::string(RNum) + ".vsp" ;
	char fname[256];
	sprintf(fname, "visperim%s.vsp", RNum);
	// put file in cwd (?)
	//char visPerimPath[256];
	//visPerimPath = bfs::initial_path() / fname ;

	// copy from a path object to a C String
	//std::strncpy(VisPerim, visPerimPath.native_file_string().c_str(),
	//	sizeof VisPerim) ;

	// remove file if it exists
	//if (bfs::exists(visPerimPath))
	//{
		//SetFileAttributes(VisPerim, FILE_ATTRIBUTE_NORMAL);
	//	remove(visPerimPath);
	//}
	MemCount = 0;
	//ReleaseSemaphore(hVisperimSemaphore, 1, &prevct);
	//	VP=fopen(VisPerim, "w");
	//	fclose(VP);
	//	DeleteFile(VisPerim);
}


void Farsite5::ResetDuration()
{
	SetSimulationDuration(0);
}

void Farsite5::FreeAllVectorThemes()
{
	CurVect = FirstVect;
	for (long i = 0; i < NumVectorThemes; i++)
	{
		NextVect = (VectorStorage *) CurVect->next;
		if (CurVect)
			delete[] CurVect;
		CurVect = NextVect;
	}
	if (CurVect)
		delete[] CurVect;
	FirstVect = 0;
	CurVect = 0;
	NextVect = 0;
	NumVectorThemes = 0;
}

void Farsite5::FreeAllCompoundCrews()
{
	//long i;

	for (long i = 0; i < NumCompoundCrews; i++)
	{
		if (compoundcrew[i])
		{
			delete[] compoundcrew[i]->CrewIndex; //free(compoundcrew[i]->CrewIndex);
			delete[] compoundcrew[i]->Multiplier; //free(compoundcrew[i]->Multiplier);
			delete compoundcrew[i]; //free(compoundcrew[i]);
		}
		compoundcrew[i] = 0;
	}
	NumCompoundCrews = 0;
}

bool Farsite5::InquireInBurnPeriod(double SimTime)
{
	if (!rbp)
		return true;

	for (long i = LastAccess + 1; i < NumRelativeData; i++)
	{
		if (SimTime > rbp[i].Start)
		{
			if (SimTime <= rbp[i].End + 1)
			{
				LastAccess = i - 1;

				return true;
			}
		}
		else
			break;
	}

	return false;
}
/**********************************************************************
* NOTE: This function will handle end of year wrap.
* Calculates number of minutes from the Farsite Start date/time
* to the start date/time of the burn periods and stores them - see
* code below.
**********************************************************************/
void Farsite5::ConvertAbsoluteToRelativeBurnPeriod()
{
	long i, days;
	double End, Start, StartMins, hrs, mins;
	//RelativeBurnPeriod* trbp;

	if (abp == 0)
		return;
	if (rbp)
		delete[] rbp;//GlobalFree(rbp);//free(rbp);
	StartMins = (GetJulianDays(startmonth) + startday - 1) * 1440;
	mins = modf(starthour / 100.0, &hrs);
	StartMins += (hrs * 60.0 + mins);
	NumRelativeData = NumAbsoluteData;
	rbp = new RelativeBurnPeriod[NumRelativeData];
	for(i = 0; i < NumRelativeData; i++)
	{
		days = GetJulianDays(abp[i].Month);
		days += abp[i].Day - 1;
		End = Start = days * 1440;
		mins = modf(abp[i].Start / 100.0, &hrs);
		Start += (hrs * 60.0 + mins * 100.0);
		mins = modf(abp[i].End / 100.0, &hrs);
		End += (hrs * 60.0 + mins * 100.0);

		if(StartMins > Start)
		{
			Start = Start + 365 * 24 * 60;
			End = End + 365 * 24 * 60;
		}
		rbp[i].Start = Start - StartMins;
		rbp[i].End = End - StartMins;

	}
	LastAccess = -1;

}
long Farsite5::GetMinMonth()
{
	long i;
	long minmonth = FirstMonth[0].all;
	long NStat = GetNumStations();

	for (i = 0; i < NStat; i++)
	{
		if (FirstMonth[i].all > minmonth)
			minmonth = FirstMonth[i].all;
	}

	return minmonth;
}


long Farsite5::GetMinDay()
{
	long i;
	long minday = FirstDay[0].all;
	long NStat = GetNumStations();
	long minmonth = GetMinMonth();

	for (i = 0; i < NStat; i++)
	{
		if (FirstMonth[i].all == minmonth)
		{
			if (FirstDay[i].all > minday)
				minday = FirstDay[i].all;
		}
	}

	return minday;
}
long Farsite5::GetEndMonth()
{
	return endmonth;
}

long Farsite5::GetEndDay()
{
	return endday;
}

long Farsite5::GetEndHour()
{
	return endhour;
}

long Farsite5::GetEndDate()
{
	return enddate;
}

long Farsite5::GetEndMin()
{
	return endmin;
}

void Farsite5::FreeBurnPeriod()
{
	NumRelativeData = 0;
	NumAbsoluteData = 0;
	if (abp)
		delete[] abp;	//free(abp);
	abp = 0;
	if (rbp)
		delete[] rbp;	//free(rbp);
	rbp = 0;
	DownTime = 0.0;
	LastAccess = -1;
}

void Farsite5::FlatSimulateInitiateTerminate()
{
	Execute_InitiateTerminate();
}

/**************************************************************************
*
* Ret: 1 OK,
*      < 0 Error Number, to get error message use CFarsite.GetErrorMessage(errnum)
*         or icf.ErrorMessage(errnum) to get error text
***************************************************************************/
int Farsite5::LaunchFarsite(void)
{
int i_Ret;
	if(!Ignition.GetLightsLandscape())
		return e_EMS_FARSITE_NO_IGNITION;
   Ignition.ShapeInput();
   if(strlen(icf.cr_FarsiteBarrier) > 0)
   {
	   SetBarrier(icf.cr_FarsiteBarrier);
		//ShapefileBarrier barrier(this);
		//barrier.setFileName(icf.cr_FarsiteBarrier);
		//barrier.load();
   }
	AllocStationGrid(1, 1);   	   // initialize station grid to 1
	ResetDuration();
	FlatSimulateInitiateTerminate();
	//SetModelParams();
	//on these in the start data case and end date case
	SetEndDate(GetJulianDays(GetEndMonth()) + GetEndDay());
	SetStartDate(GetJulianDays(GetStartMonth()) + GetStartDay());//+GetStartHour()/100);
	SetSimulationDuration(ConvertActualTimeToSimtime(GetEndMonth(),
							GetEndDay(), GetEndHour(), GetEndMin(), false));
	//time(&timeLaunchFarsite);
	timeLaunchFarsite = clock();

    i_Ret =	Execute_StartRestart();

	timeFinish = clock();

    return i_Ret;    /* 1 = ok,  < 0 Error occured */
}

/*****************************************************************/
bool Farsite5::AllocBurnPeriod(long Num)
{
	FreeBurnPeriod();
	NumAbsoluteData = Num;
	if(Num <= 0)
		return Num;
	abp = new AbsoluteBurnPeriod[Num];
	if (abp == NULL)
		return false;
	memset(abp, 0x0, Num * sizeof(AbsoluteBurnPeriod));


	//long i, FirstDay, StartDay, EndDay, NumDays, LastDay;
/*	long i,FirstDay, NumDays, LastDay;
	//long mo, dy, JDay, maxdy;
	long mo, dy, JDay;

	FirstDay = GetJulianDays(GetMinMonth());
	FirstDay += GetMinDay();
	LastDay = GetJulianDays(GetMaxMonth());
	LastDay += GetMaxDay();
	if (FirstDay < LastDay)
		NumDays = LastDay - FirstDay;
	else
		NumDays = 365 - FirstDay + LastDay;
	if (Num < NumDays)
		Num = NumDays;

	abp = new AbsoluteBurnPeriod[Num];
	if (abp == NULL)
		return false;
	memset(abp, 0x0, Num * sizeof(AbsoluteBurnPeriod));
	NumAbsoluteData = Num;

	mo = GetMinMonth();
	dy = GetMinDay();
	for (i = 0; i < NumDays; i++)
	{
		abp[i].Month = mo;
		abp[i].Day = dy;
		abp[i].Start = 0;
		abp[i].End = 2400;
		dy++;
		JDay = dy + GetJulianDays(mo);
		if (mo < 12)
		{
			if (JDay > GetJulianDays(mo + 1))
			{
				dy = 1;
				mo++;
			}
		}
		else
		{
			if (JDay > 365)
			{
				dy = 1;
				mo = 1;
			}
		}
	}*/

	return true;
}


/*void Farsite5::FreeBurnPeriod()
{
	NumRelativeData = 0;
	NumAbsoluteData = 0;
	if (abp)
		delete[] abp;	//free(abp);
	abp = 0;
	if (rbp)
		delete[] rbp;	//free(rbp);
	rbp = 0;
	DownTime = 0.0;
	LastAccess = -1;
}*/


void Farsite5::SetBurnPeriod(long num, long mo, long dy, long yr, long start, long end)
{
	//long i;
	if (abp)
	{
		//for (i = 0; i < NumAbsoluteData; i++)
		//{
		//	num = i;
		//	if (abp[i].Month == mo && abp[i].Day == dy)
		//		break;
		//}
		abp[num].Month = mo;
		abp[num].Day = dy;
        abp[num].Year = yr;
		abp[num].Start = start;
		abp[num].End = end;
	}
}

long Farsite5::GetMaxMonth()
{
	long maxmonth = LastMonth[0].all;
	long numstations = GetNumStations();

	for (long i = 0; i < numstations; i++)
	{
		if (LastMonth[i].all < maxmonth)
			maxmonth = LastMonth[i].all;
	}

	return maxmonth;
}

long Farsite5::GetMaxDay()
{
	long i;
	long maxday = LastDay[0].all;
	long numstations = GetNumStations();
	long maxmonth = GetMaxMonth();

	for (i = 0; i < numstations; i++)
	{
		if (LastMonth[i].all == maxmonth && maxday > LastDay[i].all)
			maxday = LastDay[i].all;
	}

	return maxday;
}

int Farsite5::SetIgnitionFileName(char *_ignitionFileName)
{
	strcpy(icf.cr_FarsiteIgnition, _ignitionFileName);
	strcpy(Ignition.ifile, icf.cr_FarsiteIgnition);
	return CreateIgnitionGrid();
}

int Farsite5::SetBarrierFileName(char *_barrierFileName)
{
	strcpy(icf.cr_FarsiteBarrier, _barrierFileName);

	return 1;
}

int Farsite5::SetBarrier(char *_barrierFileName)
{
	SetBarrierFileName(_barrierFileName);
	//new way to deal with barriers: Set Landscape fuels to non-burnable
	if(!ignitionGrid)
	{
		return 0;//must load ignitions first!
	}
		long i, j, NumPts, NumAlloc, NumSeg, cellx, celly;
		double x, y, x1, x2, y1, y2, dx, dy, dist, fract, res, linedist;
		res = Header.XResol;
		dist=res/IGNITON_GRID_LINEDIST_DIVISOR;
		double tWest = GetLoEast();
		double tSouth = GetHiNorth();
		//westUTM = pFarsite->ConvertEastingOffsetToUtm(West);//*MetersToKm);
			//fprintf(otpfile, "%s %lf\n", "YLLCORNER",
		//southUTM = pFarsite->ConvertNorthingOffsetToUtm(South);//*MetersToKm);
		double west = ConvertEastingOffsetToUtm(tWest), north = ConvertNorthingOffsetToUtm(tSouth), *pts = NULL;
		long NumCols = ignitionCols, NumRows = ignitionRows;
		SHPHandle	hSHP;
		double 	adfMinBound[4], adfMaxBound[4];
		SHPObject *pShape;
		int nShapes, shapeType;
		hSHP = SHPOpen( _barrierFileName, "rb" );
		if( hSHP == NULL )
		{
			return FALSE;
		}
		SHPGetInfo( hSHP, &nShapes, &shapeType, adfMinBound, adfMaxBound );
		NumAlloc=0;
		for(int s = 0; s < nShapes; s++ )
		{
			pShape = SHPReadObject( hSHP, s );
			//NumPts=0;
			switch(pShape->nSHPType)
			{
			case SHPT_POINT:
			{
				x = pShape->padfX[0];
				y = pShape->padfY[0];
				//MTT_RotateVectorPoint(&x, &y, true);
				cellx=(long) ((x-west)/res);     // zero based
				celly=(long) ((north-y)/res);//-1;
				//celly=(long) ((north-y)/res)-1;
				if(cellx<0 || cellx>NumCols-1)
					break;
				if(celly<0 || celly>NumRows-1)
					break;
				ignitionGrid[celly][cellx]= 0.0;//nodeSpread->GetStartTime();
			}
				break;
			case SHPT_MULTIPOINT:
				break;
			case SHPT_ARC:
			case SHPT_POLYGON:
			{
				xPolygon poly;
				for(int p = 0; p < pShape->nParts; p++)
				{
					int pEnd = (p < pShape->nParts - 1) ? pShape->panPartStart[p + 1] : pShape->nVertices;
					NumPts = pEnd - pShape->panPartStart[p];// + 1;
					if(pts)
						delete[] pts;
					pts=new double[NumPts*2];
					i = 0;
					for(int v = pShape->panPartStart[p]; v < pEnd; v++)
					{
						pts[i*2] = pShape->padfX[v];
						pts[i*2+1]= pShape->padfY[v];
						i++;
					}
					// copy zeros to cells crossed by line
					x1=pts[0];
					y1=pts[1];
					poly.AllocVertex(NumPts);
					poly.SetVertex(x1, y1, 0);
					for(i=1; i<NumPts; i++)
					{
						x2=pts[i*2];
						y2=pts[i*2+1];
						poly.SetVertex(x2, y2, i);
						dx=(x1-x2);
						dy=(y1-y2);
						linedist=sqrt(pow2(dx)+pow2(dy));
						NumSeg=(long) (linedist/dist)+1;
						fract=dist/linedist;//((double) NumSeg);
						for(j=0; j<NumSeg; j++)
						{
							x=x1-dx*fract*((double) j);
							y=y1-dy*fract*((double) j);
							//MTT_RotateVectorPoint(&x, &y, true);
							cellx=(long) ((x-west)/res);     // zero based
							celly=(long) ((north-y)/res);//-1;
							//celly=(long) ((north-y)/res)-1;
							if(cellx<0 || cellx>NumCols-1)
								continue;
							if(celly<0 || celly>NumRows-1)
								continue;
							//if(nodeSpread->DataExist(cellx, celly))
							//{
								long Position = GetCellPosition(x, y);
								SetCellFuel(Position, 91);
							//igCell = CellData(x, y, igCell, igCrown, igGround, &garbage);
								//float fuel = igCell.f;// NumRows - celly - 1);

							//if(ignitionGrid[celly][cellx] != 1.0 && igCell.f > 0 && (igCell.f < 90 || igCell.f > 99))
								ignitionGrid[celly][cellx] = 0.0;//.Time[0]=nodeSpread->GetStartTime();
							//}
						}
						x1=x2;
						y1=y2;
					}
					if(pShape->nSHPType == SHPT_POLYGON && this->icf.i_FarsiteFillBarriers)
						FillBarrierPolygon(&poly); // Fills polygon with 0's
				}
			}
			default:
				break;
			}
			SHPDestroyObject(pShape);

			//fscanf(fin, "%s", testend);
			//fscanf(fin, "%s", testend);    // new id tag
			//if(feof(fin))
			//	break;
		}
		if(pts)
			delete[] pts;
		SHPClose(hSHP);
		return 1;
}

int Farsite5::WriteArrivalTimeGrid(char *trgName)
{
	strcpy(RasterArrivalTime, trgName);
	//burn.rast->SelectMemOutputs(5);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_ARRIVALTIME);
	//return 1;
}

int Farsite5::WriteIntensityGrid(char *trgName)
{
	strcpy(RasterFireIntensity, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_FIREINTENSITY);
	//return 1;
}

int Farsite5::WriteFlameLengthGrid(char *trgName)
{
	strcpy(RasterFlameLength, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_FLAMELENGTH);
	//return 1;
}

int Farsite5::WriteSpreadRateGrid(char *trgName)
{
	strcpy(RasterSpreadRate, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_SPREADRATE);
	//return 1;
}

int Farsite5::WriteSpreadDirectionGrid(char *trgName)
{
	strcpy(RasterFireDirection, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_FIREDIRECTION);
	//return 1;
}

int Farsite5::WriteHeatPerUnitAreaGrid(char *trgName)
{
	strcpy(RasterHeatPerArea, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_HEATPERAREA);
	//return 1;
}

int Farsite5::WriteReactionIntensityGrid(char *trgName)
{
	strcpy(RasterReactionIntensity, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_REACTIONINTENSITY);
	//return 1;
}

int Farsite5::WriteCrownFireGrid(char *trgName)
{
	strcpy(RasterCrownFire, trgName);
	burn.rast->setHeaderType(5) ;
	// save current state of object to map
	burn.rast->SetRastData() ;
	return burn.rast->WriteFile(RAST_CROWNFIRE);
}


int Farsite5::WriteArrivalTimeGridBinary(char *trgName)
{
	return burn.rast->WriteArrivalTimeBinary(trgName);
}

int Farsite5::WriteIntensityGridBinary(char *trgName)
{
	return burn.rast->WriteIntensityBinary(trgName);
}
int Farsite5::WriteFlameLengthGridBinary(char *trgName)
{
	return burn.rast->WriteFlameLengthBinary(trgName);
}
int Farsite5::WriteSpreadRateGridBinary(char *trgName)
{
	return burn.rast->WriteSpreadRateBinary(trgName);
}
int Farsite5::WriteSpreadDirectionGridBinary(char *trgName)
{
	return burn.rast->WriteSpreadDirectionBinary(trgName);
}
int Farsite5::WriteHeatPerUnitAreaGridBinary(char *trgName)
{
	return burn.rast->WriteHeatPerUnitAreaBinary(trgName);
}
int Farsite5::WriteReactionIntensityGridBinary(char *trgName)
{
	return burn.rast->WriteReactionIntensityBinary(trgName);
}
int Farsite5::WriteCrownFireGridBinary(char *trgName)
{
	return burn.rast->WriteCrownFireBinary(trgName);
}

int Farsite5::CreateIgnitionGrid()
{
	if(ignitionGrid)
	{
		for(long r = 0; r < ignitionRows; r++)
			delete[] ignitionGrid[r];
		delete[] ignitionGrid;
		ignitionGrid = NULL;
	}
	ignitionRows = Header.numnorth;
	ignitionCols = Header.numeast;
	ignitionGrid = new float*[ignitionRows];
	for(long r = 0; r < ignitionRows; r++)
	{
		ignitionGrid[r] = new float[ignitionCols];
		for(long c = 0; c < ignitionCols; c++)
			ignitionGrid[r][c] = NODATA_VAL;
	}
	m_nCellsLit = 0;
	celldata igCell;
	crowndata igCrown;
	grounddata igGround;
	long garbage;
		long i, j, NumPts, NumAlloc, NumSeg, cellx, celly;
		double x, y, x1, x2, y1, y2, dx, dy, dist, fract, res, linedist;
		res = Header.XResol;
		dist=res/IGNITON_GRID_LINEDIST_DIVISOR;
		double tWest = GetLoEast();
		//double tSouth = GetLoNorth();
		double tNorth = GetHiNorth();
		//westUTM = pFarsite->ConvertEastingOffsetToUtm(West);//*MetersToKm);
			//fprintf(otpfile, "%s %lf\n", "YLLCORNER",
		//southUTM = pFarsite->ConvertNorthingOffsetToUtm(South);//*MetersToKm);
		double west = ConvertEastingOffsetToUtm(tWest),
			//north = ConvertNorthingOffsetToUtm(tSouth), *pts = NULL;
			north = ConvertNorthingOffsetToUtm(tNorth), *pts = NULL;
		long NumCols = ignitionCols, NumRows = ignitionRows;
		SHPHandle	hSHP;
		double 	adfMinBound[4], adfMaxBound[4];
		SHPObject *pShape;
		int nShapes, shapeType;
		hSHP = SHPOpen( Ignition.ifile, "rb" );
		if( hSHP == NULL )
		{
			return FALSE;
		}
		SHPGetInfo( hSHP, &nShapes, &shapeType, adfMinBound, adfMaxBound );
		NumAlloc=0;
		for(int s = 0; s < nShapes; s++ )
		{
			pShape = SHPReadObject( hSHP, s );
			//NumPts=0;
			switch(pShape->nSHPType)
			{
			case SHPT_POINT:
			{
				x = pShape->padfX[0];
				y = pShape->padfY[0];
				//MTT_RotateVectorPoint(&x, &y, true);
				cellx=(long) ((x-west)/res);     // zero based
				celly=(long) ((north-y)/res);//-1;
				//celly=(long) ((north-y)/res)-1;
				if(cellx<0 || cellx>NumCols-1)
					break;
				if(celly<0 || celly>NumRows-1)
					break;
				ignitionGrid[celly][cellx]= 1.0;//nodeSpread->GetStartTime();
			}
				break;
			case SHPT_MULTIPOINT:
				break;
			case SHPT_ARC:
			case SHPT_POLYGON:
			{
				xPolygon poly;
				for(int p = 0; p < pShape->nParts; p++)
				{
					int pEnd = (p < pShape->nParts - 1) ? pShape->panPartStart[p + 1] : pShape->nVertices;
					NumPts = pEnd - pShape->panPartStart[p];// + 1;
					if(pts)
						delete[] pts;
					pts=new double[NumPts*2];
					i = 0;
					for(int v = pShape->panPartStart[p]; v < pEnd; v++)
					{
						pts[i*2] = pShape->padfX[v];
						pts[i*2+1]= pShape->padfY[v];
						i++;
					}
					// copy zeros to cells crossed by line
					x1=pts[0];
					y1=pts[1];
					poly.AllocVertex(NumPts);
					poly.SetVertex(x1, y1, 0);
					for(i=1; i<NumPts; i++)
					{
						x2=pts[i*2];
						y2=pts[i*2+1];
						poly.SetVertex(x2, y2, i);
						dx=(x1-x2);
						dy=(y1-y2);
						linedist=sqrt(pow2(dx)+pow2(dy));
						NumSeg=(long) (linedist/dist)+1;
						fract=dist/linedist;//((double) NumSeg);
						for(j=0; j<NumSeg; j++)
						{
							x=x1-dx*fract*((double) j);
							y=y1-dy*fract*((double) j);
							//MTT_RotateVectorPoint(&x, &y, true);
							cellx=(long) ((x-west)/res);     // zero based
							celly=(long) ((north-y)/res);//-1;
							//celly=(long) ((north-y)/res)-1;
							if(cellx<0 || cellx>NumCols-1)
								continue;
							if(celly<0 || celly>NumRows-1)
								continue;
							//if(nodeSpread->DataExist(cellx, celly))
							//{
							igCell = CellData(x, y, igCell, igCrown, igGround, &garbage);
								//float fuel = igCell.f;// NumRows - celly - 1);

							if(ignitionGrid[celly][cellx] != 1.0 && igCell.f > 0 && (igCell.f < 90 || igCell.f > 99))
								ignitionGrid[celly][cellx] = 1.0;//.Time[0]=nodeSpread->GetStartTime();
							//}
						}
						x1=x2;
						y1=y2;
					}
					FillIgnitionPolygon(&poly, 1.0); // Fills polygon with 1's
				}
			}
			default:
				break;
			}
			SHPDestroyObject(pShape);

			//fscanf(fin, "%s", testend);
			//fscanf(fin, "%s", testend);    // new id tag
			//if(feof(fin))
			//	break;
		}
		if(pts)
			delete[] pts;
		SHPClose(hSHP);

		//now make sure we have something lit.
		for(long r = 0; r < ignitionRows; r++)
		{
			for(long c = 0; c < ignitionCols; c++)
			{
				if(ignitionGrid[r][c] > 0.0)
				{
					m_nCellsLit++;
					Ignition.SetLightsLandscape(true);
				}
			}
		}
	if(m_nCellsLit > 0)
		return 1;

	return e_EMS_FARSITE_NO_IGNITION; //no ignition cells....
}

void Farsite5::FillBarrierPolygon(xPolygon *poly)
{
	//celldata igCell;
	//crowndata igCrown;
	//grounddata igGround;
	//long garbage;
	long i, j;//, k, m;
	// double StartTime=pMTT->GetStartTime(), xpt, ypt;
	double xpt, ypt;
	double res=Header.XResol, west, north;
	long CXmin, CXmax, CYmin, CYmax;

	west=Header.WestUtm;//nodeSpread->GetGridCorner(GRID_WESTCORNER);
	north=Header.NorthUtm;//nodeSpread->GetGridCorner(GRID_NORTHCORNER);
	//west = pFlamMap->GetWest();
	//north = pFlamMap->GetNorth();
	// find cells index that includes bounding box
	CXmin=(long) ((poly->Xmin-west)/res)-1;
	CYmax=(long) ((north-poly->Ymin)/res)+1;
	CXmax=(long) ((poly->Xmax-west)/res)+1;
	CYmin=(long) ((north-poly->Ymax)/res)-1;

	// make sure bounding cells indices are within range
	if(CXmin<0)
		CXmin=0;
	if(CYmin<0)
		CYmin=0;
	if(CXmax>ignitionCols)
		CXmax=ignitionCols;
	if(CYmax>ignitionRows)
		CYmax=ignitionRows;

	// loop for all cells within bounding rectangle
	for(i=CYmin; i<CYmax; i++)
	{
		ypt=north-((double) i*res) - res/2;
		for(j=CXmin; j<CXmax; j++)
		{
			xpt=(double) j*res+west + res/2;
			if(poly->Inside(xpt, ypt))
			{
				//if(nodeSpread->DataExist(j, i))
				//float fuel = pFlamMap->GetLayerValueByCell(FUEL, j, i);
				long Position = GetCellPosition(xpt, ypt);
				SetCellFuel(Position, 91);
				ignitionGrid[i][j] = 0.0;
			}
		}
	}
}

void Farsite5::FillIgnitionPolygon(xPolygon *poly, float val)
{
	//celldata igCell;
	//crowndata igCrown;
	//grounddata igGround;
	//long garbage;
	long i, j;//, k, m;
	// double StartTime=pMTT->GetStartTime(), xpt, ypt;
	double xpt, ypt;
	double res=Header.XResol, west, north;
	long CXmin, CXmax, CYmin, CYmax;

	west=Header.WestUtm;//nodeSpread->GetGridCorner(GRID_WESTCORNER);
	north=Header.NorthUtm;//nodeSpread->GetGridCorner(GRID_NORTHCORNER);
	//west = pFlamMap->GetWest();
	//north = pFlamMap->GetNorth();
	// find cells index that includes bounding box
	CXmin=(long) ((poly->Xmin-west)/res)-1;
	CYmax=(long) ((north-poly->Ymin)/res)+1;
	CXmax=(long) ((poly->Xmax-west)/res)+1;
	CYmin=(long) ((north-poly->Ymax)/res)-1;

	// make sure bounding cells indices are within range
	if(CXmin<0)
		CXmin=0;
	if(CYmin<0)
		CYmin=0;
	if(CXmax>ignitionCols)
		CXmax=ignitionCols;
	if(CYmax>ignitionRows)
		CYmax=ignitionRows;

	celldata igCell;
	crowndata igCrown;
	grounddata igGround;
	long garbage;
	// loop for all cells within bounding rectangle
	for(i=CYmin; i<CYmax; i++)
	{
		ypt=north-((double) i*res) - res/2;
		for(j=CXmin; j<CXmax; j++)
		{
			xpt=(double) j*res+west + res/2;
			if(poly->Inside(xpt, ypt))
			{
				igCell = CellData(xpt, ypt, igCell, igCrown, igGround, &garbage);
				//if(nodeSpread->DataExist(j, i))
				//float fuel = pFlamMap->GetLayerValueByCell(FUEL, j, i);
				if(ignitionGrid[i][j] != 1.0 && igCell.f > 0 && (igCell.f < 90 || igCell.f > 99))
					ignitionGrid[i][j] = val;
			}
		}
	}
}
int Farsite5::WriteIgnitionGrid(char *trgName)
{
	if(ignitionGrid)
	{
		FILE *fout = fopen(trgName, "wt");
		if(!fout)
		{
			return e_EMS_FILE_OPEN_ERROR;
		}
		//double west = Header.WestUtm, north = Header.NorthUtm, *pts = NULL;
	//long outCols = nodeSpread->GetNumCols(), outRows = nodeSpread->GetNumRows();
		double outRes = Header.XResol,
			outWest = Header.WestUtm,//nodeSpread->West,
			outSouth =  Header.SouthUtm;//nodeSpread->South;
		if(fprintf(fout, "ncols %ld\n", ignitionCols) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "nrows %ld\n", ignitionRows) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "xllcorner %lf\n", outWest) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "yllcorner %lf\n", outSouth) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "cellsize %lf\n", outRes) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "NODATA_VALUE -9999\n") < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		for(long j=0; j<ignitionRows; j++)
		{
			for(long k=0; k<ignitionCols; k++)
			{     //burnfreq[0][j*ncols+k]/=(double) NumRiskThreads;// already handled
				//if(pFireScenario->
				if(fprintf(fout, "%f ", ignitionGrid[j][k]) < 0)
					return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
			}
			if(fprintf(fout, "\n") < 0)
				return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		}
		//fclose(fout);

		return CloseAndReturn(fout, 1);
	}
	return e_EMS_OUTPUT_DOES_NOT_EXIST;
}

void Farsite5::CleanPerimeters()
{
	CPerimeterData *delPerim, *nextPerim = perimeters;
	while(nextPerim)
	{
		delPerim = nextPerim;
		nextPerim = delPerim->GetNext();
		delete delPerim;
	}
	perimeters = lastPerimeter = 0;
	if(perimeter2)
		delete[] perimeter2;
	for(int f = 0; f < this->numfires; f++)
	{
		if(perimeter1[f])
			delete[] perimeter1[f];
	}
}

void Farsite5::AddCurrPerimeter()
{
	int NumVertices;//, NumParts;
	//int		*PartsArray;
	long InOut, count;//, NumRecord;
	long NumVertex;
	//char DataBaseID[64] = "";
	//char Description[256];
	double xpt, ypt;//, ros;

	InOut = GetInout(CurrentFire);
	if (InOut == 0 || InOut == 3)
		return;

	NumVertex = GetNumPoints(CurrentFire);
	NumVertices = NumVertex + 1;
	CPerimeterData *newData = new CPerimeterData();
	newData->SetNumPts(NumVertex + 1);

	for (count = 0; count < NumVertex; count++)
	{
			xpt = GetPerimeter1Value(CurrentFire, count, XCOORD);
			ypt = GetPerimeter1Value(CurrentFire, count, YCOORD);
			newData->SetPoint(count, xpt, ypt);
	}
	xpt = GetPerimeter1Value(CurrentFire, 0, XCOORD);
	ypt = GetPerimeter1Value(CurrentFire, 0, YCOORD);
	newData->SetPoint(count, xpt, ypt);
	newData->SetFireType(InOut);
	shape.ConvertSimTime(burn.SIMTIME);
	newData->SetMonth(shape.Month);
	newData->SetDay(shape.Day);
	newData->SetHour(shape.Hour);
	newData->SetElapsedMinutes(burn.SIMTIME);
	if(lastPerimeter)
	{
		lastPerimeter->SetNext(newData);
		lastPerimeter = newData;
	}
	else
	{
		lastPerimeter = perimeters = newData;
	}
	/*DBFWriteStringAttribute(hDBF, NumRecord, 0, Description);
	DBFWriteIntegerAttribute(hDBF, NumRecord, 1, Month);
	DBFWriteIntegerAttribute(hDBF, NumRecord, 2, Day);
	DBFWriteIntegerAttribute(hDBF, NumRecord, 3, Hour);
	DBFWriteDoubleAttribute(hDBF, NumRecord, 4, SimTime);*/

}

int Farsite5::WritePerimetersShapeFile(char *trgName)
{

	int NumVertices, NumParts = 0;
	//int		*PartsArray;
	long NumRecord;
	char DataBaseID[64] = "";
	char Description[256];
	double* VertexX = 0, * VertexY = 0, * VertexZ = 0;
	SHPHandle hSHP;
	SHPObject* pSHP;
	DBFHandle hDBF;

	int nShapeType = SHPT_ARC;
	hSHP = SHPCreate(trgName, nShapeType);
	if (hSHP == NULL)
		return 0;
	// Create the database.
	char dbfName[256];
	strcpy(dbfName, trgName);
	//strlwr(dbfName);
	char *p = strrchr(dbfName,'.');
	strcpy(p, ".dbf");
	hDBF = DBFCreate(dbfName);
	if (hDBF == NULL)
	{
		SHPClose(hSHP);
		return 0;
	}
	sprintf(DataBaseID, "%s", "Fire_Type");
	DBFAddField(hDBF, DataBaseID, FTString, 64, 0);
	sprintf(DataBaseID, "%s", "Month");
	DBFAddField(hDBF, DataBaseID, FTInteger, 32, 0);
	sprintf(DataBaseID, "%s", "Day");
	DBFAddField(hDBF, DataBaseID, FTInteger, 32, 0);
	sprintf(DataBaseID, "%s", "Hour");
	DBFAddField(hDBF, DataBaseID, FTInteger, 32, 0);
	sprintf(DataBaseID, "%s", "Elapsed_Minutes");
	DBFAddField(hDBF, DataBaseID, FTDouble, 32, 4);
	/*NumVertex = pFarsite->GetNumPoints(CurrentFire);
	if (BarExport)
	{
		NumVertex /= 2;
		NumVertices = NumVertex;
	}
	else
		NumVertices = NumVertex + 1;
	VertexX = new double[NumVertex + 1];
	VertexY = new double[NumVertex + 1];
	//VertexZ=new double[NumVertex+1];
	NumParts = 0;
	//PartsArray=new int[2];
	//PartsArray[0]=0;
	for (count = 0; count < NumVertex; count++)
	{
		if (!BarExport)
		{
			xpt = pFarsite->GetPerimeter1Value(CurrentFire, count, XCOORD);
			ypt = pFarsite->GetPerimeter1Value(CurrentFire, count, YCOORD);
			//ros=GetPerimeter1Value(CurrentFire, count, ROSVAL);
		}
		else
		{
			xpt = pFarsite->GetPerimeter2Value(count, XCOORD);
			ypt = pFarsite->GetPerimeter2Value(count, YCOORD);
			//ros=0.0;
		}
		VertexX[count] = pFarsite->ConvertEastingOffsetToUtm(xpt);
		VertexY[count] = pFarsite->ConvertNorthingOffsetToUtm(ypt);
		//VertexZ[count]=ros;
	}
	if (!BarExport)    // don't repeat first point on barrier line
	{
		xpt = pFarsite->GetPerimeter1Value(CurrentFire, 0, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(CurrentFire, 0, YCOORD);
		//ros=GetPerimeter1Value(CurrentFire, 0, ROSVAL);
		VertexX[count] = pFarsite->ConvertEastingOffsetToUtm(xpt);
		VertexY[count] = pFarsite->ConvertNorthingOffsetToUtm(ypt);
		//VertexZ[count]=ros;
	}*/
	LastAccess = -1;
	CPerimeterData *perimData = perimeters;
	while(perimData)
	{
		if(perimData->GetFireType() != 1)
		{
			perimData = perimData->GetNext();
			continue;
		}
		//if(!InquireInBurnPeriod(perimData->GetElapsedMinutes()))
		if(!InquireInBurnPeriod(perimData->GetElapsedMinutes()) && !InquireInBurnPeriod(perimData->GetElapsedMinutes()- GetActualTimeStep() + 1))
		{
			perimData = perimData->GetNext();
			continue;
		}
		NumVertices = perimData->GetNumPts();
		VertexX = new double[NumVertices];
		VertexY = new double[NumVertices];
		VertexZ=new double[NumVertices];
		for(int i = 0; i < NumVertices; i++)
		{
			VertexX[i] = perimData->GetPointX(i);
			VertexY[i] = perimData->GetPointY(i);
			VertexZ[i] = 0.0;
		}
		pSHP = SHPCreateObject(nShapeType, -1, NumParts, NULL, NULL, NumVertices,
			VertexX, VertexY, VertexZ, NULL);
		SHPWriteObject(hSHP, -1, pSHP);
		SHPDestroyObject(pSHP);

		NumRecord = DBFGetRecordCount(hDBF);

		if (VertexX)
			delete[] VertexX;
		if (VertexY)
			delete[] VertexY;
		if (VertexZ)
			delete[] VertexZ;
		switch(perimData->GetFireType())
		{
			case 1:
				strcpy(Description, "Expanding Fire");
				break;
			case 2:
				strcpy(Description, "Enclave Fire");
				break;
			case 3:
				strcpy(Description, "Barrier");
				break;
			default:
				Description[0] = 0;
		}
		DBFWriteStringAttribute(hDBF, NumRecord, 0, Description);
		DBFWriteIntegerAttribute(hDBF, NumRecord, 1, perimData->GetMonth());
		DBFWriteIntegerAttribute(hDBF, NumRecord, 2, perimData->GetDay());
		DBFWriteIntegerAttribute(hDBF, NumRecord, 3, perimData->GetHour());
		DBFWriteDoubleAttribute(hDBF, NumRecord, 4, perimData->GetElapsedMinutes());
		perimData = perimData->GetNext();
	}

	SHPClose(hSHP);
	DBFClose(hDBF);

	return 1;
}



/***********************************************************************************************/
int Farsite5::WriteTimingsFile(char *trgName)
{
  double  duration;
   duration = (double)(timeFinish - timeLaunchFarsite) / CLOCKS_PER_SEC;

	//SYSTEM_INFO sysinf;
	//GetSystemInfo(&sysinf);
	//long tprocs = sysinf.dwNumberOfProcessors;
	//CString procStr;
	//CPURegistry cpu;
	//cpu.QueryCPUInfo();
	FILE *bmFile = fopen(trgName, "wt");
	if(!bmFile)
		return e_EMS_FILE_OPEN_ERROR;
	fprintf(bmFile, "Farsite BenchMark Results\n");
	//fprintf(bmFile, "System Info:\n");
	//unsigned int winMajor, winMinor, osBuild, osPlatform;
   // OSVERSIONINFO osvi;
   // BOOL bIsWindowsXPorLater;

    //ZeroMemory(&osvi, sizeof(OSVERSIONINFO));
    //osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);

    //GetVersionEx(&osvi);
	//winMajor = osvi.dwMajorVersion;
	//winMinor = osvi.dwMinorVersion;
	//osPlatform = osvi.dwPlatformId;
	//osBuild = osvi.dwBuildNumber;
	//if(fprintf(bmFile, "\tWinMajor: %d, WinMinor: %d, Platform: %d, Build: %d\n",
	//	winMajor, winMinor, osPlatform, osBuild) < 0)
	//	return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
/*	char tmpStr[512];
	strcpy(tmpStr, cpu.GetCPUName().c_str());
	if(fprintf(bmFile, "\tProcessor: %s\n", tmpStr) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	strcpy(tmpStr, cpu.GetVendorName().c_str());
	if(fprintf(bmFile, "\tVendor: %s\n", tmpStr) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tFamily: %d\n", cpu.GetCPUFamily()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tModel: %d\n", cpu.GetCPUModel()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tStepping: %d\n", cpu.GetCPUStepping()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSpeed (MHz): %d\n", cpu.GetSpeedMHz()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);*/
	//fprintf(bmFile, "\tProcessor: %s\n", cpu.GetCPUName());
	//fprintf(bmFile, "\tProcessor: %s\n", cpu.GetCPUName());
	//char pTmp[512];
	//wchar_t wpTmp[512];
	//CW2A str(procStr.GetBuffer());
	//strcpy(pTmp, str);
	//strcpy(pTmp, procStr.GetBuffer());
	//fprintf(bmFile, "%s\n", pTmp);
	if(fprintf(bmFile, "\n\nSimulation Info:\n") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tLandscape File: %s\n", this->LandFName) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tLandscape Resolution: %.1f\n", this->GetCellResolutionX()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tLandscape Size: %ld x %ld\n", this->GetNumEast(), this->GetNumNorth()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tInputs File: %s\n", 	InputsFName) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tFarsite Start Time: %02ld %02ld %04ld\n", this->GetStartMonth(), this->GetStartDay(), this->GetStartHour()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tFarsite End Time:   %02ld %02ld %04ld\n", this->GetEndMonth(), this->GetEndDay(), this->GetEndHour()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tTotal Sim Time:     %lf\n", this->GetSimulationDuration()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tIgnition File:      %s\n", icf.cr_FarsiteIgnition) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(strlen(icf.cr_FarsiteBarrier) > 0)
	{
		if(fprintf(bmFile, "\tBarriers File:      %s\n", icf.cr_FarsiteBarrier) < 0)
			return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	}
	else
	{
		if(fprintf(bmFile, "\tBarriers File:      None\n") < 0)
			return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	}
	if(fprintf(bmFile, "\tTime Step:                  %f\n", GetActualTimeStep()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tDistance Resolution:        %f\n", GetDistRes()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tPerimeter Resolution:       %f\n", this->GetPerimRes()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tEnable Spotting:            %s\n", (this->EnableSpotting(GETVAL) == true) ? "Yes" : "No") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSpot Probability:           %f\n", (this->EnableSpotting(GETVAL) == true) ? this->PercentIgnition(GETVAL)/100.0 : 0.0) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSpot Ignition Delay:        %f\n", this->IgnitionDelay(GETVAL)) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSpot Grid Resolution:        %ld\n", (this->m_BSG.Grid == NULL) ? 0 :m_BSG.l_res) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSpotting Seed:               %d\n", this->icf.i_SpottingSeed) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tMinimum Spotting Distance:  %f\n", this->icf.f_FarsiteMinSpotDistance) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tFoliar Moisture:            %.4f\n", this->GetFoliarMC()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tEnable Crowning:            %s\n", (this->EnableCrowning(GETVAL) == 1) ? "Yes" : "No") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tCrown Fire Method:          %s\n", (this->GetCrownFireCalculation() == 0) ? "Finney" : "ScottRheinhart") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\t                            %s\n", (LinkDensityWithCrownCover(GETVAL) == 0) ? "Crown Density NOT LINKED to Crown Cover" : "Crown Density LINKED to Crown Cover") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tDist Check Method:          %s\n", (this->DistanceCheckMethod(GETVAL) == 1) ? "Fire Level" : "SimLevel") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tAcceleration:               %s\n", (this->AccelerationON() == true) ? "On" : "Off") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tBacking Spread:             %s\n", (this->ConstantBackingSpreadRate(GETVAL) == 0) ? "Calculated from Elliptical Dimensions" : "Calculated from No Wind/No Slope") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSimulation Started:         %ld%s%ld %ld:00\n", GetStartMonth(), "/",
			GetStartDay(), GetStartHour() / 100) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tSimulation Ended:           %s\n", CurTime) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tElapsed Time:               %s\n", ElTime) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tRestore Ignitions:          %s\n", (IgnitionResetAtRestart(GETVAL) == 0) ? "No" : "Yes") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tPreserve Inactive Enclaves: %s\n", (PreserveInactiveEnclaves(GETVAL) == true) ? "Yes" : "No") < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tTree Species Number:        %ld\n", this->GetCanopySpecies()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	if(fprintf(bmFile, "\tConditoning Start:          %ld/%ld\n", this->conditmonth, this->conditday) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);

	if(fprintf(bmFile, "\tBurn Periods:           %ld\n", this->GetNumBurnPeriods()) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	for(int i = 0; i < GetNumBurnPeriods(); i++)
	{
		long bpMo, bpDy, bpStart, bpEnd;
		this->GetBurnPeriod(i, &bpMo, &bpDy, &bpStart, &bpEnd);
		if(fprintf(bmFile, "          %02ld %02ld %04ld %04ld\n", bpMo, bpDy, bpStart, bpEnd) < 0)
			return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	}
	//some internals
	if(fprintf(bmFile, "\tRelative Burn Periods:      %ld\n", this->NumRelativeData) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	//double bpStart, bpEnd;
	for(int i = 0; i < this->NumRelativeData; i++)
	{
		double bpStart, bpEnd;
		bpStart = this->rbp[i].Start;
		bpEnd = this->rbp[i].End;
		if(fprintf(bmFile, "          %f - %f\n", bpStart, bpEnd) < 0)
			return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	}
	//output weather internals
	if(fprintf(bmFile, "\tInternal Weather:            %ld\n", MaxWeatherObs[0] - 20) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	for(int i = 0; i < MaxWeatherObs[0] - 20; i++)
	{
		if(fprintf(bmFile, "%ld %ld %.0f %ld %ld %.0f %.0f %ld %ld %.0f %ld %ld\n",
			wtrdt[0][i].mo,
			wtrdt[0][i].dy,
			wtrdt[0][i].rn,
			wtrdt[0][i].t1,
			wtrdt[0][i].t2,
			wtrdt[0][i].T1,
			wtrdt[0][i].T2,
			wtrdt[0][i].H1,
			wtrdt[0][i].H2,
			wtrdt[0][i].el,
			wtrdt[0][i].tr1,
			wtrdt[0][i].tr2
			) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	}
	//output wind internals
	//if(fprintf(bmFile, "\tInternal Winds:              %ld\n", (MaxWindObs[0]) / 2) < 0)
	if(fprintf(bmFile, "\tInternal Winds:              %ld\n", MaxWindObs[0] - 1) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	//for(int i = 0; i < (MaxWindObs[0]) / 2; i++)
	for(int i = 0; i < MaxWindObs[0] - 1; i++)
	{
		if(fprintf(bmFile, "%ld %ld %ld %.0f %ld %ld\n",
			wddt[0][i].mo,
			wddt[0][i].dy,
			wddt[0][i].hr,
			wddt[0][i].ws,
			wddt[0][i].wd,
			wddt[0][i].cl
			) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);
	}

	//output run time
	if(fprintf(bmFile, "\n\tTotal Farsite Run Time: %.3lf Seconds\n", duration) < 0)
		return CloseAndReturn(bmFile, e_EMS_FILE_WRITE_ERROR);


	return CloseAndReturn(bmFile, 1);
}

long Farsite5::GetCrownFireCalculation()
{
	return CrownFireCalculation;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: LoadCrownFireMethod - Load Crown Fire Method
* Desc: This can be Finney or Reinhardt,
*       Finney is used as default.
*   In: cr_CroFirMet....text argument as set in the command file
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void Farsite5::LoadCrownFireMethod (char cr_CroFirMet[])
{
/* check if Reinhardt was set in cmd file, else we'll assume Finney was set or  */
/* the no switch was found and so Finney is used as default                     */
   if ( !strcmpi (cr_CroFirMet, e_CFM_ScotRein) )
	  SetCrownFireCalculation ( 1 );   /* if scott reinhardt           */
   else
      SetCrownFireCalculation ( 0 );   /* else assume finney           */
}

long Farsite5::SetCrownFireCalculation(long Type)
{
	CrownFireCalculation=Type;

	return CrownFireCalculation;
}

long Farsite5::GetNumBurnPeriods()
{
	return NumAbsoluteData;
}

bool Farsite5::GetBurnPeriod(long num, long* mo, long* dy, long* start, long* end)
{
	if (abp)
	{
		*mo = abp[num].Month;
		*dy = abp[num].Day;
		*start = abp[num].Start;
		*end = abp[num].End;
	}
	else
	{
		*mo = *dy = *start = *end = 0;

		return false;
	}

	return true;
}



int Farsite5::CmMapEnvironment(int themeno, double mapTime, char *outName)
{
	double when=mapTime;//burn.SIMTIME;
	double DuffMx;

	long i, j, numy, numx;//, themeno;
	char name[256]="";
	float *map;
	double res, xll, yll, xpt, ypt, north, west, mult=1.0;
	double tmult=1.0, tadd=0.0, wmult=1.0;
	EnvironmentData lmw;
	FILE *outfile;

	mult=1.0;

	strcpy(name, outName);//TEnvMap.FileName);

	res=mult*GetCellResolutionX();

	numy=(GetHiNorth()-GetLoNorth())/res;
	numx=(GetHiEast()-GetLoEast())/res;
	north=GetHiNorth();
	west=GetLoEast();
	xll=GetWestUtm();
	yll=GetNorthUtm()-res*numy;
	map=new float[numx*numy];
	for(i=0; i<numy; i++)
	{
		ypt=north-i*res-res/2.0;
		xpt=west+res/2.0;
		for(j=0; j<numx; j++)
		{
			burn.fe->GetLandscapeData(xpt, ypt);

// Larry
//			burn.fe->GetFireEnvironment(burn.env, when, true);
			burn.fe->GetFireEnvironment( when, true);


			if(burn.fe->ld.elev==-9999 || burn.fe->ld.fuel<=0
				|| (burn.fe->ld.fuel >= 90 && burn.fe->ld.fuel <= 99))
			{	map[i*numx+j]=-9999;
			xpt+=res;

			continue;
			}
			burn.fe->GetEnvironmentData(&lmw);

			switch(themeno)
			{	case 0:
			map[i*numx+j]=lmw.ones;
			break;
			case 1:
				map[i*numx+j]=lmw.tens;
				break;
			case 2:
				map[i*numx+j]=lmw.hundreds;
				break;
			case 3:
				map[i*numx+j]=lmw.thousands;
				break;
			case 4:
				DuffMx=-0.347+6.42*lmw.hundreds;
				if(DuffMx<0.1)
					DuffMx=0.1;
				map[i*numx+j]=DuffMx;
				break;
			case 5:
				map[i*numx+j]=lmw.windspd*wmult;
				break;
			case 6:
				map[i*numx+j]=(lmw.temp-tadd)/tmult;
				break;
			case 7:
				map[i*numx+j]=lmw.humid;
				break;
			case 8:
				map[i*numx+j]=lmw.solrad;
				break;
			};
			xpt+=res;
		}
	}

	outfile=fopen(name, "w");
	fprintf(outfile, "ncols %ld\n", numx);
	fprintf(outfile, "nrows %ld\n", numy);
	fprintf(outfile, "xllcorner %lf\n", xll);
	fprintf(outfile, "yllcorner %lf\n", yll);
	fprintf(outfile, "cellsize %lf\n", res);
	fprintf(outfile, "NODATA_VALUE -9999\n");

	for(i=0; i<numy; i++)
	{
		for(j=0; j<numx; j++)
		{
			fprintf(outfile, "%lf ", map[i*numx+j]);
		}
			fprintf(outfile, "\n");

}


	fclose(outfile);
	delete[] map;


	return 1;
}

int Farsite5::ExportMoistureDataText(char *trgName)
{
bool b;

  b = this->cfmc.ExportMoistureDataText(trgName, this->LandFName);
  if ( b == true )
  		return 1;
	 return 0;
}

int Farsite5::WriteSpotDataFile(char *trgName)
{
	FILE *fout = fopen(trgName, "wt");
	if(!fout)
		return e_EMS_FILE_OPEN_ERROR;
	if(fprintf(fout, "launchTime, launchX, launchY, landTime, landX, landY, FlightTime, Distance\n") < 0)
		return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
	CSpotData spotData;
	CSpotList::iterator i;
	double dx;
	for(i = spotList.begin(); i != spotList.end(); ++i)
	{
		spotData = *i;
		dx = (spotData.launchX - spotData.landX) * (spotData.launchX - spotData.landX) + (spotData.launchY - spotData.landY) * (spotData.launchY - spotData.landY);
		if(dx > 0.0)
			dx = sqrt(dx);
		else
			dx = 0.0;
		if(fprintf(fout, "%f, %f, %f, %f, %f, %f, %f, %f\n", spotData.launchTime, spotData.launchX, spotData.launchY,
			spotData.landTime, spotData.landX, spotData.landY,
			spotData.landTime - spotData.launchTime,
			dx) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
	}
	return CloseAndReturn(fout, 1);
}

int Farsite5::WriteSpotShapeFile(char *trgName)
{
	long NumRecord;
	SHPHandle hSHP;
	SHPObject* pSHP;
	DBFHandle hDBF;
	double VertexX[1], VertexY[1], VertexZ[1];
	char DataBaseID[64] = "";

	int nShapeType = SHPT_POINT;
	hSHP = SHPCreate(trgName, nShapeType);
	if (hSHP == NULL)
		return 0;
	// Create the database.
	char dbfName[256];
	strcpy(dbfName, trgName);
	//strlwr(dbfName);
	char *p = strrchr(dbfName,'.');
	strcpy(p, ".dbf");
	hDBF = DBFCreate(dbfName);
	if (hDBF == NULL)
	{
		SHPClose(hSHP);
		return 0;
	}
	sprintf(DataBaseID, "%s", "Land_Time");
	DBFAddField(hDBF, DataBaseID, FTDouble, 32, 4);
	CSpotData spotData;
	CSpotList::iterator i;
	for(i = spotList.begin(); i != spotList.end(); ++i)
	{
		spotData = *i;
		VertexX[0] = spotData.landX;
		VertexY[0] = spotData.landY;
		VertexZ[0] = 0;
		pSHP = SHPCreateObject(nShapeType, -1, 1, NULL, NULL, 1,
			VertexX, VertexY, VertexZ, NULL);
		SHPWriteObject(hSHP, -1, pSHP);
		SHPDestroyObject(pSHP);
		NumRecord = DBFGetRecordCount(hDBF);
		DBFWriteDoubleAttribute(hDBF, NumRecord, 0, spotData.landTime);
	}
	SHPClose(hSHP);
	DBFClose(hDBF);
	return 1;
}

/*************************************************************************************/
double Farsite5::SetFarsiteProgress(double newProgress)
{
#ifdef WIN32
	EnterCriticalSection(&progressCS);
#endif
	progress = newProgress;
	double tmp = progress;
#ifdef WIN32
	LeaveCriticalSection(&progressCS);
#endif
	return tmp;
}

/*************************************************************
* This function will get called from within a critical section
**************************************************************/
float Farsite5::GetFarsiteProgress()
{
	return (float) this->progress;
}


/*****************************************************************/
int Farsite5::CancelFarsite(void)
{
	LEAVEPROCESS = true;       /* Tell farsite it's lights out */
 this->cfmc.Terminate();    /* Cond DLL - cancel if running */

#ifdef WIN32

 this->WN2.Cancel();        /* WindNinja - cancel if running */
#endif
	return 1;
}





int Farsite5::AddIgnitionToSpotGrid()
{
	if(!ignitionGrid)
		return 0;
	if(!m_BSG.Grid)
		return 0;
	double igEast, igWest, igNorth, igSouth;
	double west = GetLoEast();
	double north = GetHiNorth();
	//double south = GetLoNorth();
	double res = GetCellResolutionX();
	//long row, col;
	for(int r = 0; r < this->ignitionRows; r++)
	{
		for(int c = 0; c < this->ignitionCols; c++)
		{
			if(ignitionGrid[r][c] > 0.0)
			{
				igWest = west + c * res;
				igEast = igWest + res;
				igNorth = north - r * res;
				//igSouth = south + r * res;
				//igNorth = igSouth + res;
				igSouth = igNorth - res;
				while(igWest <= igEast)
				{
					igNorth = igSouth + res;
					while(igNorth >= igSouth)
					{
						m_BSG.Set(igWest, igNorth, (char *)"I");
						igNorth -= m_BSG.l_res;
					}
					igWest += m_BSG.l_res;
				}
			}
		}
	}
	return 1;
}

int Farsite5::WriteSpotGrid(char *trgName)
{
	if(m_BSG.Grid)
	{
		FILE *fout = fopen(trgName, "wt");
		if(!fout)
		{
			return e_EMS_FILE_OPEN_ERROR;
		}
		//double west = Header.WestUtm, north = Header.NorthUtm, *pts = NULL;
	//long outCols = nodeSpread->GetNumCols(), outRows = nodeSpread->GetNumRows();
		double outRes = this->m_BSG.l_res, //Header.XResol,
			outWest = Header.WestUtm,//nodeSpread->West,
			outSouth =  Header.SouthUtm;//nodeSpread->South;
		if(fprintf(fout, "ncols %ld\n", this->m_BSG.lN_East) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "nrows %ld\n", this->m_BSG.lN_North) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "xllcorner %lf\n", outWest) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "yllcorner %lf\n", outSouth) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "cellsize %lf\n", outRes) < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(fout, "NODATA_VALUE -9999\n") < 0)
			return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		for(long j=m_BSG.lN_North - 1; j >= 0; j--)
		{
			for(long k=0; k<m_BSG.lN_East; k++)
			{     //burnfreq[0][j*ncols+k]/=(double) NumRiskThreads;// already handled
				//if(pFireScenario->
				if(fprintf(fout, "%d ", m_BSG.Grid[j][k]) < 0)
					return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
			}
			if(fprintf(fout, "\n") < 0)
				return CloseAndReturn(fout, e_EMS_FILE_WRITE_ERROR);
		}
		//fclose(fout);

		return CloseAndReturn(fout, 1);
	}
	return e_EMS_OUTPUT_DOES_NOT_EXIST;
}
void Farsite5::WritePerimeter1Shapefile(int num, long curFire)
{
	char fName1[MAX_PATH];//, fName2[MAX_PATH];
	sprintf(fName1, "%.0f_%ld_%d_Perim1.shp", burn.SIMTIME, curFire, num);
	//sprintf(fName2, "%.0f_%ld_Perim2.shp", burn.SIMTIME, curFire);
	SHPHandle hSHP;
	hSHP = SHPCreate(fName1, SHPT_ARC);
	double *VertexX, *VertexY, *VertexZ;
	int n = this->GetNumPoints(curFire);
		VertexX = new double[n];
		VertexY = new double[n];
		VertexZ=new double[n];
		for(int i = 0; i < n; i++)
		{
			VertexX[i] = GetPerimeter1Value(curFire, i, XCOORD);
			VertexY[i] = GetPerimeter1Value(curFire, i, YCOORD);
			VertexZ[i] = 0.0;
		}
		SHPObject *pSHP = SHPCreateObject(SHPT_ARC, -1, 0, NULL, NULL, n,
			VertexX, VertexY, VertexZ, NULL);
		SHPWriteObject(hSHP, -1, pSHP);
		SHPDestroyObject(pSHP);
		delete[] VertexX;
		delete[] VertexY;
		delete[] VertexZ;
	SHPClose(hSHP);
}

void Farsite5::WritePerimeter2Shapefile(int num, long curFire)
{
	char fName2[MAX_PATH];
	//sprintf(fName1, "%.0f_%ld_Perim1.shp", burn.SIMTIME, curFire);
	sprintf(fName2, "%.0f_%ld_%d_Perim2.shp", burn.SIMTIME, curFire, num);
	SHPHandle hSHP;
	hSHP = SHPCreate(fName2, SHPT_ARC);
	double *VertexX, *VertexY, *VertexZ;
	int n = this->GetNumPoints(curFire);//(int)GetPerimeter2Value(-1, -1);//curFire;//
		VertexX = new double[n];
		VertexY = new double[n];
		VertexZ=new double[n];
		double startX, startY;
		startX = GetPerimeter2Value(0, XCOORD);
		startY = GetPerimeter2Value(0, YCOORD);
		for(int i = 0; i < n; i++)
		{
			VertexX[i] = GetPerimeter2Value(i, XCOORD);
			VertexY[i] = GetPerimeter2Value(i, YCOORD);
			VertexZ[i] = 0.0;
			/*if( i > 0 && VertexX[i] == startX && VertexY[i] == startY)
			{
				n = i;
				break;
			}*/
		}
		SHPObject *pSHP = SHPCreateObject(SHPT_ARC, -1, 0, NULL, NULL, n,
			VertexX, VertexY, VertexZ, NULL);
		SHPWriteObject(hSHP, -1, pSHP);
		SHPDestroyObject(pSHP);
		delete[] VertexX;
		delete[] VertexY;
		delete[] VertexZ;
	SHPClose(hSHP);
}

void Farsite5::WritePerimeter1CSV(int num, long CurFire)
{
	char fName1[MAX_PATH];//, fName2[MAX_PATH];
	sprintf(fName1, "%.0f_%ld_%d_Perim1.csv", burn.SIMTIME, CurFire, num);
	FILE *out = fopen(fName1, "wt");
	if(!out)
	{
		printf("Error opening Perimeter1 CSV file for writing.\nFile: %s", fName1);
		return;
	}
	fprintf(out, "PointNum, X, Y, ros, fli\n");
	double x, y, r, f;
	int n = this->GetNumPoints(CurFire);
		for(int i = 0; i < n; i++)
		{
			x = GetPerimeter1Value(CurFire, i, XCOORD);
			y = GetPerimeter1Value(CurFire, i, YCOORD);
			r = GetPerimeter1Value(CurFire, i, ROSVAL);
			f = GetPerimeter1Value(CurFire, i, FLIVAL);
			fprintf(out, "%d, %f, %f, %f, %f\n", i, x, y, r, f);
		}
		fclose(out);
}
void Farsite5::WritePerimeter2CSV(int num, long curFire)
{
	char fName1[MAX_PATH];//, fName2[MAX_PATH];
	sprintf(fName1, "%.0f_%ld_%d_Perim2.csv", burn.SIMTIME, curFire, num);
	FILE *out = fopen(fName1, "wt");
	if(!out)
	{
		printf("Error opening Perimeter2 CSV file for writing.\nFile: %s", fName1);
		return;
	}
	fprintf(out, "PointNum, X, Y, ros, fli\n");
	double x, y, r, f;
	int n = this->GetNumPoints(curFire);
		for(int i = 0; i < n; i++)
		{
			x = GetPerimeter2Value(i, XCOORD);
			y = GetPerimeter2Value(i, YCOORD);
			r = GetPerimeter2Value(i, ROSVAL);
			f = GetPerimeter2Value(i, FLIVAL);
			fprintf(out, "%d, %f, %f, %f, %f\n", i, x, y, r, f);
		}
		fclose(out);
}

int Farsite5::GetNumIgnitionCells()
{
	return m_nCellsLit;
}
