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
#include "fsxwatm.h"
#include "Farsite5.h"
#include <math.h>
#include <algorithm>
//#include <boost/filesystem/path.hpp>
//#include <boost/filesystem/operations.hpp>

//namespace bfs = boost::filesystem ; 


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	AtmosphereGrid Structure Functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


AtmosphereGrid::AtmosphereGrid(long numgrids, Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
	for(int i = 0; i < 6; i++)
		atmosphere[i].pFarsite = pFarsite;
	NumGrids = numgrids;
	switch (NumGrids)
	{
	case 0:
		StartGrid = 6;                  
		AtmGridWND = false;
		AtmGridWTR = false;
		break;
	case 3:
		StartGrid = 3;
		AtmGridWND = true;
		AtmGridWTR = false;
		break;
	case 6:
		StartGrid = 0;
		AtmGridWND = true;
		AtmGridWTR = true;
		break;
	default:
		StartGrid = 6;
		AtmGridWND = false;
		AtmGridWTR = false;
		break;
	}
	TimeIntervals = 0;
	Month = 0;
	Day = 0;
	Hour = 0;
	Metric = 0;
}


AtmosphereGrid::~AtmosphereGrid()
{
	long i;

	for (i = StartGrid; i < 6; i++)
		atmosphere[i].FreeAtmGrid();
	if (Month)
		delete[] Month;//free(Month);
	if (Day)
		delete[] Day;//free(Day);
	if (Hour)
		delete[] Hour;//free(Hour);
}


void AtmosphereGrid::FreeAtmData()
{
	long i;

	for (i = StartGrid; i < 6; i++)
		atmosphere[i].FreeAtmGrid();
	if (Month)
		delete[] Month;//free(Month);
	if (Day)
		delete[] Day;//free(Day);
	if (Hour)
		delete[] Hour;//free(Hour);
	Month = 0;
	Day = 0;
	Hour = 0;
	Metric = 0;
}



bool AtmosphereGrid::ReadInputTable(char *InputFileName)
{
	char InputTemp[256];
	char InputHumid[256];
	char InputRain[256];
	char InputCloud[256];
	char InputSpd[256];
	char InputDir[256];
	char FileName[256];
	char UnitsString[256] = "";
	long i, j, fpos;

	TimeIntervals = 0;
	memset(ErrMsg, 0x0, sizeof(ErrMsg));
	if ((InputTable = fopen(InputFileName, "r")) != NULL)
	{
		fscanf(InputTable, "%s", InputTemp);	// get by header
		fscanf(InputTable, "%s", UnitsString);
		
		// convert units to lower case
		std::transform(UnitsString, UnitsString+strlen(UnitsString), 
			UnitsString, tolower);
		fpos = ftell(InputTable);
		if (!strcmp(UnitsString, "metric"))
			Metric = 1;
		else if (!strcmp(UnitsString, "english"))   // english default
			Metric = 0;//fseek(InputTable, fpos, SEEK_SET);
		else
		{
			rewind(InputTable);
			fscanf(InputTable, "%s", InputTemp);	// get by header
			fpos = ftell(InputTable);
		}
		while (!feof(InputTable))		// count number of time intervals
		{
			fscanf(InputTable, "%ld", &month);
			if (feof(InputTable))
				break;
			if (NumGrids < 6)
			{
				fscanf(InputTable, "%ld %ld %s %s %s ", &day, &hour, InputSpd,
					InputDir, InputCloud);
				/*if (!bfs::exists(bfs::path(InputSpd)))
					sprintf(ErrMsg, "%s %s", InputSpd,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputDir)))
					sprintf(ErrMsg, "%s %s", InputDir,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputCloud)))
					sprintf(ErrMsg, "%s %s", InputCloud,
						"Can't Be Opened or Read");*/
			}
			else
			{
				fscanf(InputTable, "%ld %ld %s %s %s %s %s %s", &day, &hour,
					InputTemp, InputHumid, InputRain, InputSpd, InputDir,
					InputCloud);
				/*if (!bfs::exists(bfs::path(InputTemp)))
					sprintf(ErrMsg, "%s %s", InputTemp,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputHumid)))
					sprintf(ErrMsg, "%s %s", InputHumid,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputRain)))
					sprintf(ErrMsg, "%s %s", InputRain,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputSpd)))
					sprintf(ErrMsg, "%s %s", InputSpd,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputDir)))
					sprintf(ErrMsg, "%s %s", InputDir,
						"Can't Be Opened or Read");
				if (!bfs::exists(bfs::path(InputCloud)))
					sprintf(ErrMsg, "%s %s", InputCloud,
						"Can't Be Opened or Read");*/
			}
			if (strlen(ErrMsg) > 0)
			{
				fclose(InputTable);
				return false;
			}
			TimeIntervals++;
		};
		Month = new long[TimeIntervals];//(long *) calloc(TimeIntervals, sizeof(long));
		Day = new long[TimeIntervals];//(long *) calloc(TimeIntervals, sizeof(long));
		Hour = new long[TimeIntervals];//(long *) calloc(TimeIntervals, sizeof(long));
		//rewind(InputTable);
		fseek(InputTable, fpos, SEEK_SET);
		if (NumGrids < 6)
			fscanf(InputTable, "%ld %ld %ld %s %s %s ", &month, &day, &hour,
				InputSpd, InputDir, InputCloud);
		else
			fscanf(InputTable, "%ld %ld %ld %s %s %s %s %s %s", &month, &day,
				&hour, InputTemp, InputHumid, InputRain, InputSpd, InputDir,
				InputCloud);
		for (j = StartGrid;
			j < 6;
			j++)   // set header information in each file
		{
			memset(FileName, 0x0, sizeof(FileName));
			switch (j)
			{
			case 0:
				strcpy(FileName, InputTemp); break;
			case 1:
				strcpy(FileName, InputHumid); break;
			case 2:
				strcpy(FileName, InputRain); break;
			case 3:
				strcpy(FileName, InputSpd); break;
			case 4:
				strcpy(FileName, InputDir); break;
			case 5:
				strcpy(FileName, InputCloud); break;
			}
			if ((ThisFile = fopen(FileName, "r")) == NULL)
			{
				sprintf(ErrMsg, "%s %s", FileName, "Can't Be Opened or Read");
				fclose(InputTable);

				return false;
			}
			else if (!ReadHeaders(j))
			{
				sprintf(ErrMsg, "%s %s", FileName, "Header Not GRASS or GRID");
				fclose(InputTable);

				return false;
			}
			else if (!atmosphere[j].AllocAtmGrid(TimeIntervals))
			{
				sprintf(ErrMsg, "%s %s", FileName,
					"Memory Not Sufficient for File");
				for (i = StartGrid; i < j; i++)
					atmosphere[i].FreeAtmGrid();
				fclose(InputTable);

				return false;
			}
		}
		for (i = 0; i < TimeIntervals; i++)
		{
			Month[i] = month;
			Day[i] = day;
			Hour[i] = hour;
			for (j = StartGrid; j < 6; j++)
			{
				memset(FileName, 0x0, sizeof(FileName));
				switch (j)
				{
				case 0:
					strcpy(FileName, InputTemp); break;
				case 1:
					strcpy(FileName, InputHumid); break;
				case 2:
					strcpy(FileName, InputRain); break;
				case 3:
					strcpy(FileName, InputSpd); break;
				case 4:
					strcpy(FileName, InputDir); break;
				case 5:
					strcpy(FileName, InputCloud); break;
				}
				if ((ThisFile = fopen(FileName, "r")) == NULL)
				{
					sprintf(ErrMsg, "%s %s", FileName,
						"File Not Found or Cannot Be Read");
					fclose(InputTable);

					return false;
				}
				if (!CompareHeader(j))
				{
					sprintf(ErrMsg, "%s %s", FileName,
						"Header Not Same For File Type");
					fclose(InputTable);

					return false;
				}
				if (!SetAtmosphereValues(i, j))
				{
					sprintf(ErrMsg, "%s %s", FileName,
						"Error Reading File Values");
					fclose(InputTable);

					return false;
				}
			}
			if (NumGrids == 6)
				fscanf(InputTable, "%ld %ld %ld %s %s %s %s %s %s", &month,
					&day, &hour, InputTemp, InputHumid, InputRain, InputSpd,
					InputDir, InputCloud);
			else
				fscanf(InputTable, "%ld %ld %ld %s %s %s", &month, &day,
					&hour, InputSpd, InputDir, InputCloud);
		};// while(!feof(InputTable));

		fclose(InputTable);
	}
	else
		return false;

	return true;
}


bool AtmosphereGrid::ReadHeaders(long FileNumber)
{
	char TestString[256];
	char CompGrass[] = "north:";
	char CompGrid[] = "ncols";
	double north, south, east, west, xres, yres;
	long rows, cols;

	fscanf(ThisFile, "%s", TestString);
	std::transform(TestString, TestString+strlen(TestString), TestString, tolower) ; 
	if (!strcmp(TestString, CompGrass))	// grass file
	{
		fscanf(ThisFile, "%lf", &north);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &south);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &east);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &west);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%ld", &rows);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%ld", &cols);
		xres = (east - west) / (double) cols;
		yres = (north - south) / (double) rows;
		atmosphere[FileNumber].SetHeaderInfo(north, south, east, west, xres,
								yres);
	}
	else if (!(strcmp(TestString, CompGrid)))
	{
		fscanf(ThisFile, "%ld", &cols);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%ld", &rows);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &west);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &south);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &xres);
		yres = xres;
		east = west + (double) cols * xres;
		north = south + (double) rows * yres;
		atmosphere[FileNumber].SetHeaderInfo(north, south, east, west, xres,
								yres);
	}
	else
	{
		fclose(ThisFile);

		return false;
	}

	fclose(ThisFile);

	return true;
}


bool AtmosphereGrid::CompareHeader(long FileNumber)
{
	char TestString[256];
	char CompGrass[] = "north:";
	char CompGrid[] = "ncols";
	double north, south, east, west, xres, yres;
	long rows, cols;

	fscanf(ThisFile, "%s", TestString);
	std::transform(TestString, TestString+strlen(TestString), TestString, tolower) ;
	if (!strcmp(TestString, CompGrass))	// grass file
	{
		fscanf(ThisFile, "%lf", &north);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &south);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &east);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &west);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%ld", &rows);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%ld", &cols);
		xres = (east - west) / (double) cols;
		yres = (north - south) / (double) rows;
		if (!atmosphere[FileNumber].CompareHeader(north, south, east, west,
										xres, yres))
		{
			fclose(ThisFile);

			return false;
		}
	}
	else if (!(strcmp(TestString, CompGrid)))
	{
		fscanf(ThisFile, "%ld", &cols);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%ld", &rows);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &west);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &south);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%lf", &xres);
		fscanf(ThisFile, "%s", TestString);
		fscanf(ThisFile, "%s", TestString);
		yres = xres;
		east = west + (double) cols * xres;
		north = south + (double) rows * yres;
		if (!atmosphere[FileNumber].CompareHeader(north, south, east, west,
										xres, yres))
		{
			fclose(ThisFile);

			return false;
		}
	}
	else
	{
		fclose(ThisFile);

		return false;
	}
	if (FileNumber == 3)		// set gridded weather dimensions for wind display
	{
		double SouthDiff = fabs(south - pFarsite->GetSouthUtm());
		double WestDiff = fabs(west - pFarsite->GetWestUtm());
		double SouthOffset = SouthDiff - ((long) (SouthDiff / yres)) * yres;
		double WestOffset = WestDiff - ((long) (WestDiff / xres)) * xres;
		long EastNumber, NorthNumber;

		if (pFarsite->GetEastUtm() < east)
			EastNumber = (long)((pFarsite->GetEastUtm() - pFarsite->GetWestUtm() - WestOffset) / xres);
		else
			EastNumber = (long)((east - pFarsite->GetWestUtm() - WestOffset) / xres);
		if (pFarsite->GetNorthUtm() < north)
			NorthNumber = (long)((pFarsite->GetNorthUtm() - pFarsite->GetSouthUtm() - SouthOffset) /
				yres);
		else
			NorthNumber = (long)((north - pFarsite->GetSouthUtm() - SouthOffset) / yres);
		if (EastNumber < atmosphere[FileNumber].XNumber)
			EastNumber++;
		if (NorthNumber < atmosphere[FileNumber].YNumber)
			NorthNumber++;
		pFarsite->SetGridNorthOffset(SouthOffset);
		pFarsite->SetGridEastOffset(WestOffset);
		pFarsite->SetGridEastDimension(EastNumber);
		pFarsite->SetGridNorthDimension(NorthNumber);
		//SetGridEastDimension(atmosphere[FileNumber].XNumber);
		//SetGridNorthDimension(atmosphere[FileNumber].YNumber);
	}

	return true;
}


bool AtmosphereGrid::SetAtmosphereValues(long timeinterval, long filenumber)
{
	double value;

	for (long i = 0; i < atmosphere[filenumber].NumCellsPerGrid; i++)
	{
		fscanf(ThisFile, "%lf", &value);
		if (Metric)
		{
			switch (filenumber)
			{
			case 0:
				value *= 1.0;
				value += 32.0;
				break;
			case 2:
				value *= 3.93;
				break;
			case 3:
				value *= .5402;  // convert 10m to 20ft, and kmph to miph
				break;
			}
		}
		if (!atmosphere[filenumber].SetAtmValue(i, timeinterval, (short) value))
		{
			fclose(ThisFile);

			return false;
		}
	}
	fclose(ThisFile);

	return true;
}


bool AtmosphereGrid::GetAtmosphereValue(long FileNumber, double xpt,
	double ypt, long time, short* value)
{
	if (!atmosphere[FileNumber].GetAtmValue(xpt, ypt, time, value))
	{
		switch (FileNumber)
		{
		case 0:
			*value = 70; break;  // default values for ATM variables if no data
		case 1:
			*value = 40; break;
		default:
			*value = 0; break;
		}

		return false;
	}

	return true;
}


void AtmosphereGrid::GetResolution(long FileNumber, double* resx, double* resy)
{
	atmosphere[FileNumber].GetResolution(resx, resy);
}


long AtmosphereGrid::GetTimeIntervals()
{
	return TimeIntervals;
}

long AtmosphereGrid::GetAtmMonth(long count)
{
	if (count < TimeIntervals)
		return Month[count];

	return -1;
}

long AtmosphereGrid::GetAtmDay(long count)
{
	if (count < TimeIntervals)
		return Day[count];

	return -1;
}

long AtmosphereGrid::GetAtmHour(long count)
{
	if (count < TimeIntervals)
		return Hour[count];

	return -1;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
//	Atmosphere Structure Functions
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


Atmosphere::Atmosphere()
{
	North = 0;
	South = 0;
	East = 0;
	West = 0;
	ResolutionX = 0;
	ResolutionY = 0;
	Value = 0;
}


Atmosphere::~Atmosphere()
{
}


void Atmosphere::SetHeaderInfo(double north, double south, double east,
	double west, double resolutionx, double resolutiony)
{
	North = pFarsite->ConvertUtmToNorthingOffset(north);
	South = pFarsite->ConvertUtmToNorthingOffset(south);
	East = pFarsite->ConvertUtmToEastingOffset(east);
	West = pFarsite->ConvertUtmToEastingOffset(west);
	ResolutionX = resolutionx;
	ResolutionY = resolutiony;
	XNumber = (long)((East - West) / ResolutionX);
	YNumber = (long)((North - South) / ResolutionY);
	NumCellsPerGrid = (long)(XNumber * YNumber);
}


bool Atmosphere::GetHeaderInfo(double* north, double* south, double* east,
	double* west, double* resolutionx, double* resolutiony)
{
	*north = North;
	*south = South;
	*east = East;
	*west = West;
	*resolutionx = ResolutionX;
	*resolutiony = ResolutionY;

	return true;
}


void Atmosphere::GetResolution(double* resolutionx, double* resolutiony)
{
	*resolutionx = ResolutionX;
	*resolutiony = ResolutionY;
}


bool Atmosphere::CompareHeader(double north, double south, double east,
	double west, double resolutionx, double resolutiony)
{
	if (North != pFarsite->ConvertUtmToNorthingOffset(north))
		return false;
	if (South != pFarsite->ConvertUtmToNorthingOffset(south))
		return false;
	if (East != pFarsite->ConvertUtmToEastingOffset(east))
		return false;
	if (West != pFarsite->ConvertUtmToEastingOffset(west))
		return false;
	if (ResolutionX != resolutionx)
		return false;
	if (ResolutionY != resolutiony)
		return false;

	return true;
}



bool Atmosphere::AllocAtmGrid(long timeintervals)
{
	long AllocNumber = (timeintervals + 1) * NumCellsPerGrid;

	if ((Value = new short[AllocNumber]) == NULL)//(short *) calloc(AllocNumber, sizeof(short)))==NULL)	// allocate 2D grid
	{
		Value = 0;
		return false;
	}

	return true;
}


bool Atmosphere::FreeAtmGrid()
{
	if (Value)
		delete[] Value;//free(Value);
	else
		return false;

	Value = 0;

	return true;
}


bool Atmosphere::GetAtmValue(double xpt, double ypt, long time, short* value)
{
	if (xpt<West || xpt>East)
		return false;
	if (ypt<South || ypt>North)
		return false;
	CellX = ((long) ((xpt - West) / ResolutionX));
	CellY = ((long) ((North - ypt) / ResolutionY));
	*value = Value[time * NumCellsPerGrid + (CellY * XNumber + CellX)];

	return true;
}


bool Atmosphere::SetAtmValue(long CellNumber, long time, short value)
{
	if (Value)
	{
		Value[time * NumCellsPerGrid + CellNumber] = (short) value;

		return true;
	}

	return false;
}


