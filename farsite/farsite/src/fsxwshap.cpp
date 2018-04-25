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
//	FSXWSHAP.CPP   Create ARCVIEW Shape files
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environemntal Management
//******************************************************************************

//#include "fsxwshap.h"
#include "Farsite5.h"


ShapeFileUtils::ShapeFileUtils(Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
	ShapeFileName = 0;
	DataBaseName = 0;
	Polygons = 1;
	BarSeparate = 0;
	VisOnly = 1;
	BarExport = false;
	NewFile = true;
}


ShapeFileUtils::~ShapeFileUtils()
{
	if (ShapeFileName)
		free(ShapeFileName);
	if (DataBaseName)
		free(DataBaseName);
	//FreeSHPArrays();
}


void ShapeFileUtils::ResetShape()
{
	if (ShapeFileName)
		free(ShapeFileName);
	if (DataBaseName)
		free(DataBaseName);
	ShapeFileName = 0;
	DataBaseName = 0;
	Polygons = 1;
	BarSeparate = 0;
	VisOnly = 1;
	BarExport = false;
	NewFile = true;
}


void ShapeFileUtils::InitializeShapeExport(char* FileName, long poly)
{
	// for one-time export of file types
	long len;

	ShapeFileName = strdup(FileName);
	len = strlen(ShapeFileName);
	if (ShapeFileName[len - 4] == '.')
	{
		//DataBaseName = (char *) GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT,
		//							(len + 1) * sizeof(char));
		DataBaseName = (char *)malloc((len + 1) * sizeof(char));
		memset(DataBaseName,0X0,(len + 1) * sizeof(char));
		//memset(DataBaseName, 0x0, sizeof(DataBaseName));
		strcpy(DataBaseName, ShapeFileName);
		DataBaseName[len - 3] = 'd';
		DataBaseName[len - 2] = 'b';
		DataBaseName[len - 1] = 'f';
		ShapeFileName[len - 3] = 's';     // make sure .SHP is appended for DDE
		ShapeFileName[len - 2] = 'h';
		ShapeFileName[len - 1] = 'p';
	}
	else
	{
		//DataBaseName = (char *) GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT,
		//							(len + 5) * sizeof(char));
		DataBaseName = (char *) malloc((len + 5) * sizeof(char));
		memset(DataBaseName ,0X0,((len + 5) * sizeof(char)));
		//memset(DataBaseName, 0x0, sizeof(DataBaseName));
		strcpy(DataBaseName, ShapeFileName);
		strcat(DataBaseName, ".dbf");
		free(ShapeFileName);
		ShapeFileName = strdup(DataBaseName);
		ShapeFileName[len - 3] = 's';
		ShapeFileName[len - 2] = 'h';	// make sure .SHP is appended for DDE
		ShapeFileName[len - 1] = 'p';
	}
	Polygons = poly;
	//if(poly==0)
	BarExport = true;	// only write lines for barrier output
	//else
	//	BarExport=false;
	NewFile = true;
}


long ShapeFileUtils::ImportShapeData(const char* FileName, long PolygonType)
{
	SHPHandle hSHP;
	SHPObject* pSHP;
	int nShapeType, nEntities;
	long i, j, k, start, end;
	double xpt, ypt, fli;
	double maxx, maxy, minx, miny;
	double MinBound[4], MaxBound[4];

	hSHP = SHPOpen(FileName, "rb");
	if (hSHP == NULL)
		return false;

	SHPGetInfo(hSHP, &nEntities, &nShapeType, MinBound, MaxBound);
	if (nShapeType == SHPT_POINT ||
		nShapeType == SHPT_POINTZ ||
		nShapeType == SHPT_POINTM ||
		nShapeType == SHPT_MULTIPOINT ||
		nShapeType == SHPT_MULTIPOINTZ ||
		nShapeType == SHPT_MULTIPOINTM)
	{
		SHPClose(hSHP);

		return -1;
	}
	if (PolygonType < 3)
		fli = 0.0;
	else
		fli = -1.0;
	for (i = 0; i < nEntities; i++)
	{
		pSHP = SHPReadObject(hSHP, i);
		if (pSHP->nVertices > 0)
		{
			for (k = 0; k < pSHP->nParts; k++)
			{
				start = pSHP->panPartStart[k];
				if (k < pSHP->nParts - 1)
					end = pSHP->panPartStart[k + 1];
				else
					end = pSHP->nVertices;
				if (end - start <= 0)
					continue;
				pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), (end - start) + 1);
				for (j = start; j < end; j++)
				{
					xpt = pFarsite->ConvertUtmToEastingOffset(pSHP->padfX[j]);
					ypt = pFarsite->ConvertUtmToNorthingOffset(pSHP->padfY[j]);
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), j - start, xpt, ypt);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), j - start, -1.0, fli);
					if (j == start)
					{
						minx = maxx = xpt;
						miny = maxy = ypt;
					}
					else
					{
						if (xpt < minx)
							minx = xpt;
						else if (xpt > maxx)
							maxx = xpt;
						if (ypt < miny)
							miny = ypt;
						else if (ypt > maxy)
							maxy = ypt;
					}
				}
				pFarsite->SetPerimeter1(pFarsite->GetNewFires(), j - start, minx, maxx);
				pFarsite->SetFireChx(pFarsite->GetNewFires(), j - start, miny, maxy);
				pFarsite->SetInout(pFarsite->GetNewFires(), PolygonType);
				pFarsite->SetNumPoints(pFarsite->GetNewFires(), end - start);//pSHP->nVertices);
				pFarsite->IncNewFires(1);
			}
		}
		SHPDestroyObject(pSHP);
	}
	SHPClose(hSHP);

	if (nShapeType == SHPT_POLYGON ||
		nShapeType == SHPT_POLYGONZ ||
		nShapeType == SHPT_POLYGONM ||
		nShapeType == SHPT_MULTIPATCH)
		i = 0;
	else
		i = 1;

	return i;
}


char* ShapeFileUtils::GetShapeFileName()
{
	// for DDE data handle update
	return ShapeFileName;
}


void ShapeFileUtils::RetrieveExportFileInfo()
{
	// retrieve filename from simulation output information
	long len;

	if (ShapeFileName)
		free(ShapeFileName);
	if (DataBaseName)
		free(DataBaseName);
	ShapeFileName = strdup(pFarsite->GetShapeFileChx(&VisOnly, &Polygons, &BarSeparate));
	len = strlen(ShapeFileName);
	if (ShapeFileName[len - 4] == '.')
	{
		//DataBaseName = (char *) GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT,
		//							(len + 1) * sizeof(char));
		DataBaseName = (char *) malloc((len + 1) * sizeof(char));
		memset(DataBaseName,0X0,((len + 1) * sizeof(char)));
		//memset(DataBaseName, 0x0, sizeof(DataBaseName));
		strcpy(DataBaseName, ShapeFileName);
		DataBaseName[len - 3] = 'd';
		DataBaseName[len - 2] = 'b';
		DataBaseName[len - 1] = 'f';
		ShapeFileName[len - 3] = 's';     // make sure .SHP is appended for DDE
		ShapeFileName[len - 2] = 'h';
		ShapeFileName[len - 1] = 'p';
	}
	else
	{
		//DataBaseName = (char *) GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT,
		//							(len + 5) * sizeof(char));
		DataBaseName = (char *) malloc((len + 5) * sizeof(char));
		memset(DataBaseName,0X0,((len + 5) * sizeof(char)));
		//memset(DataBaseName, 0x0, sizeof(DataBaseName));
		strcpy(DataBaseName, ShapeFileName);
		strcat(DataBaseName, ".dbf");
		free(ShapeFileName);
		ShapeFileName = strdup(DataBaseName);
		ShapeFileName[len - 3] = 's';
		ShapeFileName[len - 2] = 'h';	// make sure .SHP is appended for DDE
		ShapeFileName[len - 1] = 'p';
	}

	NewFile = true;
}


bool ShapeFileUtils::ShapeData(long CurrentFire, double SimTime)
{
	int NumVertices, NumParts;
	//int		*PartsArray;
	long InOut, count, NumRecord;
	long NumVertex;
	char DataBaseID[64] = "";
	char Description[256];
	double xpt, ypt;//, ros;
	double* VertexX = 0, * VertexY = 0, * VertexZ = 0;
	SHPHandle hSHP;
	SHPObject* pSHP;
	DBFHandle hDBF;

	InOut = pFarsite->GetInout(CurrentFire);
	if (InOut == 0)
		return true;

	if (NewFile)
	{
		// create new shapefile
		if (BarExport)
		{
			if (Polygons == 1)
				BarExport = false;
		}
		else
			RetrieveExportFileInfo();
		if (Polygons)
			nShapeType = SHPT_POLYGON;
		else
			nShapeType = SHPT_ARC;
		hSHP = SHPCreate(ShapeFileName, nShapeType);
		if (hSHP == NULL)
			return false;
		SHPClose(hSHP);
		// Create the database.
		hDBF = DBFCreate(DataBaseName);
		if (hDBF == NULL)
			return false;
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
		DBFClose(hDBF);
		NewFile = false;
	}
	hSHP = SHPOpen(ShapeFileName, "rb+");
	hDBF = DBFOpen(DataBaseName, "rb+");
	NumVertex = pFarsite->GetNumPoints(CurrentFire);
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
	//if(PartsArray)
	//	delete[] PartsArray;
	switch (InOut)
	{
	case 1:
		sprintf(Description, "%s", "Expanding Fire");
		break;
	case 2:
		sprintf(Description, "%s", "Enclave Fire");
		break;
	case 3:
		sprintf(Description, "%s", "Barrier");
		break;
	default:
		memset(Description, 0x0, sizeof(Description));
		break;
	}
	ConvertSimTime(SimTime);
	DBFWriteStringAttribute(hDBF, NumRecord, 0, Description);
	DBFWriteIntegerAttribute(hDBF, NumRecord, 1, Month);
	DBFWriteIntegerAttribute(hDBF, NumRecord, 2, Day);
	DBFWriteIntegerAttribute(hDBF, NumRecord, 3, Hour);
	DBFWriteDoubleAttribute(hDBF, NumRecord, 4, SimTime);
	SHPClose(hSHP);
	DBFClose(hDBF);

	return true;
}


void ShapeFileUtils::ConvertSimTime(double SimTime)
{
	double Hr, Dy;
	long min;

	Month = pFarsite->GetStartMonth();
	Day = 0;
	Hour = 0;

	if (SimTime >= 1440.0)
	{
		Dy = SimTime / 1440.0;
		Day = (long) Dy;
		SimTime -= (1440 * Day);					 // truncate to nearest day
	}
	if (SimTime >= 60.0)
	{
		Hr = SimTime / 60.0;
		Hour = (long) Hr;
		SimTime -= (60 * Hour);					 // truncate to nearesat hour
	}
	min = (long) SimTime;//*60.0);  							   // minutes left over

	Hour += (pFarsite->GetStartHour() / 100);
	if (Hour >= 24)
	{
		Hour -= 24;
		Day++;
	}
	Day += pFarsite->GetStartDay();
	long days, oldMonth;
	do
	{
		oldMonth = Month;
		switch (Month)
		{
		case 1:
			days = 31; break;			 // days in each Monthnth, ignoring leap year
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
		if (Day > days)
		{
			Day -= days;
			Month++;
			if (Month > 12)
				Month -= 12;
		}
	}
	while (Month != oldMonth);						// allows startup of current clock at any time, will find cur month

	Hour *= 100;
	Hour += min;
}

