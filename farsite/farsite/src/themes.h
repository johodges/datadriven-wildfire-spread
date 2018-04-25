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
//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#ifndef THEMES
#define THEMES

#include <stdlib.h>
//#include <windows.h>
#include <time.h>
#include <sys/timeb.h>

/*
class ColorRamp
{
	long i, NumColors;
	double ColorIncR, ColorIncG, ColorIncB;
	COLORREF* Colors;

	void GetColorChanges(double NumColors, long R, long G, long B, long Var);

public:
	bool Reverse;
	long ColorType;
	double Min, Max;
	long Rval, Gval, Bval;
	char ColorFile[256];

	ColorRamp(long NumColors, long R, long G, long B, long Var, double Min,
		double Max, bool Reverse);
	~ColorRamp();

	COLORREF GetColor(long order);
	void SetRamp(long NumColors, long R, long G, long B, long Var, double Max);
	long GetNumColors();
	bool SetSpecificColor(long numcolor, long R, long G, long B);
	//void SetNumCats(long numcats);
	//long GetCat(long Num);
	//void CopyCats(long *cats);
	//long GetNumCats();
	//long CategoriesOK(long OK);
};
*/

class GridTheme
{
public:
	char Name[256];
	long Continuous;
	long RedVal, GreenVal, BlueVal, VarVal, NumColors, MaxBrite, ColorChange;
	bool WantNewRamp, WantNewColor, LcpAscii, OnOff, OnOff3d, Changed3d,
		ConvertFuelColors;
	long Cats[100], NumCats, CatsOK, Priority;
	long LegendNum;
	double MaxVal, MinVal;
//	ColorRamp* ramp;

	GridTheme();
	~GridTheme();
	void CreateRamp();
	void DeleteRamp();
	//bool GetColor(double value, COLORREF* colr);
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
class Farsite5;

class LandscapeTheme : public GridTheme
{
	void FillCats();
	void SortCats();
	Farsite5 *pFarsite;
public:
	long NumAllCats[10];
	long AllCats[10][100];
	double maxval[10], minval[10];

	LandscapeTheme(bool Analyze, Farsite5 *_pFarsite);
	void CopyStats(long layer);
	void ReadStats();
	void AnalyzeStats();
};

//LandscapeTheme *GetLandscapeTheme()

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/*
class RasterTheme : public GridTheme
{
	void FillCats();
	void SortCats();

public:
	double* map;
	double rW, rE, rN, rS, rCellSizeX, rCellSizeY, rMaxVal, rMinVal;
	long rRows, rCols;

	RasterTheme();
	~RasterTheme();
	bool SetTheme(const char *Name);
};
*/

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

struct VectorTheme
{
	long VectorNum;
	char FileName[256];
	long FileType;
	long Permanent;
	//COLORREF Color;
	int PenStyle;
	int PenWidth;
	bool OnOff;
	bool OnOff3d;
	bool Changed;
	long Priority;
};


#endif // THEMES
