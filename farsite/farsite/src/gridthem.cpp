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
//	Color Legend stuff for FARSITE Window
//
//------------------------------------------------------------------------------
#include "themes.h"
#include <math.h>
#include <string.h>
#include "Farsite5.h"

GridTheme::GridTheme()
{
	//ramp = 0;
	RedVal = 50;
	GreenVal = 0;
	BlueVal = 0;
	VarVal = 0;
	NumColors = 12;
	MaxBrite = 255;
	LegendNum = -1;
	OnOff = false;
	OnOff3d = false;
	WantNewRamp = true;                     
	WantNewColor = false;
	CatsOK = false;
	Changed3d = false;
	ConvertFuelColors = false;
	Priority = 0;
}

GridTheme::~GridTheme()
{
	//if (ramp)
	//	delete ramp;
	//ramp = 0;
}

void GridTheme::CreateRamp()
{
	//bool Reverse = false;
	//double MinBrite = 20;
     /*
	if (ramp)
	{
		RedVal = ramp->Rval;
		GreenVal = ramp->Gval;
		BlueVal = ramp->Bval;
		MinBrite = ramp->Min;
		Reverse = ramp->Reverse;
		MaxBrite = ramp->Max;
		delete ramp;
	}
	ramp = 0;
     */
	if (!Continuous)
	{
		NumColors = NumCats;
		VarVal = 18;
	}
	//ramp = new ColorRamp(NumColors, RedVal, GreenVal, BlueVal, VarVal,
	//			MinBrite, MaxBrite, Reverse);
	//WantNewRamp=true;
}


void GridTheme::DeleteRamp()
{
	//if (ramp)
	//	delete ramp;
	//ramp = 0;
}


/*
bool GridTheme::GetColor(double value, COLORREF* colr)
{
	bool yesno = true;
	long k, num;

	if (value < 0)//Maxval==Minval || Maxval<Minval)
	{
		*colr = ramp->GetColor(0);

		return true;
	}

	if (!Continuous)
	{
		for (k = 0; k < NumCats; k++)
		{
			if (ceil(value) == Cats[k])
				break;
		}
		if (WantNewColor)
		{
			if (k != ColorChange)
				yesno = false;
			else
				*colr = ramp->GetColor(k);
		}
		else
			//if(WantNewRamp)
			*colr = ramp->GetColor(k);
	}
	else
	{
		if (MaxVal == MinVal)
			num = 0;
		else
			num = (value - MinVal) / (MaxVal - MinVal) * NumColors;
		if (MaxVal != NumColors)
			num += 1;
		if (num > NumColors)
			num = NumColors;
		if (WantNewColor)
		{
			if (num != ColorChange)
				yesno = false;
			else
				*colr = ramp->GetColor(num);//Colors[num];
		}
		else
			//if(WantNewRamp)
			*colr = ramp->GetColor(num);
	}

	return yesno;
}
*/
//------------------------------------------------------------------------------
//
//   COLOR RAMP FUNCTIONS
//
//------------------------------------------------------------------------------

/*
ColorRamp::ColorRamp(long numcolors, long R, long G, long B, long Var,
	double min, double max, bool reverse)
{
	Min = min;
	Max = max;
	NumColors = 0;
	Colors = 0;
	Reverse = reverse;
	SetRamp(numcolors, R, G, B, Var, max);
	memset(ColorFile, 0x0, sizeof(ColorFile));
}

void ColorRamp::GetColorChanges(double numcolors, long R, long G, long B,
	long Var)
{
	Rval = R;
	Gval = G;
	Bval = B;

	switch (Var)
	{
	case 0:
		ColorIncG = ColorIncB = 0.0;
		ColorIncR = (double) (Max - R) / (double) numcolors;
		Min = R;
		break;
	case 1:
		ColorIncR = ColorIncB = 0.0;
		ColorIncG = (double) (Max - G) / (double) numcolors;
		Min = G;
		break;
	case 2:
		ColorIncR = ColorIncG = 0.0;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		Min = B;
		break;
	case 3:
		ColorIncB = 0.0;
		ColorIncR = (double) (Max - R) / (double) numcolors;
		ColorIncG = (double) (Max - G) / (double) numcolors;
		Min = R;
		break;
	case 4:
		ColorIncR = 0.0;
		ColorIncG = (double) (Max - R) / (double) numcolors;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		Min = R;
		break;
	case 5:
		ColorIncG = 0.0;
		ColorIncR = (double) (Max - G) / (double) numcolors;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		Min = G;
		break;
	case 6:
		ColorIncR = (double) (Max - R) / (double) numcolors;
		ColorIncG = (double) (Max - G) / (double) numcolors;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		Min = R;
		break;
	case 7:
		ColorIncR = 0.0;
		ColorIncG = (double) (Max - G) / (double) numcolors;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		break;
	case 8:
		ColorIncG = 0.0;
		ColorIncR = (double) (Max - R) / (double) numcolors;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		break;
	case 9:
		ColorIncB = 0.0;
		ColorIncR = (double) (Max - R) / (double) numcolors;
		ColorIncG = (double) (Max - G) / (double) numcolors;
		break;
	case 10:
		ColorIncR = ColorIncG = 0.0;
		ColorIncB = (double) (Max - B) / (double) numcolors;
		break;
	case 11:
		ColorIncG = ColorIncB = 0.0;
		ColorIncR = (double) (Max - R) / (double) numcolors;
		break;
	case 12:
		ColorIncR = ColorIncB = 0.0;
		ColorIncG = (double) (Max - G) / (double) numcolors;
		break;
	}
}


void ColorRamp::SetRamp(long numcolors, long R, long G, long B, long Var,
	double max)
{
	long j, tvar;
	double tmax, tmin;
	long red, green, blue;
	double Range;
	Max = max;
	if (Colors)
		delete[] Colors;//GlobalFree(Colors);
	Colors = 0;
	Colors = new COLORREF[numcolors + 1];//(COLORREF *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, (numcolors+1)*sizeof(COLORREF));
	NumColors = numcolors;

	if (Var < 7)
	{
		GetColorChanges(NumColors, R, G, B, Var);
		Colors[0] = RGB(0, 0, 0);
		for (i = 1; i <= NumColors; i++)
		{
			Colors[i] = RGB(R + ((long) (double) i * ColorIncR),
							G + ((long) (double) i * ColorIncG),
							B + ((long) (double) i * ColorIncB));
		}
	}
	else if (Var < 13)
	{
		Range = (double) NumColors / 2.0;
		tmin = Min;
		GetColorChanges(Range, R, G, B, Var - 7);
		Colors[0] = RGB(0, 0, 0);
		i = j = 1;
		while (j <= Range)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		R = GetRValue(Colors[(long) Range]);
		G = GetGValue(Colors[(long) Range]);
		B = GetBValue(Colors[(long) Range]);
		GetColorChanges(Range, R, G, B, Var);
		i = 1;
		while (j <= Range * 2.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		Min = tmin;
		Rval = Gval = Bval = Min;
	}
	else if (Var < 17)
	{
		switch (Var)
		{
		case 13:
			tvar = 0; break;
		case 14:
			tvar = 0; break;
		case 15:
			tvar = 2; break;
		case 16:
			tvar = 5; break;
		}
		Range = (double) NumColors / 3.0;
		tmin = Min;
		GetColorChanges(Range, R, G, B, tvar);
		Colors[0] = RGB(0, 0, 0);
		i = j = 1;
		while (j <= Range)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		R = GetRValue(Colors[(long) Range]);
		G = GetGValue(Colors[(long) Range]);
		B = GetBValue(Colors[(long) Range]);
		tmax = max;
		switch (Var)
		{
		case 13:
			tvar = 2; break;
		case 14:
			tvar = 1; break;
		case 15:
			tvar = 1; break;
		case 16:
			tvar = 2; tmax = Max; Max = 0; break;
		}
		GetColorChanges(Range, R, G, B, tvar);
		Max = tmax;
		i = 1;
		while (j <= Range * 2.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		R = GetRValue(Colors[(long) (2.0 * Range)]);
		G = GetGValue(Colors[(long) (2.0 * Range)]);
		B = GetBValue(Colors[(long) (2.0 * Range)]);
		switch (Var)
		{
		case 13:
			tvar = 0; tmax = Max; Max = 0;  break;
		case 14:
			tvar = 0; tmax = Max; Max = 0;  break;
		case 15:
			tvar = 2; tmax = Max; Max = 0;  break;
		case 16:
			tvar = 1; tmax = Max; break;
		}
		GetColorChanges(Range, R, G, B, tvar);
		Min = tmin;
		Max = tmax;
		i = 1;
		while (j <= Range * 3.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		Rval = Gval = Bval = Min;
	}
	else if (Var < 18)  // full spectrum
	{
		 	//if(NumColors<6)
		 	//{    if(Colors)
		 	//		delete[] Colors;//GlobalFree(Colors);
		 	//     NumColors=numcolors=6;
		 	//	Colors=new COLORREF[numcolors+1];//(COLORREF *) GlobalAlloc(GMEM_FIXED |  GMEM_ZEROINIT, (numcolors+1)*sizeof(COLORREF));
		 	//}
		 	Range = (double) NumColors / 6.0;
		 	tmin = Min;
		 	GetColorChanges(Range, R, G, B, 0);
		 	Colors[0] = RGB(0, 0, 0);
		 	j = i = 1;
		while (j <= Range)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		 	R = Max;//255
		 	GetColorChanges(Range, R, G, B, 1);
		 	i = 1;
		while (j <= Range * 2.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		 	R = tmin;//0;
		 	G = Max;//255;
		 	GetColorChanges(Range, R, G, B, 0);
		 	i = 1;
		while (j <= Range * 3.0)
		{
			red = R - (long) ((double) i * ColorIncR);
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		 	R = tmin;//0;
		 	G = Max;//255;
		 	B = tmin;//0;
		 	GetColorChanges(Range, R, G, B, 2);
		 	i = 1;
		while (j <= Range * 4.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		 	R = tmin;//0;
		 	G = tmin;//0;
		 	B = Max;//255;
		 	GetColorChanges(Range, R, G, B, 2);
		 	i = 1;
		while (j <= Range * 5.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G - (long) ((double) i * ColorIncG);
			if (green < 0)
				green = 0;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		};
		 	R = tmin;//0;
		 	G = tmin;//0;
		 	B = Max;//255;
		 	GetColorChanges(Range, R, G, B, 0);
		 	i = 1;
		while (j <= Range * 6.0)
		{
			red = R + (long) ((double) i * ColorIncR);
			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			green = G + (long) ((double) i * ColorIncG);
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			blue = B + (long) ((double) i * ColorIncB);
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			Colors[j++] = RGB(red, green, blue);
			i++;
		}
		 	Min = tmin;
		 	Rval = Gval = Bval = Min;
	}
	else
	{
		 	//struct timeb tb;			// time struct for randomizing wind direction
		 	//ftime(&tb);
		 	//srand(tb.time+tb.millitm);
		SYSTEMTIME tb;
		 	GetSystemTime(&tb);
		 	srand(tb.wMilliseconds);

		for (i = 1; i <= NumColors; i++)
		{
			R = rand() % 200 + 56;
			G = rand() % 200 + 56;
			B = rand() % 200 + 56;
			Colors[i] = RGB(R, G, B);
		}
	}
	ColorType = Var;

	if (Var < 18 && Reverse)
	{
		COLORREF tcolr;
		for (i = 1; i <= numcolors / 2; i++)
		{
			j = numcolors - i + 1;
			tcolr = Colors[i];
			Colors[i] = Colors[j];
			Colors[j] = tcolr;
		}
	}
}


ColorRamp::~ColorRamp()
{
	if (Colors)
		delete[] Colors;//GlobalFree(Colors);
	Colors = 0;
}

COLORREF ColorRamp::GetColor(long order)
{
	return Colors[order];
}


bool ColorRamp::SetSpecificColor(long numcolor, long R, long G, long B)
{
	if (numcolor >= NumColors + 1)
		return false;

	Colors[numcolor] = RGB(R, G, B);
	//ColorChange=numcolor;
	//WantNewColor=true;

	return true;
}


long ColorRamp::GetNumColors()
{
	return NumColors;
}
*/



//------------------------------------------------------------------------------
//
//	Landscape Theme functions
//
//------------------------------------------------------------------------------


LandscapeTheme::LandscapeTheme(bool Analyze, Farsite5 *_pFarsite) : GridTheme()
{
	pFarsite = _pFarsite;
	long i;
		strcpy(Name, pFarsite->GetLandFileName());
	//GetLandFilePath().native_file_string().copy(Name, sizeof(Name));

	for (i = 0; i < 10; i++)
//		ZeroMemory(&AllCats[i], 100 * sizeof(long));
		memset(&AllCats[i],0x0, 100 * sizeof(long));

	if (Analyze)
		AnalyzeStats();
	else
		ReadStats();
	CopyStats(F_DATA);
	CreateRamp();
	WantNewRamp = false;
}


void LandscapeTheme::ReadStats()
{
	NumAllCats[0] = pFarsite->Header.numelev;
	NumAllCats[1] = pFarsite->Header.numslope;
	NumAllCats[2] = pFarsite->Header.numaspect;
	NumAllCats[3] = pFarsite->Header.numfuel;
	NumAllCats[4] = pFarsite->Header.numcover;
	NumAllCats[5] = pFarsite->Header.numheight;
	NumAllCats[6] = pFarsite->Header.numbase;
	NumAllCats[7] = pFarsite->Header.numdensity;
	NumAllCats[8] = pFarsite->Header.numduff;
	NumAllCats[9] = pFarsite->Header.numwoody;
	memcpy(&AllCats[0], pFarsite->Header.elevs, 100 * sizeof(long));
	memcpy(&AllCats[1], pFarsite->Header.slopes, 100 * sizeof(long));
	memcpy(&AllCats[2], pFarsite->Header.aspects, 100 * sizeof(long));
	memcpy(&AllCats[3], pFarsite->Header.fuels, 100 * sizeof(long));
	memcpy(&AllCats[4], pFarsite->Header.covers, 100 * sizeof(long));
	memcpy(&AllCats[5], pFarsite->Header.heights, 100 * sizeof(long));
	memcpy(&AllCats[6], pFarsite->Header.bases, 100 * sizeof(long));
	memcpy(&AllCats[7], pFarsite->Header.densities, 100 * sizeof(long));
	memcpy(&AllCats[8], pFarsite->Header.duffs, 100 * sizeof(long));
	memcpy(&AllCats[9], pFarsite->Header.woodies, 100 * sizeof(long));
	maxval[0] = pFarsite->Header.hielev;
	minval[0] = pFarsite->Header.loelev;
	maxval[1] = pFarsite->Header.hislope;
	minval[1] = pFarsite->Header.loslope;
	maxval[2] = pFarsite->Header.hiaspect;
	minval[2] = pFarsite->Header.loaspect;
	maxval[3] = pFarsite->Header.hifuel;
	minval[3] = pFarsite->Header.lofuel;
	maxval[4] = pFarsite->Header.hicover;
	minval[4] = pFarsite->Header.locover;
	maxval[5] = pFarsite->Header.hiheight;
	minval[5] = pFarsite->Header.loheight;
	maxval[6] = pFarsite->Header.hibase;
	minval[6] = pFarsite->Header.lobase;
	maxval[7] = pFarsite->Header.hidensity;
	minval[7] = pFarsite->Header.lodensity;
	maxval[8] = pFarsite->Header.hiduff;
	minval[8] = pFarsite->Header.loduff;
	maxval[9] = pFarsite->Header.hiwoody;
	minval[9] = pFarsite->Header.lowoody;
	Continuous = 0;
}


void LandscapeTheme::AnalyzeStats()
{
	long i, j;

	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 100; j++)
			AllCats[i][j] = -2;
	}

	FillCats();
	SortCats();
	pFarsite->Header.numelev = NumAllCats[0];
	pFarsite->Header.numslope = NumAllCats[1];
	pFarsite->Header.numaspect = NumAllCats[2];
	pFarsite->Header.numfuel = NumAllCats[3];
	pFarsite->Header.numcover = NumAllCats[4];
	pFarsite->Header.numheight = NumAllCats[5];
	pFarsite->Header.numbase = NumAllCats[6];
	pFarsite->Header.numdensity = NumAllCats[7];
	pFarsite->Header.numduff = NumAllCats[8];
	pFarsite->Header.numwoody = NumAllCats[9];
	memcpy(pFarsite->Header.elevs, &AllCats[0], 100 * sizeof(long));
	memcpy(pFarsite->Header.slopes, &AllCats[1], 100 * sizeof(long));
	memcpy(pFarsite->Header.aspects, &AllCats[2], 100 * sizeof(long));
	memcpy(pFarsite->Header.fuels, &AllCats[3], 100 * sizeof(long));
	memcpy(pFarsite->Header.covers, &AllCats[4], 100 * sizeof(long));
	memcpy(pFarsite->Header.heights, &AllCats[5], 100 * sizeof(long));
	memcpy(pFarsite->Header.bases, &AllCats[6], 100 * sizeof(long));
	memcpy(pFarsite->Header.densities, &AllCats[7], 100 * sizeof(long));
	memcpy(pFarsite->Header.duffs, &AllCats[8], 100 * sizeof(long));
	memcpy(pFarsite->Header.woodies, &AllCats[9], 100 * sizeof(long));
	pFarsite->Header.hielev = (long)maxval[0];
	pFarsite->Header.loelev = (long)minval[0];
	pFarsite->Header.hislope = (long)maxval[1];
	pFarsite->Header.loslope = (long)minval[1];
	pFarsite->Header.hiaspect = (long)maxval[2];
	pFarsite->Header.loaspect = (long)minval[2];
	pFarsite->Header.hifuel = (long)maxval[3];
	pFarsite->Header.lofuel = (long)minval[3];
	pFarsite->Header.hicover = (long)maxval[4];
	pFarsite->Header.locover = (long)minval[4];
	pFarsite->Header.hiheight = (long)maxval[5];
	pFarsite->Header.loheight = (long)minval[5];
	pFarsite->Header.hibase = (long)maxval[6];
	pFarsite->Header.lobase = (long)minval[6];
	pFarsite->Header.hidensity = (long)maxval[7];
	pFarsite->Header.lodensity = (long)minval[7];
	pFarsite->Header.hiduff = (long)maxval[8];
	pFarsite->Header.loduff = (long)minval[8];
	pFarsite->Header.hiwoody = (long)maxval[9];
	pFarsite->Header.lowoody = (long)minval[9];
	Continuous = 0;
}

void LandscapeTheme::CopyStats(long layer)
{
	memcpy(Cats, AllCats[layer], 99 * sizeof(long));
	NumCats = NumAllCats[layer];
	MaxVal = maxval[layer];
	MinVal = minval[layer];
	if (NumCats > 0)
		CatsOK = true;
	else
		CatsOK = false;
}


void LandscapeTheme::FillCats()
{
	long i, j, k, m, pos;
	double x, y, resx, resy;
	celldata cell;
	crowndata cfuel;
	grounddata gfuel;

	resx = pFarsite->GetCellResolutionX();
	resy = pFarsite->GetCellResolutionY();
	for (m = 0; m < 10; m++)
	{
		maxval[m] = -1e100;
		minval[m] = 1e100;
		NumAllCats[m] = 0;
		for (k = 0; k < 100; k++)
			AllCats[m][k] = -1;
	}
	for (i = 0; i < pFarsite->GetNumNorth(); i++)
	{
		y = pFarsite->GetHiNorth() - i * resy - resy / 2.0;
		for (j = 0; j < pFarsite->GetNumEast(); j++)
		{
			x = pFarsite->GetLoEast() + j * resx + resx / 2.0;
			pFarsite->CellData(x, y, cell, cfuel, gfuel, &pos);
			AllCats[0][NumAllCats[0]] = cell.e;
			AllCats[1][NumAllCats[1]] = cell.s;
			AllCats[2][NumAllCats[2]] = cell.a;
			AllCats[3][NumAllCats[3]] = cell.f;
			AllCats[4][NumAllCats[4]] = cell.c;
			if (pFarsite->HaveCrownFuels())
			{
				if (cfuel.h >= 0)
					AllCats[5][NumAllCats[5]] = cfuel.h;///10.0;
				if (cfuel.b >= 0)
					AllCats[6][NumAllCats[6]] = cfuel.b;///10;
				if (cfuel.p >= 0)
					AllCats[7][NumAllCats[7]] = cfuel.p;
			}
			if (pFarsite->HaveGroundFuels())
			{
				if (gfuel.d >= 0)
					AllCats[8][NumAllCats[8]] = gfuel.d;
				if (gfuel.w >= 0)
					AllCats[9][NumAllCats[9]] = gfuel.w;
			}
			for (m = 0; m < 10; m++)
			{
				if (maxval[m] < AllCats[m][NumAllCats[m]])
					maxval[m] = AllCats[m][NumAllCats[m]];
				if (AllCats[m][NumAllCats[m]] >= 0)
				{
					if (minval[m] > AllCats[m][NumAllCats[m]])
						minval[m] = AllCats[m][NumAllCats[m]];
				}
			}
			for (m = 0; m < 10; m++)
			{
				if (NumAllCats[m] > 98)
					continue;
				for (k = 0; k < NumAllCats[m]; k++)
				{
					if (AllCats[m][NumAllCats[m]] == AllCats[m][k])
						break;
				}
				if (k == NumAllCats[m])
					NumAllCats[m]++;
			}
		}
	}
	for (m = 0; m < 10; m++)
	{
		if (NumAllCats[m] > 98)
			NumAllCats[m] = -1;
	}
}


void LandscapeTheme::SortCats()
{
	long i, j, m;
	long SwapCats[101];

	for (m = 0; m < 10; m++)
	{
		if (NumAllCats[m] < 0)
			continue;
		memcpy(SwapCats, AllCats[m], 100 * sizeof(long));
		for (i = 0; i < NumAllCats[m] - 1; i++)
		{
			for (j = i + 1; j < NumAllCats[m]; j++)
			{
				if (SwapCats[j] < SwapCats[i])
				{
					SwapCats[100] = SwapCats[i];
					SwapCats[i] = SwapCats[j];
					SwapCats[j] = SwapCats[100];
				}
			}
		}
		AllCats[m][0] = 0;
		for (i = 0; i < NumAllCats[m]; i++)
			AllCats[m][i + 1] = SwapCats[i];
		minval[m] = AllCats[m][1];
		if (minval[m] < 0)
			minval[m] = 0;
	}
}

