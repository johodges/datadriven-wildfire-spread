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
//#include "fswindow.h"
#include "fsx4.hpp"
#include "Farsite5.h"
#include <stdlib.h>

const double PI = acos(-1.0);



LandscapeData::LandscapeData(Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
}

void LandscapeData::FuelConvert(short fl)
{
	ld.fuel = pFarsite->GetFuelConversion(fl);
}


void LandscapeData::ElevConvert(short el)
{
	ld.elev = el;
	if (ld.elev == -9999)
		return;

	switch (pFarsite->GetTheme_Units(E_DATA))
	{
	case 0:
		// meters default
		break;
	case 1:
		// feet
		ld.elev =(short) (ld.elev/3.2804);
		break;
	}
}

void LandscapeData::SlopeConvert(short sl)
{
	ld.slope = sl;
	if (ld.slope == -9999)
	{
		ld.slope =(short) 0.0;
		return;
	}

	switch (pFarsite->GetTheme_Units(S_DATA))
	{
	case 0:
		// degrees default
		break;
	case 1:
		// percent
		double fraction, ipart;
		double slopef;

		slopef = atan((double) ld.slope / 100.0) / PI * 180.0;
		ld.slope =(short) slopef;
		fraction = modf(slopef, &ipart);
		if (fraction >= 0.5)
			ld.slope++;
		break;
	}
}


void LandscapeData::AspectConvert(short as)
{
	if (as == -9999)
	{
		ld.aspect = 0;
		ld.aspectf = 0.0;
		return;
	}
	ld.aspect = as;
	ld.aspectf = (double) as;
	switch (pFarsite->GetTheme_Units(A_DATA))
	{
	case 0:
		// grass 1-25 counterclockwise from east
		if (ld.aspect != 25)
			ld.aspectf = (2.0 * PI - (ld.aspectf / 12.0 * PI)) +
				(7.0 * PI / 12.0);  /* aspect from GRASS, east=1, north=7, west=13, south=19 */
		else
		{
			ld.aspectf = 25.0;
			ld.slope = 0;
		}
		break;
	case 1:
		// degrees 0 to 360 counterclockwise from east
		ld.aspectf = (2.0 - ld.aspectf / 180.0) * PI + PI / 2.0;
		break;
	case 2:
		ld.aspectf = ld.aspectf / 180.0 * PI;     // arcinfo, degrees AZIMUTH
		break;
	}
	if (ld.aspectf > (2.0 * PI))
		ld.aspectf -= (2.0 * PI);
}



void LandscapeData::CoverConvert(short cov)
{
	ld.cover = cov;
	if (pFarsite->GetTheme_Units(C_DATA) == 0)
	{
		switch (ld.cover)
		{
		case 99:
			ld.cover = 0; break;
		case 1:
			ld.cover = 10; break;
		case 2:
			ld.cover = 30; break;
		case 3:
			ld.cover = 60; break;
		case 4:
			ld.cover = 75; break;
		default:
			ld.cover = 0; break;
		}
	}
}


void LandscapeData::HeightConvert(short height)
{
	if (pFarsite->HaveCrownFuels())
	{
		if (height >= 0)
		{
			short units = pFarsite->GetTheme_Units(H_DATA);

			ld.height = (double) height / 10.0;
			if (units == 2 || units == 4)
				ld.height /= 3.280839;
			if (ld.height > 100.0)
				ld.height /= 10.0;	// probably got wrong units
		}
		else
			ld.height = pFarsite->GetDefaultCrownHeight();
	}
	else
		ld.height = pFarsite->GetDefaultCrownHeight();
}

void LandscapeData::BaseConvert(short base)
{
	if (pFarsite->HaveCrownFuels())
	{
		if (base >= 0)
		{
			short units = pFarsite->GetTheme_Units(B_DATA);
			ld.base = (double) base / 10.0;
			if (units == 2 || units == 4)
				ld.base /= 3.280839;
			if (ld.base > 100)
				ld.base /= 10.0;	// probably got wrong units
		}
		else
			ld.base = pFarsite->GetDefaultCrownBase();
	}
	else
		ld.base = pFarsite->GetDefaultCrownBase();
}

void LandscapeData::DensityConvert(short density)
{
	if (pFarsite->HaveCrownFuels())
	{
		if (density >= 0)
		{
			short units = pFarsite->GetTheme_Units(P_DATA);
			ld.density = ((double) density) / 100.0;
			if (units == 2 || units == 4)
				ld.density *= 1.601847;	// convert 10lb/ft3 to kg/m3
			if (ld.density > 1.0)
				ld.density /= 100.0;	// probably got wrong units
			if (pFarsite->LinkDensityWithCrownCover(GETVAL))
				ld.density *= ((double) ld.cover) / 100.0;
		}
		else
			ld.density = pFarsite->GetDefaultCrownBD(ld.cover);
	}
	else
		ld.density = pFarsite->GetDefaultCrownBD(ld.cover);
}

void LandscapeData::DuffConvert(short duff)
{
	if (pFarsite->HaveGroundFuels())
	{
		if (duff >= 0)
		{
			ld.duff = (double) duff / 10.0;
			if (pFarsite->GetTheme_Units(D_DATA) == 2)
				ld.duff *= 2.2417088978002777;
		}
		else
			ld.duff = 0.0;
	}
	else
		ld.duff = 0.0;
}

void LandscapeData::WoodyConvert(short woody)
{
	if (pFarsite->HaveGroundFuels())
		ld.woody = woody;
	else
		ld.woody = 0;
}









