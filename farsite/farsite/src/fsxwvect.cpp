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
#include "fsx4.hpp"
#include "Farsite5.h"

const double PI = acos(-1.0);
//extern const double PI;


Vectorize::Vectorize(Farsite5 *_pFarsite) : OutputFile(_pFarsite)
{
	pFarsite = _pFarsite;
}


Vectorize::~Vectorize()
{
}

void Vectorize::VectData(long CurrentFire, double SimTime)
{
	CurFire = CurrentFire;
	t = SimTime;
	ConvF();		   			// computer english/metric conversions
	ArcFileFormat();
}


void Vectorize::ArcFileFormat()
{
	otpfile = fopen(pFarsite->GetVectorFileName(), "a");
	if (otpfile == NULL)
		return;

	if (CurFire > -1)
	{
		double firstx, firsty;
		double days = 0, hours = 0, mins;

		I = pFarsite->GetNumPoints(CurFire);
		if (t >= 1440)
		{
			days = t / 1440.0;
			t -= (((long) days) * 1440);
		}
		if (t >= 60)
		{
			hours = t / 60.0;
			t -= (((long) hours) * 60);
		}
		mins = (long) t;
		fprintf(otpfile, "%02ld%02ld%02ld%05ld\n", (long) days, (long) hours,
			(long) mins, CurFire + 1); 		  // time as polygon label
		//		fprintf(otpfile, "%06ld%05ld\n", ((long) t), CurFire+1); 		  // time as polygon label
		i = 0;
		GetXY();
		fprintf(otpfile, "%13.6lf %13.6lf\n", x, y);   // XY coordinates
		firstx = x; firsty = y;
		for (i = 1; i < I; i++)
		{
			GetXY();
			fprintf(otpfile, "%13.6lf %13.6lf\n", x, y);   	  // XY coordinates
		}
		fprintf(otpfile, "%13.6lf %13.6lf\n", firstx, firsty);   // XY coordinates
		fprintf(otpfile, "%s\n", "END");				  // end of polygon block
	}
	else
		fprintf(otpfile, "%s\n", "END");				  // end of file
	fclose(otpfile);
}


void Vectorize::OptionalFileFormat()
{
	if (CurFire > -1)
	{
		I = pFarsite->GetNumPoints(CurFire);
		for (i = 0; i < I; i++)
		{
			GetXY();
			GetRF();
			SelectOutputs(2);
		}
	}
}


void Vectorize::GetXY()
{
	x = pFarsite->ConvertEastingOffsetToUtm(pFarsite->GetPerimeter1Value(CurFire, i, XCOORD));
	y = pFarsite->ConvertNorthingOffsetToUtm(pFarsite->GetPerimeter1Value(CurFire, i, YCOORD));
	if (pFarsite->GetFileOutputOptions(RAST_FIREDIRECTION))
	{
		double xl, yl, xn, yn;
		long j, k;
		j = i - 1;
		k = i + 1;
		if (j < 0)
			j = pFarsite->GetNumPoints(CurFire) - 1;
		if (k > pFarsite->GetNumPoints(CurFire))
			k = 0;
		xl = pFarsite->ConvertEastingOffsetToUtm(pFarsite->GetPerimeter1Value(CurFire, j, XCOORD));
		yl = pFarsite->ConvertNorthingOffsetToUtm(pFarsite->GetPerimeter1Value(CurFire, j, YCOORD));
		xn = pFarsite->ConvertEastingOffsetToUtm(pFarsite->GetPerimeter1Value(CurFire, k, XCOORD));
		yn = pFarsite->ConvertNorthingOffsetToUtm(pFarsite->GetPerimeter1Value(CurFire, k, YCOORD));
		if (pow2(xn - xl) + pow2(yn - yl) < 1.0e-6)
			d = 0;
		else
			d = (long) (360 * atan2(-(xn - xl), (yn - yl)) / (2.0 * PI));
		if (d < 0)
			d += 360;
		d -= 90;
		if (d < 0)
			d += 360;
	}
}


void Vectorize::GetRF()
{
	r = fabs(pFarsite->GetPerimeter1Value(CurFire, i, ROSVAL));
	f = fabs(pFarsite->GetPerimeter1Value(CurFire, i, FLIVAL));
}
