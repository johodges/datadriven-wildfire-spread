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
//	FSXWRAST.CPP	Rasterizing Interpolation functions for FARSITE
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environemntal Management
//******************************************************************************


#include <time.h>
#include "fsx4.hpp"
#include "Farsite5.h"

const double PI = acos(-1.0);
//extern const double PI;


Rasterize::Rasterize(Farsite5 *_pFarsite) : OutputFile(_pFarsite), APolygon(_pFarsite)
{
	pFarsite = _pFarsite;
	xmid = ymid = 0;
	//AllocRastData(200);
}


Rasterize::~Rasterize()
{
}

void Rasterize::rasterinit(long CurrentFire, long ExNumPoints, double SIMTIME,
	double TimeIncrement, double CuumTimeIncrement)
{
	GetRasterExtent();
	count = CurrentFire;
	ExNumPts = ExNumPoints;
	CurTime = (SIMTIME / 60.0) + CuumTimeIncrement / 60.0;    // time in hours.decimals
	LastTime = CurTime - TimeIncrement / 60.0;
	//if(GetRastFormat()==3)
	//	otpfile=fopen(GetRasterFileName(0), "a");
	ConvF();						 // calculate conversion factors (::SelectOutputs
	raster();						 // rasterize the last time increment
	//if(GetRastFormat()==3)
	//	fclose(otpfile);
}


void Rasterize::RasterReset()
{
	filepos = 0;
	pFarsite->CanSetRasterResolution(1);		// true
}


void Rasterize::raster()
{
	// ROS IS RASTERIZED AS WELL AS TIME OF ARRIVAL
	// MUST BE ABLE TO SWITCH RASTERIZING TO SELECT FLI, FL, AND OTHER FIRE CHX'S

	long inc, noeast, nonorth;
	long nump, E, N, inside, newnump;
	double xpt1, xpt2, xpt1n, xpt2n, ypt1, ypt2, ypt1n, ypt2n, sx, sy;
	double xcross, ycross;
	double ros1 = 0, ros2, ros2n, ros1n = 0, rcx1, rcx1n;
	double fli1 = 0, fli2, fli2n, fli1n = 0, rcx2, rcx2n;
	double xmax, ymax, xmin, ymin;
	double RastResX, RastResY;
	double emax, emin, nmax, nmin;  //posit=0,

	pFarsite->GetRastRes(&RastResX, &RastResY);
	if (ExNumPts > 0)				// TERMINATION OF INWARD BURNING FIRE
	{
		pFarsite->AllocPerimeter1(count, ExNumPts);
		pFarsite->SetNumPoints(count, ExNumPts);
		for (inc = 0; inc < ExNumPts; inc++)		 // fill in perim1 array
		{
			xpt1 = pFarsite->GetPerimeter2Value(inc, XCOORD);
			ypt1 = pFarsite->GetPerimeter2Value(inc, YCOORD);
			pFarsite->SetPerimeter1(count, inc, xpt1, ypt1);
			ros1 = pFarsite->GetPerimeter2Value(inc, ROSVAL);
			fli1 = pFarsite->GetPerimeter2Value(inc, FLIVAL);
			rcx1 = pFarsite->GetPerimeter2Value(inc, RCXVAL);
			pFarsite->SetFireChx(count, inc, ros1, fli1);
			pFarsite->SetReact(count, inc, rcx1);
		}

		double dist, lowdist; // very large initial value
		FMaxMin(0, &xmax, &xmin);
		FMaxMin(1, &ymax, &ymin);
		rastdata(xmax, ymax, &emax, &nmax, 1);  	// rounds off to nearest raster
		rastdata(xmin, ymin, &emin, &nmin, 0);  	// rounds off to nearest raster
		noeast = (long)(emax - emin) / (long)RastResX + 1;
		nonorth = (long)(nmax - nmin) / (long)RastResY + 1;
		startx = emin + RastResX / 2.0;
		for (E = 0; E <= noeast; E++)
		{
			//startx=((double) E)*RastResX+emin+RastResX/2.0;				// startx, and starty are
			starty = nmin + RastResY / 2.0;
			for (N = 0;
				N <= nonorth;
				N++)  		  // XUtilities public data members
			{
				//starty=((double) N)*RastResY+nmin+RastResY/2.0;
				inside = APolygon::Overlap(count);
				if (inside)
				{
					xpt1 = pFarsite->GetPerimeter1Value(count, 0, XCOORD);
					ypt1 = pFarsite->GetPerimeter1Value(count, 0, YCOORD);
					r = pFarsite->GetPerimeter1Value(count, 0, ROSVAL);
					f = pFarsite->GetPerimeter1Value(count, 0, FLIVAL);
					rx = pFarsite->GetPerimeter1Value(count, 0, RCXVAL);
					lowdist = pow2(xpt1 - startx) + pow2(ypt1 - starty);
					if (lowdist > 0.0)
					{
						xpt1n = xpt1;
						ypt1n = ypt1;
					}
					else
					{
						xpt1n = xpt1 + 0.1; 		 //arbitrary offset
						ypt1n = ypt1 + 0.1; 		 // to avoid blowing up atan2
					}
					for (inc = 1;
						inc < ExNumPts;
						inc++)   // find distance from all vertices
					{
						xpt1 = pFarsite->GetPerimeter1Value(count, inc, XCOORD);
						ypt1 = pFarsite->GetPerimeter1Value(count, inc, YCOORD);
						dist = pow2(xpt1 - startx) + pow2(ypt1 - starty);
						if (dist < lowdist) 	   	   // select the closest point values
						{
							lowdist = dist;
							r = pFarsite->GetPerimeter1Value(count, inc, ROSVAL);
							f = pFarsite->GetPerimeter1Value(count, inc, FLIVAL);
							rx = pFarsite->GetPerimeter1Value(count, inc, RCXVAL);
							if (lowdist > 0.0)
							{
								xpt1n = xpt1;
								ypt1n = ypt1;
							}
						}
					}
					x = APolygon::startx;
					y = APolygon::starty;
					t = CurTime;
					if (pow2(x - xpt1n) + pow2(y - ypt1n) < 1.0e-6)
						d = 0;
					else
						d = (long)
							(360.0 * atan2((ypt1n - y), -(xpt1n - x)) /
							(2.0 * PI));
					//	d=(long) (360.0*atan2(y-ypt1n, xpt1n-x)/(2.0*PI));
					if (d < 0)
						d += 360;
					//SelectOutputs(GetRastFormat());  // outputs to raster file
					SetRastData(x, y, t, f, r, rx, d);
				}
				starty += RastResY;
			}
			startx += RastResX;
		}
		nump = 0;
	}
	else
		nump = pFarsite->GetNumPoints(count);  	  // exit raster procedure
	if (nump != 0)
	{
		pFarsite->GetPerimeter2(0, &xpt1, &ypt1, &ros1, &fli1, &rcx1);
		xpt2 = pFarsite->GetPerimeter1Value(count, 0, XCOORD);
		ypt2 = pFarsite->GetPerimeter1Value(count, 0, YCOORD);
		ros2 = pFarsite->GetPerimeter1Value(count, 0, ROSVAL);
		fli2 = pFarsite->GetPerimeter1Value(count, 0, FLIVAL);
		rcx2 = pFarsite->GetPerimeter1Value(count, 0, RCXVAL);
		for (inc = 1; inc <= nump; inc++)
		{
			if (inc < nump)
				newnump = inc;
			else
				newnump = 0;
			pFarsite->GetPerimeter2(newnump, &xpt1n, &ypt1n, &ros1n, &fli1n, &rcx1n);
			xpt2n = pFarsite->GetPerimeter1Value(count, newnump, XCOORD);
			ypt2n = pFarsite->GetPerimeter1Value(count, newnump, YCOORD);
			ros2n = pFarsite->GetPerimeter1Value(count, newnump, ROSVAL);
			fli2n = pFarsite->GetPerimeter1Value(count, newnump, FLIVAL);
			rcx2n = pFarsite->GetPerimeter1Value(count, newnump, RCXVAL);
			if (ros1 < 0.0)
			{
				ros1 = ros2;	 // in case start value was 100% ROS
				fli1 = fli2;
			}
			if (ros1n < 0.0)
			{
				ros1n = ros2n;
				fli1n = fli2n;
			}
			//if(fli1<0.0) fli1=fli2;     // in case start value was 100% FLI
			//if(fli1n<0.0) fli1n=fli2n;
			box[0][0] = xpt1;		// write points in clockwise circle
			box[0][1] = ypt1;
			box[1][0] = xpt2;
			box[1][1] = ypt2;
			box[2][0] = xpt2n;
			box[2][1] = ypt2n;
			box[3][0] = xpt1n;
			box[3][1] = ypt1n;
			box[0][4] = ros1;		// write points in clockwise circle
			box[1][4] = ros2;
			box[2][4] = ros2n;
			box[3][4] = ros1n;
			box[0][5] = fabs(fli1); 	// write points in clockwise circle
			box[1][5] = fabs(fli2);
			box[2][5] = fabs(fli2n);
			box[3][5] = fabs(fli1n);
			box[0][6] = rcx1;		// write points in clockwise circle
			box[1][6] = rcx2;
			box[2][6] = rcx2n;
			box[3][6] = rcx1n;

			/***** ELIMINATE CROSSED BOXES FOR INTERPOLATION */
			SamePoint = false;
			if (xpt1 == xpt2 && ypt1 == ypt2)
				SamePoint = true;
			if (xpt1n == xpt2n && ypt1n == ypt2n)
				SamePoint = true;
			if (xpt1 == xpt1n && ypt1 == ypt1n)
				SamePoint = true;
			if (xpt2 == xpt2n && ypt2 == ypt2n)
				SamePoint = true;
			if (SamePoint == false)
			{
				if (Cross(xpt1, ypt1, xpt2, ypt2, xpt1n, ypt1n, xpt2n, ypt2n,
						&xcross, &ycross))
				{
					box[1][0] = xpt2n;    // switch coordinates & values
					box[1][1] = ypt2n;
					box[2][0] = xpt2;
					box[2][1] = ypt2;
					box[1][4] = ros2n;
					box[2][4] = ros2;
					box[1][5] = fabs(fli2n);
					box[2][5] = fabs(fli2);
					box[1][6] = rcx2n;
					box[2][6] = rcx2;
				}
				else if (Cross(xpt1, ypt1, xpt1n, ypt1n, xpt2, ypt2, xpt2n,
							ypt2n, &xcross, &ycross))
				{
					box[2][0] = xpt1n;    // switch coordinates & values
					box[2][1] = ypt1n;
					box[3][0] = xpt2n;
					box[3][1] = ypt2n;
					box[2][4] = ros1n;
					box[3][4] = ros2n;
					box[2][5] = fabs(fli1n);
					box[3][5] = fabs(fli2n);
					box[2][6] = rcx1n;
					box[3][6] = rcx2n;
				}
			}
			if (box[0][0] == box[2][0] && box[0][1] == box[2][1])
			{
				TempBox[0] = box[1][0];
				TempBox[1] = box[1][1];
				box[1][0] = box[2][0];
				box[1][1] = box[2][1];
				box[2][0] = TempBox[0];
				box[2][1] = TempBox[1];
				TempBox[0] = box[1][4];
				TempBox[1] = box[1][5];
				box[1][4] = box[2][4];
				box[1][5] = box[2][5];
				box[2][4] = TempBox[0];
				box[2][5] = TempBox[1];
				TempBox[0] = box[1][6];
				//box[1][6] = box[4][6];
				//box[4][6] = TempBox[0];
				//Changed SB 3/2/10
				box[1][6] = box[3][6];
				box[3][6] = TempBox[0];
			}
			else if (box[1][0] == box[3][0] && box[1][1] == box[3][1])
			{
				TempBox[0] = box[0][0];
				TempBox[1] = box[0][1];
				box[0][0] = box[3][0];
				box[0][1] = box[3][1];
				box[3][0] = TempBox[0];
				box[3][1] = TempBox[1];
				TempBox[0] = box[0][4];
				TempBox[1] = box[0][5];
				box[0][4] = box[3][4];
				box[0][5] = box[3][5];
				box[3][4] = TempBox[0];
				box[3][5] = TempBox[1];
				TempBox[0] = box[0][6];
				box[0][6] = box[3][6];
				box[3][6] = TempBox[0];
			}
			xmax = findmax(xpt1, xpt1n, xpt2, xpt2n);
			ymax = findmax(ypt1, ypt1n, ypt2, ypt2n);
			xmin = findmin(xpt1, xpt1n, xpt2, xpt2n);
			ymin = findmin(ypt1, ypt1n, ypt2, ypt2n);
			rastdata(xmax, ymax, &emax, &nmax, 1);  	// rounds off to nearest raster
			rastdata(xmin, ymin, &emin, &nmin, 0);  	// rounds off to nearest raster
			noeast = (long)(emax - emin) / (long)RastResX + 1;
			nonorth = (long)(nmax - nmin) / (long)RastResY + 1;
			startx = emin + RastResX / 2.0;

			for (E = 0; E < noeast; E++)
			{
				starty = nmin + RastResY / 2.0;
				for (N = 0;
					N < nonorth;
					N++)   		 	// XUtilities public data members
				{
					inside = Overlap();
					if (inside)
					{
						sy =ymid = starty; sx = xmid = startx;
						Interpolate(true);
						starty = sy; startx = sx;
					}
					starty += RastResY;
				}
				startx += RastResX;
			}
			xpt1 = xpt1n; ypt1 = ypt1n; xpt2 = xpt2n; ypt2 = ypt2n;
			ros1 = ros1n; ros2 = ros2n; fli1 = fli1n; fli2 = fli2n;
		}
	}
	//SelectMemOutputs(GetRastFormat());	// write to raster file
	if (ExNumPts > 0)
	{
		ExNumPts = 0;
		pFarsite->FreePerimeter1(count);
		pFarsite->SetNumPoints(count, 0);
	}
}


void Rasterize::RasterizePostFrontal(double Resolution, char* FileName,
	bool ViewPort, long ThemeNumber)
{
	bool OutOfRange = false;
	long i, j, k, m, n, q, numy, numx, nrings;
	long noeast, nonorth, locx, locy, loc;
	long E, N, inside;
	float* map;
	double PFNorth, PFEast, WRate1, WRate2, WRateF1, WRateF2;
	double xpt1, ypt1, xpt2, ypt2, xpt1n, ypt1n, xpt2n, ypt2n;
	double xcross, ycross;
	double xmax, ymax, xmin, ymin, sx, sy;
	double emax, emin, nmax, nmin;  //posit=0,
	double pm25f = 67.4 - 0.95 * 66.8;
	double pm25s = 67.4 - 0.75 * 66.8;
	double pm10f = 1.18 * pm25f;
	double pm10s = 1.18 * pm25s;
	double ch4f = 42.7 - 0.95 * 43.2;
	double ch4s = 42.7 - 0.75 * 43.2;
	double coF = 961 - 0.95 * 984.0;
	double coS = 961 - 0.75 * 984.0;
	double co2f = 0.95 * 1833.0;
	double co2s = 0.75 * 1833.0;
	double ConvWt = 0.001;  		   //kg to Mg
	double ConvHeat = 18.6; 		   //kg to MJ
	double ConvArea = 10000.0;  	   //m2 to ha
	double ConvEmit = 1.0;

	if (ThemeNumber > 1)
		ConvEmit = 0.001;			 //g to kg
	if (pFarsite->AccessOutputUnits(GETVAL))		// if English Output Units
	{
		ConvWt = 0.001102311;   		// kg to tons
		ConvHeat = 1.76328;			// kg to MBTU
		ConvArea = 4046.8564;				// m2 to ac
	}
	FireRing* ring;

	CurTime = pFarsite->GetTemporaryTimeStep();
	LastTime = 0;
	PFRes = Resolution;
	/*if (ViewPort)
	{
		PFNorth = pFarsite->GetViewNorth();
		PFSouth = GetViewSouth();
		PFEast = GetViewEast();
		PFWest = GetViewWest();
		locx = (long) ((PFWest - GetLoEast()) / PFRes);
		PFWest = ((double) locx) * PFRes + GetLoEast();
		locx = (long) ((PFEast - GetLoEast()) / PFRes);
		PFEast = ((double) locx) * PFRes + GetLoEast();
		locy = (long) ((PFSouth - GetLoNorth()) / PFRes);
		PFSouth = ((double) locy) * PFRes + GetLoNorth();
		locy = (long) ((PFNorth - GetLoNorth()) / PFRes);
		PFNorth = ((double) locy) * PFRes + GetLoNorth();
	}
	else
	{*/
		PFNorth = pFarsite->GetHiNorth();
		PFSouth = pFarsite->GetLoNorth();
		PFEast = pFarsite->GetHiEast();
		PFWest = pFarsite->GetLoEast();
	//}
	numy = (long)(PFNorth - PFSouth) / (long)PFRes;
	numx = (long)(PFEast - PFWest) / (long)PFRes;

	map = new float[numx * numy];
	for (i = 0; i < numy; i++)  	// initialize to -1.0;
	{
		for (j = 0; j < numx; j++)
			map[i * numx + j] = -1.0;
	}

	for (i = 0; i < pFarsite->GetNumFires(); i++)
	{
		xmin = pFarsite->GetPerimeter1Value(i, pFarsite->GetNumPoints(i), XCOORD);
		xmax = pFarsite->GetPerimeter1Value(i, pFarsite->GetNumPoints(i), YCOORD);
		ymin = pFarsite->GetPerimeter1Value(i, pFarsite->GetNumPoints(i), ROSVAL);
		ymax = pFarsite->GetPerimeter1Value(i, pFarsite->GetNumPoints(i), FLIVAL);
		starty = ymax;
		do
		{
			startx = xmin;
			n = (long) ((PFNorth - starty) / PFRes);
			if (n < 0 || n >= numy)
			{
				starty -= PFRes;
				continue;
			}
			do
			{
				if (APolygon::Overlap(i))
				{
					q = (long) ((startx - PFWest) / PFRes);
					if (q < 0 || q >= numx)
					{
						startx += PFRes;
						continue;
					}
					map[n * numx + q] = -2.0;
				}
				startx += PFRes;
			}
			while (startx <= xmax);
			starty -= PFRes;
		}
		while (starty >= ymin);
	}

	nrings = pFarsite->GetNumRings();
	for (i = 0; i < nrings; i++)
	{
		ring = pFarsite->GetRing(i);
		if (ring->perimpoints == NULL)
			continue;
		for (j = 0; j < ring->NumFires; j++)
		{
			if (ring->NumPoints == NULL)
				continue;
			if (j == 0)
				n = 0;
			else
				n = ring->NumPoints[j - 1];
			for (k = n; k < ring->NumPoints[j]; k++)
			{
				q = k + 1;
				if (q > ring->NumPoints[j] - 1)
					q = n;

				if (ring->perimpoints[k].Area > 1e-9)     // kg/m2/min
				{
					WRate1 = ring->perimpoints[k].hist.CurWtRemoved /
						pFarsite->GetTemporaryTimeStep() /
						ring->perimpoints[k].Area;
					WRateF1 = ring->perimpoints[k].hist.FlameWtRemoved /
						pFarsite->GetTemporaryTimeStep() /
						ring->perimpoints[k].Area;
				}
				else
				{
					WRate1 = 0.0;
					WRateF1 = 0.0;
				}
				if (ring->perimpoints[q].Area > 1e-9)     // kg/m2/min
				{
					WRate2 = ring->perimpoints[q].hist.CurWtRemoved /
						pFarsite->GetTemporaryTimeStep() /
						ring->perimpoints[q].Area;
					WRateF2 = ring->perimpoints[q].hist.FlameWtRemoved /
						pFarsite->GetTemporaryTimeStep() /
						ring->perimpoints[q].Area;
				}
				else
				{
					WRate2 = 0.0;
					WRateF2 = 0.0;
				}

				box[0][0] = xpt1 = ring->perimpoints[k].x1;		// write points in clockwise circle
				box[0][1] = ypt1 = ring->perimpoints[k].y1;
				box[1][0] = xpt2 = ring->perimpoints[k].x2;
				box[1][1] = ypt2 = ring->perimpoints[k].y2;
				box[2][0] = xpt2n = ring->perimpoints[q].x2;
				box[2][1] = ypt2n = ring->perimpoints[q].y2;
				box[3][0] = xpt1n = ring->perimpoints[q].x1;
				box[3][1] = ypt1n = ring->perimpoints[q].y1;

				// check that points around box are within viewport
				for (m = 0; m < 4; m++)
				{
					if (box[m][0] < PFWest || box[m][0] > PFEast)
					{
						OutOfRange = true;
						break;
					}
					if (box[m][1] < PFSouth || box[m][1] > PFNorth)
					{
						OutOfRange = true;
						break;
					}
				}
				if (OutOfRange)
				{
					OutOfRange = false;

					continue;
				}
				box[0][4] = WRate1;		// write Total points in clockwise circle
				box[1][4] = WRate2;
				box[2][4] = WRate2;
				box[3][4] = WRate1;
				box[0][5] = WRateF1;		// write Flaming points in clockwise circle
				box[1][5] = WRateF2;
				box[2][5] = WRateF2;
				box[3][5] = WRateF1;
				box[0][6] = WRate1;		// write points in clockwise circle
				box[1][6] = WRate2;
				box[2][6] = WRate2;
				box[3][6] = WRate1;

				/***** ELIMINATE CROSSED BOXES FOR INTERPOLATION */
				SamePoint = false;
				if (xpt1 == xpt2 && ypt1 == ypt2)
					SamePoint = true;
				if (xpt1n == xpt2n && ypt1n == ypt2n)
					SamePoint = true;
				if (xpt1 == xpt1n && ypt1 == ypt1n)
					SamePoint = true;
				if (xpt2 == xpt2n && ypt2 == ypt2n)
					SamePoint = true;
				if (SamePoint == false)
				{
					if (Cross(xpt1, ypt1, xpt2, ypt2, xpt1n, ypt1n, xpt2n,
							ypt2n, &xcross, &ycross))
					{
						box[1][0] = xpt2n;    // switch coordinates & values
						box[1][1] = ypt2n;
						box[2][0] = xpt2;
						box[2][1] = ypt2;
					}
					else if (Cross(xpt1, ypt1, xpt1n, ypt1n, xpt2, ypt2,
								xpt2n, ypt2n, &xcross, &ycross))
					{
						box[2][0] = xpt1n;    // switch coordinates & values
						box[2][1] = ypt1n;
						box[3][0] = xpt2n;
						box[3][1] = ypt2n;
					}
				}
				if (box[0][0] == box[2][0] && box[0][1] == box[2][1])
				{
					TempBox[0] = box[1][0];
					TempBox[1] = box[1][1];
					box[1][0] = box[2][0];
					box[1][1] = box[2][1];
					box[2][0] = TempBox[0];
					box[2][1] = TempBox[1];
					TempBox[0] = box[1][4];
					TempBox[1] = box[1][5];
					box[1][4] = box[2][4];
					box[1][5] = box[2][5];
					box[2][4] = TempBox[0];
					box[2][5] = TempBox[1];
					TempBox[0] = box[1][6];
					//box[1][6] = box[4][6];
					//box[4][6] = TempBox[0];
					//Changed SB 3/2/10
					box[1][6] = box[3][6];
					box[3][6] = TempBox[0];
				}
				else if (box[1][0] == box[3][0] && box[1][1] == box[3][1])
				{
					TempBox[0] = box[0][0];
					TempBox[1] = box[0][1];
					box[0][0] = box[3][0];
					box[0][1] = box[3][1];
					box[3][0] = TempBox[0];
					box[3][1] = TempBox[1];
					TempBox[0] = box[0][4];
					TempBox[1] = box[0][5];
					box[0][4] = box[3][4];
					box[0][5] = box[3][5];
					box[3][4] = TempBox[0];
					box[3][5] = TempBox[1];
					TempBox[0] = box[0][6];
					box[0][6] = box[3][6];
					box[3][6] = TempBox[0];
				}
				xmax = findmax(xpt1, xpt1n, xpt2, xpt2n);
				ymax = findmax(ypt1, ypt1n, ypt2, ypt2n);
				xmin = findmin(xpt1, xpt1n, xpt2, xpt2n);
				ymin = findmin(ypt1, ypt1n, ypt2, ypt2n);
				PFrastdata(xmax, ymax, &emax, &nmax, 1);	  // rounds off to nearest raster
				PFrastdata(xmin, ymin, &emin, &nmin, 0);	  // rounds off to nearest raster
				noeast = (long)(emax - emin) / (long)PFRes + 1;
				nonorth = (long)(nmax - nmin) / (long)PFRes + 1;

				startx = emin + PFRes / 2.0;
				for (E = 0; E < noeast; E++)
				{
					starty = nmin + PFRes / 2.0;
					for (N = 0;
						N < nonorth;
						N++)   		 	// XUtilities public data members
					{
						inside = Overlap();
						if (inside)
						{
							sx = xmid = startx;
							sy = ymid = starty;
							Interpolate(false);
							//loc=((long) ((PFNorth-sy)/PFRes))*numx+(long) (sx/PFRes);
							locy = ((long) ((PFNorth - sy) / PFRes));
							locx = (long) ((sx - PFWest) / PFRes);
							loc = locy * numx + locx;
							switch (ThemeNumber)
							{
							case 0:
								map[loc] = (WrT * 18.60) * ConvHeat;	// MJ, or MBtu
								break;
							case 1:
								map[loc] = WrT * ConvWt;
								break;
							case 2:
								map[loc] = WfT / WrT;
								break;
							case 3:
								map[loc] = ((WrT - WfT) * pm25s + WfT * pm25f) * ConvWt * ConvEmit;
								break;
							case 4:
								map[loc] = ((WrT - WfT) * pm10s + WfT * pm10f) * ConvWt * ConvEmit;
								break;
							case 5:
								map[loc] = ((WrT - WfT) * ch4s + WfT * ch4f) * ConvWt * ConvEmit;
								break;
							case 6:
								map[loc] = ((WrT - WfT) * coS + WfT * coF) * ConvWt * ConvEmit;
								break;
							case 7:
								map[loc] = ((WrT - WfT) * co2s + WfT * co2f) * ConvWt * ConvEmit;
								break;
							}
							map[loc] *= ConvArea;
							startx = sx;
							starty = sy;
						}
						starty += PFRes;
					}
					startx += PFRes;
				}
			}
		}
	}

	FILE* outfile = fopen(FileName, "w");
	fprintf(outfile, "ncols %ld\n", numx);
	fprintf(outfile, "nrows %ld\n", numy);
	fprintf(outfile, "xllcorner %lf\n", pFarsite->ConvertEastingOffsetToUtm(PFWest));
	fprintf(outfile, "yllcorner %lf\n", pFarsite->ConvertNorthingOffsetToUtm(PFSouth));
	fprintf(outfile, "cellsize %lf\n", PFRes);
	fprintf(outfile, "NODATA_VALUE -1\n");

	for (i = 0; i < numy; i++)
	{
		for (j = 0; j < numx; j++)
		{
			fprintf(outfile, "%lf ", map[i * numx + j]);
			fprintf(outfile, "\n");
		}
	}
	fclose(outfile);
	delete[] map;
}


bool Rasterize::Interpolate(bool WriteFile)
{
	double diffx, diffy, xcross, ycross, xc1, yc1, xc2, yc2, xc3, yc3, xc4,
		yc4;
	double xpt1, xpt2, xpt1n, xpt2n, ypt1, ypt2, ypt1n, ypt2n;

	xmid = startx; ymid = starty;
	xpt1 = box[0][0];
	ypt1 = box[0][1];
	xpt2 = box[1][0];
	ypt2 = box[1][1];
	xpt2n = box[2][0];
	ypt2n = box[2][1];
	xpt1n = box[3][0];
	ypt1n = box[3][1];


	// perpendicular to segment
	if ((pow2(xpt1n - xpt1) + pow2(ypt1n - ypt1)) < 1.0e-6)
		d = 0;
	else
		d = (long)
			(360.0 * atan2((ypt1n - ypt1), -(xpt1n - xpt1)) / (2.0 * PI));
	if (d < 0)
		d += 360;

	/*
	if(SamePoint==false)
	{    startx=xpt1-(xpt1-xpt2n)/2.0;		// to determine if concave box, and use
	starty=ypt1-(ypt1-ypt2n)/2.0;			// all distance weighting if so
	if(!Overlap())
	{	AllDistanceWeighting();
		return true;
	}
	else
	{	startx=xpt2-(xpt2-xpt1n)/2.0;
		starty=ypt2-(ypt2-ypt1n)/2.0;
		if(!Overlap())
		{	AllDistanceWeighting();
			return true;
		}
		 }
	}
	*/
	Cross(xpt1, ypt1, xpt2, ypt2, xpt1n, ypt1n, xpt2n, ypt2n, &xcross, &ycross);
	Cross(xcross, ycross, xmid, ymid, xpt1, ypt1, xpt1n, ypt1n, &xc1, &yc1);
	Cross(xcross, ycross, xmid, ymid, xpt2, ypt2, xpt2n, ypt2n, &xc2, &yc2);
	Cross(xpt1, ypt1, xpt1n, ypt1n, xpt2, ypt2, xpt2n, ypt2n, &xcross, &ycross);
	Cross(xcross, ycross, xmid, ymid, xpt1, ypt1, xpt2, ypt2, &xc3, &yc3);
	Cross(xcross, ycross, xmid, ymid, xpt1n, ypt1n, xpt2n, ypt2n, &xc4, &yc4);
	diffx = xmid - xc1;
	diffy = ymid - yc1;
	W1 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - xc2;
	diffy = ymid - yc2;
	W2 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - xc3;
	diffy = ymid - yc3;
	W3 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - xc4;
	diffy = ymid - yc4;
	W4 = sqrt(pow2(diffx) + pow2(diffy));
	TimeWt();
	box[0][2] = xc3;		// write points in clockwise circle
	box[0][3] = yc3;
	box[1][2] = xc2;
	box[1][3] = yc2;
	box[2][2] = xc4;
	box[2][3] = yc4;
	box[3][2] = xc1;
	box[3][3] = yc1;
	AreaWt();						// try area weighting first, but use
	if (num != -1)
		DistWt();   	 		// distance weighting if 1 leg is 0
	//x=xmid; y=ymid; t=WT; f=WfT; r=WrT, rx=WxT;
	if (WriteFile)
		SetRastData(xmid, ymid, WT, WfT, WrT, WxT, d);
	//if(WriteFile)
	//	SelectOutputs(GetRastFormat());    // outputs to raster file

	return true;
}


void Rasterize::AllDistanceWeighting()
{
	double wb1, wb2, wb3, wb4;
	double wt1, wt2, WTotal;

	xmid = startx; ymid = starty;
	diffx = xmid - box[0][0];
	diffy = ymid - box[0][1];
	W1 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - box[1][0];
	diffy = ymid - box[1][1];
	W2 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - box[2][0];
	diffy = ymid - box[2][1];
	W3 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - box[3][0];
	diffy = ymid - box[3][1];
	W4 = sqrt(pow2(diffx) + pow2(diffy));

	WTotal = W1 + W2 + W3 + W4;
	if (W1 + W2 > 0.0)
	{
		wt1 = WTotal / (W1 + W2);
		if (W3 + W4 > 0.0)
		{
			wt2 = WTotal / (W3 + W4);
			WT = wt1 + wt2;
			wt1 = wt1 / WT * LastTime;
			wt2 = wt2 / WT * CurTime;
			WT = wt1 + wt2;
		}
		else
			WT = CurTime;
	}
	else
		WT = LastTime;

	if (W1 > 0.0)
	{
		wb1 = WTotal / W1;
		if (W2 > 0.0)
		{
			wb2 = WTotal / W2;
			if (W3 > 0.0)
			{
				wb3 = WTotal / W3;
				if (W4 > 0.0)
				{
					wb4 = WTotal / W4;
					WTotal = wb1 + wb2 + wb3 + wb4;
					wb1 = wb1 / WTotal;
					wb2 = wb2 / WTotal;
					wb3 = wb3 / WTotal;
					wb4 = wb4 / WTotal;
					WrT = wb1 * box[0][4] +
						wb2 * box[1][4] +
						wb3 * box[2][4] +
						wb4 * box[3][4];
					WfT = wb1 * box[0][5] +
						wb2 * box[1][5] +
						wb3 * box[2][5] +
						wb4 * box[3][5];
					WxT = wb1 * box[0][6] +
						wb2 * box[1][6] +
						wb3 * box[2][6] +
						wb4 * box[3][6];
				}
				else
				{
					WrT = box[3][4];
					WfT = box[3][5];
					WxT = box[3][6];
				}
			}
			else
			{
				WrT = box[2][4];
				WfT = box[2][5];
				WxT = box[2][6];
			}
		}
		else
		{
			WrT = box[1][4];
			WfT = box[1][5];
			WxT = box[1][6];
		}
	}
	else
	{
		WrT = box[0][4];
		WfT = box[0][5];
		WxT = box[0][6];
	}
	if (t < -1.0)
		t *= 1.0;
	x = xmid; y = ymid; t = WT; f = WfT; r = WrT; rx = WxT;
	//SelectOutputs(pFarsite->GetRastFormat());    // outputs to raster file
	SelectOutputs(1);    // outputs to raster file
}


void Rasterize::TriangleArea(void)
{
	double Area, x0, y0, x1, y1, x2, y2, W[3];
	int numx;
	long i;

	for (num = 0; num < 4; num++)
	{
		x0 = box[num][0];
		y0 = box[num][1];
		x1 = box[num][2];
		y1 = box[num][3];
		if (num > 0)
			numx = num - 1;
		else
			numx = 3;
		x2 = box[numx][2];
		y2 = box[numx][3];
		Area = fabs(.5 * (xmid * y0 -
			x0 * ymid +
			x0 * y1 -
			x1 * y0 +
			x1 * ymid -
			xmid * y1));
		Area += fabs(.5 * (xmid * y0 -
			x0 * ymid +
			x0 * y2 -
			x2 * y0 +
			x2 * ymid -
			xmid * y2));
		switch (num)
		{
		case 0:
			W1 = Area; break;
		case 1:
			W2 = Area; break;
		case 2:
			W3 = Area; break;
		case 3:
			W4 = Area; break;
		}
	}
	if (W1 < 1e-6)
		W1 = 1e-6;
	if (W2 < 1e-6)
		W2 = 1e-6;
	if (W3 < 1e-6)
		W3 = 1e-6;
	if (W4 < 1e-6)
		W4 = 1e-6;
	WRT = W1 + W2 + W3 + W4;
	W1 = WRT / W1;
	W2 = WRT / W2;
	W3 = WRT / W3;
	W4 = WRT / W4;
	WRT = W1 + W2 + W3 + W4;
	for (i = 4; i < 7; i++)
	{
		Wf1 = W1 / WRT * box[0][i];
		Wf2 = W2 / WRT * box[1][i];
		Wf3 = W3 / WRT * box[2][i];
		Wf4 = W4 / WRT * box[3][i];
		W[i - 4] = Wf1 + Wf2 + Wf3 + Wf4;
	}
	WrT = W[0];
	WfT = W[1];
	WxT = W[2];
}


void Rasterize::TimeWt(void)
{
	double w1 = W1, w2 = W2;

	WT = w1 + w2;
	if (w1 > 0)
	{
		w1 = WT / w1;
		if (w2 > 0)
		{
			w2 = WT / w2;
			WT = w1 + w2;
			w1 = (w1 / WT) * LastTime;
			w2 = (w2 / WT) * CurTime;
			WT = w1 + w2;
		}
		else
			WT = CurTime;
	}
	else
		WT = LastTime;
}



void Rasterize::AreaWt(void)
{
	if (W1 < .0001)
		num = 3;
	else
	{
		if (W2 < .0001)
			num = 1;
		else
		{
			if (W3 < .0001)
				num = 0;
			else
			{
				if (W4 < .0001)
					num = 2;
				else
				{
					TriangleArea();
					num = -1;
				}
			}
		}
	}
}



void Rasterize::DistWt()
{
	// distance weighting instead of area weighting, because 1 leg is~0
	int numx;
	double x1 = box[num][0], y1 = box[num][1];
	double Wx1, Wx2;
	double ros1 = box[num][4], fli1 = box[num][5], rcx1 = box[num][6];
	if (num == 3)
		numx = 0;
	else
		numx = num + 1;
	double x2 = box[numx][0], y2 = box[numx][1];
	double ros2 = box[numx][4], fli2 = box[numx][5], rcx2 = box[numx][6];


	diffx = xmid - x1;
	diffy = ymid - y1;
	W1 = sqrt(pow2(diffx) + pow2(diffy));
	diffx = xmid - x2;
	diffy = ymid - y2;
	W2 = sqrt(pow2(diffx) + pow2(diffy));
	WRT = W1 + W2;
	if (W1 > 0)
	{
		W1 = WRT / W1;
		if (W2 > 0)
		{
			W2 = WRT / W2;
			WRT = W1 + W2;
			Wr1 = (W1 / WRT) * ros1;
			Wr2 = (W2 / WRT) * ros2;
			Wf1 = (W1 / WRT) * fli1;
			Wf2 = (W2 / WRT) * fli2;
			Wx1 = (W1 / WRT) * rcx1;
			Wx2 = (W2 / WRT) * rcx2;
			WrT = Wr1 + Wr2;
			WfT = Wf1 + Wf2;
			WxT = Wx1 + Wx2;
		}
		else
		{
			WrT = ros2;
			WfT = fli2;
			WxT = rcx2;
		}
	}
	else
	{
		WrT = ros1;
		WfT = fli1;
		WxT = rcx1;
	}
}



bool Rasterize::Cross(double xpt1, double ypt1, double xpt2, double ypt2,
	double xpt1n, double ypt1n, double xpt2n, double ypt2n, double* newx,
	double* newy)
{
	double xdiff1, ydiff1, xdiff2, ydiff2, ycept1, ycept2;
	double slope1, slope2, ycommon, xcommon;
	bool BadIntersection = false;

	xdiff1 = xpt2 - xpt1;
	ydiff1 = ypt2 - ypt1;
	//if (fabs(xdiff1) < 1e-9)
	if (IsTiny(xdiff1))
		xdiff1 = 0.0;
	if (xdiff1 != 0.0)
	{
		slope1 = ydiff1 / xdiff1;
		ycept1 = ypt1 - (slope1 * xpt1);
	}
	else
		slope1 = ydiff1;					 // SLOPE NON-ZERO
	xdiff2 = xpt2n - xpt1n;
	ydiff2 = ypt2n - ypt1n;
	//if (fabs(xdiff2) < 1e-9)
	if (IsTiny(xdiff2))
		xdiff2 = 0.0;
	if (xdiff2 != 0.0)
	{
		slope2 = ydiff2 / xdiff2;
		ycept2 = ypt1n - (slope2 * xpt1n);
	}
	else
		slope2 = ydiff2;					// SLOPE NON-ZERO
	if (slope1 == slope2)
	{
		if (xdiff1 != 0.0)
		{
			*newy = ymid - slope1 * xmid;				// ycross is slope of parallel lines and becomes y-intercept
			*newx = 0;
		}
		else
		{
			*newy = ypt1;				 // could also be ypt2 because slope is vertical
			*newx = xmid;
		}
	}
	else
	{
		if (xdiff1 != 0.0 && xdiff2 != 0.0)
		{
			xcommon = (ycept2 - ycept1) / (slope1 - slope2);
			*newx = xcommon;
			ycommon = ycept1 + slope1 * xcommon;
			*newy = ycommon;
			if ((*newx > xpt1 && *newx < xpt2) || (*newx<xpt1 && *newx>xpt2))
			{
				if ((*newx > xpt1n && *newx < xpt2n) ||
					(*newx<xpt1n && *newx>xpt2n))
					BadIntersection = true;
			}
		}
		else
		{
			if (xdiff1 == 0.0 && xdiff2 != 0.0)
			{
				ycommon = slope2 * xpt1 + ycept2;
				*newx = xpt1;
				*newy = ycommon;
				if ((*newy > ypt1 && *newy < ypt2) ||
					(*newy<ypt1 && *newy>ypt2))
				{
					if ((*newx > xpt1n && *newx < xpt2n) ||
						(*newx<xpt1n && *newx>xpt2n))
						BadIntersection = true;
				}
			}
			else
			{
				if (xdiff1 != 0.0 && xdiff2 == 0.0)
				{
					ycommon = slope1 * xpt1n + ycept1;
					*newx = xpt1n;
					*newy = ycommon;
					if ((*newy > ypt1n && *newy < ypt2n) ||
						(*newy<ypt1n && *newy>ypt2n))
					{
						if ((*newx > xpt1 && *newx < xpt2) ||
							(*newx<xpt1 && *newx>xpt2))
							BadIntersection = true;
					}
				}
			}
		}
	}

	return BadIntersection;
}


long Rasterize::Overlap(void)
{
	// determines if pont is inside quadrilateral
	long count = 0, count1, inside = 0;
	double Aangle = 0.0, Bangle;
	double CuumAngle = 0.0, DiffAngle;//, RefAngle=0.0;//, Zangle=0, Xangle=0;
	double Sxpt, Sypt;//, Sxptn=0.0, Syptn=0.0;//, Do=0;

	while (count < 4)
	{
		Sxpt = box[count][0];
		Sypt = box[count][1];
		Aangle = direction(Sxpt, Sypt);
		count++;
		if (Aangle != 999.9)
			break;
		else
			return 1;	// point is on one of the coordinates, and "inside"
	}
	for (count1 = count; count1 <= 4; count1++)
	{
		if (count1 == 4)
		{
			Sxpt = box[count - 1][0];
			Sypt = box[count - 1][1];
		}
		else
		{
			Sxpt = box[count1][0];
			Sypt = box[count1][1];
		}
		Bangle = direction(Sxpt, Sypt);
		if (Bangle != 999.9)
		{
			DiffAngle = Bangle - Aangle;
			if (DiffAngle > PI)
				DiffAngle = -(2.0 * PI - DiffAngle);
			else if (DiffAngle < -PI)
				DiffAngle = (2.0 * PI + DiffAngle);
			else if (DiffAngle == PI || DiffAngle == -PI)
				return 1; // point is on the segment

			CuumAngle -= DiffAngle;
			Aangle = Bangle;
		}
		else
			return 1;	// point is on one of the coordinates, and "inside"
	}
	if (fabs(CuumAngle) > 0.1)
		//	if(CuumAngle>PI)
		inside = 1;

	return inside;
}


double Rasterize::findmax(double pt1, double pt1n, double pt2, double pt2n)
{
	double ptmax;

	if (pt1 > pt1n)
		ptmax = pt1;
	else
		ptmax = pt1n;
	if (pt2 > ptmax)
		ptmax = pt2;
	if (pt2n > ptmax)
		ptmax = pt2n;

	return ptmax;
}


double Rasterize::findmin(double pt1, double pt1n, double pt2, double pt2n)
{
	double ptmin;

	if (pt1 < pt1n)
		ptmin = pt1;
	else
		ptmin = pt1n;
	if (pt2 < ptmin)
		ptmin = pt2;
	if (pt2n < ptmin)
		ptmin = pt2n;

	return ptmin;
}


void Rasterize::FMaxMin(int Coord, double* ptmax, double* ptmin)
{
	double pt2;

	*ptmax = *ptmin = pFarsite->GetPerimeter2Value(0, Coord);
	for (long i = 1; i < ExNumPts; i++)
	{
		pt2 = pFarsite->GetPerimeter2Value(i, Coord);
		if (*ptmax > pt2)
			*ptmax = pt2;
		if (*ptmin < pt2)
			*ptmin = pt2;
	}
}


void Rasterize::PFrastdata(double east, double north, double* eastf,
	double* northf, long MaxMin)
{
	// rounds off east and north to nearest raster
	long easti, northi;

	*eastf = east;
	*northf = north;
	east = (east - PFWest) / PFRes;
	north = (north - PFSouth) / PFRes;
	easti = (long) east;
	northi = (long) north;
	if (MaxMin)
	{
		if (((double) easti) < east)
			easti++;
		if (((double) northi) < north)
			northi++;
	}
	*eastf = ((double) easti) * PFRes + PFWest;
	*northf = ((double) northi) * PFRes + PFSouth;
}


void Rasterize::rastdata(double east, double north, double* eastf,
	double* northf, long MaxMin)
{
	// rounds off east and north to nearest raster
	long easti, northi;
	double RastResX, RastResY;

	pFarsite->GetRastRes(&RastResX, &RastResY);
	*eastf = east;
	*northf = north;
	east = (east - West) / RastResX;
	north = (north - South) / RastResY;
	easti = (long) east;
	northi = (long) north;
	if (MaxMin)
	{
		if (((double) easti) < east)
			easti++;
		if (((double) northi) < north)
			northi++;
	}
	*eastf = ((double) easti) * RastResX + West;
	*northf = ((double) northi) * RastResY + South;
}


