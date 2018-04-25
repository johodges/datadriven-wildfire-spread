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
//	FSXWBAR.CPP    Use of Vector Barriers to Fire Spread
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environemntal Management
//******************************************************************************


#include <stdlib.h>
#include <math.h>
#include <memory.h>
//#include "fsxw.hpp"
//#include "fsglbvar.h"
#include "fsxwbar.h"
#include "Farsite5.h"

VectorBarrier::VectorBarrier(Farsite5 *_pFarsite)
{
	Barrier = 0;                              
	BufBarrier = 0;
	NumVertices = 0;
	DistanceResolution = 0;
	pFarsite = _pFarsite;
}


VectorBarrier::~VectorBarrier()
{
	FreeBarrier();
}


bool VectorBarrier::AllocBarrier(long VertNumber)
{
	if (VertNumber <= 0)
		return false;
	else
		NumVertices = VertNumber;
	if (Barrier)
		free(Barrier);
	Barrier = (double *) malloc(VertNumber * 2 * sizeof(double));
	memset(Barrier,0x0,VertNumber*2*sizeof(double));
	if (!Barrier)
		return false;

	/*     BufBarrier=(double*) calloc(VertNumber*4, sizeof(double));
		 if(!BufBarrier)
		 	return false;
	*/
	return true;
}


void VectorBarrier::FreeBarrier()
{
	if (Barrier)
		free(Barrier);
	if (BufBarrier)
		free(BufBarrier);
	Barrier = 0;
	BufBarrier = 0;
}


void VectorBarrier::SetBarrierVertex(long VertNumber, double xpt, double ypt)
{
	VertNumber *= 2;
	Barrier[VertNumber] = xpt;
	Barrier[VertNumber + 1] = ypt;
	/*if(VertNumber>1)
	{    double xdiff, ydiff;
		xdiff=Barrier[VertNumber-2]-xpt;
		 ydiff=Barrier[VertNumber-1]-ypt;
		BarrierDistance+=sqrt(xdiff*xdiff+ydiff*ydiff);
	}*/
}


bool VectorBarrier::ReDiscretizeBarrier()
{
	long i, VertCount, NumNewVertices;
	double xdiff, ydiff, testdist;
	double xpt1, xpt2, ypt1, ypt2, newx, newy;
	double MaxDist, MaxDistSquared;

	MaxDist = pFarsite->GetDistRes() * pFarsite->MetricResolutionConvert();
	MaxDistSquared = MaxDist * MaxDist;

	do
	{
		NumNewVertices = 0;
		VertCount = 0;

		if ((BufBarrier = (double *) malloc( NumVertices * 4 * sizeof(double))) ==
			NULL)
			return false;
		memset(BufBarrier,0x0,NumVertices*4*sizeof(double));
		xpt1 = Barrier[0];
		ypt1 = Barrier[1];
		for (i = 1; i < NumVertices; i++)
		{
			xpt2 = Barrier[i * 2];
			ypt2 = Barrier[i * 2 + 1];
			xdiff = xpt1 - xpt2;
			ydiff = ypt1 - ypt2;
			testdist = xdiff * xdiff + ydiff * ydiff;
			if (testdist > MaxDistSquared)
			{
				BufBarrier[VertCount * 2] = xpt1;
				BufBarrier[VertCount * 2 + 1] = ypt1;
				VertCount++;
				newx = xpt1 - xdiff / 2.0;
				newy = ypt1 - ydiff / 2.0;
				BufBarrier[VertCount * 2] = newx;
				BufBarrier[VertCount * 2 + 1] = newy;
				VertCount++;
				NumNewVertices++;
			}
			else
			{
				BufBarrier[VertCount * 2] = xpt1;
				BufBarrier[VertCount * 2 + 1] = ypt1;
				VertCount++;
			}
			xpt1 = xpt2;
			ypt1 = ypt2;
		}
		BufBarrier[VertCount * 2] = xpt1;
		BufBarrier[VertCount * 2 + 1] = ypt1;
		VertCount++;
		free(Barrier);
		Barrier = 0;
		AllocBarrier(VertCount);
		for (i = 0; i < VertCount; i++)
			SetBarrierVertex(i, BufBarrier[i * 2], BufBarrier[i * 2 + 1]);
		free(BufBarrier);
	}
	while (NumNewVertices > 0);

	BufBarrier = (double *) malloc( NumVertices * 4 * sizeof(double));
	memset(BufBarrier,0x0,NumVertices * 4 * sizeof(double));


	return true;
}


void VectorBarrier::BufferBarrier(double DistResMult)
{
	long i, j;
	double xpt, ypt, xptl, yptl, xptn, yptn;
	double xdiffl, xdiffn, ydiffl, ydiffn, tempx, tempy;
	double xdiff, ydiff, dist, distl, distn;
	double xbuf1, ybuf1;
	double xbuf2, ybuf2;
	double A1, A2, A3, DR;

	ReDiscretizeBarrier();

	DistanceResolution = pFarsite->GetDistRes() * DistResMult /
		2.0 * pFarsite->MetricResolutionConvert();
	xpt = xptl = Barrier[0];
	ypt = yptl = Barrier[1];
	xptn = Barrier[2];
	yptn = Barrier[3];

	for (i = 1; i <= NumVertices; i++)
	{
		xdiffl = xpt - xptl;
		ydiffl = ypt - yptl;
		xdiffn = xpt - xptn;
		ydiffn = ypt - yptn;
		distl = sqrt(xdiffl * xdiffl + ydiffl * ydiffl);
		distn = sqrt(xdiffn * xdiffn + ydiffn * ydiffn);

		if (distl > 0.0 && distn > 0.0)
		{
			if (distl < distn)
			{
				tempx = xpt - xdiffn * distl / distn;
				tempy = ypt - ydiffn * distl / distn;
				xdiff = xptl - tempx;
				ydiff = yptl - tempy;
			}
			else
			{
				if (distn < distl)
				{
					tempx = xpt - xdiffl * distn / distl;
					tempy = ypt - ydiffl * distn / distl;
					xdiff = tempx - xptn;
					ydiff = tempy - yptn;
				}
				else
				{
					xdiff = xptl - xptn;
					ydiff = yptl - yptn;
				}
			}
		}
		else
		{
			xdiff = xptl - xptn;
			ydiff = yptl - yptn;
		}

		dist = sqrt(xdiff * xdiff + ydiff * ydiff);
		if (fabs(xdiffl) < 1e-9 && fabs(ydiffl) < 1e-9)
			A1 = 0.0;
		else
			A1 = atan2((xdiffl), (ydiffl));
		if (fabs(xdiff) < 1e-9 && fabs(ydiff) < 1e-9)
			A2 = 0.0;
		else
			A2 = atan2((xdiff), (ydiff));
		A3 = cos(A1) * cos(A2) + sin(A1) * sin(A2);
		if (fabs(A3) > 1e-2 && distl > 1e-9)
			DR = DistanceResolution / fabs(A3);
		else
			DR = DistanceResolution;
		if (dist == 0.0)
			dist = 1.0;
		xbuf1 = xpt - DR / dist * ydiff;	// perpendicular to xpt,ypt
		ybuf1 = ypt + DR / dist * xdiff;
		xbuf2 = xpt + DR / dist * ydiff; 	// perpendicular to xpt,ypt
		ybuf2 = ypt - DR / dist * xdiff;

		//else
		//{	xbuf1=xbuf2=xpt;
		//	ybuf1=ypt+DistanceResolution;
		//     ybuf2=ypt-DistanceResolution;
		//}
		BufBarrier[(i - 1) * 2] = xbuf2;
		BufBarrier[(i - 1) * 2 + 1] = ybuf2;
		BufBarrier[(NumVertices * 2 - i) * 2] = xbuf1;
		BufBarrier[(NumVertices * 2 - i) * 2 + 1] = ybuf1;

		if (i < NumVertices - 1)
			j = i + 1;
		else
			j = NumVertices - 1;
		xptl = xpt;
		yptl = ypt;
		xpt = xptn;
		ypt = yptn;
		xptn = Barrier[j * 2];
		yptn = Barrier[j * 2 + 1];
	}
}


bool VectorBarrier::ReBufferBarriers()
{
	// public access
	if (DistanceResolution == 0.0)
		return false;


	DiffRes = pFarsite->GetDistRes() - DistanceResolution;

	for (long i = 0; i < pFarsite->GetNumFires(); i++)
	{
		if (pFarsite->GetInout(i) == 3)
			ReBuffer(i);
	}
	DistanceResolution = pFarsite->GetDistRes();

	return true;
}


void VectorBarrier::ReBuffer(long BarrierNumber)
{
	// private access if change in distance resolution, then change barrier with from old one
	if (DiffRes > 0.0)
	{
		long i, k;
		double xpt, ypt, xptl, yptl, xptn, yptn;
		double xdiffl, xdiffn, ydiffl, ydiffn, tempx, tempy;
		double xdiff, ydiff, dist, distl, distn;
		double xbuf1, ybuf1;
		//double xbuf2, ybuf2;

		NumVertices = pFarsite->GetNumPoints(BarrierNumber);

		xptl = pFarsite->GetPerimeter1Value(BarrierNumber, NumVertices - 1, XCOORD);
		yptl = pFarsite->GetPerimeter1Value(BarrierNumber, NumVertices - 1, YCOORD);
		xpt = pFarsite->GetPerimeter1Value(BarrierNumber, 0, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(BarrierNumber, 0, YCOORD);

		for (i = 0; i < NumVertices; i++)
		{
			if (i < NumVertices - 1)
				k = i + 1;
			else
				k = 0;
			xptn = pFarsite->GetPerimeter1Value(BarrierNumber, k, XCOORD);
			yptn = pFarsite->GetPerimeter1Value(BarrierNumber, k, YCOORD);
			xdiffl = xpt - xptl;
			ydiffl = ypt - yptl;
			xdiffn = xpt - xptn;
			ydiffn = ypt - yptn;
			distl = sqrt(xdiffl * xdiffl + ydiffl * ydiffl);
			distn = sqrt(xdiffn * xdiffn + ydiffn * ydiffn);

			if (distl > 0.0 && distn > 0.0)
			{
				if (distl < distn)
				{
					tempx = xpt - xdiffn * distl / distn;
					tempy = ypt - ydiffn * distl / distn;
					xdiff = xptl - tempx;
					ydiff = yptl - tempy;
				}
				else
				{
					if (distn < distl)
					{
						tempx = xpt - xdiffl * distn / distl;
						tempy = ypt - ydiffl * distn / distl;
						xdiff = tempx - xptn;
						ydiff = tempy - yptn;
					}
					else
					{
						xdiff = xptl - xptn;
						ydiff = yptl - yptn;
					}
				}
			}
			else
			{
				xdiff = xptl - xptn;
				ydiff = yptl - yptn;
			}

			dist = sqrt(xdiff * xdiff + ydiff * ydiff);
			if (fabs(xdiff) > 0.0)
			{
				xbuf1 = xpt - DiffRes / dist * ydiff;	// perpendicular to xpt,ypt
				ybuf1 = ypt + DiffRes / dist * xdiff;
			}
			else
			{
				xbuf1 = xpt;
				ybuf1 = ypt + DiffRes;
			}
			pFarsite->SetPerimeter1(BarrierNumber, i, xbuf1, ybuf1);
			xptl = xpt;
			yptl = ypt;
			xpt = xptn;
			ypt = yptn;
		}
	}
}


bool VectorBarrier::TransferBarrier(long NumFire)
{
	long i;//, ThisFire;
	double xpt, ypt;

	if (NumVertices > 0)
	{
		//ThisFire=GetNewFires();
		pFarsite->AllocPerimeter1(NumFire, NumVertices * 2 + 1);
		pFarsite->SetInout(NumFire, 3);
		pFarsite->SetNumPoints(NumFire, NumVertices * 2);
		//IncNewFires(1);
		for (i = 0; i < NumVertices*2; i++)
		{
			xpt = BufBarrier[i * 2];
			ypt = BufBarrier[i * 2 + 1];
			pFarsite->SetPerimeter1(NumFire, i, xpt, ypt);
			pFarsite->SetFireChx(NumFire, i, 0.0, -1.0);
			pFarsite->SetReact(NumFire, i, 0.0);
		}
	}
	FreeBarrier();

	return true;
}




