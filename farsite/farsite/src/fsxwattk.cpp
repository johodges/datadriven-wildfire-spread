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
// 	FSXWATTK.CPP	Suppression Capabilities
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environmental Management
//******************************************************************************

#include "fsxwattk.h"
#include "fsx4.hpp"
//#include "ftestrc.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cctype>
#include "Farsite5.h"

const double PI = acos(-1.0);
//extern bool GroundRes_ON;

//------------------------------------------------------------------------------
//
//
//	Crew Stuff
//
//
//------------------------------------------------------------------------------

Crew::Crew()
{
	long i;

	for (i = 0; i < 51; i++)		// initialize LineProduction Rates
		LineProduction[i] = 0.0;
	memset(CrewName, 0x0, sizeof(CrewName));
	Units = 1;
	FlameLimit = 2.0;
	Compound = -1;
}


static long NumCrews = 0;
//static Crew*	crew[200];
Crew* crew[200];


//------------------------------------------------------------------------------
//
//
//	Attack Class Functions
//
//
//------------------------------------------------------------------------------


Attack::Attack(Farsite5 *_pFarsite) : ld(_pFarsite), vectorbarrier(_pFarsite)
{
	NumInsertPoints = 0;
	LineOffset = -1.0;
	LineRate = 0;
	pFarsite = _pFarsite;
	//struct timeb tb;
	//tzset();
	//::ftime(&tb);
	//srand(tb.time+tb.millitm);						// initialize random number
	//SYSTEMTIME tb;
	//GetSystemTime(&tb);
	//srand(tb.wMilliseconds);
}

Attack::~Attack()
{
}

long Attack::ProblemQuad()
{
	double testx, testy;


	if (Cross(xpt1, ypt1, xpt1n, ypt1n, xpt2, ypt2, xpt2n, ypt2n, &testx,
			&testy))
	{
		// extinguish 1n and travel directly to 2n and along next perim

		return 1;
	}
	else if (Cross(xpt1, ypt1, xpt2, ypt2, xpt1n, ypt1n, xpt2n, ypt2n, &testx,
				&testy))
	{
		if (sqrt(pow2(xpt1 - xpt1n) + pow2(ypt1 - ypt1n)) > 1e-9)
			return 1;      // extinguish 1n and travel directly to 2n and along next perim
	}
	else if (!Cross(xpt1, ypt1, xpt2n, ypt2n, xpt1n, ypt1n, xpt2, ypt2,
				&testx, &testy))
	{
		if (sqrt(pow2(xpt1 - xpt1n) + pow2(ypt1 - ypt1n)) > 1e-9)
			return 2;      // go to atleast 1n and 2n and along next perim
	}


	return 0;
}


long Attack::FindCurrentPoint()
{
	long i;
	double dist, mindist = -1.0;

	if (attack->IndirectLine1[2] == -1.0)	// first time for attack
	{
		attack->IndirectLine1[2] = 1;   // change value for subsequent attacks
		for (i = 0; i < pFarsite->GetNumPoints(attack->FireNumber); i++)
		{
			xpt1 = pFarsite->GetPerimeter2Value(i, XCOORD);
			ypt1 = pFarsite->GetPerimeter2Value(i, YCOORD);
			// use existing xpt1,ypt1 coords from last timestep for comparison
			if (xpt1 == attack->IndirectLine1[0] &&
				ypt1 == attack->IndirectLine1[1])
			{
				attack->CurrentPoint = i;
				break;
			}
		}
	}
	else
	{
		for (i = 0; i < pFarsite->GetNumPoints(attack->FireNumber); i++)
		{
			if (pFarsite->GetPerimeter2Value(i, FLIVAL) < 0.0)
			{
				xpt1 = pFarsite->GetPerimeter2Value(i, XCOORD);
				ypt1 = pFarsite->GetPerimeter2Value(i, YCOORD);
				// use existing xpt1,ypt1 coords from last timestep for comparison
				dist = pow2(xpt1 - attack->IndirectLine1[0]) +
					pow2(ypt1 -
						attack->IndirectLine1[1]);
				if (dist == 0.0) //xpt1==attack->IndirectLine1[0] && ypt1==attack->IndirectLine1[1])
				{
					attack->CurrentPoint = i;

					break;
				}
				else if (mindist < 0.0 || dist < mindist)
				{
					mindist = dist;
					attack->CurrentPoint = i;  // use closest point
				}
			}
		}
		if (i == pFarsite->GetNumPoints(attack->FireNumber)) // couldn't find dead point
		{
			if (attack->CurrentPoint >= i)
				attack->CurrentPoint = 0;
		}
	}

	return attack->CurrentPoint;
}


void Attack::CalcChordArcRatio(double LastX, double LastY)
{
	// computes ratio of circular arc to chord distance for given angle
	//  used to correct linear approximation to line construction within quadrilateral
	double PerpDist1 = 0, PerpDist2 = 0, PerpDiff, TanDist1, TanDist2;
	double RefAngle, Angle, AddDist1, AddDist2, DiagDist, TestDist;
	double SinDiff, CosDiff, SpreadDist1, SpreadDist2;

	ChordArcRatio = 1.0;			// Attack:: Data Member

	return;

	TanDist1 = sqrt(pow2(xpt1 - xpt2) + pow2(ypt1 - ypt2));
	if (TanDist1 == 0.0)
		return;
	RefAngle = atan2(ypt1 - ypt2, xpt1 - xpt2);

	SpreadDist1 = sqrt(pow2(xpt1n - xpt1) + pow2(ypt1n - ypt1));
	if (SpreadDist1 > 0.0)
	{
		Angle = atan2(ypt1 - ypt1n, xpt1 - xpt1n);
		SinDiff = sin(RefAngle) * cos(Angle) - cos(RefAngle) * sin(Angle);
		PerpDist1 = fabs(SinDiff * SpreadDist1);
	}
	CosDiff = cos(RefAngle) * cos(Angle) + sin(RefAngle) * sin(Angle);
	RefAngle += PI;

	SpreadDist2 = sqrt(pow2(LastX - xpt2) + pow2(LastY - ypt2));
	if (SpreadDist2 > 0.0)
	{
		Angle = atan2(ypt2 - LastY, xpt2 - LastX);
		SinDiff = sin(RefAngle) * cos(Angle) - cos(RefAngle) * sin(Angle);
		PerpDist2 = fabs(SinDiff * SpreadDist2);
	}
	if (SpreadDist2 < PerpDist2)
		PerpDist2 = SpreadDist2;
	PerpDiff = PerpDist2 - PerpDist1;

	if (fabs(PerpDiff) < 0.01)
		return;

	/*
	BaseDist=TanDist1;
	// correct for perpendicular projection 2
	AddDist2=sqrt(pow2(SpreadDist2)-pow2(PerpDist2));
	if(SpreadDist2>PerpDist2)   	   // eliminate potential precision problem
	{	DiagDist=pow2(xpt2n-xpt1)+pow2(ypt2n-ypt1); 	   // diagonal distance across quadrilateral
		 TestDist=pow2(TanDist1)+pow2(PerpDist2);   		// hypotenuse
		 if(TestDist<DiagDist)	// must decrease PerpDist1
			  BaseDist+=AddDist2;
		 else					// must increase BaseDist AND increase PerpDist1
			  BaseDist-=AddDist2;
	}
	// correct for perpendicular projection 1
	AddDist1=sqrt(pow2(SpreadDist1)-pow2(PerpDist1));
	if(SpreadDist1>PerpDist1)	// could go either way
	{    DiagDist=pow2(xpt1n-xpt2)+pow2(ypt1n-ypt2);		// diagonal distance across quadrilateral
		 TestDist=pow2(TanDist1)+pow2(PerpDist1);   		// hypotenuse
		 if(TestDist<DiagDist)
	{//	BaseDist+=AddDist1;
	   	dRdX=PerpDiff/(BaseDist+AddDist1);
		 	PerpDist1+=dRdX*AddDist1;		// increase perp height at start point (x1, y1)
		 }
		 else
		 {//	BaseDist-=AddDist1;
	   	dRdX=PerpDiff/(BaseDist-AddDist1);
		 	PerpDist1-=dRdX*AddDist1;		// reduce perp height at start point (x1, y1)
		 }
	}
	PerpDiff=PerpDist2-PerpDist1;	// corrected
	if(fabs(PerpDiff)<0.01)
		return;
	if(BaseDist<0.0)		// problem quadrangle
		return;
	double PDFsq, PDSsq, Lsq, Line, LDiff, LHyp, g, inc, Drop;
	PDSsq=pow2(PerpDist1);
	PDFsq=pow2(PerpDiff);
	Lsq=pow2(LastX-xpt1)+pow2(LastY-ypt1);
	//TanDist2=sqrt(Lsq);
	if(PerpDiff>0.0)
	   g=inc=BaseDist*0.1;
	else
	   g=inc=-BaseDist*0.1;
	do
	{	LHyp=pow2(BaseDist+g)+PDSsq+PDFsq+pow2(g);
		LDiff=Lsq-LHyp;
		 if(PerpDiff>0.0)
		 {    if(LDiff<0.0)
			{	g-=inc;
			 	inc/=2.0;
		 		 g+=inc;
			}
			 else
		 		g+=inc;
		 }
		 else
		 {    if(LDiff>0.0)
			{	g-=inc;
			 	inc/=2.0;
		 		 g+=inc;
			}
			 else
		 		g+=inc;
		 }
	} while(fabs(LDiff)>1e-6);
	Drop=sqrt(pow2(g)+PDFsq);
	Angle=asin(Drop/sqrt(Lsq));
	if(Angle>0.0)
			ChordArcRatio=fabs(sin(Angle)/Angle); // ratio of arc dist to chord dist
	*/

	// modify PerpDiff depending on angle of SpreadDist1
	if (SpreadDist1 > PerpDist1)
	{
		TanDist2 = sqrt(pow2(LastX - xpt1n) + pow2(LastY - ypt1n));
		if (TanDist2 > 0.0)
		{
			AddDist1 = fabs(CosDiff * PerpDiff /
						sqrt(pow2(TanDist2) - pow2(PerpDiff)));
			DiagDist = pow2(xpt1n - xpt2) + pow2(ypt1n - ypt2); 	   // diagonal distance across quadrilateral
			TestDist = pow2(TanDist1) + pow2(PerpDist1);		   // hypotenuse
			if (TestDist > DiagDist)
				PerpDiff += AddDist1;
			else
				PerpDiff -= AddDist1;
		}
	}

	if (PerpDiff > 0.0)
	{
		// modify TanDist1 depending on angle of SpreadDist2
		AddDist2 = sqrt(pow2(SpreadDist2) - pow2(PerpDist2));  // additional distance
		DiagDist = pow2(LastX - xpt1) + pow2(LastY - ypt1); 	   // diagonal distance across quadrilateral
		TestDist = pow2(TanDist1) + pow2(PerpDist2);		   // hypotenuse
		if (TestDist > DiagDist)
			TanDist1 -= AddDist2;
		else
			TanDist1 += AddDist2;
		if (TanDist1 > 0.0)
		{
			//-----------------------------------

			//Angle=fabs(atan(PerpDiff/TanDist1));     // or.....

			//-------------------------------------

			if (PerpDist1 < PerpDist2)
			{
				TestDist = sqrt(pow2(TanDist1) + DiagDist);
				Angle = fabs(acos(PerpDist1 / TestDist) -
							acos(PerpDist2 / TestDist));
			}
			else
			{
				TestDist = sqrt(pow2(TanDist1) + pow2(PerpDist1));
				Angle = fabs(acos(PerpDist2 / TestDist) -
							acos(PerpDist1 / TestDist));
			}

			if (Angle > 0.0)
				ChordArcRatio = fabs(sin(Angle) / Angle); // ratio of arc dist to chord dist
		}
	}
}

long Attack::CheckOverlapAttacks(long* Attacks, double TimeStep)
{
	long i;

	for (i = 0; i < pFarsite->GetNumAttacks(); i++)
	{
		// 1. find out how many direct attacks are on same fire
		// 2. find nearest point on fire for each fire
		// 3. estimate end point after next time step
		//	  (must move past fli=-1 points
		// 4. if line routes overlap at all, alloc Attacks array with indices to Attacks
		// 5. return number of attacks that will overlap
	}

	return 0;
}


bool Attack::DirectAttack(AttackData* atk, double TimeStep)
{
	// perform direct attack
	double HorizPerimDist, DiagonalDist, SurfDist, LineRate, SuppressionDist;
	double FlameLength, FlankDist, LineDist = 0.0;
	double ratio, PreviousSpreadRatio, CuumSpreadDist;
	double newx, newy;
	long i, j, k, posit, FuelType;
	//	bool ContinueAttack=true;
	bool Replace1n = false;
	bool Replace2n = false;
	long PointToInsert;
	double ReplaceX1n, ReplaceY1n, ReplaceR1n, ReplaceF1n, ReplaceC1n;
	double ReplaceX2n, ReplaceY2n, ReplaceR2n, ReplaceF2n, ReplaceC2n;
	//double DistRes=GetDistRes()*2.0;
	double ExtendXn, ExtendYn; //ExtendX, ExtendY;
	double newdistn, totaldistn, NewPt2Dist, OriginalPt2Dist;
	double OriginalTimeStep;
	celldata tempcell;
	crowndata tempcrown;
	grounddata tempground;

	attack = atk;
	FindCurrentPoint();
	PreviousSpreadRatio = CuumSpreadDist = 0.0;
	pFarsite->GetPerimeter2(attack->CurrentPoint, &xpt1, &ypt1, &ros1, &fli1, &rcx1);
	xpt1n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint,
				XCOORD);
	ypt1n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint,
				YCOORD);
	ros1n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint,
				ROSVAL);
	fli1n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint,
				FLIVAL);
	rcx1n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint,
				RCXVAL);

	OriginalTimeStep = TimeStep;
	while (TimeStep > 0.0)
	{
		if (attack->Reverse)
		{
			attack->NextPoint = attack->CurrentPoint - 1;
			if (attack->NextPoint < 0)
				attack->NextPoint = pFarsite->GetNumPoints(attack->FireNumber) - 1;
		}
		else
		{
			attack->NextPoint = attack->CurrentPoint + 1;
			if (attack->NextPoint > pFarsite->GetNumPoints(attack->FireNumber) - 1)
				attack->NextPoint = 0;
		}
		pFarsite->GetPerimeter2(attack->NextPoint, &xpt2, &ypt2, &ros2, &fli2, &rcx2);
		fli2n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint,
					FLIVAL);
		if (fli2 < 0.0 || fli2n < 0.0)	// can't suppress extinguished fire
		{
			if (fli1n == 0.0)
				fli1n = 0.001;
			if (fli1n > 0.0)
				pFarsite->SetFireChx(attack->FireNumber, attack->CurrentPoint, ros1n,
					-fli1n);
			//attack->CurrentPoint=-1;
			k = attack->NextPoint;
			for (i = 1; i < pFarsite->GetNumPoints(attack->FireNumber); i++)
			{
				if (attack->Reverse)
				{
					j = attack->CurrentPoint - i;
					if (j < 0)
						j = pFarsite->GetNumPoints(attack->FireNumber) - i;
				}
				else
				{
					j = i + attack->CurrentPoint;
					if (j > pFarsite->GetNumPoints(attack->FireNumber) - 1)
						j -= pFarsite->GetNumPoints(attack->FireNumber);
				}
				fli1 = pFarsite->GetPerimeter2Value(j, FLIVAL);
				if (fli1 >= 0.0)
				{
					fli1n = pFarsite->GetPerimeter1Value(attack->FireNumber, j, FLIVAL);
					if (fli1n >= 0.0)
					{
						attack->CurrentPoint = k;
						pFarsite->GetPerimeter2(attack->CurrentPoint, &xpt1, &ypt1,
							&ros1, &fli1, &rcx1);
						xpt1n = pFarsite->GetPerimeter1Value(attack->FireNumber,
									attack->CurrentPoint, XCOORD);
						ypt1n = pFarsite->GetPerimeter1Value(attack->FireNumber,
									attack->CurrentPoint, YCOORD);
						ros1n = pFarsite->GetPerimeter1Value(attack->FireNumber,
									attack->CurrentPoint, ROSVAL);
						fli1n = pFarsite->GetPerimeter1Value(attack->FireNumber,
									attack->CurrentPoint, FLIVAL);
						rcx1n = pFarsite->GetPerimeter1Value(attack->FireNumber,
									attack->CurrentPoint, RCXVAL);
						if (fli1n == 0.0)
							fli1n = 0.001;
						break;
					}
				}
				k = j;
			}
			if (i == pFarsite->GetNumPoints(attack->FireNumber))	// searched fire and no unburned points found
				return false;
			else
				continue;
		}
		xpt2n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint,
					XCOORD);
		ypt2n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint,
					YCOORD);
		ros2n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint,
					ROSVAL);
		fli2n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint,
					FLIVAL);
		rcx2n = pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint,
					RCXVAL);

		OriginalPt2Dist = sqrt(pow2(xpt2n - xpt2) + pow2(ypt2n - ypt2));
		if (PreviousSpreadRatio > 0.0)  // form new pro-rated xpt2
		{
			CuumSpreadDist = OriginalPt2Dist * PreviousSpreadRatio;
			xpt2 = xpt2 - (xpt2 - xpt2n) * PreviousSpreadRatio;
			ypt2 = ypt2 - (ypt2 - ypt2n) * PreviousSpreadRatio;
			PreviousSpreadRatio = 0.0;
		}
		pFarsite->CellData(xpt1, ypt1, tempcell, tempcrown, tempground, &posit);
		FuelType = pFarsite->GetFuelConversion(tempcell.f);	//CheckAltFuels(tempcell.f, xpt1, ypt1)); no longer valid for version 4.0
		if (FuelType > 0)
		{
			LineRate = crew[attack->CrewNum]->LineProduction[FuelType - 1] * pFarsite->MetricResolutionConvert();
			if (ros2 == 0.0)   // create temporary false point for calcs
			{
				xpt2n = xpt2 - (xpt1 - xpt1n);
				ypt2n = ypt2 - (ypt1 - ypt1n);
			}
		}
		else
		{
			attack->CurrentPoint = attack->NextPoint;
			xpt1 = xpt2;
			ypt1 = ypt2;
			xpt1n = xpt2n;
			ypt1n = ypt2n;
			ros1 = ros2;
			ros1n = ros2n;
			fli1 = fli2;
			fli1n = fli2n;
			rcx1 = rcx2;
			rcx1n = rcx2n;
			continue;
		}
		if (LineDist < 1e-9)//==0.0)
		{
			LineDist = LineRate * TimeStep;
			if (LineDist < 0.001)      // don't prosecute attack with such small line construction
				return true;
			attack->LineBuilt += (LineDist / pFarsite->MetricResolutionConvert());
		}
		DiagonalDist = sqrt(pow2(xpt1 - xpt2n) + pow2(ypt1 - ypt2n));
		HorizPerimDist = sqrt(pow2(xpt1 - xpt2) + pow2(ypt1 - ypt2));
		ConvertLandData(tempcell.s, tempcell.a);
		SurfDist = CalculateSlopeDist(xpt1, ypt1, xpt2n, ypt2n);
		LineDist *= DiagonalDist / SurfDist;
		CalcChordArcRatio(xpt2n, ypt2n);
		LineDist *= ChordArcRatio;
		FlameLength = 0.0775 * pow(fabs(fli1), 0.46);
		FlankDist = sqrt(pow2(xpt1n - xpt1) + pow2(ypt1n - ypt1));

		if ((posit = ProblemQuad()) > 0)
		{
			if (posit == 1)
			{
				HorizPerimDist = sqrt(pow2(xpt1 - xpt2n) + pow2(ypt1 - ypt2n));
				if (HorizPerimDist > 0.0)
				{
					if (LineDist >= HorizPerimDist)
					{
						//TimeStep*=(1.0-HorizPerimDist/LineDist);
						//LineDist-=HorizPerimDist;
						TimeStep = 0.0;
						ReplaceX2n = newx = xpt2n;
						ReplaceY2n = newy = ypt2n;
						ReplaceR2n = ros2n;
						ReplaceF2n = fli2n;
						ReplaceC2n = rcx2n;
						Replace2n = true;
						PointToInsert = 0;
					}
					else if (LineDist < HorizPerimDist)
					{
						newx = xpt1 -
							(xpt1 - xpt2n) * LineDist /
							HorizPerimDist;
						newy = ypt1 -
							(ypt1 - ypt2n) * LineDist /
							HorizPerimDist;
						PointToInsert = 3;
						TimeStep = 0.0;
					}
				}
				else
				{
					newx = xpt2n;
					newy = ypt2n;
					PointToInsert = 0;
					TimeStep = 0.0;
				}
				ReplaceX1n = xpt1;//n;
				ReplaceY1n = ypt1;//n;
				ReplaceR1n = ros1;//n;
				ReplaceF1n = fli1;//n;
				ReplaceC1n = rcx1;//n;
				//Replace1n=true;   // incase starting with this point
			}
			else
			{
				//HorizPerimDist=sqrt(pow2(xpt1-xpt1n)+pow2(ypt1-ypt1n));
				//HorizPerimDist+=sqrt(pow2(xpt1n-xpt2n)+pow2(ypt1n-ypt2n));
				//if(LineDist<HorizPerimDist)
				//{	LineDist=HorizPerimDist;		// make at least that much line
				TimeStep = 0.0;
				//}
				//TimeStep*=(1.0-HorizPerimDist/LineDist);
				//LineDist-=HorizPerimDist;
				//if(fli1<0)
				//     pFarsite->SetPerimeter2(attack->CurrentPoint, xpt1, ypt1, ros1, -fli1);
				ReplaceX1n = xpt1n;
				ReplaceY1n = ypt1n;
				ReplaceR1n = ros1n;
				ReplaceF1n = fli1n;
				ReplaceC1n = rcx1n;
				ReplaceX2n = newx = xpt2n;
				ReplaceY2n = newy = ypt2n;
				ReplaceR2n = ros2n;
				ReplaceF2n = fli2n;
				ReplaceC2n = rcx2n;
				Replace1n = true;   	   //really means it
				Replace2n = true;
				PointToInsert = 0;
			}
		}
		else if (LineDist < DiagonalDist)
		{
			if (xpt1 == xpt1n && ypt1 == ypt1n)
			{
				xpt1n = xpt1 - (xpt2 - xpt2n);	// form new pt1n from dims of pt2n
				ypt1n = ypt1 - (ypt2 - ypt2n);
			}
			if (FlameLength < crew[attack->CrewNum]->FlameLimit)
			{
				if (IterateIntersection(LineDist, xpt2n, ypt2n, xpt1, ypt1,
						xpt1n, ypt1n, &newx, &newy))
					PointToInsert = 1;
				else
					PointToInsert = 2;
			}
			else	// flank only;
			{
				ratio = LineDist / FlankDist;
				if (ratio > 1.0)
					ratio = 1.0;
				newx = xpt1 - (xpt1 - xpt1n) * ratio;
				newy = ypt1 - (ypt1 - ypt1n) * ratio;
				PointToInsert = 2;
				pFarsite->WriteAttackLog(attack, 3, 0, 0);
			}
			ReplaceX1n = xpt1;
			ReplaceY1n = ypt1;
			ReplaceR1n = ros1;
			ReplaceF1n = fli1;
			ReplaceC1n = rcx1;
			//Replace1n=true;     // in case starting with this point
			TimeStep = 0.0;
		}
		else if (LineDist > DiagonalDist)
		{
			if (FlameLength < crew[attack->CrewNum]->FlameLimit)
			{
				if (LineDist > HorizPerimDist)
					ratio = LineDist * 2.0;
				else
					ratio = HorizPerimDist * 2.0;     // extend linedist out from x1,y1 past x2,y2 by double LineDist
				//ExtendX=xpt1-(xpt1-xpt2)*(ratio/HorizPerimDist);
				//ExtendY=ypt1-(ypt1-ypt2)*(ratio/HorizPerimDist);
				ExtendXn = xpt1n - (xpt1n - xpt2n) * (ratio / HorizPerimDist);
				ExtendYn = ypt1n - (ypt1n - ypt2n) * (ratio / HorizPerimDist);
				//  			 	xdiff=xpt2-xpt2n;   			  // make sure its long enough
				//  			 	ydiff=ypt2-ypt2n;
				//  				ExtendXn=ExtendX-xdiff;//-(xpt1n-xpt2n)/(ratio/HorizPerimDist);
				//  				ExtendYn=ExtendY-ydiff;//-(ypt1n-ypt2n)/(ratio/HorizPerimDist);

				if (IterateIntersection(LineDist, ExtendXn, ExtendYn, xpt1,
						ypt1, xpt2n, ypt2n, &newx, &newy))
				{
					Cross(newx, newy, xpt1, ypt1, xpt2, ypt2, xpt2n, ypt2n,
						&ReplaceX2n, &ReplaceY2n);
					newx = ReplaceX2n;
					newy = ReplaceY2n;
				}
				else
				{
					ReplaceX2n = newx;
					ReplaceY2n = newy;
				}
				Replace2n = true;
				ReplaceR2n = ros2n;
				ReplaceF2n = fli2n;
				ReplaceC2n = rcx2n;
				SuppressionDist = sqrt(pow2(ReplaceX2n - xpt1) +
									pow2(ReplaceY2n - ypt1));
				if (SuppressionDist > LineDist)
					SuppressionDist = LineDist;
				TimeStep *= (1.0 - SuppressionDist / LineDist);
				LineDist -= SuppressionDist;
				PointToInsert = 0;
			}
			else	// flank only;
			{
				//pFarsite->SetPerimeter2(attack->CurrentPoint, xpt1, ypt1, ros1, -fli1);
				ratio = LineDist / FlankDist;
				if (ratio > 1.0)
					ratio = 1.0;
				newx = xpt1 - (xpt1 - xpt1n) * ratio;
				newy = ypt1 - (ypt1 - ypt1n) * ratio;
				TimeStep = 0.0;
				PointToInsert = 2;
				LineDist = 0.0;
			}
			ReplaceX1n = xpt1;
			ReplaceY1n = ypt1;
			ReplaceR1n = ros1;
			ReplaceF1n = fli1;
			ReplaceC1n = rcx1;
			//Replace1n=true;
		}
		else	// equal
		{
			ReplaceX1n = xpt1;
			ReplaceY1n = ypt1;
			ReplaceR1n = ros1;
			ReplaceF1n = fli1;
			ReplaceX2n = newx = xpt2n;
			ReplaceY2n = newy = ypt2n;
			ReplaceR2n = ros2n;
			ReplaceF2n = fli2n;
			ReplaceC2n = rcx2n;
			//Replace1n=true;
			Replace2n = true;
			PointToInsert = 0;
			TimeStep = 0.0;
		}
		if (ReplaceF1n > 0.0)
			ReplaceF1n *= -1.0;
		if (attack->AttackTime > 0.0)
		{
			pFarsite->SetPerimeter1(attack->FireNumber, attack->CurrentPoint,
				ReplaceX1n, ReplaceY1n);
			if (ReplaceF1n == 0.0)
				ReplaceF1n = -0.001;
			pFarsite->SetFireChx(attack->FireNumber, attack->CurrentPoint, ReplaceR1n,
				ReplaceF1n);
			pFarsite->SetReact(attack->FireNumber, attack->CurrentPoint, ReplaceC1n);
		}
		else
		{
			if (!Replace1n)
			{
				if (fli1n<0.0 && fli1>0.0)
					ReplaceF1n = fabs(ReplaceF1n);
			}
			else
				Replace1n = false;
			pFarsite->SetPerimeter2(attack->CurrentPoint, ReplaceX1n, ReplaceY1n,
				ReplaceR1n, ReplaceF1n, rcx1);
		}
		switch (PointToInsert)
		{
		case 0:
			break;
		case 1:
			newdistn = sqrt(pow2(newx - xpt1n) + pow2(newy - ypt1n));
			totaldistn = sqrt(pow2(xpt2n - xpt1n) + pow2(ypt2n - ypt1n));
			ros1n = ros1n * (1.0 - newdistn / totaldistn) +
				ros2n * newdistn /
				totaldistn;
			fli1n = fabs(fli1n) * (1.0 - newdistn / totaldistn) +
				fabs(fli2n) * newdistn /
				totaldistn;
			rcx1n = rcx1n * (1.0 - newdistn / totaldistn) +
				rcx2n * newdistn /
				totaldistn;
			InsertPerimeterPoint(newx, newy, ros1n, -fli1n, rcx1n, 1);
			break;
		case 2:
			newdistn = sqrt(pow2(newx - xpt1) + pow2(newy - ypt1));
			totaldistn = sqrt(pow2(xpt1n - xpt1) + pow2(ypt1n - ypt1));
			ros1n = ros1 * (1.0 - newdistn / totaldistn) +
				ros1n * newdistn /
				totaldistn;
			fli1n = fabs(fli1) * (1.0 - newdistn / totaldistn) +
				fabs(fli1n) * newdistn /
				totaldistn;
			rcx1n = rcx1 * (1.0 - newdistn / totaldistn) +
				rcx1n * newdistn /
				totaldistn;
			InsertPerimeterPoint(newx, newy, ros1n, -fli1n, rcx1n, 2);
			break;
		case 3:
			newdistn = sqrt(pow2(newx - xpt1) + pow2(newy - ypt1));
			totaldistn = sqrt(pow2(xpt2n - xpt1) + pow2(ypt2n - ypt1));
			ros1n = ros1 * (1.0 - newdistn / totaldistn) +
				ros2n * newdistn /
				totaldistn;
			fli1n = fabs(fli1) * (1.0 - newdistn / totaldistn) +
				fabs(fli2n) * newdistn /
				totaldistn;
			rcx1n = rcx1 * (1.0 - newdistn / totaldistn) +
				rcx2n * newdistn /
				totaldistn;
			InsertPerimeterPoint(newx, newy, ros1n, -fli1n, rcx1n, 3);
			break;
		}
		if (Replace2n)
		{
			if (ReplaceF2n == 0.0)
				ReplaceF2n = 0.001;
			if (ReplaceF2n > 0.0)
				ReplaceF2n *= -1.0;
			pFarsite->SetPerimeter1(attack->FireNumber, attack->NextPoint, ReplaceX2n,
				ReplaceY2n);
			pFarsite->SetFireChx(attack->FireNumber, attack->NextPoint, ReplaceR2n,
				ReplaceF2n);
			pFarsite->SetReact(attack->FireNumber, attack->NextPoint, ReplaceC2n);
		}
		if (TimeStep > 0.0)
		{
			PointToInsert = 0;
			Replace2n = false;
			attack->CurrentPoint = attack->NextPoint;
			NewPt2Dist = sqrt(pow2(newx - xpt2) + pow2(newy - ypt2));
			if (OriginalPt2Dist < 1e-9)//==0.0)
				PreviousSpreadRatio = 1.0;
			else
			{
				CuumSpreadDist += NewPt2Dist;
				PreviousSpreadRatio = CuumSpreadDist / OriginalPt2Dist;    //xdiff
				//PreviousSpreadRatio=ydiff/xdiff;
				if (PreviousSpreadRatio > 1.0)
					PreviousSpreadRatio = 1.0;
			}
			xpt1n = xpt2n;
			ypt1n = ypt2n;
			ros1 = ros2;
			ros1n = ros2n;
			fli1 = fli2;
			fli1n = fli2n;
			rcx1 = rcx2;
			rcx1n = rcx2n;
		}
		xpt1 = attack->IndirectLine1[0] = newx;
		ypt1 = attack->IndirectLine1[1] = newy;
		/*
			if(attack->Reverse)
			{    fli1=pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint, FLIVAL);
				 if(fli1>0.0)
			  {	ros1=pFarsite->GetPerimeter1Value(attack->FireNumber, attack->CurrentPoint, ROSVAL);
				pFarsite->SetFireChx(attack->FireNumber, attack->NextPoint, ros1, fli1);
				}
			}
			else
			{	fli1=pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint, FLIVAL);
			   if(fli1>0.0)
			  {	ros1=pFarsite->GetPerimeter1Value(attack->FireNumber, attack->NextPoint, ROSVAL);
				pFarsite->SetFireChx(attack->FireNumber, attack->NextPoint, ros1, fli1);
				}
			}
			*/
	}
	//ReplaceDeadPoints();
	attack->AttackTime += OriginalTimeStep;

	return true;
}


bool Attack::IndirectAttack(AttackData* atk, double TimeStep)
{
	// perform indirect attack
	long i, posit, NewPoint, PointCount;
	double ratio, LineRate, HorizDist, SurfDist;//, SegmentDist;
	double LineDist = 0.0;
	double DistRes = pFarsite->GetDistRes();
	celldata tempcell;
	crowndata tempcrown;
	grounddata tempground;
	bool CrewIsActive = true;
	bool DistanceCheck, NewEndPoint;
	long StartPoint;
	long FuelType;

	attack = atk;
	StartPoint = attack->CurrentPoint;
	if (StartPoint > attack->NumPoints - 1)
		StartPoint = attack->CurrentPoint = attack->NumPoints - 1;
	NewEndPoint = false;
	DistanceCheck = false;
	xpt1 = attack->IndirectLine1[attack->CurrentPoint * 2];
	ypt1 = attack->IndirectLine1[attack->CurrentPoint * 2 + 1];
	if (attack->Burnout)
	{
		if (attack->BurnLine[0] == -1)
			attack->BurnLine[0] = StartPoint;
	}
	attack->AttackTime += TimeStep;
	while (TimeStep > 0.0)    // perform distance checking in line building rate
	{
		if (!DistanceCheck)
		{
			attack->CurrentPoint++;
			if (attack->CurrentPoint < attack->NumPoints)
				DistanceCheck = false;
			else
			{
				attack->CurrentPoint--;
				CrewIsActive = false;
				break;
			}
		}
		else
			DistanceCheck = false;
		xpt2 = attack->IndirectLine1[attack->CurrentPoint * 2];
		ypt2 = attack->IndirectLine1[attack->CurrentPoint * 2 + 1];
		pFarsite->CellData(xpt1, ypt1, tempcell, tempcrown, tempground, &posit);
		FuelType = pFarsite->GetFuelConversion(tempcell.f); //CheckAltFuels(tempcell.f, xpt1, ypt1));
		if (FuelType > 0)
			LineRate = crew[attack->CrewNum]->LineProduction[FuelType - 1] * pFarsite->MetricResolutionConvert();
		else
			LineRate = 0.0;
		if (LineRate == 0.0)
		{
			LineDist = sqrt(pow2(xpt1 - xpt2) + pow2(ypt1 - ypt2));
			if (LineDist > DistRes)
			{
				xpt1 = xpt1 - (xpt1 - xpt2) * DistRes / LineDist;
				ypt1 = ypt1 - (ypt1 - ypt2) * DistRes / LineDist;
				LineDist = 0.0;
				DistanceCheck = true;
			}
			else
			{
				xpt1 = xpt2;
				ypt1 = ypt2;
			}

			continue;
		}
		HorizDist = sqrt(pow2(xpt1 - xpt2) + pow2(ypt1 - ypt2));
		if (LineDist < 1e-9)//==0.0)
		{
			LineDist = LineRate * TimeStep;
			attack->LineBuilt += (LineDist / pFarsite->MetricResolutionConvert());
		}
		ConvertLandData(tempcell.s, tempcell.a);
		SurfDist = CalculateSlopeDist(xpt1, ypt1, xpt2, ypt2);
		if (SurfDist > 0)
			LineDist *= HorizDist / SurfDist;
		else
			continue;

		if (LineDist < HorizDist)
		{
			if (LineDist > DistRes)
			{
				ratio = DistRes / LineDist;
				TimeStep *= (1.0 - ratio);
				LineDist -= DistRes;
				ratio = DistRes / HorizDist;
				DistanceCheck = true;
				LineDist *= SurfDist / HorizDist;			// reset original linedist ratio
				//NumInsertPoints++;
			}
			else
			{
				ratio = LineDist / HorizDist;
				TimeStep = 0.0;
				NewEndPoint = true;
				NumInsertPoints++;
			}
			xpt1 = xpt1 - (xpt1 - xpt2) * ratio;
			ypt1 = ypt1 - (ypt1 - ypt2) * ratio;
		}
		else if (LineDist > HorizDist)
		{
			xpt1 = xpt2;
			ypt1 = ypt2;
			TimeStep *= (1.0 - HorizDist / LineDist);
			LineDist -= HorizDist;
			LineDist *= SurfDist / HorizDist;				// reset original linedist ratio
		}
		else
			TimeStep = 0.0;
	}
	posit = attack->CurrentPoint - StartPoint;
	if (NewEndPoint) 	// insert new mid point on original line
	{
		posit = 1;
		if ((attack->IndirectLine2 = new double[(attack->NumPoints + 1) * 2]) !=
			NULL)//(double *) calloc((attack->NumPoints+1)*2, sizeof(double)))!=NULL)
		{
			NewPoint = 0;
			for (i = 0; i < attack->NumPoints; i++)
			{
				attack->IndirectLine2[(i + NewPoint) * 2] = attack->IndirectLine1[i * 2];
				attack->IndirectLine2[(i + NewPoint) * 2 + 1] = attack->IndirectLine1[i * 2 + 1];
				if (i == attack->CurrentPoint - 1)
				{
					NewPoint = 1;
					attack->IndirectLine2[(i + NewPoint) * 2] = xpt1;
					attack->IndirectLine2[(i + NewPoint) * 2 + 1] = ypt1;
				}
			}
			attack->NumPoints++;
			delete[] attack->IndirectLine1; //free(attack->IndirectLine1);
			attack->IndirectLine1 = new double[(attack->NumPoints + 1) * 2];//(double *) calloc(attack->NumPoints*2, sizeof(double));
			for (i = 0; i < attack->NumPoints; i++)
			{
				attack->IndirectLine1[i * 2] = attack->IndirectLine2[i * 2];
				attack->IndirectLine1[i * 2 + 1] = attack->IndirectLine2[i * 2 + 1];
			}
			delete[] attack->IndirectLine2; //free(attack->IndirectLine2);
			attack->IndirectLine2 = 0;
			//attack->CurrentPoint+=NumInsertPoints;
		}
	}
	if (posit > 0)
	{
		vectorbarrier.AllocBarrier(attack->CurrentPoint + 1);
		for (PointCount = 0;
			PointCount <= attack->CurrentPoint;
			PointCount++)//(i=StartPoint; i<PointNum; i++)
		{
			vectorbarrier.SetBarrierVertex(PointCount,
							attack->IndirectLine1[PointCount * 2],
							attack->IndirectLine1[PointCount * 2 + 1]);
		}
		vectorbarrier.BufferBarrier(1.2);
		if (attack->FireNumber > -1)  // if not first time through
		{
			pFarsite->FreePerimeter1(attack->FireNumber);
			//pFarsite->SetInout(attack->FireNumber, 0);
			//pFarsite->SetNumPoints(attack->FireNumber, 0);
			//IncSkipFires(1);
		}
		else
		{
			attack->FireNumber = pFarsite->GetNewFires();
			pFarsite->IncNewFires(1);
		}
		vectorbarrier.TransferBarrier(attack->FireNumber);
		BoundingBox();
		if (attack->Burnout)
		{
			attack->BurnLine[1] = attack->CurrentPoint;
			BurnOut();
			if (!CrewIsActive)
			{
				if (attack->BurnLine[1] > attack->BurnLine[0])
				{
					CrewIsActive = true;
					attack->BurnDelay = 0;
				}
			}
		}
	}
	else if (!CrewIsActive)
	{
		if (attack->Burnout)
		{
			attack->BurnLine[1] = attack->CurrentPoint;
			BurnOut();
		}
	}

	return CrewIsActive;
}


bool Attack::ParallelAttack(AttackData* atk, double TimeStep)
{
	// perform direct attack
	bool CrewIsActive = true, WriteLine;
	long NumHullPts, HullPt, TempNum, PointCount;
	double* Hull1, * Hull2;

	double HorizPerimDist, DiagonalDist, SurfDist, LineRate, SuppressionDist;
	double FlankDist, LineDist = 0.0;
	double ratio, PreviousSpreadRatio, CuumSpreadDist;
	double newx, newy;
	long i, posit, FuelType;
	double ReplaceX2n, ReplaceY2n;
	double xdiff, ydiff, ExtendXn, ExtendYn;  //ExtendX, ExtendY;
	double OriginalTimeStep, NewPt2Dist, OriginalPt2Dist;
	celldata tempcell;
	crowndata tempcrown;
	grounddata tempground;
	attack = atk;

	//alloc double the number of points so that density control can rediscretize
	if ((Hull2 = new double[pFarsite->GetNumPoints(attack->FireNumber) * 6]) == NULL) //(double *) calloc(pFarsite->GetNumPoints(attack->FireNumber)*6, sizeof(double)))==NULL)
		return false;
	GetConvexHull(Hull2, &NumHullPts);
	if ((Hull1 = new double[pFarsite->GetNumPoints(attack->FireNumber) * 4]) == NULL) //(double *) calloc(pFarsite->GetNumPoints(attack->FireNumber)*4, sizeof(double)))==NULL)
	{
		if (Hull2)
			delete[] Hull2;//free(Hull2);

		return false;
	}
	ExpandConvexHulls(Hull1, Hull2, NumHullPts);
	if (!attack->Burnout)
		HullDensityControl(Hull2, Hull1, &NumHullPts);
	FindHullPoint(Hull2, NumHullPts, &HullPt);
	if (!AllocParallelLine(Hull1, NumHullPts, TimeStep))
	{
		if (Hull1)
			delete[] Hull1;//free(Hull1);
		if (Hull2)
			delete[] Hull2;//free(Hull2);

		return false;
	}

	if (attack->Burnout && attack->BurnLine[0] == -1)
	{
		if (attack->NumPoints > 1)
			attack->BurnLine[0] = attack->NumPoints - 1;
		else
			attack->BurnLine[0] = 0;
	}
	PreviousSpreadRatio = CuumSpreadDist = 0.0;
	xpt1 = attack->IndirectLine2[0];
	ypt1 = attack->IndirectLine2[1];
	//xpt1n=Hull2[attack->CurrentPoint*3];
	//ypt1n=Hull2[attack->CurrentPoint*3+1];
	fli1n = fli1 = Hull2[attack->CurrentPoint * 3 + 2];
	OriginalTimeStep = TimeStep;
	while (TimeStep > 0.0)
	{
		if (attack->Reverse)
		{
			attack->NextPoint = attack->CurrentPoint - 1;
			if (attack->NextPoint < 0)
				attack->NextPoint = NumHullPts - 1;
		}
		else
		{
			attack->NextPoint = attack->CurrentPoint + 1;
			if (attack->NextPoint > NumHullPts - 1)
				attack->NextPoint = 0;
		}
		xpt2 = Hull1[attack->NextPoint * 2];
		ypt2 = Hull1[attack->NextPoint * 2 + 1];
		fli2 = Hull2[attack->NextPoint * 3 + 2];
		if (fli2 <= 0.0)	// can't suppress extinguished fire
		{
			for (i = 0; i < NumHullPts; i++)
			{
				fli2n = Hull2[i * 3 + 2];  //pFarsite->GetPerimeter2Value(j, FLIVAL);
				if (fli2n > 0.0)
					break;
			}
			if (i == NumHullPts)				// searched fire and no unburned points found
			{
				CrewIsActive = false;
				OriginalTimeStep -= TimeStep;  	// set correct time elapsed

				break;
			}
		}
		xpt2n = Hull2[attack->NextPoint * 3];
		ypt2n = Hull2[attack->NextPoint * 3 + 1];
		fli2n = Hull2[attack->NextPoint * 3 + 2];

		if (OriginalTimeStep == TimeStep)		// first time through
		{
			xdiff = xpt2 - xpt2n;
			ydiff = ypt2 - ypt2n;
			xpt1n = xpt1 - xdiff;
			ypt1n = ypt1 - ydiff;
		}

		/*
			if(OriginalTimeStep==TimeStep)		// first time through
			{	xpt1n=Hull1[attack->CurrentPoint*2];
				ypt1n=Hull1[attack->CurrentPoint*2+1];
				 OrigDist=sqrt(pow2(xpt2-xpt1n)+pow2(ypt2-ypt1n));
				 PropDist=sqrt(pow2(xpt2-xpt1)+pow2(ypt2-ypt1));
			  xpt1n=Hull2[attack->CurrentPoint*3];
			  ypt1n=Hull2[attack->CurrentPoint*3+1];
				 if(OrigDist>0.0)
			  {	PropDist/=OrigDist;
					xpt1n=xpt2n-(xpt2n-xpt1n)*sqrt(pow2(xpt2n-xpt1n)+pow2(ypt2n-ypt1n))*PropDist;
				   ypt1n=ypt2n-(ypt2n-ypt1n)*sqrt(pow2(xpt2n-xpt1n)+pow2(ypt2n-ypt1n))*PropDist;
				 }
			}
			*/

		OriginalPt2Dist = sqrt(pow2(xpt2 - xpt2n) + pow2(ypt2 - ypt2n));
		if (PreviousSpreadRatio > 0.0)  // form new pro-rated xpt2
		{
			CuumSpreadDist = OriginalPt2Dist * PreviousSpreadRatio;
			xpt2 = xpt2 - (xpt2 - xpt2n) * PreviousSpreadRatio;
			ypt2 = ypt2 - (ypt2 - ypt2n) * PreviousSpreadRatio;
			PreviousSpreadRatio = 0.0;
		}
		pFarsite->CellData(xpt1, ypt1, tempcell, tempcrown, tempground, &posit);
		FuelType = pFarsite->GetFuelConversion(tempcell.f);//CheckAltFuels(tempcell.f, xpt1, ypt1));
		ros2 = pFarsite->GetPerimeter2Value(labs((long)fli2), (long)ROSVAL);
		if (FuelType > 0)
		{
			LineRate = crew[attack->CrewNum]->LineProduction[FuelType - 1] * pFarsite->MetricResolutionConvert();
			if (ros2 == 0.0)   // create temporary false point for calcs
			{
				xpt2n = xpt2 - (xpt1 - xpt1n);
				ypt2n = ypt2 - (ypt1 - ypt1n);
			}
		}
		else
		{
			attack->CurrentPoint = attack->NextPoint;
			xdiff = xpt1 - xpt2;
			ydiff = ypt1 - ypt2;
			if (pow2(xdiff) + pow2(ydiff) > 0.0)
			{
				WriteLine = true;
				if (attack->NumPoints >= NumInsertPoints)
					WriteLine = AllocParallelLine(Hull1, NumHullPts,
									OriginalTimeStep);
				if (WriteLine)
				{
					attack->IndirectLine1[attack->NumPoints * 2] = xpt2;
					attack->IndirectLine1[attack->NumPoints * 2 + 1] = ypt2;
					attack->NumPoints++;
				}
			}
			xpt1 = xpt2;
			ypt1 = ypt2;
			xpt1n = xpt2n;
			ypt1n = ypt2n;
			ros1 = ros2;
			ros1n = ros2n;
			fli1 = fli2;
			fli1n = fli2n;

			continue;
		}
		if (LineDist < 1e-9)//==0.0)
		{
			LineDist = LineRate * TimeStep;
			if (LineDist < 0.001)      // don't prosecute attack with such small line construction
			{
				if (Hull1)
					delete[] Hull1;//free(Hull1);
				if (Hull2)
					delete[] Hull1;//free(Hull2);
				return true;
			}
			attack->LineBuilt += (LineDist / pFarsite->MetricResolutionConvert());
		}
		DiagonalDist = sqrt(pow2(xpt1 - xpt2n) + pow2(ypt1 - ypt2n));
		HorizPerimDist = sqrt(pow2(xpt1 - xpt2) + pow2(ypt1 - ypt2));
		ConvertLandData(tempcell.s, tempcell.a);
		SurfDist = CalculateSlopeDist(xpt1, ypt1, xpt2n, ypt2n);
		LineDist *= DiagonalDist / SurfDist;
		CalcChordArcRatio(xpt2n, ypt2n);
		LineDist *= ChordArcRatio;
		//FlameLength=0.0775*pow(fabs(fli1), 0.46);
		FlankDist = sqrt(pow2(xpt1n - xpt1) + pow2(ypt1n - ypt1));

		if (fli2 <= 0.0)	// on extinguished point
		{
			if (SurfDist - LineDist > 1e-3)
			{
				xpt2n = xpt1 - (xpt1 - xpt2n) * LineDist / SurfDist;
				ypt2n = ypt1 - (ypt1 - ypt2n) * LineDist / SurfDist;
				//xpt2n=xpt2n-(xpt2n-xpt1)*LineDist/SurfDist;
				//ypt2n=ypt2n-(ypt2n-ypt1)*LineDist/SurfDist;
				TimeStep = 0.0;
			}
			else
			{
				TimeStep *= (1.0 - SurfDist / LineDist);
				LineDist -= SurfDist;
				attack->CurrentPoint = attack->NextPoint;
				if (TimeStep < 1e-6)
					TimeStep = 0.0;
			}
			xdiff = xpt1 - xpt2n;
			ydiff = ypt1 - ypt2n;
			if (pow2(xdiff) + pow2(ydiff) > 0.0)
			{
				WriteLine = true;
				if (attack->NumPoints >= NumInsertPoints)
					WriteLine = AllocParallelLine(Hull1, NumHullPts,
									OriginalTimeStep);
				if (WriteLine)
				{
					attack->IndirectLine1[attack->NumPoints * 2] = xpt2n;
					attack->IndirectLine1[attack->NumPoints * 2 + 1] = ypt2n;
					attack->IndirectLine2[0] = xpt2n;
					attack->IndirectLine2[1] = ypt2n;
					attack->NumPoints++;
				}
			}
			xpt1 = xpt2n;
			ypt1 = ypt2n;
			xpt1n = xpt2n;
			ypt1n = ypt2n;
			ros1 = ros2;
			ros1n = ros2n;
			fli1 = fli2;
			fli1n = fli2n;

			continue;
		}

		if ((posit = ProblemQuad()) > 0)
		{
			if (posit == 1)
			{
				HorizPerimDist = sqrt(pow2(xpt1 - xpt2n) + pow2(ypt1 - ypt2n));
				if (HorizPerimDist > 0.0)
				{
					if (LineDist >= HorizPerimDist)
					{
						TimeStep = 0.0;
						ReplaceX2n = newx = xpt2n;
						ReplaceY2n = newy = ypt2n;
					}
					else if (LineDist < HorizPerimDist)
					{
						newx = xpt1 -
							(xpt1 - xpt2n) * LineDist /
							HorizPerimDist;
						newy = ypt1 -
							(ypt1 - ypt2n) * LineDist /
							HorizPerimDist;
						TimeStep = 0.0;
					}
				}
				else
				{
					newx = xpt2n;
					newy = ypt2n;
					TimeStep = 0.0;
				}
				ReplaceX2n = newx;
				ReplaceY2n = newy;
			}
			else
			{
				TimeStep = 0.0;
				ReplaceX2n = newx = xpt2n;
				ReplaceY2n = newy = ypt2n;
			}
		}
		else if (LineDist < DiagonalDist)
		{
			if (xpt1 == xpt1n && ypt1 == ypt1n)
			{
				xpt1n = xpt1 - (xpt2 - xpt2n);	// form new pt1n from dims of pt2n
				ypt1n = ypt1 - (ypt2 - ypt2n);
			}
			IterateIntersection(LineDist, xpt2n, ypt2n, xpt1, ypt1, xpt1n,
				ypt1n, &newx, &newy);
			ReplaceX2n = newx;
			ReplaceY2n = newy;
			TimeStep = 0.0;
		}
		else if (LineDist > DiagonalDist)
		{
			if (LineDist > HorizPerimDist)
				ratio = LineDist * 2.0;
			else
				ratio = HorizPerimDist * 2.0;     // extend linedist out from x1,y1 past x2,y2 by double LineDist
			//     		ExtendX=xpt1-(xpt1-xpt2)*(ratio/HorizPerimDist);
			//			ExtendY=ypt1-(ypt1-ypt2)*(ratio/HorizPerimDist);
			ExtendXn = xpt1n - (xpt1n - xpt2n) * (ratio / HorizPerimDist);
			ExtendYn = ypt1n - (ypt1n - ypt2n) * (ratio / HorizPerimDist);
			//  		   xdiff=xpt2-xpt2n;				 // make sure its long enough
			//  		   ydiff=ypt2-ypt2n;
			//  		   ExtendXn=ExtendX-xdiff-(xpt1n-xpt2n)/(ratio/HorizPerimDist);
			//  		   ExtendYn=ExtendY-ydiff-(ypt1n-ypt2n)/(ratio/HorizPerimDist);

			if (IterateIntersection(LineDist, ExtendXn, ExtendYn, xpt1, ypt1,
					xpt2n, ypt2n, &newx, &newy))
			{
				if (Cross(newx, newy, xpt1, ypt1, xpt2, ypt2, xpt2n, ypt2n,
						&ReplaceX2n, &ReplaceY2n))
				{
					newx = ReplaceX2n;
					newy = ReplaceY2n;
				}
				else
				{
					ReplaceX2n = newx = xpt2n;
					ReplaceY2n = newy = ypt2n;
				}
			}
			else
			{
				ReplaceX2n = newx;
				ReplaceY2n = newy;
			}
			SuppressionDist = sqrt(pow2(ReplaceX2n - xpt1) +
								pow2(ReplaceY2n - ypt1));
			if (SuppressionDist > LineDist)
				SuppressionDist = LineDist;
			TimeStep *= (1.0 - SuppressionDist / LineDist);
			LineDist -= SuppressionDist;
		}
		else	// equal
		{
			ReplaceX2n = newx = xpt2n;
			ReplaceY2n = newy = ypt2n;
			TimeStep = 0.0;
		}
		xdiff = xpt1 - ReplaceX2n;
		ydiff = ypt1 - ReplaceY2n;
		SuppressionDist = sqrt(pow2(xdiff) + pow2(ydiff));
		if (SuppressionDist >= FlankDist / 2.0)
		{
			WriteLine = true;
			if (attack->NumPoints >= NumInsertPoints)
				WriteLine = AllocParallelLine(Hull1, NumHullPts,
								OriginalTimeStep);
			if (WriteLine)
			{
				attack->IndirectLine1[attack->NumPoints * 2] = ReplaceX2n;
				attack->IndirectLine1[attack->NumPoints * 2 + 1] = ReplaceY2n;
				attack->NumPoints++;
			}
		}
		if (TimeStep > 0.0)
		{
			attack->CurrentPoint = attack->NextPoint;
			NewPt2Dist = sqrt(pow2(newx - xpt2) + pow2(newy - ypt2));
			if (OriginalPt2Dist < 1e-9)//==0.0)
				PreviousSpreadRatio = 1.0;
			else
			{
				CuumSpreadDist += NewPt2Dist;
				PreviousSpreadRatio = CuumSpreadDist / OriginalPt2Dist;
				if (PreviousSpreadRatio > 1.0)
					PreviousSpreadRatio = 1.0;
			}
			xpt1n = xpt2n;
			ypt1n = ypt2n;
			ros1 = ros2;
			ros1n = ros2n;
			fli1 = fli2;
			fli1n = fli2n;
		}
		xpt1 = attack->IndirectLine2[0] = newx;
		ypt1 = attack->IndirectLine2[1] = newy;
	}
	if (attack->NumPoints > 1)
	{
		vectorbarrier.AllocBarrier(attack->NumPoints);
		for (PointCount = 0;
			PointCount < attack->NumPoints;
			PointCount++)//(i=StartPoint; i<PointNum; i++)
		{
			vectorbarrier.SetBarrierVertex(PointCount,
							attack->IndirectLine1[PointCount * 2],
							attack->IndirectLine1[PointCount * 2 + 1]);
		}
		vectorbarrier.BufferBarrier(1.2);
		if (attack->LineNumber > -1)  // if not first time through
			pFarsite->FreePerimeter1(attack->LineNumber);
		else
		{
			attack->LineNumber = pFarsite->GetNewFires();
			pFarsite->IncNewFires(1);
		}
		vectorbarrier.TransferBarrier(attack->LineNumber);
		TempNum = attack->FireNumber;
		attack->FireNumber = attack->LineNumber;
		BoundingBox();
		attack->FireNumber = TempNum;
		//--------------------------------------------------------------
		//	Burnout Now Conducted in FarProc3 like Indirect Attack
		//--------------------------------------------------------------

		if (attack->Burnout)
		{
			attack->BurnLine[1] = attack->NumPoints - 1;
			//	BurnOut();
			if (!CrewIsActive)
			{
				if (attack->BurnLine[1] > attack->BurnLine[0])
				{
					//CrewIsActive=true;
					attack->BurnDelay = 0;
				}
			}
		}
	}
	if (!CrewIsActive)
	{
		if (attack->Burnout)
		{
			attack->BurnLine[1] = attack->NumPoints - 1;
			BurnOut();
		}
	}
	attack->AttackTime += OriginalTimeStep;
	// save attack->CurrentPoint from fire perimeter so that it can be located on revision
	attack->CurrentPoint = attack->NumPoints - 1;
	if (Hull1)
		delete[] Hull1;//free(Hull1);
	if (Hull2)
		delete[] Hull2;//free(Hull2);

	return CrewIsActive;
}


void Attack::ConvertLandData(short slope, short aspect)
{
	ld.SlopeConvert(slope);
	ld.AspectConvert(aspect);
}


/*
double Attack::CalculateSlopeDist(double x1, double y1, double x2, double y2)
{// calculates horizontal distance from slope, aspect and coords of 2 pts
	 double Rh;
	 double Angle, Theta;
	 double xt, yt;
	 double PI=3.14159265358979324;
	 double slopef=((double) ld.ld.slope/180.0)*PI;
	 double HorizDist;

	 xt=x1-x2;
	 yt=y1-y2;
	 HorizDist=sqrt(pow2(xt)+pow2(yt));

	 Angle=atan2(-xt, -yt);
	 if(Angle<0.0)
		  Angle+=2.0*PI;
	Theta=ld.ld.aspectf-Angle;

	 if(fabs(Theta)<0.0000000001)     // precision problem
		  Theta=0;

	 Rh=sqrt(pow2(HorizDist*cos(Theta)/cos(slopef))+pow2(HorizDist*sin(Theta)));

	 return Rh;
}
*/


double Attack::CalculateSlopeDist(double x1, double y1, double x2, double y2)
{
	// calculates horizontal distance from slope, aspect and coords of 2 pts

	double Theta;
	double Angle;
	double hx, hy, ds;
	//double XSign=1.0, YSign=1.0;
	double Mx = -1.0, My = -1.0;
	//     double PI=3.14159265358979324;
	double slopef = ((double) ld.ld.slope / 180.0) * PI;
	double xt, yt, HorizDist;

	xt = x1 - x2;
	yt = y1 - y2;
	HorizDist = sqrt(pow2(xt) + pow2(yt));

	if (ld.ld.slope == 0 || HorizDist < 1e-9)//==0.0)
		return HorizDist;

	// calculate angles on horizontal coordinate system

	Angle = atan2(-xt, -yt);
	if (Angle < 0.0)
		Angle += 2.0 * PI;
	Theta = ld.ld.aspectf - Angle;

	// calculate angles on local surface coordinate system
	Theta = -atan2(cos(Theta) / cos(slopef), sin(Theta)) + PI / 2.0;
	//Angle=aspectf-Theta;
	//if(Angle<0.0)
	//     Angle+=2.0*PI;

	// calculate distance difference
	ds = HorizDist * cos(Theta) * (1.0 - cos(slopef));

	// if(quadrants 1 and 3
	if ((ld.ld.aspectf <= PI / 2.0) ||
		(ld.ld.aspectf > PI && ld.ld.aspectf <= 3.0 * PI / 2.0))
	{
		if (cos(Angle) != 0.0 || Angle == PI)
		{
			if (tan(Angle) < 0.0)
			{
				if (tan(Angle) < tan(ld.ld.aspectf - PI / 2.0))
					My = 1.0;
				else
					Mx = 1.0;
			}
		}
	}
	else // if quadrants 2 and 4
	{
		if (cos(Angle) != 0.0)
		{
			if (tan(Angle) > 0.0 && Angle != PI)
			{
				if (tan(Angle) > tan(ld.ld.aspectf - PI / 2.0))
					My = 1.0;
				else
					Mx = 1.0;
			}
		}
	}

	// transform x and y components to local surface coordinates
	hx = fabs(xt) - Mx * fabs(ds * sin(ld.ld.aspectf));
	hy = fabs(yt) - My * fabs(ds * cos(ld.ld.aspectf));

	//determine signs of horiz x and y components
	//if(xt<0.0)
	//    XSign=-1.0;
	//if(yt<0.0)
	//     YSign=-1.0;

	// substitute surface for horizontal components
	//xt=hx*XSign;
	//yt=hy*YSign;

	return sqrt(pow2(hx) + pow2(hy));
}


bool Attack::IndirectSegment()
{
	return true;
}


void Attack::BoundingBox()
{
	double xpt, ypt, Xlo, Xhi, Ylo, Yhi;
	long NumFire = attack->FireNumber;//pFarsite->GetNewFires()-1;
	long NumPoint = pFarsite->GetNumPoints(NumFire);

	Xlo = Xhi = pFarsite->GetPerimeter1Value(NumFire, 0, XCOORD);
	Ylo = Yhi = pFarsite->GetPerimeter1Value(NumFire, 0, YCOORD);
	for (int i = 1; i < NumPoint; i++)
	{
		xpt = pFarsite->GetPerimeter1Value(NumFire, i, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(NumFire, i, YCOORD);
		if (xpt < Xlo)
			Xlo = xpt;
		if (xpt > Xhi)
			Xhi = xpt;
		if (ypt < Ylo)
			Ylo = ypt;
		if (ypt > Yhi)
			Yhi = ypt;
	}
	pFarsite->SetPerimeter1(NumFire, NumPoint, Xlo, Xhi);
	pFarsite->SetFireChx(NumFire, NumPoint, Ylo, Yhi);
}


void Attack::InsertPerimeterPoint(double newx, double newy, double nros,
	double nfli, double nrcx, long InsertType)
{
	long i, j, TestPoint;
	long numpts = pFarsite->GetNumPoints(attack->FireNumber);
	long numfire = attack->FireNumber;
	double xpt, ypt, ros, fli, rcx;
	double ProportionalDist, x2, y2;

	if (nfli == 0.0)
		nfli = -0.001;
	if (attack->Reverse)
		TestPoint = attack->CurrentPoint;
	else
		TestPoint = attack->NextPoint;
	j = 0;
	attack->IndirectLine2 = new double[(numpts + 1) * NUMDATA];//(double *) calloc((numpts+1)*NUMDATA, sizeof(double));
	switch (InsertType)
	{
	case 1:
		ProportionalDist = sqrt(pow2(newx - xpt1n) + pow2(newy - ypt1n)) /
			sqrt(pow2(xpt2n - xpt1n) +
				pow2(ypt2n -
					ypt1n));
		x2 = xpt1 - (xpt1 - xpt2) * ProportionalDist;
		y2 = ypt1 - (ypt1 - ypt2) * ProportionalDist;
		break;
	case 2:
		//ProportionalDist=sqrt(pow2(newx-xpt1)+pow2(newy-ypt1))/
		// 	 sqrt(pow2(xpt1n-xpt1)+pow2(ypt1n-ypt1));
		x2 = xpt1;
		y2 = ypt1;
		break;
	case 3:
		//ProportionalDist=sqrt(pow2(newx-xpt1)+pow2(newy-ypt1))/
		// 	 sqrt(pow2(xpt2n-xpt1)+pow2(ypt2n-ypt1));
		x2 = xpt1;
		y2 = ypt1;
		break;
	}
	for (i = 0; i < numpts; i++)
	{
		pFarsite->GetPerimeter2(i, &xpt, &ypt, &ros, &fli, &rcx);
		if (i == TestPoint)
		{
			attack->IndirectLine2[i * 4] = x2;
			attack->IndirectLine2[i * 4 + 1] = y2;
			attack->IndirectLine2[i * 4 + 2] = ros;
			//if(fli<0.0)
			//	attack->IndirectLine2[i*4+3]=-fli;
			//else
			//	attack->IndirectLine2[i*4+3]=fli;
			attack->IndirectLine2[i * 4 + 3] = fabs(fli);
			attack->IndirectLine2[i * 4 + 4] = rcx;
			j = 1;
		}
		attack->IndirectLine2[(i + j) * 4] = xpt;
		attack->IndirectLine2[(i + j) * 4 + 1] = ypt;
		attack->IndirectLine2[(i + j) * 4 + 2] = ros;
		attack->IndirectLine2[(i + j) * 4 + 3] = fli;
		attack->IndirectLine2[(i + j) * 4 + 4] = rcx;
	}
	if (j == 0)
	{
		attack->IndirectLine2[i * 4] = x2;
		attack->IndirectLine2[i * 4 + 1] = y2;
		attack->IndirectLine2[i * 4 + 2] = ros;
		attack->IndirectLine2[i * 4 + 3] = fabs(fli);
		attack->IndirectLine2[i * 4 + 4] = rcx;
	}
	//FreePerimeter2();
	numpts++;
	pFarsite->AllocPerimeter2(numpts);
	for (i = 0; i < numpts; i++)
	{
		xpt = attack->IndirectLine2[i * 4];
		ypt = attack->IndirectLine2[i * 4 + 1];
		ros = attack->IndirectLine2[i * 4 + 2];
		fli = attack->IndirectLine2[i * 4 + 3];
		rcx = attack->IndirectLine2[i * 4 + 4];
		pFarsite->SetPerimeter2(i, xpt, ypt, ros, fli, rcx);
	}
	j = 0;
	numpts--;
	for (i = 0; i < numpts; i++)
	{
		if (i == TestPoint)
		{
			attack->IndirectLine2[i * 4] = newx;
			attack->IndirectLine2[i * 4 + 1] = newy;
			attack->IndirectLine2[i * 4 + 2] = nros;
			attack->IndirectLine2[i * 4 + 3] = -fabs(nfli);  // always negative
			attack->IndirectLine2[i * 4 + 4] = nfli;  // always negative
			j = 1;
		}
		attack->IndirectLine2[(i + j) * 4] = pFarsite->GetPerimeter1Value(numfire, i,
												XCOORD);
		attack->IndirectLine2[(i + j) * 4 + 1] = pFarsite->GetPerimeter1Value(numfire,
													i, YCOORD);
		attack->IndirectLine2[(i + j) * 4 + 2] = pFarsite->GetPerimeter1Value(numfire,
													i, ROSVAL);
		attack->IndirectLine2[(i + j) * 4 + 3] = pFarsite->GetPerimeter1Value(numfire,
													i, FLIVAL);
		attack->IndirectLine2[(i + j) * 4 + 4] = pFarsite->GetPerimeter1Value(numfire,
													i, RCXVAL);
	}
	if (j == 0)
	{
		attack->IndirectLine2[i * 4] = newx;
		attack->IndirectLine2[i * 4 + 1] = newy;
		attack->IndirectLine2[i * 4 + 2] = nros;
		attack->IndirectLine2[i * 4 + 3] = -fabs(nfli);
		attack->IndirectLine2[i * 4 + 4] = nrcx;
	}
	numpts++;
	pFarsite->FreePerimeter1(numfire);
	pFarsite->AllocPerimeter1(numfire, numpts + 1);
	pFarsite->SetNumPoints(numfire, numpts);
	for (i = 0; i < numpts; i++)
	{
		xpt = attack->IndirectLine2[i * 4];
		ypt = attack->IndirectLine2[i * 4 + 1];
		ros = attack->IndirectLine2[i * 4 + 2];
		fli = attack->IndirectLine2[i * 4 + 3];
		rcx = attack->IndirectLine2[i * 4 + 4];
		pFarsite->SetPerimeter1(numfire, i, xpt, ypt);
		pFarsite->SetFireChx(numfire, i, ros, fli);
		pFarsite->SetReact(numfire, i, rcx);
	}
	delete[] attack->IndirectLine2;//free(attack->IndirectLine2);
	attack->IndirectLine2 = 0;
	/*
		 if(attack->Reverse)
		{    attack->CurrentPoint--;
		   	if(attack->CurrentPoint<0)
			  	attack->CurrentPoint=numpts-1;
		 }
		 else
		 {	attack->CurrentPoint++;
			if(attack->CurrentPoint>numpts-1)
				attack->CurrentPoint=0;
		 }
	*/
}


bool Attack::IterateIntersection(double LineDist, double extendx,
	double extendy, double xstart, double ystart, double xend, double yend,
	double* newx, double* newy)
{
	bool NeedNewPoint = true;
	double CloseDist = sqrt(pow2(extendx - xstart) + pow2(extendy - ystart));
	double TotalDist = sqrt(pow2(extendx - xend) + pow2(extendy - yend));
	double StartDist = sqrt(pow2(xstart - xend) + pow2(ystart - yend));
	double NewTotalDist;
	double SlideDist;
	double NewLineDist = LineDist;
	double oldx, oldy, inc;
	double Tolerance;
	long NumIter = 0;

	Tolerance = fabs(CloseDist - NewLineDist) / 10.0;
	if (Tolerance > 0.1)
		Tolerance = 0.1;
	else if (Tolerance < 1e-9)//==0.0)
		Tolerance = 0.1;

	oldx = extendx;
	oldy = extendy;
	inc = TotalDist / 4.0;
	NewTotalDist = TotalDist - inc * 2.0;
	// reconvert line distance to slopeing distance
	LineDist *= CalculateSlopeDist(xstart, ystart, extendx, extendy) /
		CloseDist;
	while (fabs(CloseDist - NewLineDist) > Tolerance)
	{
		NumIter++;
		*newx = oldx - (oldx - xend) * (1.0 - NewTotalDist / TotalDist);
		*newy = oldy - (oldy - yend) * (1.0 - NewTotalDist / TotalDist);
		CloseDist = sqrt(pow2(*newx - xstart) + pow2(*newy - ystart));
		SlideDist = sqrt(pow2(*newx - xend) + pow2(*newy - yend));
		NewLineDist = LineDist * CloseDist / CalculateSlopeDist(xstart,
												ystart, *newx, *newy);
		//CalcChordArcRatio(*newx, *newy);
		//NewLineDist*=ChordArcRatio			;
		if (CloseDist == NewLineDist)
			break;
		else
		{
			if (CloseDist < NewLineDist)
			{
				NewTotalDist += inc;
				*newx = oldx;
				*newy = oldy;
			}
			else
			{
				NewTotalDist -= inc;
				oldx = *newx;
				oldy = *newy;
				inc /= 2.0;
			}
		}
		if (SlideDist < Tolerance)
		{
			if (NewLineDist <= StartDist)
			{
				*newx = xend - (xend - xstart) * (1.0 - LineDist / StartDist);
				*newy = yend - (yend - ystart) * (1.0 - LineDist / StartDist);
				NeedNewPoint = false;
				break;
			}
		}
		if (inc < 0.001)
		{
			*newx = xend;
			*newy = yend;
			NeedNewPoint = false;
			break;
		}
	}
	if (NumIter == 0) // compute *newx,*newy if fails condition in while loop 1st time
	{
		*newx = oldx - (oldx - xend) * (1.0 - NewTotalDist / TotalDist);
		*newy = oldy - (oldy - yend) * (1.0 - NewTotalDist / TotalDist);
	}

	return NeedNewPoint;
}


void Attack::ConductBurnout(AttackData* atk)
{
	attack = atk;

	if (attack->Burnout)
	{
		if (attack->BurnLine[0] == -1)
		{
			if (attack->NumPoints > 1)
				attack->BurnLine[0] = attack->NumPoints - 1;
			else
				attack->BurnLine[0] = 0;
		}
		attack->BurnLine[1] = attack->NumPoints - 1;
		BurnOut();
	}
}


void Attack::BurnOut()
{
	//     bool EndEarly=false;
	long i, j, k;
	long NewFire, NewFirstPoint;
	long LagPoint, NewPoints;
	long start = attack->BurnLine[0];
	long end = attack->BurnLine[1];
	double Delay;
	double xpt, ypt, xptl, yptl, xptn, yptn;
	double xbuf1, ybuf1;
	double xbuf2, ybuf2;
	double DistRes, OldOffset = LineOffset;

	Delay = (double) attack->BurnDelay * pFarsite->MetricResolutionConvert();
	if (end < 0)
		return;

	do
	{
		LineOffset = ((double) (rand() % 22490) + 7500.0) / 15000.0;	// largest is 2.0m, smallest is 0.5
	}
	while (fabs(LineOffset - OldOffset) < 0.5);

	if (start == end)
		start -= 1;
	if (start < 0)
		start = 0;
	NewFirstPoint = end;//-1;
	//if(NewFirstPoint<0)
	//	NewFirstPoint=0;

	i = 1;
	LagPoint = 0;
	xpt2 = xpt2n = attack->IndirectLine1[end * 2];
	ypt2 = ypt2n = attack->IndirectLine1[end * 2 + 1];
	do
	{
		xpt1 = attack->IndirectLine1[(end - i) * 2];
		ypt1 = attack->IndirectLine1[(end - i) * 2 + 1];
		LineRate = sqrt(pow2(xpt2n - xpt1) + pow2(ypt2n - ypt1));
		if (LineRate > Delay)
		{
			xpt2n = xpt2n - (xpt2n - xpt1) * Delay / LineRate;
			ypt2n = ypt2n - (ypt2n - ypt1) * Delay / LineRate;
			LagPoint = i;
			NewFirstPoint = end - i - 1;
			if (NewFirstPoint < 0)
				NewFirstPoint = 0;
			break;
		}
		else
		{
			Delay -= LineRate;
			xpt2n = xpt1;
			ypt2n = ypt1;
		}
		i++;
	}
	while ((end - i) >= start);

	if (Delay > LineRate)
		return;
	end -= LagPoint;
	if (end <= start)
		return;


	/* start */
	long AddPoints = 0;
	double TotalDist, IncDist, CuumDist;//,DelayDist,  ResidDist;
	DistRes = pFarsite->GetDistRes() / 1.2 * pFarsite->MetricResolutionConvert();	// make a little finer
	TotalDist = 0.0;//CuumDist=0.0;
	//DelayDist=pow2(attack->BurnDelay);
	xpt = xptl = attack->IndirectLine1[start * 2];
	ypt = yptl = attack->IndirectLine1[start * 2 + 1];
	if (start > 0)
	{
		xpt = attack->IndirectLine1[(start + 1) * 2];
		ypt = attack->IndirectLine1[(start + 1) * 2 + 1];
		xpt = xpt - (xpt - xptl) * 0.5;
		ypt = ypt - (ypt - yptl) * 0.5;
	}
	for (i = start; i <= end; i++)
	{
		if (i < end)
		{
			xptn = attack->IndirectLine1[(i + 1) * 2];
			yptn = attack->IndirectLine1[(i + 1) * 2 + 1];
		}
		else if (LagPoint > 0)
		{
			xptn = xpt2n;
			yptn = ypt2n;
		}
		else
			break;
		CuumDist = sqrt(pow2(xpt - xptn) + pow2(ypt - yptn));
		TotalDist += CuumDist;
		AddPoints += (long) (CuumDist / DistRes);
		xpt = xptn;
		ypt = yptn;
	}
	NewPoints = end - start + 1 + AddPoints;
	if (LagPoint > 0)
		NewPoints += 1;

	if (NewPoints < 2)    // can't have fewer points than 2
		return;
	/* end */

	xpt = xptl;
	ypt = yptl;
	if (start > 0)
	{
		xpt = attack->IndirectLine1[(start + 1) * 2];
		ypt = attack->IndirectLine1[(start + 1) * 2 + 1];
		xpt = xpt - (xpt - xptl) * 0.5;
		ypt = ypt - (ypt - yptl) * 0.5;
	}

	//NewPoints=end-start+1;
	pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), 2 * NewPoints + 1);
	k = 0;
	for (i = start; i <= end + 1; i++)
	{
		j = i + 1;
		if (j <= end)
		{
			xptn = attack->IndirectLine1[j * 2];
			yptn = attack->IndirectLine1[j * 2 + 1];
		}
		else if (LagPoint > 0)
		{
			xptn = xpt2n;
			yptn = ypt2n;
		}
		else
			break;

		BufferBurnout(xptl, yptl, xpt, ypt, xptn, yptn, &xbuf1, &ybuf1,
			&xbuf2, &ybuf2);
		if (attack->BurnDirection == 1)
		{
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 2 * NewPoints - 1 - k, xbuf2, ybuf2);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), 2 * NewPoints - 1 - k, 0.0, -1.0);
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), k, xbuf1, ybuf1);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), k, 0.0, 0.0);
		}
		/*{    pFarsite->SetPerimeter1(pFarsite->GetNewFires(), k, xbuf2, ybuf2);
						pFarsite->SetFireChx(pFarsite->GetNewFires(), k, 0.0, -1.0);
						pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 2*NewPoints-1-k, xbuf1, ybuf1);
					 pFarsite->SetFireChx(pFarsite->GetNewFires(), 2*NewPoints-1-k, 0.0, 0.0);
				}*/
		else
		{
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 2 * NewPoints - 1 - k, xbuf1, ybuf1);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), 2 * NewPoints - 1 - k, 0.0, 0.0);
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), k, xbuf2, ybuf2);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), k, 0.0, -1.0);
		}
		/*{    pFarsite->SetPerimeter1(pFarsite->GetNewFires(), k, xbuf1, ybuf1);
			   	pFarsite->SetFireChx(pFarsite->GetNewFires(), k, 0.0, 0.0);
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 2*NewPoints-1-k, xbuf2, ybuf2);
				 pFarsite->SetFireChx(pFarsite->GetNewFires(), 2*NewPoints-1-k, 0.0, -1.0);
			}*/
		k++;
		xptl = xpt;
		yptl = ypt;

		IncDist = pow2(xpt - xptn) + pow2(ypt - yptn);
		if (IncDist > pow2(DistRes))
		{
			IncDist = sqrt(IncDist);
			xpt = xpt - (xpt - xptn) * DistRes / IncDist;
			ypt = ypt - (ypt - yptn) * DistRes / IncDist;
			i--;
		}
		else
		{
			xpt = xptn;
			ypt = yptn;
			if (LagPoint == 0)
				i++;
		}
	}
	//pFarsite->SetNumPoints(pFarsite->GetNewFires(), 2*NewPoints);
	if (k < NewPoints)	// no
	{
		pFarsite->FreePerimeter1(pFarsite->GetNewFires());

		return;
	}
	pFarsite->SetNumPoints(pFarsite->GetNewFires(), 2 * k);
	pFarsite->SetInout(pFarsite->GetNewFires(), 1);
	NewFire = attack->FireNumber;      // save to restore after BoundingBox
	attack->FireNumber = pFarsite->GetNewFires();
	pFarsite->IncNewFires(1);
	BoundingBox();
	attack->FireNumber = NewFire;
	attack->BurnLine[0] = NewFirstPoint;
}


void Attack::BufferBurnout(double xptl, double yptl, double xpt, double ypt,
	double xptn, double yptn, double* xbuf1, double* ybuf1, double* xbuf2,
	double* ybuf2)
{
	double xdiffl, xdiffn, ydiffl, ydiffn, tempx, tempy;
	double xdiff, ydiff, dist, distl, distn;
	double Sign;
	double DistanceResolution;
	double A1, A2, A3, DR;

	DistanceResolution = pFarsite->GetDistRes() * 0.6 * pFarsite->MetricResolutionConvert();	// can't reduce width because center
	//DistanceResolution=GetDistRes()*1.2*pFarsite->MetricResolutionConvert();
	// of buffer is calculated line
	if (attack->BurnDirection == 0)	// left
		Sign = 1.0;
	else
		Sign = -1.0;
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
	if (dist < 1e-9) //=0.0)
		dist = 1.0;
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
	*xbuf1 = xpt + Sign * (DR + LineOffset) / dist * ydiff;	// perpendicular to xpt,ypt
	*ybuf1 = ypt - Sign * (DR + LineOffset) / dist * xdiff;
	*xbuf2 = xpt + Sign * LineOffset / dist * ydiff;					// perpendicular to xpt,ypt
	*ybuf2 = ypt - Sign * LineOffset / dist * xdiff;
	//*xbuf2=xpt+Sign*(DistanceResolution-LineOffset)/dist*ydiff;	// perpendicular to xpt,ypt
	//*ybuf2=ypt-Sign*(DistanceResolution-LineOffset)/dist*xdiff;
}


/*double Attack::pow2(double input)
{
	return input * input;
}*/

bool Attack::Cross(double xpt1, double ypt1, double xpt2, double ypt2,
	double xpt1n, double ypt1n, double xpt2n, double ypt2n, double* newx,
	double* newy)
{
	double xdiff1, ydiff1, xdiff2, ydiff2, ycept1, ycept2;
	double slope1, slope2, ycommon, xcommon;
	bool BadIntersection = false;

	xdiff1 = xpt2 - xpt1;
	ydiff1 = ypt2 - ypt1;
	if (fabs(xdiff1) < 1e-9)
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
	if (fabs(xdiff2) < 1e-9)
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
		/*  if(xdiff1!=0)
			{    *newy=ymid-slope1*xmid;		// ycross is slope of parallel lines and becomes y-intercept
				*newx=0;
			}
			else
			{   	*newy=ypt1;				 // could also be ypt2 because slope is vertical
				*newx=xmid;
				  }
				  */
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


long Attack::FindHullPoint(double* Hull, long NumHullPts, long* HullPt)
{
	bool RecordMinDist;
	long i, j, k, AbsoluteMinPoint;
	double dist, distl, distn;
	double AbsoluteMinDist = -1.0, DirectMinDist = -1.0;
	double startx, starty;

	for (i = 0; i < NumHullPts; i++)
	{
		xpt1 = Hull[i * 3];
		ypt1 = Hull[i * 3 + 1];
		if (xpt1 == attack->IndirectLine2[0] &&
			ypt1 == attack->IndirectLine2[1])
		{
			*HullPt = attack->CurrentPoint = i;
			DirectMinDist = 1.0;		// prevent exiting to external part of fire
			break;
		}
		else
		{
			RecordMinDist = true;
			startx = attack->IndirectLine2[0];
			starty = attack->IndirectLine2[1];
			dist = pow2(xpt1 - startx) + pow2(ypt1 - starty);
			if (AbsoluteMinDist < 0.0 || dist < AbsoluteMinDist)
			{
				AbsoluteMinDist = dist;
				AbsoluteMinPoint = i;
			}
			if (DirectMinDist < 0.0 || dist < DirectMinDist)
			{
				startx = startx - (startx - xpt1) * 0.01;	// offset a little so don't intersect dead pts
				starty = starty - (starty - ypt1) * 0.01;
				xpt2 = pFarsite->GetPerimeter2Value(0, XCOORD);
				ypt2 = pFarsite->GetPerimeter2Value(0, YCOORD);
				for (j = 1; j <= pFarsite->GetNumPoints(attack->FireNumber); j++)
				{
					k = j;
					if (j == pFarsite->GetNumPoints(attack->FireNumber))
						k = 0;
					xpt2n = pFarsite->GetPerimeter2Value(k, XCOORD);
					ypt2n = pFarsite->GetPerimeter2Value(k, YCOORD);
					// test to see if crosses fire if so then don't use this point
					if (Cross(startx, starty, xpt1, ypt1, xpt2, ypt2, xpt2n,
							ypt2n, &xpt1n, &ypt1n))
					{
						RecordMinDist = false;

						break;
					}
					xpt2 = xpt2n;
					ypt2 = ypt2n;
				}
				if (RecordMinDist)
				{
					if (DirectMinDist < 0.0 || dist < DirectMinDist)
					{
						DirectMinDist = dist;
						*HullPt = attack->CurrentPoint = i;  // use closest point
					}
				}
			}
		}
	}
	if (DirectMinDist < 0.0)	// if cannot see directly outside
	{
		*HullPt = attack->CurrentPoint = AbsoluteMinPoint;
		attack->NumPoints = 0;
		attack->IndirectLine2[0] = pFarsite->GetPerimeter1Value(attack->FireNumber,
									AbsoluteMinPoint, XCOORD);
		attack->IndirectLine2[1] = pFarsite->GetPerimeter1Value(attack->FireNumber,
									AbsoluteMinPoint, YCOORD);
		attack->IndirectLine2[2] = -1.0;	 // reset indirect line
		attack->BurnLine[0] = -1;
		if (attack->IndirectLine1)
			delete[] attack->IndirectLine1;//free(attack->IndirectLine1);
		attack->IndirectLine1 = 0;
		attack->LineNumber = (long)-1.0;			// set new fire number for barrier
	}

	if (attack->IndirectLine2[2] < 0.0)
	{
		attack->IndirectLine2[2] = 1;
		return attack->CurrentPoint;
	}

	i = attack->CurrentPoint + 1;
	if (i > NumHullPts - 1)
		i = 0;
	xpt1 = Hull[i * 3];
	ypt1 = Hull[i * 3 + 1];
	distn = pow2(xpt1 - attack->IndirectLine2[0]) +
		pow2(ypt1 -
			attack->IndirectLine2[1]);			// dist to point
	distl = pow2(xpt1 - Hull[attack->CurrentPoint * 3]) +
		pow2(ypt1 -
			Hull[attack->CurrentPoint * 3 + 1]);  		// total dist
	if (distn < distl)  								 	// on this segment
	{
		if (attack->Reverse)
			*HullPt = attack->CurrentPoint = i;
	}
	else if (distn > distl)
	{
		//if(!attack->Reverse)
		//{	attack->CurrentPoint-=1;
		//	if(attack->CurrentPoint<0)
		//     	attack->CurrentPoint+=NumHullPts;
		*HullPt = attack->CurrentPoint;

		//}
	}

	return attack->CurrentPoint;
}


bool Attack::AllocParallelLine(double* Hull, long NumHullPts, double TimeStep)
{
	long i, j, NewNumPts;
	double* TempLine = 0;
	double Rate, MaxRate = -1.0, MaxDist;
	double xptl, yptl, xpt, ypt;

	// determine maximum line prod rate for dimensioning array
	for (i = 0; i < 51; i++)
	{
		Rate = crew[attack->CrewNum]->LineProduction[i];
		if (Rate > MaxRate)
			MaxRate = Rate;
	}

	// calcualate distance along Hull line
	MaxDist = pow2(TimeStep * MaxRate * pFarsite->MetricResolutionConvert());  // maximum horizontal distance of line production squared
	Rate = 0.0;
	xptl = Hull[attack->CurrentPoint * 2];
	yptl = Hull[attack->CurrentPoint * 2 + 1];
	for (i = 1; i < NumHullPts; i++)
	{
		if (attack->Reverse)
		{
			j = attack->CurrentPoint - i;
			if (j < 0)
				j += NumHullPts;
		}
		else
		{
			j = attack->CurrentPoint + i;
			if (j > NumHullPts - 1)
				j -= NumHullPts;
		}
		xpt = Hull[j * 2];
		ypt = Hull[j * 2 + 1];
		Rate += pow2(xptl - xpt) + pow2(yptl - ypt);
		xptl = xpt;
		yptl = ypt;
		if (Rate > MaxDist)
			break;
	}
	NewNumPts = attack->NumPoints + i + 4;

	// transfer and save existing parallel line points
	if (attack->NumPoints > 0 && attack->IndirectLine1)
	{
		if ((TempLine = new double[attack->NumPoints * 2]) != NULL)//(double *) calloc(attack->NumPoints*2, sizeof(double)))!=NULL)
		{
			for (i = 0; i < attack->NumPoints; i++)
			{
				TempLine[i * 2] = attack->IndirectLine1[i * 2];
				TempLine[i * 2 + 1] = attack->IndirectLine1[i * 2 + 1];
			}
			delete[] attack->IndirectLine1;//free(attack->IndirectLine1);
			attack->IndirectLine1 = 0;
		}
		else
			return false;
	}

	// Reallocate IndirectLine2 and transfer existing points back from TempLine
	if ((attack->IndirectLine1 = new double[NewNumPts * 2]) != NULL)//(double *) calloc(NewNumPts*2, sizeof(double)))!=NULL)
	{
		NumInsertPoints = NewNumPts;			  // record number of points
		for (i = 0; i < attack->NumPoints; i++)
		{
			attack->IndirectLine1[i * 2] = TempLine[i * 2];
			attack->IndirectLine1[i * 2 + 1] = TempLine[i * 2 + 1];
		}
		if (TempLine)
			delete[] TempLine;//free(TempLine);
		if (attack->NumPoints == 0)
		{
			attack->IndirectLine1[0] = attack->IndirectLine2[0] = Hull[attack->CurrentPoint * 2]; //attack->IndirectLine2[0];
			attack->IndirectLine1[1] = attack->IndirectLine2[1] = Hull[attack->CurrentPoint * 2 + 1]; //attack->IndirectLine2[1];
			attack->NumPoints++;
		}
	}
	else
		return false;

	return true;
}


void Attack::GetConvexHull(double* Hull, long* NewHullPts)
{
	long i;
	long FireNum = attack->FireNumber;
	long NumPoints = pFarsite->GetNumPoints(attack->FireNumber);
	long NumHullPts;
	long NorthPt, SouthPt, EastPt, WestPt;
	double xpt, ypt, Xlo, Xhi, Ylo, Yhi;

	Xlo = Xhi = pFarsite->GetPerimeter1Value(FireNum, 0, XCOORD);
	Ylo = Yhi = pFarsite->GetPerimeter1Value(FireNum, 0, YCOORD);
	NorthPt = 0;
	SouthPt = 0;
	EastPt = 0;
	WestPt = 0;
	for (i = 1; i < NumPoints; i++)
	{
		xpt = pFarsite->GetPerimeter1Value(FireNum, i, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(FireNum, i, YCOORD);
		if (xpt < Xlo)
		{
			Xlo = xpt;
			WestPt = i;
		}
		if (xpt > Xhi)
		{
			Xhi = xpt;
			EastPt = i;
		}
		if (ypt < Ylo)
		{
			Ylo = ypt;
			SouthPt = i;
		}
		if (ypt > Yhi)
		{
			Yhi = ypt;
			NorthPt = i;
		}
	}
	pFarsite->SetPerimeter1(FireNum, NumPoints, Xlo, Xhi);
	pFarsite->SetFireChx(FireNum, NumPoints, Ylo, Yhi);

	NumHullPts = 0;
	FindHullQuadrant(Hull, &NumHullPts, WestPt, SouthPt, PI);
	FindHullQuadrant(Hull, &NumHullPts, SouthPt, EastPt, PI / 2.0);
	FindHullQuadrant(Hull, &NumHullPts, EastPt, NorthPt, 0.0);
	FindHullQuadrant(Hull, &NumHullPts, NorthPt, WestPt, 3.0 * PI / 2.0);
	*NewHullPts = NumHullPts;
}


void Attack::FindHullQuadrant(double* Hull, long* NumPts, long StartPt,
	long EndPt, double RefAngle)
{
	long i, j, Mult, ConvexStartPt, ConvexEndPt;
	long ConPt, ConOrder, OldPt;
	long FireNum = attack->FireNumber;
	long NumPoints = pFarsite->GetNumPoints(attack->FireNumber);
	long NumHullPts = *NumPts;
	double xpt, ypt;
	double AngleOut, AngleIn, Angle, DiffAngle;
	//   double PerimRes, Dist, PDist;
	APolygon ap(pFarsite);

	//   PerimRes==GetPerimRes()*2.0*pFarsite->MetricResolutionConvert();
	Hull[NumHullPts * 3] = xpt1 = pFarsite->GetPerimeter1Value(FireNum, StartPt, XCOORD);
	Hull[NumHullPts * 3 + 1] = ypt1 = pFarsite->GetPerimeter1Value(FireNum, StartPt,
										YCOORD);
	if (pFarsite->GetPerimeter2Value(StartPt, FLIVAL) < 0.0)
		Mult = -1;
	else
		Mult = 1;
	Hull[NumHullPts * 3 + 2] = StartPt * Mult;
	OldPt = StartPt;
	NumHullPts++;
	ap.startx = pFarsite->GetPerimeter1Value(FireNum, StartPt, XCOORD);
	ap.starty = pFarsite->GetPerimeter1Value(FireNum, StartPt, YCOORD);
	AngleOut = RefAngle;
	xpt = pFarsite->GetPerimeter1Value(FireNum, EndPt, XCOORD);
	ypt = pFarsite->GetPerimeter1Value(FireNum, EndPt, YCOORD);
	AngleIn = ap.direction(xpt, ypt);
	DiffAngle = AngleOut - AngleIn;
	if (DiffAngle < 0.0)
		DiffAngle += 2.0 * PI;
	ConvexStartPt = StartPt + 1;
	ConvexEndPt = EndPt;
	if (ConvexEndPt < ConvexStartPt)
		ConvexEndPt += NumPoints - ConvexStartPt + StartPt + 1;
	ConOrder = -1;
	for (j = ConvexStartPt; j < ConvexEndPt; j++)
	{
		i = j;
		if (i > NumPoints - 1)
			i -= NumPoints;
		xpt = pFarsite->GetPerimeter1Value(FireNum, i, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(FireNum, i, YCOORD);
		Angle = AngleOut - ap.direction(xpt, ypt);
		if (Angle < 0.0)
			Angle += 2.0 * PI;
		if (Angle < DiffAngle)
		{
			DiffAngle = Angle;
			ConPt = i;
			ConOrder = j;
		}
		if (j == ConvexEndPt - 1 && ConOrder > -1)
		{
			if (ConOrder < ConvexEndPt)
			{
				j = ConOrder;
				ap.startx = pFarsite->GetPerimeter1Value(FireNum, ConPt, XCOORD);
				ap.starty = pFarsite->GetPerimeter1Value(FireNum, ConPt, YCOORD);
				xpt = pFarsite->GetPerimeter1Value(FireNum, EndPt, XCOORD);
				ypt = pFarsite->GetPerimeter1Value(FireNum, EndPt, YCOORD);
				AngleIn = ap.direction(xpt, ypt);
				DiffAngle = AngleOut - AngleIn;
				if (DiffAngle < 0.0)
					DiffAngle += 2.0 * PI;
			}
			if (pFarsite->GetPerimeter2Value(ConPt, FLIVAL) < 0.0)
				Mult = -1;
			else
				Mult = 1;
			/*if(labs(ConPt-OldPt)>1)
					{	xpt=pFarsite->GetPerimeter1Value(FireNum, ConPt, XCOORD);
					   ypt=pFarsite->GetPerimeter1Value(FireNum, ConPt, YCOORD);
					 Dist=sqrt(pow2(xpt-xpt1)+pow2(ypt-ypt1));
					 PDist=Dist/PerimRes;
					   for(k=1; k<(PDist-0.5); k++)
					  {	Hull[NumHullPts*3]=xpt1-(xpt1-xpt)*(double) k/PDist;
						Hull[NumHullPts*3+1]=ypt1-(ypt1-ypt)*(double) k/PDist;
						 Hull[NumHullPts*3+2]=ConPt*Mult;
						   NumHullPts++;
					  }
					}*/
			Hull[NumHullPts * 3] = xpt1 = pFarsite->GetPerimeter1Value(FireNum, ConPt,
											XCOORD);
			Hull[NumHullPts * 3 + 1] = ypt1 = pFarsite->GetPerimeter1Value(FireNum,
												ConPt, YCOORD);
			Hull[NumHullPts * 3 + 2] = ConPt * Mult;
			NumHullPts++;
			ConOrder = -1;
			OldPt = ConPt;
		}
	}
	*NumPts = NumHullPts;
}


void Attack::ExpandConvexHulls(double* Hull1, double* Hull2, long NumHullPts)
{
	long i, j, k, ConPt;
	double Distance;
	double xptl, yptl;
	double xptn, yptn;
	double xpt, ypt, xh, yh, fh, oldx, oldy;
	double firstx, firsty, xdiff, ydiff;
	double xdiffl, ydiffl, xdiffn, ydiffn;
	double tempx, tempy, distl, distn, dist;
	double A1, A2, A3, DR;

	/*
	FILE *newfile=fopen("hulls1.vct", "a");
	fprintf(newfile, "%s\n", "10");
	for(i=0; i<NumHullPts; i++)
	{	xh=ConvertEastingOffsetToUtm(Hull2[i*3]);
		yh=ConvertNorthingOffsetToUtm(Hull2[i*3+1]);
	fprintf(newfile, "%lf %lf\n", xh, yh);
	}
	fprintf(newfile, "%s\n", "END");
	fclose(newfile);
	*/

	Distance = ((double) attack->FireDist + pFarsite->GetDistRes() / 2.0) * pFarsite->MetricResolutionConvert();
	j = NumHullPts - 1;
	xptl = Hull2[j * 3];
	yptl = Hull2[j * 3 + 1];
	xpt = firstx = Hull2[0];
	ypt = firsty = Hull2[1];
	for (i = 0; i < NumHullPts; i++)
	{
		k = i + 1;
		if (k > NumHullPts - 1)
		{
			xptn = firstx;
			yptn = firsty;
		}
		else
		{
			xptn = Hull2[k * 3];
			yptn = Hull2[k * 3 + 1];
		}
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
		if (dist < 1e-9) //==0.0)
			dist = 1.0;
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
			DR = Distance / fabs(A3);
		else
			DR = Distance;

		ConPt = labs((long) Hull2[i * 3 + 2]);
		xh = pFarsite->GetPerimeter2Value(ConPt, XCOORD);
		yh = pFarsite->GetPerimeter2Value(ConPt, YCOORD);
		fh = pFarsite->GetPerimeter2Value(ConPt, FLIVAL);
		oldx = xpt;
		oldy = ypt;
		if (fh < 0.0)
		{
			xpt = xh;
			ypt = yh;
		}
		Hull2[i * 3] = xpt - DR / dist * ydiff;
		Hull2[i * 3 + 1] = ypt + DR / dist * xdiff;
		xdiffl = xpt - xh;
		ydiffl = ypt - yh;
		Hull1[i * 2] = Hull2[i * 3] - xdiffl;
		Hull1[i * 2 + 1] = Hull2[i * 3 + 1] - ydiffl;
		/*
			xbuf1=xpt-DistanceResolution/dist*ydiff;	// perpendicular to xpt,ypt
			ybuf1=ypt+DistanceResolution/dist*xdiff;
			xbuf2=xpt+DistanceResolution/dist*ydiff; 	// perpendicular to xpt,ypt
			ybuf2=ypt-DistanceResolution/dist*xdiff;
			*/

		xptl = oldx;//xpt;
		yptl = oldy;//ypt;
		xpt = xptn;
		ypt = yptn;
	}


	/*     FILE *newfile=fopen("hulls.vct", "a");
		 fprintf(newfile, "%s\n", "10");
		 for(i=0; i<NumHullPts; i++)
		 {	xh=ConvertEastingOffsetToUtm(Hull2[i*3]);
		 	yh=ConvertNorthingOffsetToUtm(Hull2[i*3+1]);
			fprintf(newfile, "%lf %lf\n", xh, yh);
		 }
		 fprintf(newfile, "%s\n", "END");
		 fprintf(newfile, "%s\n", "20");
		 for(i=0; i<NumHullPts; i++)
		 {	xh=ConvertEastingOffsetToUtm(Hull1[i*2]);
		 	yh=ConvertNorthingOffsetToUtm(Hull1[i*2+1]);
			fprintf(newfile, "%lf %lf\n", xh, yh);
		 }
		 fprintf(newfile, "%s\n", "END");
		fclose(newfile);
	*/
}


void Attack::HullDensityControl(double* Hull1, double* Hull2, long* numhullpts)
{
	bool NewPointAdded;
	long i, j, k, NumHullPts = *numhullpts;
	double xpt, ypt, xpt1, ypt1, xpt2, ypt2, fli, fli1;
	double dist, PerimRes;

	PerimRes = pFarsite->GetPerimRes() * 2.0 * pFarsite->MetricResolutionConvert();
	do
	{
		NewPointAdded = false;
		for (i = 0; i < NumHullPts; i++)
		{
			j = i + 1;
			if (j >= NumHullPts)
				j = 0;
			dist = sqrt(pow2(Hull1[i * 3] - Hull1[j * 3]) +
					pow2(Hull1[i * 3 + 1] - Hull1[j * 3 + 1]));
			if (dist > PerimRes)
			{
				xpt1 = Hull1[i * 3] - (Hull1[i * 3] - Hull1[j * 3]) / 2.0;
				ypt1 = Hull1[i * 3 + 1] -
					(Hull1[i * 3 + 1] - Hull1[j * 3 + 1]) /
					2.0;
				fli1 = (long) ((Hull1[i * 3 + 2] + Hull1[j * 3 + 2]) / 2.0);
				xpt2 = Hull2[i * 2] - (Hull2[i * 2] - Hull2[j * 2]) / 2.0;
				ypt2 = Hull2[i * 2 + 1] -
					(Hull2[i * 2 + 1] - Hull2[j * 2 + 1]) /
					2.0;
				for (k = i + 1; k < NumHullPts; k++)
				{
					xpt = Hull1[k * 3];
					ypt = Hull1[k * 3 + 1];
					fli = Hull1[k * 3 + 2];
					Hull1[k * 3] = xpt1;
					Hull1[k * 3 + 1] = ypt1;
					Hull1[k * 3 + 2] = fli1;
					xpt1 = xpt;
					ypt1 = ypt;
					fli1 = fli;
					xpt = Hull2[k * 2];
					ypt = Hull2[k * 2 + 1];
					Hull2[k * 2] = xpt2;
					Hull2[k * 2 + 1] = ypt2;
					xpt2 = xpt;
					ypt2 = ypt;
				}
				Hull1[NumHullPts * 3] = xpt1;
				Hull1[NumHullPts * 3 + 1] = ypt1;
				Hull1[k * 3 + 2] = fli1;
				Hull2[NumHullPts * 2] = xpt2;
				Hull2[NumHullPts * 2 + 1] = ypt2;
				NumHullPts++;
				i++;						// force over newly added point
				if (NumHullPts >= pFarsite->GetNumPoints(attack->FireNumber) * 2 - 1)   // alloc safety
				{
					NewPointAdded = false;
					break;
				}
				else
					NewPointAdded = true;
			}
		}
	}
	while (NewPointAdded);

	*numhullpts = NumHullPts;
}


/*
void Attack::ReplaceDeadPoints()
{// replaces new with old point if fli was <0.0, indicating it was extinuished
	for(long i=0; i<pFarsite->GetNumPoints(attack->FireNumber); i++)
	 {    fli1=pFarsite->GetPerimeter2Value(i, FLIVAL);
		if(fli1<0.0)
		  {    pFarsite->GetPerimeter2(i, &xpt1, &ypt1, &ros1, &fli1);
		  	pFarsite->SetPerimeter1(attack->FireNumber, i, xpt1, ypt1);
			   pFarsite->SetFireChx(attack->FireNumber, i, ros1, fli1);
		  }
	 }
}
*/


//------------------------------------------------------------------------------
//
//	Global Supression Access Functions
//
//------------------------------------------------------------------------------

/*static long NumAttacks = 0;
static long AttackCounter = 0;
AttackData* FirstAttack;
AttackData* NextAttack;
AttackData* CurAttack;
AttackData* LastAttack;
AttackData* Reassignment = 0;
char AttackLog[256];


void LoadAttacks(AttackData attackdata)
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


long SetupIndirectAttack(long crewnum, double* startpt, long numpts)
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


long ResetIndirectAttack(AttackData* atk, double* coords, long numpts)
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


long SetupDirectAttack(long crewnum, long FireNum, double* coords)
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


long ResetDirectAttack(AttackData* atk, long FireNum, double* coords)
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
	if (pFarsite->GetInputMode() == PARALLELLOCATION ||
		pFarsite->GetInputMode() == RELOCATEPARALLEL)
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
	for (i = 0; i < pFarsite->GetNumPoints(CurAttack->FireNumber); i++)
	{
		xpt1 = pFarsite->GetPerimeter1Value(CurAttack->FireNumber, i, XCOORD);
		ypt1 = pFarsite->GetPerimeter1Value(CurAttack->FireNumber, i, YCOORD);
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
	for (i = 0; i < pFarsite->GetNumPoints(atk->FireNumber); i++)
	{
		xpt1 = pFarsite->GetPerimeter1Value(CurAttack->FireNumber, i, XCOORD);
		ypt1 = pFarsite->GetPerimeter1Value(CurAttack->FireNumber, i, YCOORD);
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
				(double) pFarsite->GetNumPoints(atk->FireNumber) /
				2.0)
			{
				atk->Reverse = true;
				//			else
				//  				atk->Reverse=false;
			}
		}
		else if (atk->CurrentPoint -
			atk->NextPoint <
			(double) pFarsite->GetNumPoints(atk->FireNumber) /
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
			atk->IndirectLine1[0] = pFarsite->GetPerimeter1Value(atk->FireNumber,
										atk->CurrentPoint, XCOORD);
			atk->IndirectLine1[1] = pFarsite->GetPerimeter1Value(atk->FireNumber,
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
			atk->IndirectLine2[0] = pFarsite->GetPerimeter1Value(atk->FireNumber,
										atk->CurrentPoint, XCOORD);
			atk->IndirectLine2[1] = pFarsite->GetPerimeter1Value(atk->FireNumber,
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

long GetNumAttacks()
{
	// returns the number of active attacks
	return NumAttacks;
}

long GetFireNumberForAttack(long AttackCounter)
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


AttackData* GetAttackForFireNumber(long NumFire, long StartAttackNum,
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

void SetNewFireNumberForAttack(long oldnumfire, long newnumfire)
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

long GetNumCrews()
{
	return NumCrews;
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

AttackData* GetAttackByOrder(long OrdinalAttackNum, bool IndirectOnly)
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


AttackData* GetAttack(long AttackCounter)
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


void CancelAttack(long AttackCounter)
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


AttackData* GetReassignedAttack()
{
	return Reassignment;
}

void ReassignAttack(AttackData* atk)
{
	Reassignment = atk;
}


void FreeAllAttacks()
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


void WriteAttackLog(AttackData* atk, long type, long var1, long var2)
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


bool LoadCrews(char* FileName, bool AppendList)
{
	FILE* crewfile;
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

	return true;
}

/*
bool LoadCrews(char *FileName, bool AppendList)
{
	FILE* crewfile;
	 char garbage[256];
	 char ch[2]="";
	 double RateMult, rate;
	 long fuelnumber, crewnumber;

	if((crewfile=fopen(FileName, "r"))!=NULL)
	{    long i, j;
		  j=GetNumCrews();
		  if(!AppendList)
		  {    for(i=0; i<j; i++)
				  FreeCrew(0);
		  }

		  ch[0]=getc(crewfile);
		  do
	 	{    ch[0]=getc(crewfile);
			   if(feof(crewfile))
			   	break;
			   crewnumber=SetNewCrew();
		  	if(crewnumber==-1)
				return false;
			   crew[crewnumber]=GetCrew(crewnumber);
			   crew[crewnumber]->Compound=-1;
//  		  	for(i=0; i<51; i++)		// initialize LineProduction Rates
//     			crew[crewnumber]->LineProduction[i]=0.0;
//  			 fscanf(crewfile, "%1c", ch);
//     		fscanf(crewfile, "%1c", ch);
			memset(crew[crewnumber]->CrewName, 0x0, sizeof(crew[crewnumber]->CrewName));
		 	do
		 	{    strcat(crew[crewnumber]->CrewName, ch);
			   	ch[0]=getc(crewfile);
				//fscanf(crewfile, "%1c", ch);
			  } while(strcmp(ch, "#"));
		 	fscanf(crewfile, "%s", garbage);
			  if(!strcmp(garbage, "CHAINS_PER_HOUR"))
			  {    crew[crewnumber]->Units=3;
			   	RateMult=0.27709;
			   }
			  else if(!strcmp(garbage, "METERS_PER_MINUTE"))
			  {    crew[crewnumber]->Units=1;
			   	RateMult=1.0;
			   }
			  else if(!strcmp(garbage, "FEET_PER_MINUTE"))
			  {    crew[crewnumber]->Units=2;
			   	RateMult=0.3048;
			   }
			fscanf(crewfile, "%s", garbage);
			  if(!strcmp(strupr(garbage), "FLAME_LIMIT"))
			  	fscanf(crewfile, "%lf", &crew[crewnumber]->FlameLimit);
			  //else if(!strcmp(strupr(garbage), "SLOPE_LIMIT"))
			  //	fscanf(crewfile, "%lf", &crew[crewnumber]->SlopeLimit);
		 	//fscanf(crewfile, "%s", garbage);
			  //if(!strcmp(strupr(garbage), "SLOPE_LIMIT"))
			  //	fscanf(crewfile, "%lf", &crew[crewnumber]->SlopeLimit);
			  //else if(!strcmp(strupr(garbage), "FLAME_LIMIT"))
			  //	fscanf(crewfile, "%lf", &crew[crewnumber]->FlameLimit);
			if(crew[crewnumber]->Units==2 || crew[crewnumber]->Units==3)
			  	crew[crewnumber]->FlameLimit*=0.3048;
			  do
			{    fscanf(crewfile, "%ld", &fuelnumber);
				if(fuelnumber<51)
					{	fscanf(crewfile, "%lf", &rate);
					crew[crewnumber]->LineProduction[fuelnumber-1]=rate*RateMult;
					}
					else if(fuelnumber==99)
						break;
			} while(!feof(crewfile));
			   ch[0]=getc(crewfile);
			   ch[0]=getc(crewfile);
		}while(!strncmp(ch, "#", 1)); //(!feof(crewfile));
		fclose(crewfile);
	 }
	 else
		  return false;

	 return true;
}


Crew* GetCrew(long CrewNumber)
{
	if (crew[CrewNumber])
		return crew[CrewNumber];

	return 0;
}


long SetNewCrew()
{
	// doesn't require crew number
	if ((crew[NumCrews] = new Crew) != NULL)//(Crew *) calloc(1, sizeof(Crew)))!=NULL)
		return NumCrews++;

	return -1;
}

long SetCrew(long CrewNumber)
{
	// requires crew number for allocation
	if ((crew[CrewNumber] = new Crew) != NULL)//(Crew *) calloc(1, sizeof(Crew)))!=NULL)
		return NumCrews++;

	return -1;
}


void FreeAllCrews()
{
	long i;

	for (i = 0; i < NumCrews; i++)
		FreeCrew(0);
	NumCrews = 0;
}


void FreeCrew(long CrewNumber)
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


static long const MaxNumCompoundCrews = 200;
CompoundCrew* compoundcrew[200];
static long NumCompoundCrews = 0;

long GetNumCompoundCrews()
{
	return NumCompoundCrews;
}


long SetCompoundCrew(long CrewNumber, char *crewname)
{
	 if(NumCompoundCrews>=MaxNumCompoundCrews)
	 	return -1;

	if((compoundcrew[CrewNumber]=new CompoundCrew)!=NULL)//(CompoundCrew *) calloc(1, sizeof(CompoundCrew)))!=NULL)
	 {    compoundcrew[CrewNumber]->CrewIndex=new long[COMPOUNDINC];//(long *) calloc(COMPOUNDINC, sizeof(long));
		compoundcrew[CrewNumber]->Multiplier=new double[COMPOUNDINC];//(double *) calloc(COMPOUNDINC, sizeof(double));
		  compoundcrew[CrewNumber]->NumTotalCrews=COMPOUNDINC;
		  compoundcrew[CrewNumber]->NumCurrentCrews=0;
		  if((compoundcrew[CrewNumber]->CompCrew=SetNewCrew())>-1)
		  {	memset(crew[compoundcrew[CrewNumber]->CompCrew]->CrewName, 0x0,
			   	sizeof(crew[compoundcrew[CrewNumber]->CompCrew]->CrewName));
			   strcpy(crew[compoundcrew[CrewNumber]->CompCrew]->CrewName, crewname);
			   crew[compoundcrew[CrewNumber]->CompCrew]->Compound=CrewNumber;
		  }

	 	return NumCompoundCrews++;
	 }

	 return -1;
}


long SetCompoundCrew(long GroupNumber, char* crewname, long ExistingCrewNumber)
{
	if (NumCompoundCrews >= MaxNumCompoundCrews)
		return -1;

	long Val = 0;

	if (GroupNumber >= NumCompoundCrews)
	{
		if ((compoundcrew[GroupNumber] = new CompoundCrew) == NULL)
			return -1;
		compoundcrew[GroupNumber]->CrewIndex = new long[COMPOUNDINC];//(long *) calloc(COMPOUNDINC, sizeof(long));
		compoundcrew[GroupNumber]->Multiplier = new double[COMPOUNDINC];//(double *) calloc(COMPOUNDINC, sizeof(double));
		compoundcrew[GroupNumber]->NumTotalCrews = COMPOUNDINC;
		compoundcrew[GroupNumber]->NumCurrentCrews = 0;
		Val = 1;
	}

	if (ExistingCrewNumber == -1)
	{
		if ((compoundcrew[GroupNumber]->CompCrew = SetNewCrew()) == -1)
			return -1;
	}
	else
		compoundcrew[GroupNumber]->CompCrew = ExistingCrewNumber;
	memset(crew[compoundcrew[GroupNumber]->CompCrew]->CrewName, 0x0,
		sizeof(crew[compoundcrew[GroupNumber]->CompCrew]->CrewName));
	strcpy(crew[compoundcrew[GroupNumber]->CompCrew]->CrewName, crewname);
	crew[compoundcrew[GroupNumber]->CompCrew]->Compound = GroupNumber;

	if (Val == 0)
		return NumCompoundCrews;

	return NumCompoundCrews++;
}


bool AddToCompoundCrew(long CrewNumber, long NewCrew, double Mult)
{
	long* tempindx;
	double* tempmult;
	long numcur = compoundcrew[CrewNumber]->NumCurrentCrews;

	if (numcur == compoundcrew[CrewNumber]->NumTotalCrews)
	{
		if ((tempindx = new long[numcur]) == NULL)	//if((tempindx=(long *) calloc(numcur, sizeof(long)))==NULL)
			return false;
		if ((tempmult = new double[numcur]) == NULL) //((tempmult=(long *) calloc(numcur, sizeof(double)))==NULL)
			return false;
		memcpy(tempindx, compoundcrew[CrewNumber]->CrewIndex,
			numcur * sizeof(long));
		memcpy(tempmult, compoundcrew[CrewNumber]->Multiplier,
			numcur * sizeof(double));
		delete[] compoundcrew[CrewNumber]->CrewIndex;//free(compoundcrew[CrewNumber]->CrewIndex);
		delete[] compoundcrew[CrewNumber]->Multiplier; //free(compoundcrew[CrewNumber]->Multiplier);
		compoundcrew[CrewNumber]->CrewIndex = new long[numcur + COMPOUNDINC];	// (long *) calloc(numcur+COMPOUNDINC, sizeof(long));
		compoundcrew[CrewNumber]->Multiplier = new double[numcur + COMPOUNDINC];	// (double *) calloc(numcur+COMPOUNDINC, sizeof(double));
		compoundcrew[CrewNumber]->NumTotalCrews = numcur + COMPOUNDINC;
		memcpy(compoundcrew[CrewNumber]->CrewIndex, tempindx,
			numcur * sizeof(long));
		memcpy(compoundcrew[CrewNumber]->Multiplier, tempmult,
			numcur * sizeof(double));
		delete[] tempindx;//free(tempindx);
		delete[] tempmult;//free(tempmult);
	}
	compoundcrew[CrewNumber]->CrewIndex[numcur] = NewCrew;
	compoundcrew[CrewNumber]->Multiplier[numcur] = Mult;
	compoundcrew[CrewNumber]->NumCurrentCrews++;

	return true;
}

void CalculateCompoundRates(long CrewNumber)
{
	double Rate, Cost, MaxLen;
	long i, j;
	long numcur = compoundcrew[CrewNumber]->NumCurrentCrews;

	MaxLen = 0.0;
	for (i = 0; i < 51; i++)
	{
		Rate = 0.0;
		Cost = 0.0;
		for (j = 0; j < numcur; j++)
		{
			Rate += crew[compoundcrew[CrewNumber]->CrewIndex[j]]->LineProduction[i] * compoundcrew[CrewNumber]->Multiplier[j];
			Cost += crew[compoundcrew[CrewNumber]->CrewIndex[j]]->Cost;
		}
		crew[compoundcrew[CrewNumber]->CompCrew]->LineProduction[i] = Rate;
		crew[compoundcrew[CrewNumber]->CompCrew]->Cost = Cost;
	}
	for (j = 0; j < numcur; j++)
	{
		if (crew[compoundcrew[CrewNumber]->CrewIndex[j]]->FlameLimit > MaxLen)
			MaxLen = crew[compoundcrew[CrewNumber]->CrewIndex[j]]->FlameLimit;
	}
	crew[compoundcrew[CrewNumber]->CompCrew]->FlameLimit = MaxLen;
	crew[compoundcrew[CrewNumber]->CompCrew]->Units = 1;
}


CompoundCrew* GetCompoundCrew(long CrewNumber)
{
	return compoundcrew[CrewNumber];
}

void FreeCompoundCrew(long CrewNumber)
{
	long i, j, OldCompCrew;
	CompoundCrew* crw;

	crw = compoundcrew[CrewNumber];
	OldCompCrew = compoundcrew[CrewNumber]->CompCrew;
	for (i = CrewNumber; i < NumCompoundCrews - 1; i++)
	{
		compoundcrew[i] = compoundcrew[i + 1];
		if (compoundcrew[i]->CompCrew > OldCompCrew)
			compoundcrew[i]->CompCrew--;
		for (j = 0; j < compoundcrew[i]->NumCurrentCrews; j++)
		{
			if (compoundcrew[i]->CrewIndex[j] > OldCompCrew)
				compoundcrew[i]->CrewIndex[j]--;
		}
	}
	AttackData* atk;
	for (i = 0; i < GetNumAttacks(); i++)
	{
		atk = GetAttackByOrder(i, false);
		if (atk->CrewNum > OldCompCrew)
			atk->CrewNum--;
	}
	NumCompoundCrews--;
	compoundcrew[NumCompoundCrews] = crw;

	for (i = 0; i < GetNumCrews(); i++)
	{
		if (crew[i]->Compound > CrewNumber)
			crew[i]->Compound--;
	}

	if (compoundcrew[NumCompoundCrews])
	{
		FreeCrew(compoundcrew[NumCompoundCrews]->CompCrew);
		delete[] compoundcrew[NumCompoundCrews]->CrewIndex;//free(compoundcrew[NumCompoundCrews]->CrewIndex);
		delete[] compoundcrew[NumCompoundCrews]->Multiplier; //free(compoundcrew[NumCompoundCrews]->Multiplier);
		delete compoundcrew[NumCompoundCrews]; //free(compoundcrew[NumCompoundCrews]);
	}
	compoundcrew[NumCompoundCrews] = 0;
}

void FreeAllCompoundCrews()
{
	long i;

	for (i = 0; i < NumCompoundCrews; i++)
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

bool LoadCompoundCrews(char* FileName)
{
	long NumCompCrews;
	long CrewNum, NumCrews;
	long i, j, * Index;
	char Name[256] = "";
	double* Mult;
	FILE* CurrentFile;

	if ((CurrentFile = fopen(FileName, "rb")) != NULL)
	{
		fread(&NumCompCrews, sizeof(long), 1, CurrentFile);
		for (i = 0; i < NumCompCrews; i++)
		{
			fread(&CrewNum, sizeof(long), 1, CurrentFile);
			CrewNum = SetNewCrew();
			crew[CrewNum]->Compound = i;
			fread(Name, sizeof(char), 256, CurrentFile);
			SetCompoundCrew(i, Name, CrewNum);
			fread(&NumCrews, sizeof(long), 1, CurrentFile);	// numtotal
			Index = new long[NumCrews];
			Mult = new double[NumCrews];
			fread(&NumCrews, sizeof(long), 1, CurrentFile);	// numcurrent
			fread(Index, sizeof(long), NumCrews, CurrentFile);
			fread(Mult, sizeof(double), NumCrews, CurrentFile);
			for (j = 0; j < NumCrews; j++)
				AddToCompoundCrew(i, Index[j], Mult[j]);
			delete[] Index;
			delete[] Mult;
			CalculateCompoundRates(i);
		}
		fclose(CurrentFile);
	}
	else
		return false;

	return true;
}


void RemoveFromCompoundCrew(long CompNumber, long CrewNumber)
{
	long i;

	//j=CrewNumber;//compoundcrew[CompNumber]->CrewIndex[CrewNumber];

	WriteAttackLog(0, 4, GetCompoundCrew(CompNumber)->CrewIndex[CrewNumber],
		GetCompoundCrew(CompNumber)->CompCrew);
	compoundcrew[CompNumber]->NumCurrentCrews--;
	for (i = CrewNumber; i < compoundcrew[CompNumber]->NumCurrentCrews; i++)
	{
		compoundcrew[CompNumber]->CrewIndex[i] = compoundcrew[CompNumber]->CrewIndex[i + 1];
		compoundcrew[CompNumber]->Multiplier[i] = compoundcrew[CompNumber]->Multiplier[i + 1];
	}
	compoundcrew[CompNumber]->CrewIndex[compoundcrew[CompNumber]->NumCurrentCrews] = 0;
	compoundcrew[CompNumber]->Multiplier[compoundcrew[CompNumber]->NumCurrentCrews] = 0;
}

*/

//------------------------------------------------------------------------------
//
//	attack dialog functions
//
//------------------------------------------------------------------------------
/*
TTransAttackData::TTransAttackData()
{
	Dat.BurnOut = false;
	Dat.Left = false;
	Dat.Right = false;
	Dat.LineProd.LowValue = 1;
	Dat.LineProd.HighValue = 50;
	Dat.LineProd.Position = 1;
	Dat.BurnDelay.LowValue = 10;
	Dat.BurnDelay.HighValue = 500;
	Dat.BurnDelay.Position = 50;
	Dat.FireDist.LowValue = 1;
	Dat.FireDist.HighValue = 200;
	Dat.FireDist.Position = 10;
	Dat.dir = true;
	Dat.indir = false;
	Dat.pdir = false;
	Dat.gdir = false;
	//sprintf(Dat.combo.Selection, "%s", "");
}
*/


//------------------------------------------------------------------------------
//
//	Ground Resources Dialog, Load Crew, Save Crew Info
//
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//
//	Right Button List Box
//
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//
//	Reassign Resources from Right Button Click
//
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//
//	Ground Resource Dialog
//
//------------------------------------------------------------------------------



//------------------------------------------------------------------------------
//
//     Group Edit Stuff
//
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//
//	Add crew dialog
//
//------------------------------------------------------------------------------



