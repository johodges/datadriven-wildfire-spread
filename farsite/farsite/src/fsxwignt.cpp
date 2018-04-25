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
 *   - http://www.gpoaccess.gov/uscode/  (enter "cd 17USC105" in the search box.)
 */
#define IDYES 1
#define IDNO 2

#include <string.h>
#include "fsxwignt.h"
#include "Farsite5.h"
//#include "portablestrings.h"

const double PI = acos(-1.0);
//extern const double PI;

IgnitionFile::IgnitionFile(Farsite5 *_pFarsite) : IgnitionCorrect(_pFarsite)
{
	startsize = 1;
	pFarsite = _pFarsite;
	lightsLandscape = false;
}

IgnitionFile::~IgnitionFile()
{
}


void IgnitionFile::SelectFileInput()
{
	/*char Test[5] = "";
	long len = strlen(ifile);

	Test[0] = ifile[len - 4];
	Test[1] = ifile[len - 3];
	Test[2] = ifile[len - 2];
	Test[3] = ifile[len - 1];
	if (!strcmp(strlwr(Test), ".shp"))
		ShapeInput();
	else if ((IFile = fopen(ifile, "r")) != NULL)
	{
		int IResponse;
		fscanf(IFile, "%s", TestEnd);
		rewind(IFile);
		if (!strcmp(_strupr(TestEnd), "ORGANIZATION:"))
			GrassFile();
		else
		{
		 //IResponse = MessageBox(hW, "Does This Arc File contain closed POLYGON ignitions??", ifile, MB_YESNOCANCEL);
		IResponse= IDYES;
			if (IResponse == IDYES){
				ArcPoly();

			}else if (IResponse == IDNO)
			{
				IResponse = IDYES;
				if (IResponse == IDYES){
					ArcLine();
				}else if (IResponse == IDNO)
				{
					ArcPoint();

				}
			}
		}
		fclose(IFile);
	}*/
}


bool IgnitionFile::ShapeInput()
{
	SHPHandle hSHP;
	SHPObject* pSHP;
	int nShapeType, nEntities;
	long i, j, k, m, start, end, NewFire, NumPts, OldNumPts;
	double xpt, ypt;
	double xdiff, ydiff, midx, midy;
	double MinBound[4], MaxBound[4];

	hSHP = SHPOpen(ifile, "rb");
	if (hSHP == NULL)
		return false;

	SHPGetInfo(hSHP, &nEntities, &nShapeType, MinBound, MaxBound);
	for (i = 0; i < nEntities; i++)
	{
		pSHP = SHPReadObject(hSHP, i);
		if (pSHP->nVertices <= 0)
		{
			SHPDestroyObject(pSHP);
			continue;
		}

		NewFire = pFarsite->GetNewFires();
		switch (nShapeType)
		{
		case SHPT_POINT:
			// intentional fall through
		case SHPT_POINTZ:
		case SHPT_POINTM:
		case SHPT_MULTIPOINT:
		case SHPT_MULTIPOINTZ:
		case SHPT_MULTIPOINTM:
			for (j = 0; j < pSHP->nVertices; j++)
			{
				NewFire = pFarsite->GetNewFires();
				pFarsite->AllocPerimeter1(NewFire, 11);
				CenterX = pFarsite->ConvertUtmToEastingOffset(pSHP->padfX[j]);
				CenterY = pFarsite->ConvertUtmToNorthingOffset(pSHP->padfY[j]);
				for (k = 0; k <= 9; k++)
				{
					angle = (double (k) * (PI / 5.0) + PI / 2.0);
					xpt = CenterX + (cos(angle) * startsize);
					ypt = CenterY + (sin(angle) * startsize);
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), k, xpt, ypt);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), k, 0.0, 0.0);
					pFarsite->SetReact(pFarsite->GetNewFires(), k, 0.0);
				}
				pFarsite->SetNumPoints(pFarsite->GetNewFires(), 10);
				pFarsite->SetInout(NewFire, 1);
				pFarsite->IncNewFires(1);
				BoundingBox(NewFire);	  		// find bounding box for line source
			}
			break;
		case SHPT_ARC:
		case SHPT_ARCZ:
		case SHPT_ARCM:
			for (m = 0; m < pSHP->nParts; m++)
			{
				NewFire = pFarsite->GetNewFires();
				start = pSHP->panPartStart[m];
				if (m < pSHP->nParts - 1)
					end = pSHP->panPartStart[m + 1];
				else
					end = pSHP->nVertices;
				if ((end - start) == 2)
				{
					pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), 5);		// 1 more for bounding box
					CenterX = pFarsite->ConvertUtmToEastingOffset(pSHP->padfX[0]);
					CenterY = pFarsite->ConvertUtmToNorthingOffset(pSHP->padfY[0]);
					xpt = pFarsite->ConvertUtmToEastingOffset(pSHP->padfX[1]);
					ypt = pFarsite->ConvertUtmToNorthingOffset(pSHP->padfY[1]);
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 0, CenterX, CenterY);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), 0, -1.0, 0.0);	// 100% of equilibrium spread
					pFarsite->SetReact(pFarsite->GetNewFires(), 0, 0.0);
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 2, xpt, ypt);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), 2, -1.0, 0.0);	// 100% of equilibrium spread
					pFarsite->SetReact(pFarsite->GetNewFires(), 2, 0.0);
					xdiff = CenterX - xpt;
					ydiff = CenterY - ypt;
					midx = CenterX - xdiff / 2.0;   	//-xdiff/100.0;
					midy = CenterY - ydiff / 2.0;   	//-xdiff/100.0*oppslope;
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 1, midx, midy);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), 1, -1.0, 0.0);	// 100% of equilibrium spread
					pFarsite->SetReact(pFarsite->GetNewFires(), 1, 0.0);
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 3, midx, midy);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), 3, -1.0, 0.0);	// 100% of equilibrium spread
					pFarsite->SetReact(pFarsite->GetNewFires(), 3, 0.0);
					pFarsite->SetNumPoints(pFarsite->GetNewFires(), 4);
				}
				else if ((end - start) > 0)
				{
					pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), 2 * (end - start) - 1);
					count2 = 2 * (end - start) - 3;
					for (j = start; j < end; j++)
					{
						CenterX = pFarsite->ConvertUtmToEastingOffset(pSHP->padfX[j]);
						CenterY = pFarsite->ConvertUtmToNorthingOffset(pSHP->padfY[j]);
						pFarsite->SetPerimeter1(pFarsite->GetNewFires(), j - start, CenterX,
							CenterY);
						pFarsite->SetFireChx(pFarsite->GetNewFires(), j - start, -1.0, 0.0);	// 100% of equilibrium spread
						pFarsite->SetReact(pFarsite->GetNewFires(), j - start, 0.0);
						if (j > start && k < end - 1)   			   	// write points bacward in array
						{
							pFarsite->SetPerimeter1(pFarsite->GetNewFires(), count2, CenterX,
								CenterY);
							pFarsite->SetFireChx(pFarsite->GetNewFires(), count2, -1.0, 0.0);	// 100% of equilibrium spread
							pFarsite->SetReact(pFarsite->GetNewFires(), count2, 0.0);
							--count2;
						}
					}
					pFarsite->SetNumPoints(pFarsite->GetNewFires(), 2 * (end - start) - 2);
				}
				pFarsite->SetInout(pFarsite->GetNewFires(), 1);
				pFarsite->IncNewFires(1);
				BoundingBox(NewFire);	  		// find bounding box for line source
			}
			break;
		case SHPT_POLYGON:
		case SHPT_POLYGONZ:
		case SHPT_POLYGONM:
		case SHPT_MULTIPATCH:
			for (m = 0; m < pSHP->nParts; m++)
			{
				NewFire = pFarsite->GetNewFires();
				start = pSHP->panPartStart[m];
				if (m < pSHP->nParts - 1)
					end = pSHP->panPartStart[m + 1];
				else
					end = pSHP->nVertices;
				if ((end - start) <= 0)
					continue;
				pFarsite->AllocPerimeter1(NewFire, (end - start) + 1);
				for (j = start; j < end; j++)
				{
					xpt = pFarsite->ConvertUtmToEastingOffset(pSHP->padfX[j]);
					ypt = pFarsite->ConvertUtmToNorthingOffset(pSHP->padfY[j]);
					pFarsite->SetPerimeter1(NewFire, j - start, xpt, ypt);
					pFarsite->SetFireChx(NewFire, j - start, -1.0, 0.0);
					pFarsite->SetReact(NewFire, j - start, 0.0);
				}
				pFarsite->SetNumPoints(NewFire, end - start);
				pFarsite->SetInout(NewFire, 1);
				pFarsite->IncNewFires(1);
				if (arp() < 0)
					ReversePoints(1);
				BoundingBox(NewFire);	  		// find bounding box for line source
			}
			break;
		}
		SHPDestroyObject(pSHP);
		RemoveIdenticalPoints(NewFire);

		xpt = pFarsite->GetPerimeter1Value(NewFire, pFarsite->GetNumPoints(NewFire), XCOORD);
		ypt = pFarsite->GetPerimeter1Value(NewFire, pFarsite->GetNumPoints(NewFire), YCOORD);
		midx = pFarsite->GetPerimeter1Value(NewFire, pFarsite->GetNumPoints(NewFire), ROSVAL);
		midy = pFarsite->GetPerimeter1Value(NewFire, pFarsite->GetNumPoints(NewFire), FLIVAL);
		do
		{
			OldNumPts = pFarsite->GetNumPoints(NewFire);
			DensityControl(NewFire);
			NumPts = pFarsite->GetNumPoints(NewFire);
			pFarsite->FreePerimeter1(NewFire);
			pFarsite->AllocPerimeter1(NewFire, NumPts + 1);
			pFarsite->SwapFirePerims(NewFire, -NumPts);
		}
		while (NumPts > OldNumPts);
		pFarsite->SetPerimeter1(NewFire, NumPts, xpt, ypt);	// replace bounding box
		pFarsite->SetFireChx(NewFire, NumPts, midx, midy);
		pFarsite->SetReact(NewFire, NumPts, 0.0);
	}
	SHPClose(hSHP);

	return true;
}


bool IgnitionFile::GrassFile()
{
/*	char Head[30];
	long count, NewFire, OldNumPts, NumPts;
	double xpt, ypt;
	double xdiff, ydiff, midx, midy;//, oppslope;

	do
	{
		fscanf(IFile, "%s", Head);
	}
	while (strcmp(Head, "VERTI:"));
	while (!feof(IFile))
	{
		fscanf(IFile, "%s %li", Head, &NumVertex);
		NewFire = GetNewFires();
		if (!strcmp(_strupr(Head), "A"))
		{
			AllocPerimeter1(GetNewFires(), NumVertex + 1);
			for (count = 0; count < NumVertex; count++)
			{
				fscanf(IFile, "%lf %lf", &ypt, &xpt);   // grass reverses x & y
				xpt = ConvertUtmToEastingOffset(xpt);
				ypt = ConvertUtmToNorthingOffset(ypt);
				SetPerimeter1(GetNewFires(), count, xpt, ypt);
				SetFireChx(GetNewFires(), count, -1.0, 0.0);
				SetReact(GetNewFires(), count, 0.0);
			}
			SetNumPoints(GetNewFires(), count);
			SetInout(GetNewFires(), 1);
			IncNewFires(1);
			if (arp() < 0)
				ReversePoints(1);
			BoundingBox(GetNewFires() - 1);							// find bounding box for line source
		}
		else if (!strcmp(_strupr(Head), "L"))
		{
			if (NumVertex == 2)
			{
				AllocPerimeter1(GetNewFires(), 5);		// 1 more for bounding box
				fscanf(IFile, "%lf %lf", &CenterY, &CenterX);
				fscanf(IFile, "%lf %lf", &ypt, &xpt);
				CenterX = ConvertUtmToEastingOffset(CenterX);
				CenterY = ConvertUtmToNorthingOffset(CenterY);
				xpt = ConvertUtmToEastingOffset(xpt);
				ypt = ConvertUtmToNorthingOffset(ypt);
				SetPerimeter1(GetNewFires(), 0, CenterX, CenterY);
				SetFireChx(GetNewFires(), 0, -1.0, 0.0);	// 100% of equilibrium spread
				SetReact(GetNewFires(), 0, 0.0);
				SetPerimeter1(GetNewFires(), 2, xpt, ypt);
				SetFireChx(GetNewFires(), 2, -1.0, 0.0);	// 100% of equilibrium spread
				SetReact(GetNewFires(), 2, 0.0);
				xdiff = CenterX - xpt;
				ydiff = CenterY - ypt;
				midx = CenterX - xdiff / 2.0;   	//-xdiff/100.0;
				midy = CenterY - ydiff / 2.0;   	//-xdiff/100.0*oppslope;
				SetPerimeter1(GetNewFires(), 1, midx, midy);
				SetFireChx(GetNewFires(), 1, -1.0, 0.0);	// 100% of equilibrium spread
				SetReact(GetNewFires(), 1, 0.0);
				SetPerimeter1(GetNewFires(), 3, midx, midy);
				SetFireChx(GetNewFires(), 3, -1.0, 0.0);	// 100% of equilibrium spread
				SetReact(GetNewFires(), 3, 0.0);
				SetNumPoints(GetNewFires(), 4);
			}
			else
			{
				AllocPerimeter1(GetNewFires(), 2 * NumVertex - 1);		// 1 more for bounding box
				count2 = 2 * NumVertex - 3;
				for (count = 0; count < NumVertex; count++)
				{
					fscanf(IFile, "%lf %lf", &CenterY, &CenterX);
					CenterX = ConvertUtmToEastingOffset(CenterX);
					CenterY = ConvertUtmToNorthingOffset(CenterY);
					SetPerimeter1(GetNewFires(), count, CenterX, CenterY);
					SetFireChx(GetNewFires(), count, -1.0, 0.0);	// 100% of equilibrium spread
					SetReact(GetNewFires(), count, 0.0);
					if (count > 0 && count < NumVertex - 1) 			// write points bacward in array
					{
						SetPerimeter1(GetNewFires(), count2, CenterX, CenterY);
						SetFireChx(GetNewFires(), count2, -1.0, 0.0);	// 100% of equilibrium spread
						SetReact(GetNewFires(), count2, 0.0);
						--count2;
					}
				}
				SetNumPoints(GetNewFires(), 2 * NumVertex - 2);
			}
			SetInout(GetNewFires(), 1);
			IncNewFires(1);
			BoundingBox(GetNewFires() - 1);							// find bounding box for line source
			fscanf(IFile, "%s", TestEnd);
		}
		RemoveIdenticalPoints(NewFire);
		xpt = GetPerimeter1Value(NewFire, GetNumPoints(NewFire), XCOORD);
		ypt = GetPerimeter1Value(NewFire, GetNumPoints(NewFire), YCOORD);
		midx = GetPerimeter1Value(NewFire, GetNumPoints(NewFire), ROSVAL);
		midy = GetPerimeter1Value(NewFire, GetNumPoints(NewFire), FLIVAL);
		do
		{
			OldNumPts = GetNumPoints(NewFire);
			DensityControl(NewFire);
			NumPts = GetNumPoints(NewFire);
			FreePerimeter1(NewFire);
			AllocPerimeter1(NewFire, NumPts + 1);
			SwapFirePerims(NewFire, -NumPts);
		}
		while (NumPts > OldNumPts);
		SetPerimeter1(NewFire, NumPts, xpt, ypt);	// replace bounding box
		SetFireChx(NewFire, NumPts, midx, midy);
		SetReact(NewFire, NumPts, 0.0);
	}

	return true;*/
	return false;
}


bool IgnitionFile::ArcLine()
{
	long OldFires, NewFire, NumPts, OldNumPts;
	double xdiff, ydiff, midx, midy;//, oppslope;

	OldFires = NewFire = pFarsite->GetNewFires();
	fscanf(IFile, "%s", TestEnd);		// get past label location
	do
	{
		NewFire = pFarsite->GetNewFires();
		fposition1 = ftell(IFile);
		NumVertex = 0;
		do
		{
			fscanf(IFile, "%s", TestEnd);
			if (!strcmp(TestEnd, "END"))
				break;
			fscanf(IFile, "%lf", &CenterY);
			NumVertex++;
		}
		while (strcmp(TestEnd, "END"));
		if (NumVertex == 2)
		{
			fseek(IFile, fposition1, SEEK_SET);
			pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), 5);		// 1 more for bounding box
			fscanf(IFile, "%lf %lf", &CenterX, &CenterY);
			fscanf(IFile, "%lf %lf", &xpt, &ypt);
			CenterX = pFarsite->ConvertUtmToEastingOffset(CenterX);
			CenterY = pFarsite->ConvertUtmToNorthingOffset(CenterY);
			xpt = pFarsite->ConvertUtmToEastingOffset(xpt);
			ypt = pFarsite->ConvertUtmToNorthingOffset(ypt);
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 0, CenterX, CenterY);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), 0, -1.0, 0.0);	// 100% of equilibrium spread
			pFarsite->SetReact(pFarsite->GetNewFires(), 0, 0.0);
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 2, xpt, ypt);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), 2, -1.0, 0.0);	// 100% of equilibrium spread
			pFarsite->SetReact(pFarsite->GetNewFires(), 2, 0.0);
			xdiff = CenterX - xpt;
			ydiff = CenterY - ypt;
			midx = CenterX - xdiff / 2.0;   	//-xdiff/100.0;
			midy = CenterY - ydiff / 2.0;   	//-xdiff/100.0*oppslope;
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 1, midx, midy);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), 1, -1.0, 0.0);	// 100% of equilibrium spread
			pFarsite->SetReact(pFarsite->GetNewFires(), 1, 0.0);
			pFarsite->SetPerimeter1(pFarsite->GetNewFires(), 3, midx, midy);
			pFarsite->SetFireChx(pFarsite->GetNewFires(), 3, -1.0, 0.0);	// 100% of equilibrium spread
			pFarsite->SetReact(pFarsite->GetNewFires(), 3, 0.0);
			count = 3;
			pFarsite->SetNumPoints(pFarsite->GetNewFires(), 4);
			RemoveIdenticalPoints(NewFire);
		}
		else if (NumVertex > 1)
		{
			fseek(IFile, fposition1, SEEK_SET);
			pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), 2 * NumVertex - 1);		// 1 more for bounding box
			count2 = 2 * NumVertex - 3;
			for (count = 0; count < NumVertex; count++)
			{
				fscanf(IFile, "%lf %lf", &CenterX, &CenterY);
				CenterX = pFarsite->ConvertUtmToEastingOffset(CenterX);
				CenterY = pFarsite->ConvertUtmToNorthingOffset(CenterY);
				pFarsite->SetPerimeter1(pFarsite->GetNewFires(), count, CenterX, CenterY);
				pFarsite->SetFireChx(pFarsite->GetNewFires(), count, -1.0, 0.0);	// 100% of equilibrium spread
				pFarsite->SetReact(pFarsite->GetNewFires(), count, 0.0);
				if (count > 0 && count < NumVertex - 1) 				 // write points bacward in array
				{
					pFarsite->SetPerimeter1(pFarsite->GetNewFires(), count2, CenterX, CenterY);
					pFarsite->SetFireChx(pFarsite->GetNewFires(), count2, -1.0, 0.0);	// 100% of equilibrium spread
					pFarsite->SetReact(pFarsite->GetNewFires(), count2, 0.0);
					--count2;
				}
			}
			fscanf(IFile, "%s", TestEnd);
			pFarsite->SetNumPoints(pFarsite->GetNewFires(), 2 * NumVertex - 2);
			RemoveIdenticalPoints(NewFire);
		}
		if (NumVertex > 1)
		{
			pFarsite->SetInout(pFarsite->GetNewFires(), 1);
			pFarsite->IncNewFires(1);
			do
			{
				OldNumPts = pFarsite->GetNumPoints(NewFire);
				DensityControl(NewFire);
				NumPts = pFarsite->GetNumPoints(NewFire);
				pFarsite->FreePerimeter1(NewFire);
				pFarsite->AllocPerimeter1(NewFire, NumPts + 1);
				pFarsite->SwapFirePerims(NewFire, -NumPts);
			}
			while (NumPts > OldNumPts);
			BoundingBox(NewFire);
		}							// find bounding box for line source
		fscanf(IFile, "%s", TestEnd);
	}
	while (strcmp(TestEnd, "END"));
	if (OldFires == pFarsite->GetNewFires())
		printf("Line Ignition Contains only One Point Ignition NOT Imported\n");

	return true;
}


bool IgnitionFile::ArcPoly()
{
	long OldFires, NumPts, NewFire, OldNumPts;

	OldFires = NewFire = pFarsite->GetNewFires();
	fscanf(IFile, "%s", TestEnd);		// get past label location
	do
	{
		NewFire = pFarsite->GetNewFires();
		fposition1 = ftell(IFile);
		NumVertex = 0;
		do
		{
			fscanf(IFile, "%s", TestEnd);
			if (!strcmp(TestEnd, "END"))
				break;
			fscanf(IFile, "%lf", &CenterY);
			NumVertex++;
		}
		while (strcmp(TestEnd, "END"));
		if (NumVertex > 2)
		{
			fseek(IFile, fposition1, SEEK_SET);
			pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), NumVertex + 1);		// 1 more for bounding box
			for (count = 0; count < NumVertex; count++)
			{
				fscanf(IFile, "%lf %lf", &CenterX, &CenterY);
				CenterX = pFarsite->ConvertUtmToEastingOffset(CenterX);
				CenterY = pFarsite->ConvertUtmToNorthingOffset(CenterY);
				pFarsite->SetPerimeter1(pFarsite->GetNewFires(), count, CenterX, CenterY);
				pFarsite->SetFireChx(pFarsite->GetNewFires(), count, -1.0, 0.0);	// 100% of equilibrium spread
				pFarsite->SetReact(pFarsite->GetNewFires(), count, 0.0);
			}
			fscanf(IFile, "%s", TestEnd);
			pFarsite->SetNumPoints(pFarsite->GetNewFires(), NumVertex);
			pFarsite->SetInout(pFarsite->GetNewFires(), 1);
			pFarsite->IncNewFires(1);
			RemoveIdenticalPoints(NewFire);
			if (arp() < 0)
				ReversePoints(1);
			do
			{
				OldNumPts = pFarsite->GetNumPoints(NewFire);
				DensityControl(NewFire);
				NumPts = pFarsite->GetNumPoints(NewFire);
				pFarsite->FreePerimeter1(NewFire);
				pFarsite->AllocPerimeter1(NewFire, NumPts + 1);
				pFarsite->SwapFirePerims(NewFire, -NumPts);
			}
			while (NumPts > OldNumPts);
			BoundingBox(NewFire);
		}							// find bounding box for line source
		fscanf(IFile, "%s", TestEnd);
	}
	while (strcmp(TestEnd, "END"));
	if (OldFires == pFarsite->GetNewFires())
		printf("Polygon Ignition Contains only One Point Ignition NOT Imported\n");
	return true;
}


bool IgnitionFile::ArcPoint()
{
	while (!feof(IFile))
	{
		fscanf(IFile, "%s", TestEnd);
		if (strcmp(TestEnd, "END"))
		{
			fscanf(IFile, "%lf %lf", &CenterX, &CenterY);
			CenterX = pFarsite->ConvertUtmToEastingOffset(CenterX);
			CenterY = pFarsite->ConvertUtmToNorthingOffset(CenterY);
			pFarsite->AllocPerimeter1(pFarsite->GetNewFires(), 11);		// 10 points, 11th is for bounding box
			for (count = 0; count <= 9; count++)
			{
				angle = (double (count) * (PI / 5.0)) + PI / 2.0;
				xpt = CenterX + (cos(angle) * startsize);
				ypt = CenterY + (sin(angle) * startsize);
				pFarsite->SetPerimeter1(pFarsite->GetNewFires(), count, xpt, ypt);
				pFarsite->SetFireChx(pFarsite->GetNewFires(), count, 0.0, 0.0);
				pFarsite->SetReact(pFarsite->GetNewFires(), count, 0.0);
			}
			pFarsite->SetNumPoints(pFarsite->GetNewFires(), count);
			pFarsite->SetInout(pFarsite->GetNewFires(), 1);
			pFarsite->IncNewFires(1);
			BoundingBox(pFarsite->GetNewFires() - 1);
			fscanf(IFile, "%s", TestEnd);
		}
	}

	return true;
}


IgnitionCorrect::IgnitionCorrect(Farsite5 *_pFarsite) : APolygon(_pFarsite), StandardizePolygon(_pFarsite)
{
	pFarsite = _pFarsite;
}

IgnitionCorrect::~IgnitionCorrect()
{
}

void IgnitionCorrect::ReversePoints(long TYPE)
{
	long j, AFire = pFarsite->GetNewFires() - 1;
	long count, BFire = pFarsite->GetNewFires();
	double fxc, fyc, fxc2, fyc2, RosI, RosI2;
	long halfstop = pFarsite->GetNumPoints(AFire) / 2;				// truncated number of points

	switch (TYPE)
	{
	case 0:
		// transfer points in reverse to next array, +1 fires
		pFarsite->AllocPerimeter1(BFire, pFarsite->GetNumPoints(AFire)); 	// allocate new array
		for (count = 0; count < pFarsite->GetNumPoints(AFire); count++)
		{
			fxc = pFarsite->GetPerimeter1Value(AFire, count, 0);
			fyc = pFarsite->GetPerimeter1Value(AFire, count, 1);
			RosI = pFarsite->GetPerimeter1Value(AFire, count, 2);
			j = pFarsite->GetNumPoints(AFire) - (count + 1);
			pFarsite->SetPerimeter1(BFire, j, fxc, fyc);  		 // reverse points
			pFarsite->SetFireChx(BFire, j, RosI, 0);  			 // in next available array
			pFarsite->SetReact(BFire, j, 0.0);
		}   									   // set bounding box
		pFarsite->SetPerimeter1(BFire, count, pFarsite->GetPerimeter1Value(AFire, count, 0),
			pFarsite->GetPerimeter1Value(AFire, count, 1));  		 // reverse points
		pFarsite->SetFireChx(BFire, count, pFarsite->GetPerimeter1Value(AFire, count, 2),
			pFarsite->GetPerimeter1Value(AFire, count, 3)); 				 // in next available array
		pFarsite->SetReact(BFire, count, 0.0);
		pFarsite->SetNumPoints(BFire, count);		// same number of points in fire
		pFarsite->SetInout(BFire, 1); 	   		// ID fire as outward burning
		//			IncNumFires(1);
		pFarsite->IncNewFires(1);
		break;
	case 1:
		// reverse points in same array
		for (count = 0; count < halfstop; count++)
		{
			j = pFarsite->GetNumPoints(AFire) - count - 1;
			fxc = pFarsite->GetPerimeter1Value(AFire, count, 0);
			fyc = pFarsite->GetPerimeter1Value(AFire, count, 1);
			RosI = pFarsite->GetPerimeter1Value(AFire, count, 2);
			fxc2 = pFarsite->GetPerimeter1Value(AFire, j, 0);
			fyc2 = pFarsite->GetPerimeter1Value(AFire, j, 1);
			RosI2 = pFarsite->GetPerimeter1Value(AFire, j, 2);
			pFarsite->SetPerimeter1(AFire, count, fxc2, fyc2);
			pFarsite->SetFireChx(AFire, count, RosI2, 0);
			pFarsite->SetReact(AFire, j, 0.0);
			pFarsite->SetPerimeter1(AFire, j, fxc, fyc);
			pFarsite->SetFireChx(AFire, j, RosI, 0);
			pFarsite->SetReact(AFire, j, 0.0);
		}
		break;
	}
}

/*
void IgnitionCorrect::BoundingBox()
{
	double xpt, ypt, Xlo, Xhi, Ylo, Yhi;
	long NumFire=GetNewFires()-1;
	long NumPoint=GetNumPoints(NumFire);

	Xlo=Xhi=GetPerimeter1Value(NumFire, 0, 0);
	Ylo=Yhi=GetPerimeter1Value(NumFire, 0, 1);
	for(int i=1; i<NumPoint; i++)
	{	xpt=GetPerimeter1Value(NumFire, i, 0);
		ypt=GetPerimeter1Value(NumFire, i, 1);
		if(xpt<Xlo) Xlo=xpt;
		else
		{	if(xpt>Xhi)
				Xhi=xpt;
		}
		if(ypt<Ylo) Ylo=ypt;
		else
		{	if(ypt>Yhi)
			Yhi=ypt;
		}
	}
	SetPerimeter1(NumFire, NumPoint, Xlo, Xhi);
	SetFireChx(NumFire, NumPoint, Ylo, Yhi);
}
*/

double IgnitionCorrect::arp()
	// CALCULATES AREA AND PERIMETER AS A PLANIMETER (WITH TRIANGLES)
{
	long count, count1 = 0, FireNum = pFarsite->GetNewFires() - 1, numx;
	double xpt1, ypt1, xpt2, ypt2, aangle, zangle, DiffAngle;
	double newarea, area = 0.0;

	numx = pFarsite->GetNumPoints(FireNum);
	if (numx < 3)
		return area;

	startx = pFarsite->GetPerimeter1Value(FireNum, 0, 0);
	starty = pFarsite->GetPerimeter1Value(FireNum, 0, 1);
	while (count1 < numx)    // make sure that startx,starty!=x[0]y[0]
	{
		count1++;
		xpt1 = pFarsite->GetPerimeter1Value(FireNum, count1, 0);
		ypt1 = pFarsite->GetPerimeter1Value(FireNum, count1, 1);
		zangle = direction(xpt1, ypt1);
		if (zangle != 999.9)
			break;
	}
	count1++;
	for (count = count1; count < numx; count++)
	{
		xpt2 = pFarsite->GetPerimeter1Value(FireNum, count, 0);
		ypt2 = pFarsite->GetPerimeter1Value(FireNum, count, 1);
		newarea = .5 * (startx * ypt1 -
			xpt1 * starty +
			xpt1 * ypt2 -
			xpt2 * ypt1 +
			xpt2 * starty -
			startx * ypt2);
		newarea = fabs(newarea);
		aangle = direction(xpt2, ypt2);
		if (aangle != 999.9)
		{
			DiffAngle = aangle - zangle;
			if (DiffAngle > PI)
				DiffAngle = -(2.0 * PI - DiffAngle);
			else if (DiffAngle < -PI)
				DiffAngle = (2.0 * PI + DiffAngle);
			if (DiffAngle > 0.0)
				area -= newarea;
			else if (DiffAngle < 0.0)
				area += newarea;
			zangle = aangle;
		}
		xpt1 = xpt2;
		ypt1 = ypt2;
	}

	return area;
}


void StandardizePolygon::RemoveIdenticalPoints(long NumFire)
{
	long i, j, k, NumPts;
	double xpt, ypt, xptn, yptn, xn, yn;
	double xdiff, ydiff, dist, offset;
	double* perimeter1;
	double minDist = 1e-9;
	//minDist = pFarsite->GetPerimRes() / 4.0;
	NumPts = pFarsite->GetNumPoints(NumFire);
	if(NumPts > 8)//if we have at least 8 points use the user input distance
		minDist = pFarsite->icf.f_FarsiteMinIgnitionVertexDistance;//5.0;//
	for (i = 0; i < NumPts; i++)
	{
		xpt = pFarsite->GetPerimeter1Value(NumFire, i, XCOORD);
		ypt = pFarsite->GetPerimeter1Value(NumFire, i, YCOORD);
		for (j = 0; j < NumPts; j++)
		{
			if (i == j)
				continue;
			xptn = pFarsite->GetPerimeter1Value(NumFire, j, XCOORD);
			yptn = pFarsite->GetPerimeter1Value(NumFire, j, YCOORD);
			dist = sqrt(pow2(xpt - xptn) + pow2(ypt - yptn));
			if (dist < minDist)
			{
				k = j + 1;
				if (k > NumPts - 1)
					k = 0;
				xn = pFarsite->GetPerimeter1Value(NumFire, k, XCOORD);
				yn = pFarsite->GetPerimeter1Value(NumFire, k, YCOORD);
				xdiff = xptn - xn;
				ydiff = yptn - yn;
				dist = sqrt(pow2(xdiff) + pow2(ydiff));
				if (dist < minDist)
				{
					if(k > 0)
					{
						perimeter1 = pFarsite->GetPerimeter1Address(NumFire, 0);
						memcpy(&perimeter1[j * NUMDATA],
							&perimeter1[(j + 1) * NUMDATA],
							(NumPts - j) * NUMDATA * sizeof(double));
					}
					NumPts--;
				}
				else
				{
					offset = dist * 0.01;
					if (offset > 0.1)
						offset = 0.1;
					xptn = xptn - offset * xdiff / dist;
					yptn = yptn - offset * ydiff / dist;
					pFarsite->SetPerimeter1(NumFire, j, xptn, yptn);
				}
			}
		}
	}
	if (NumPts < pFarsite->GetNumPoints(NumFire))
		pFarsite->SetNumPoints(NumFire, NumPts);
}

/** this is processed from the CmdLine
* for right now
* - 1 is ArcPoly
* - 2 is ArcLine
* - 3 is ArcPoint
*/
void IgnitionFile::SelectFileInputCmdL(int type)
{
/*	char Test[5] = "";
	long len = strlen(ifile);

	Test[0] = ifile[len - 4];
	Test[1] = ifile[len - 3];
	Test[2] = ifile[len - 2];
	Test[3] = ifile[len - 1];
	if (!strcmp(_strlwr(Test), ".shp"))
		ShapeInput();
	else if ((IFile = fopen(ifile, "r")) != NULL)
	{
		fscanf(IFile, "%s", TestEnd);
		rewind(IFile);
		if (!strcmp(_strupr(TestEnd), "ORGANIZATION:"))
			GrassFile();
		else
		{
			if (type == 1){
				ArcPoly();
		        }else if (type == 2){
				ArcLine();
		        }else{
				ArcPoint();
		        }
		}
		fclose(IFile);
	}*/
	//ADD a failure to open notification
}

bool IgnitionFile::GetLightsLandscape()
{
	return lightsLandscape;
}

bool IgnitionFile::SetLightsLandscape(bool lightsLCP)
{
	lightsLandscape = lightsLCP;
	return lightsLandscape;
}
