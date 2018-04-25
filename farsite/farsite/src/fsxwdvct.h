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
#include "shapefil.h"
#include "themes.h"

//-----------------------------------------------------------------------------
//
//	Draw2DVectors Struct
//
//------------------------------------------------------------------------------

#ifndef DRAW2DVECTOR_DECS
#define DRAW2DVECTOR_DECS

class ImportVectors
{
	int prev_mode, prev_memmode;
	char TestEnd[30];
	long xc, yc, type, count;
	long OldEasti, OldNorthi;
	double xrat, yrat, MaxX, MaxY;
	double xpt, ypt;
	HBITMAP garbage;
	HPEN hPen, hMemPen, hOldPen, hOldMemPen;
	HDC hDC, hMemDC;
	FILE* compfile;

	bool TestOpenFile();
	bool DrawVisFile();
	bool DrawGrassFile();
	bool DrawArcFile();
	bool DrawShapeFile();
	void StartDrawing();
	void EndDrawing();

public:
	VectorTheme theme;

	ImportVectors(char* FileName);
	ImportVectors();
	bool TestForShapeFile();
	bool TestForGrassFile();
	bool TestForArcFile();
	bool TestForVisFile();
	bool SetArcFileType(char* Type);
	bool DrawFile();
};

#endif //DRAW2DVECTOR_DECS
