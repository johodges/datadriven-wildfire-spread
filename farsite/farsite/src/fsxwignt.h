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
#ifndef IgnitionHeader
#define IgnitionHeader

#include "fsx4.hpp"
//#include "fsglbvar.h"
#include "shapefil.h"
//#include "Farsite5.h"
class Farsite5;

class IgnitionFile : public IgnitionCorrect
{
	FILE* IFile;
	long count, count2;
	long fposition1, NumVertex;
	double startsize;
	double CenterX, CenterY, xpt, ypt, angle;//, FireArea;
	Farsite5 *pFarsite;
	bool lightsLandscape;
public:
	char TestEnd[30];
	char ifile[256];
	//HWND hW;

	IgnitionFile(Farsite5 *_pFarsite);
	~IgnitionFile();

	void SelectFileInput() ;
	bool GrassFile() ;
	bool ArcLine()  ;
	bool ArcPoly() ;
	bool ArcPoint() ;
	bool ShapeInput() ;
	void SelectFileInputCmdL(int type) ;
	bool GetLightsLandscape();
	bool SetLightsLandscape(bool lightsLCP);
};

#endif

