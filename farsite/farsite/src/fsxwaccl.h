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
//#include "fsclrsta.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <owl\dialog.h>
//#include <owl\control.h>
//#include <owl\scrollba.h>
//#include <owl\button.h>
//#include <owl\static.h>
//#include <owl\opensave.h>
//#include <owl\radiobut.h>
//#include <owl\gdiobjec.h>

typedef struct
{
	char AccelLoad[256];
	char AccelSave[256];
} AccelData;


struct TTransAccelData
{
	AccelData Dat;

	TTransAccelData();
	void ResetData();
};



