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
#ifndef LANDSCAPE_DATA_HEADER
#define LANDSCAPE_DATA_HEADER

class Farsite5;

typedef struct
{
	// structure for holding basic cell information
	short e;				 // elevation
	short s;				 // slope
	short a;				 // aspect
	short f;				 // fuel models
	short c;				 // canopy cover
}celldata;


typedef struct
{
	// structure for holding optional crown fuel information
	short h;				// canopy height
	short b;				// crown base
	short p;				// bulk density
}crowndata;


typedef struct
{
	// structure for holding duff and woody fuel information
	short d;				// duff model
	short w;				// coarse woody model
}grounddata;


struct CanopyCharacteristics
{
	// contains average values, landscape wide, temporary until themes are used
	double DefaultHeight;
	double DefaultBase;
	double DefaultDensity;
	double Height;
	double CrownBase;
	double BulkDensity;
	double Diameter;
	double FoliarMC;
	long Tolerance;
	long Species;
	CanopyCharacteristics();
};


typedef struct
{
	short elev;
	short slope;
	short aspect;
	short fuel;
	short cover;			   // READ OR DERIVED FROM LANDSCAPE DATA
	double aspectf;
	double height;
	double base;
	double density;
	double duff;
	long woody;
} LandscapeStruct;


class LandscapeData
{
	Farsite5 *pFarsite;
public:
	LandscapeData(Farsite5 *_pFarsite);
	LandscapeStruct ld;
	void ElevConvert(short);
	void SlopeConvert(short);
	void AspectConvert(short);
	void FuelConvert(short);
	void CoverConvert(short);
	void HeightConvert(short);
	void BaseConvert(short);
	void DensityConvert(short);
	void DuffConvert(short);
	void WoodyConvert(short);
};

#endif
