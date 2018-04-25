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
//	FSXWBAR.H    Use of Vector Barriers to Fire Spread
//
//
//				Copyright 1994, 1995, 1996
//				Mark A. Finney, Systems for Environemntal Management
//******************************************************************************
#ifndef VECTORBARRIERS
#define VECTORBARRIERS

class Farsite5;

class VectorBarrier
{
	long NumVertices;
	double DiffRes;
	double* Barrier;
	double* BufBarrier;
	double DistanceResolution;
	double BarrierDistance;

	bool ReDiscretizeBarrier();
	void ReBuffer(long BarrierNumber);
	void FreeBarrier();
	Farsite5 *pFarsite;

public:
	VectorBarrier(Farsite5 *_pFarsite);
	~VectorBarrier();
	bool AllocBarrier(long VertNumber);
	void SetBarrierVertex(long NumVecs, double xpt, double ypt);
	void BufferBarrier(double DistanceMultiplier);
	bool ReBufferBarriers(); // buffer the barrier by given width
	bool TransferBarrier(long NumFire);
};


#endif
