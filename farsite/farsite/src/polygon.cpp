//#include "stdafx.h"
#include "polygon.h"
#include "Farsite5.h"

const double PI=acos(-1.0);


xPolygon::xPolygon()
{
	Vertex=0;
     NumPoints=0;
}


xPolygon::~xPolygon()
{
	if(Vertex)
     	delete[] Vertex;

     Vertex=0;
     NumPoints=0;
}


bool xPolygon::AllocVertex(long Number)
{
     if(Number>NumPoints)
     {	if(Vertex)
     		delete[] Vertex;
     	Vertex=new double[Number*2];
          if(Vertex==0)
          	return false;
     }
     NumPoints=Number;

	return true;
}


bool xPolygon::SetVertex(double xpt, double ypt, long Num)
{
	if(!Vertex)
     	return false;
     if(Num>=NumPoints)
     	return false;

     Vertex[Num*2]=xpt;
     Vertex[Num*2+1]=ypt;

     if(Num==0)
     {	Xmax=Xmin=xpt;
     	Ymax=Ymin=ypt;
     }
     else
     {	if(xpt<Xmin)
     		Xmin=xpt;
     	if(xpt>Xmax)
          	Xmax=xpt;
          if(ypt<Ymin)
          	Ymin=ypt;
          if(ypt>Ymax)
          	Ymax=ypt;
     }

     return true;
}



double xPolygon::direction(double xpt1, double ypt1)
{// calculates sweep direction for angle determination
	double zangle=999.9, xdiff, ydiff;

	xdiff=xpt1-startx;
	ydiff=ypt1-starty;
    // if(fabs(xdiff)<1e-9)
     if(IsTiny(xdiff))
          xdiff=0.0;
     //if(fabs(ydiff)<1e-9)
     if(IsTiny(ydiff))
          ydiff=0.0;

	if(xdiff!=0.0)
	{	zangle=atan(ydiff/xdiff);
		if(xdiff>0.0)
			zangle=(PI/2.0)-zangle;
		else
			zangle=(3.0*PI/2.0)-zangle;
	}
	else
	{	if(ydiff>=0.0)
			zangle=0;
		else if(ydiff<0.0)
			zangle=PI;
	}

	return zangle;
}



long xPolygon::Inside(double xpt, double ypt)
{// determines if point is inside or outside a fire polygon (CurrentFire)
	long NumVertex=NumPoints;
	long count=0, count1, count2, inside=0;
	double Aangle=0.0, Bangle;
	double CuumAngle=0.0, DiffAngle;
	double Sxpt, Sypt;

     startx=xpt;
     starty=ypt;

	while(count<NumVertex)    // make sure that startx,starty!=x[0]y[0]
	{	Sxpt=Vertex[count*2];
		Sypt=Vertex[count*2+1];
		Aangle=direction(Sxpt, Sypt);
		count++;
		if(Aangle!=999.9)
			break;
	}
	for(count1=count; count1<=NumVertex; count1++)
	{    if(count1==NumVertex)
			count2=count-1;
		else
			count2=count1;
		Sxpt=Vertex[count2*2];
		Sypt=Vertex[count2*2+1];
		Bangle=direction(Sxpt, Sypt);
		if(Bangle!=999.9)
		{	DiffAngle=Bangle-Aangle;
			if(DiffAngle>PI)
				DiffAngle=-(2.0*PI-DiffAngle);
			else if(DiffAngle<-PI)
				DiffAngle=(2.0*PI+DiffAngle);
			CuumAngle-=DiffAngle;
			Aangle=Bangle;
		}
	}
	if(fabs(CuumAngle)>PI)        // if absolute value of CuumAngle is>PI
		inside=1;                // then point is inside polygon

	return inside;
}

