

#ifndef POLYGON_HEADER
#define POLYGON_HEADER

#include <stdlib.h>
#include <math.h>


class xPolygon
{
	double startx, starty;
     double *Vertex;
     long NumPoints;

public:
     double Xmax, Ymax, Xmin, Ymin;

	xPolygon();
	~xPolygon();
	long Inside(double xpt, double ypt);
	double direction(double xpt1, double ypt1);
     bool AllocVertex(long Number);
     bool SetVertex(double xpt, double ypt, long Number);
};



#endif // POLYGON
