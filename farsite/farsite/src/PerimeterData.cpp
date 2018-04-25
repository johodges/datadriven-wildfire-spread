#include "PerimeterData.h"
#include <memory.h>

CPerimeterData::CPerimeterData(void)
{
	fireType = 0;
	month = 0;
	day = 0;
	hour = 0;
	Elapsed_Minutes = 0;
	numPts = 0;
	xPts = 0;
	yPts = 0;
	next = 0;
}

CPerimeterData::~CPerimeterData(void)
{
	if(xPts)
		delete[] xPts;
	if(yPts)
		delete[] yPts;
}

void CPerimeterData::SetNumPts(long _numPoints)
{
	if(xPts)
		delete[] xPts;
	if(yPts)
		delete[] yPts;
	xPts = yPts = 0;
	numPts = _numPoints;
	if(numPts > 0)
	{
		xPts = new double[numPts];
		yPts = new double[numPts];
		memset(xPts, 0, numPts *sizeof(double));
		memset(yPts, 0, numPts *sizeof(double));
	}

}

int CPerimeterData::GetNumPts()
{
	return numPts;
}

int CPerimeterData::GetPointX(int pointNum)
{
	if(pointNum < numPts)
		return xPts[pointNum];
	return 0;
}

int CPerimeterData::GetPointY(int pointNum)
{
	if(pointNum < numPts)
		return yPts[pointNum];
	return 0;
}

void CPerimeterData::SetPoint(long pointNum, double x, double y)
{
	if(pointNum < numPts)
	{
		xPts[pointNum] = x;
		yPts[pointNum] = y;
	}
}

int CPerimeterData::SetMonth(int _month)
{
	month = _month;
	return month;
}

int CPerimeterData::GetMonth()
{
	return month;
}

int CPerimeterData::SetDay(int _day)
{
	day = _day;
	return day;
}

int CPerimeterData::GetDay()
{
	return day;
}

int CPerimeterData::SetHour(int _hour)
{
	hour = _hour;
	return hour;
}

int CPerimeterData::GetHour()
{
	return hour;
}

int CPerimeterData::SetElapsedMinutes(int _minutes)
{
	Elapsed_Minutes = _minutes;
	return Elapsed_Minutes;
}

int CPerimeterData::GetElapsedMinutes()
{
	return Elapsed_Minutes;
}

CPerimeterData *CPerimeterData::GetNext()
{
	return next;
}

CPerimeterData *CPerimeterData::SetNext(CPerimeterData *_next)
{
	next = _next;
	return next;
}

int CPerimeterData::SetFireType(int _fireType)
{
	fireType = _fireType;
	return fireType;
}

int CPerimeterData::GetFireType()
{
	return fireType;
}
