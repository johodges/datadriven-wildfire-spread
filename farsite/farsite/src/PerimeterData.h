#pragma once

class CPerimeterData
{
public:
	CPerimeterData(void);
	~CPerimeterData(void);
	void SetNumPts(long _numPoints);
	int GetNumPts();
	int GetPointX(int pointNum);
	int GetPointY(int pointNum);
	void SetPoint(long pointNum, double x, double y);
	int SetFireType(int _fireType);
	int GetFireType();
	int SetMonth(int _month);
	int GetMonth();
	int SetDay(int _day);
	int GetDay();
	int SetHour(int _hour);
	int GetHour();
	int SetElapsedMinutes(int _minutes);
	int GetElapsedMinutes();
	CPerimeterData *GetNext();
	CPerimeterData *SetNext(CPerimeterData *_next);
private:
	int fireType;
	int month;
	int day;
	int hour;
	double Elapsed_Minutes;
	long numPts;
	double *xPts;
	double *yPts;
	CPerimeterData *next;

};
