//FARSITE.h
//Defines exported class for FARSITE.dll

/*#ifdef BUILD_FARSITEDLL
#define FARSITEDLL_EXPORT __declspec(dllexport)
#else
#define FARSITEDLL_EXPORT __declspec(dllimport)
#endif */
#define FARSITEDLL_EXPORT

class Farsite5;

class FARSITEDLL_EXPORT CFarsite
{
public:
	CFarsite();
	~CFarsite();
	int SetLandscapeFile(char *_lcpFileName);
	int SetIgnition(char *shapeFileName);
	int SetBarriers(char *shapeFileName);
	int SetNumProcessors(int numThreads = 1);
	int SetStartProcessor(int _procNum);
	int GetStartProcessor();
	int GetNumIgnitionCells();

	int LoadFarsiteInputs(char *_inputFile);
	char *CommandFileError(int i_ErrNum);
	char *GetErrorMessage(int errNum);

	double GetResolution();
	long GetNumCols();
	long GetNumRows();
	double GetWest();
	double GetSouth();
	double GetEast();
	double GetNorth();

	int CanLaunchFarsite(void);
	int LaunchFarsite(void);
  double GetFarsiteProgress();
 // double GetFarsiteProgress(int *ai_RunStatus, char cr[]);
// int  GetFarsiteStatus (char cr[]);

	int CancelFarsite(void);

	//output functions
	int WriteArrivalTimeGrid(char *trgName);
	int WriteIntensityGrid(char *trgName);
	int WriteFlameLengthGrid(char *trgName);
	int WriteSpreadRateGrid(char *trgName);
	int WriteSpreadDirectionGrid(char *trgName);
	int WriteHeatPerUnitAreaGrid(char *trgName);
	int WriteReactionIntensityGrid(char *trgName);
	int WriteCrownFireGrid(char *trgName);
	int WriteArrivalTimeGridBinary(char *trgName);
	int WriteIntensityGridBinary(char *trgName);
	int WriteFlameLengthGridBinary(char *trgName);
	int WriteSpreadRateGridBinary(char *trgName);
	int WriteSpreadDirectionGridBinary(char *trgName);
	int WriteHeatPerUnitAreaGridBinary(char *trgName);
	int WriteReactionIntensityGridBinary(char *trgName);
	int WriteCrownFireGridBinary(char *trgName);
	int WriteIgnitionGrid(char *trgName);
	int WritePerimetersShapeFile(char *trgName);
	int WriteTimingsFile(char *trgName);

	int WriteOneHours(char *trgName);
	int WriteTenHours(char *trgName);
	int WriteHundredHours(char *trgName);
	int WriteMoistData(char *trgName);
	int WriteSpotDataFile(char *trgName);
	int WriteSpotShapeFile(char *trgName);
	int WriteSpotGrid(char *trgName);

private:
	Farsite5 *m_pFarsite;

};
