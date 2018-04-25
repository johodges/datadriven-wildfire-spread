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
#define STRICT
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "fsdattyp.h"
#include "ftestrc.h"
#include "fsoutput.h"
//#include "fsxwinpt.h"

#ifndef BULLSHIT
#define BULLSHIT


struct TScrollBarData
{
	int LowValue;
	int HighValue;
	int Position;
};

struct TTransWSData
{
	TTransWSData();
	int S1, S2, S3, S4, S5;
	int M1, M2, M3, M4, M5;
	int D1, D2, D3, D4, D5;
};

/*
class TEditFileKB : public TEditFile
{
  public:
	TEditFileKB(TWindow*		parent = 0,
			 int			 id = 0,
			 const char far* text = 0,
			 int x = 0, int y = 0, int w = 0, int h = 0,
			 const char far* fileName = 0,
			 TModule*   	 module = 0)
	: TEditFile(parent, id, text, x, y, w, h, fileName, module) {}
  DECLARE_RESPONSE_TABLE(TEditFileKB);
};
*/
//------------------------------------------------------------------------------
//
//	LCP Generation Class && Transfer Structure
//
//------------------------------------------------------------------------------

/*
class TWStationDlg: public TDialog
{
	TCheckBox *S1, *S2, *S3, *S4, *S5;
	TCheckBox *M1, *M2, *M3, *M4, *M5;
	TCheckBox *D1, *D2, *D3, *D4, *D5;
	void EvClose();
	void CmCancel();
	void SetupWindow();
	bool CanClose();

public:
	TWStationDlg(TWindow* AParent, const char* resID, TTransWSData& ts);
	void S1Msg();
	void S2Msg();
	void S3Msg();
	void S4Msg();
	void S5Msg();
	DECLARE_RESPONSE_TABLE(TWStationDlg);
};
*/

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

/*
class FireColorControl : public TControl
{
	TBrush *Brush;
	RECT rect;
	bool White;
	long Max, Min, intervals;

	char far* GetClassName() {return "FireColorControl";}
	void EvPaint();
	COLORREF GetFireColor(long Value);

	DECLARE_RESPONSE_TABLE(FireColorControl);
public:
	FireColorControl(TWindow* Parent, int ResId, bool White, long interval);
	void SetColorWhite();
	void DrawColors(bool white, long interval);
};


class TColorControl : public TControl
{
	TColor  Color;

	char far* GetClassName() {return "ColorControl";}
	Uint Transfer(void* buffer, TTransferDirection direction);
	void EvPaint();

	DECLARE_RESPONSE_TABLE(TColorControl);
public:
	TColorControl(TWindow* parent, int resId, TColor color);
	virtual void SetColor(TColor color);
	TColor GetColor() const {return Color;}
};
*/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


struct DisplayData
{
	int CheckMetric, CheckEnglish;
	int VCheckWhite, VCheckFLI, VCheckFL, VCheckROS, VCheckHA;
	int SVGA;		//ShadeOFF, ShadeON;
	TScrollBarData MAX, INT, FM, FR, FG, FB;
	//	char ThemeData[30];
	char CMAX[10], CCINT[10], CFM[5];
};

struct TTransDisplayData
{
	DisplayData Dat;
	//TComboBoxData theme;
	TTransDisplayData();
};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


struct LandDisplayData
{
	int GridOnOff, ShadeOnOff;
	int LN, LS, LE, LW, LNE, LNW, LSE, LSW;
	int white, lgray, dgray, black;
	int VE, VS, VA, VF, VC, VH, VB, VP;
	TScrollBarData Res;
	char RES[10];
};


struct TTransLandDisplayData
{
	LandDisplayData Dat;
	TTransLandDisplayData();
};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


struct OutputData
{
	int CheckR, CheckV;
	int OCheckTOA, OCheckFLI, OCheckFL, OCheckROS, OCheckHA, OCheckRI,
		OCheckCF, OCheckFD;
	int DispEng, DispMet, OtpEng, OtpMet;
	int CheckArcVect, CheckOptVect, CheckGrassRast, CheckGridRast,
		CheckOptRast;
	int CheckVisOnly;
	int CheckShape, CheckShapeVisOnly, CheckShapeBarSep, CheckLine, CheckPoly;
	int logfile;
	char RastStatX[15];
	char RastStatY[15];
	double xres, yres;
};


struct TTransOutputData
{
	OutputData Dat;
	TTransOutputData();
	void ResetData(bool All);
};
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


struct ModelData
{
	TScrollBarData DTS, DVTS, DPRES, DSRES;
	int ChkV1, ChkV2, Metric, English;
	char SDTS[20], SDVTS[20], SDVTS2[20], SDPRES[20], SDSRES[20];
};

struct TTransModelData
{
	ModelData Dat;
	TTransModelData();
};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

struct OptionsData
{
	TScrollBarData PIgnite;
	int Meth1, Meth2, /*Meth3,*/ ExpChk, Crown, Link, SSpot, TSpot, SGrowth,
		Back;
	char pignite[20];
};


struct TTransOptionsData
{
	OptionsData Dat;
	TTransOptionsData();
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

struct DurationData
{
	int Start, Ending;//, Elapse;
	TScrollBarData Mo, Day, Hr, Min;
	int Reset;
	char STMo[10], STDay[10], STHr[10];
	char CMo[10], CDay[10], CHr[10], CMin[10];
};

struct TTransDurationData
{
	DurationData Dat;
	TTransDurationData();
};

/*
class _USERCLASS NewDuration: public TDialog
{
	 char Line[8];
	 bool CanSetStart;
	 long Val;
	 bool Ret;
	 TCheckBox *cond;
	 TUpDown *ucmo, *ucdy, *usmo, *usdy, *ushr, *usmn, *uemo, *uedy, *uehr, *uemn;
	TStatic *scmo, *scdy, *ssmo, *ssdy, *sshr, *ssmn, *semo, *sedy, *sehr, *semn;

	void SetupWindow();
	 bool CanClose();
	 void EvClose();
	 void CmHelp();
	 void CmCancel();
	 bool CheckMonth(long *Mo, long type);
	 bool CheckDay(long *Dy, long type);
	 bool CheckHour(long *Hr, long type);
	 bool CheckMin(long *Mn, long type);
	 void SetConditioning();
	 bool Cond_Mo(TNmUpDown &nm);
	 bool Cond_Dy(TNmUpDown &nm);
	 bool Start_Mo(TNmUpDown &nm);
	 bool Start_Dy(TNmUpDown &nm);
	 bool Start_Hr(TNmUpDown &nm);
	 bool Start_Mn(TNmUpDown &nm);
	 bool End_Mo(TNmUpDown &nm);
	 bool End_Dy(TNmUpDown &nm);
	 bool End_Hr(TNmUpDown &nm);
	 bool End_Mn(TNmUpDown &nm);

	DECLARE_RESPONSE_TABLE( NewDuration );
public:
	NewDuration(TWindow *Parent, const char *resID, bool CanSetStart);
};


class TSimOptions: public TDialog
{
	TCheckBox *DurReset, *IgnReset, *RotateIgn, *ShowFires, *AdjIgns, *Preserve;
	 TEdit *ethread;
	 TUpDown *udthread;

	void SetupWindow();
	bool CanClose();
	void CmHelp();
	void CmCancel();
	void EvClose();

public:
	TSimOptions(TWindow* AParent, const char* resID);

	DECLARE_RESPONSE_TABLE(TSimOptions);
};


class TPostFrontDataWindow: public TFrameWindow
{
	SIZE TextHt;
	char test[4];
	bool CanClose();
	HDC hTempDC;
	long VNumber, PFDataType, Units;
	 double Conv;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;
	 bool LButtonDown;
	 long StartLine, EndLine;
	 long StartPos, EndPos;

	 void EvClose();
	void EvSize( Uint, TSize& );
	void EvMove(TPoint &);
	void Paint(TDC& hDC, bool, TRect&);
	void EvRButtonDown( Uint, TPoint& );
	 void EvLButtonDown(Uint, TPoint&);
	 void EvLButtonUp(Uint, TPoint&);
	 void EvMouseMove(Uint, TPoint&);
	 void SendWindowToClipboard();
	 void SaveDataToFile();
	 void SelectCaption();
	 void CalcChange();

	DECLARE_RESPONSE_TABLE(TPostFrontDataWindow);

public:
	TPostFrontDataWindow(TWindow* AParent, const char* ATitle);
	~TPostFrontDataWindow(){}
};


class TFDataWindow: public TFrameWindow
{
	SIZE TextHt;
	char test[4];
	bool CanClose();
	HDC hTempDC;
	long VNumber;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;
	 bool LButtonDown;
	 long StartLine, EndLine;
	 long StartPos, EndPos;

	 void EvClose();
	void EvSize( Uint, TSize& );
	void EvMove(TPoint &);
	void Paint(TDC& hDC, bool, TRect&);
	void EvRButtonDown(Uint, TPoint&);
	 void EvLButtonDown(Uint, TPoint&);
	 void EvLButtonUp(Uint, TPoint&);
	 void EvMouseMove(Uint, TPoint&);
	 void SendWindowToClipboard();
	 void SaveDataToFile();
	 void CalcChange();

	DECLARE_RESPONSE_TABLE(TFDataWindow);

public:
	TFDataWindow(TWindow* AParent, const char* ATitle);
	~TFDataWindow(){}
};


class TADataWindow: public TFrameWindow
{
	SIZE TextHt;
	char test[4];
	bool CanClose();
	HDC hTempDC;
	long VNumber;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;
	 bool LButtonDown;
	 long StartLine, EndLine;
	 long StartPos, EndPos;

	 void EvClose();
	void EvMove(TPoint &);
	void EvSize( Uint, TSize& );
	void Paint(TDC& hDC, bool, TRect&);
	void EvRButtonDown( Uint, TPoint& );
	 void EvLButtonDown(Uint, TPoint&);
	 void EvLButtonUp(Uint, TPoint&);
	 void EvMouseMove(Uint, TPoint&);
	 void SendWindowToClipboard();
	 void SaveDataToFile();
	 void CalcChange();

	DECLARE_RESPONSE_TABLE(TADataWindow);

public:
	TADataWindow(TWindow* AParent, const char* ATitle);
	~TADataWindow()
	{
	}
};


class TPDataWindow: public TFrameWindow
{
	SIZE TextHt;
	char test[4];
	bool CanClose();
	HDC hTempDC;
	long VNumber;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;
	 bool LButtonDown;
	 long StartLine, EndLine;
	 long StartPos, EndPos;

	 void EvClose();
	void EvMove(TPoint &);
	void EvSize( Uint, TSize& );
	void Paint(TDC& hDC, bool, TRect&);
	void EvRButtonDown( Uint, TPoint& );
	 void EvLButtonDown(Uint, TPoint&);
	 void EvLButtonUp(Uint, TPoint&);
	 void EvMouseMove(Uint, TPoint&);
	 void SendWindowToClipboard();
	 void SaveDataToFile();
	 void CalcChange();

	DECLARE_RESPONSE_TABLE(TPDataWindow);

public:
	TPDataWindow(TWindow* AParent, const char* ATitle);
	~TPDataWindow(){}
};


class TGraphWindow
{
	long xtime1, xtime2;
	long i, HIY, PrintYDir;
	long GraphXEnd, GraphXStart;
	long GraphYEnd, GraphYStart;
	double GraphRes, Conversion;
	//HPEN hPen1, hPrevPen;
	TPen *Pen;
	COLORREF colr;
	HFONT hFontx, hFonty, hFontv;				// declare logical fonts

	double GetTime(long Num);					// gets elapsed time in minutes
	void AddSegment(TDC&);  				// updates line graph without redrawing whole graph
	void GetGraphDims();					// calculate x and y axes for graph
	void DrawLines(TDC&);					// draw lines of graph
	void GetPoints(long point);				// get points from Global arrays
	void GetLabels(TDC&);

public:
	long TYPE, SN, YMAX, XMAX, number;		// type of graph (area=1 perimeter=2)
	long h1, h2, s1, s2, z1, z2;
	double xmax, ymax;
	bool RefreshBitmap;
	bool Printing;
	bool Change;

	TGraphWindow();
	~TGraphWindow();
	void GraphPaint(TDC&, HWND, HBITMAP *);			// subsitute for Paint function
	void ReSet();
};


class TGADataWindow: public TOutputs
{
	 long DERIV;

	 CustomGraph g;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;

	void Paint(TDC& hDC, bool, TRect&);
	void EvSize( Uint, TSize& );			   // resize frame window and status bar
	void EvMove(TPoint &);
	void EvRButtonDown( Uint, TPoint& );
	void EvLButtonDown( Uint, TPoint& );
	 void EvClose();
	bool CanClose();
	 void SendWindowToClipboard();
	 void GetDerivative();

	DECLARE_RESPONSE_TABLE( TGADataWindow );
public:
	TGADataWindow(TWindow * AParent, const char* ATitle);
	~TGADataWindow();
	 bool CheckChange();
};


class TGPDataWindow: public TOutputs
{
	 long DERIV;

	 CustomGraph g;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;

	void Paint(TDC& hDC, bool, TRect&);
	void EvMove(TPoint &);
	void EvSize( Uint, TSize& );			   // resize frame window and status bar
	void EvRButtonDown( Uint, TPoint& );
	void EvLButtonDown( Uint, TPoint& );
	 void EvClose();
	bool CanClose();
	 void SendWindowToClipboard();
	 void GetDerivative();

	DECLARE_RESPONSE_TABLE( TGPDataWindow );
public:
	TGPDataWindow(TWindow * AParent, const char* ATitle);
	~TGPDataWindow();
	 bool CheckChange();
};


class TGTHDataWindow: public TOutputs
{
	int StationNumber;
	TGraphWindow g;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;

	void Paint(TDC& hDC, bool, TRect&);
//	void GoPrinter();
	void EvMove(TPoint &);
	void EvSize( Uint, TSize& );			   // resize frame window and status bar
	void EvRButtonDown( Uint, TPoint& );
	void EvLButtonDown( Uint, TPoint& );
	 void EvClose();
	bool CanClose();
	 void SendWindowToClipboard();

	DECLARE_RESPONSE_TABLE( TGTHDataWindow );
public:
	TGTHDataWindow(TWindow * AParent, const char* ATitle, int StatNum);
	~TGTHDataWindow();
};


class TGFMDataWindow: public TOutputs
{
	int StationNumber;
	TGraphWindow g;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;

	void Paint(TDC& hDC, bool, TRect&);
//	void GoPrinter();
	void EvMove(TPoint &);
	void EvSize( Uint, TSize& );			   // resize frame window and status bar
	void EvRButtonDown( Uint, TPoint& );
	void EvLButtonDown( Uint, TPoint& );
	 void EvClose();
	bool CanClose();
	 void SendWindowToClipboard();

	DECLARE_RESPONSE_TABLE( TGFMDataWindow );
public:
	TGFMDataWindow(TWindow * AParent, const char* ATitle, int StatNum);
	~TGFMDataWindow();
};


double GetPostFrontalData(long Species, long Phase, long Num);

class TPostFrontGraphWindow: public TOutputs
{
	 bool *PFW_ON, ForceDraw;
	 long DERIV;
	 long MaxXDim, MaxYDim;
	 long Species;

	 CustomGraph g;
	WindowSettings *WinRect;
	WINDOWPLACEMENT wpt;

	void Paint(TDC& hDC, bool, TRect&);
	void EvSize( Uint, TSize& );			   // resize frame window and status bar
	void EvMove(TPoint &);
	void EvRButtonDown( Uint, TPoint& );
	void EvLButtonDown( Uint, TPoint& );
	 void EvClose();
	bool CanClose();
	 void SendWindowToClipboard();
	 void GetDerivative();

	DECLARE_RESPONSE_TABLE( TPostFrontGraphWindow );
public:
	TPostFrontGraphWindow(TWindow * AParent, const char* ATitle,
	 		long Option, bool *pfw_on);
	~TPostFrontGraphWindow();
	 bool CheckChange();
};



class LandInfo: public TDialog
{
	 char LCPFileName[256];
	 TButton 	*save;
	 TEdit 	*edit;

	 bool CanClose();
	 void CmCancel();
	 void CmHelp();
	 void EvClose();
	 void SetupWindow();
	void SaveFile();
	 void WriteData();
	DECLARE_RESPONSE_TABLE(LandInfo);

public:
	LandInfo(TWindow *parent, const char* resID, char *filename);
};


class ViewPortControl : public TControl
{
	RECT DefRect, VPRect;
	long NumNorth, NumEast;
	double XResol, YResol;
	COLORREF colr1, colr2;

	char far* GetClassName() {return "ViewPortControl";}
	//Uint Transfer(void* buffer, TTransferDirection direction);
	void EvPaint();

	DECLARE_RESPONSE_TABLE(ViewPortControl);
public:
	ViewPortControl(TWindow* parent, int resId);
	ViewPortControl(TWindow* parent, int resId, RECT rect,
				 long numnorth, long numeast, double xresol, double yresol);
	void DrawPort(RECT rect);
	void SetLandscape(RECT rect, long numnorth, long numeast,
							double xresol, double yresol);
};


class MergeControl: public TControl
{
	COLORREF colr[4];
	POint LandPoint[5][5];
	long NumRect;
	RECT rect;
	double West[5], East[5], North[5], South[5];
	double Width, Height;

	void EvPaint();
	char far* GetClassName() {return "MergeControl";}

	DECLARE_RESPONSE_TABLE(MergeControl);
public:
	MergeControl(TWindow* parent, int ResID);
	void SetRect(long RectNum, double UtmNorth, double UtmSouth,
						  double UtmWest, double UtmEast);
	 void ResetRect();
};


class MergeLCPFile: public TDialog
{
	TCheckBox 	*FirstFile, *SecondFile, *ThirdFile, *FourthFile, *FifthFile;
	TColorStatic 	*FirstName, *SecondName, *ThirdName, *FourthName;
	TStatic 		*FifthName;
	TStatic 		*East5, *West5, *North5, *South5, *Width5, *Height5, *Rows5, *Cols5;
	TStatic 		*Result;
	TButton 		*Exit, *Merge;
	 TGauge 		*Prog;
	 TEdit		*Desc;
	MergeControl 	*MergeCTL;
	FILE 		*file[5];
	char 		*TempName1, *TempName2, *TempName3, *TempName4, *TempName5;
	headdata 		Head[5];
	celldata 		CellDat;
	crowndata 	CrownDat;
	 grounddata 	GroundDat;
	long 		OldFilePosition[4];
	long 		CrownFuelsPresent;
	 long 		GroundFuelsPresent;
	 long 		NumAllCats[10];
	 long 		AllCats[10][100];
	 double 		maxval[10], minval[10];
	char 		Value[20];

	bool CanClose();
	void CmCancel();
	void EvClose();
	void CmHelp();
	void SetupWindow();
	bool CalcHeader5(long HeadNumber);
	void WriteHeader5();
	long GetCellPos(double east, double north, long FileNo);
	bool CellInfo(double east, double north, long FileNo);
	bool InsideFile1(long FileNum, double xpt, double ypt);
	void EliminateFile(long FileNo);
	 void UpdateProgress(double fract);
	 void FinalizeHeader(headdata *newheader);
	 void FillCats(celldata *cell, crowndata *crown, grounddata *ground);
	 void SortCats();
	 void InMsg();
	 void OutMsg();
	 char *GetFileName(char *name);
	 void AppendFileNames(char *oldname, char *newname);

public:
	MergeLCPFile(TWindow * AParent, const char * ATitle);
	~MergeLCPFile() {};

	void FileOneMessage();
	void FileTwoMessage();
	void FileThreeMessage();
	void FileFourMessage();
	void FileFiveMessage();
	void ExitMessage();
	void MergeMessage();

	DECLARE_RESPONSE_TABLE(MergeLCPFile);
};


class ExtractLCPFile: public TDialog
{
	TCheckBox 	*FirstFile, *SecondFile;
	TScrollBar 	*North, *South, *East, *West;
	TColorStatic 	*File1, *File2;
	TStatic 		*OutNorth, *InNorth, *OutSouth, *InSouth;
	TStatic 		*OutEast, *InEast, *OutWest, *InWest;
	TStatic 		*Result, *InRow, *InCol, *OutRow, *OutCol;
	ViewPortControl *vp;
	TButton 		*Exit, *Extract, *ViewPort, *In, *Out;
	 TEdit		*Desc;
	 TGauge 		*Prog;
	headdata ExHead, OutHead;
	celldata CellDat;
	crowndata CrownDat;
	 grounddata GroundDat;
	FILE *file1, *file2;
	RECT rect;
	char *TempName1, *TempName2;
	double north, south, east, west;
	long OldFilePosition, cols, rows;
	char Value[20];
	long 		CrownFuelsPresent;
	 long 		GroundFuelsPresent;
	 long 		NumAllCats[10];
	 long 		AllCats[10][100];
	 double 		maxval[10], minval[10];
	bool CurrentLandscape;

	bool CanClose();
	void EvClose();
	void CmHelp();
	void CmCancel();
	void SetupWindow();
	long GetCellPos(double east, double north);
	void CellInfo(double east, double north);
	 void FinalizeHeader(headdata *newheader);
	 void FillCats(celldata *cell, crowndata *crown, grounddata *ground);
	 void SortCats();
	 void UpdateProgress(double fract);
	 void InMsg();
	 void OutMsg();

public:
	ExtractLCPFile(TWindow* AParent, const char * ATitle);
	~ExtractLCPFile() {};

	void FileOneMessage();
	void FileTwoMessage();
	void NorthScrollMessage( Uint );
	void SouthScrollMessage( Uint );
	void EastScrollMessage( Uint );
	void WestScrollMessage( Uint );
	void ViewPortMessage();
	void EnableViewPortMessage(TCommandEnabler &vpe);
	void ExitMessage();
	void ExtractMessage();

	DECLARE_RESPONSE_TABLE(ExtractLCPFile);
};


class ViewPortDialog: public TDialog
{
	TStatic *Nw, *Ne, *Nn, *Ns;
	TStatic *Ow, *Oe, *On, *Os;
	TStatic *OutRow, *InRow, *OutCol, *InCol;
	TButton *Maximize, *Zoom, *In, *Out, *Fire;
	TScrollBar *west, *east, *north, *south;
	ViewPortControl *vp;
	RECT rect;
	long BarValue, cols, rows;
	double DCols, DRows, ResX, ResY;
	char Value[21];
	double North, South, East, West;

	bool CanClose();
	void CmCancel();
	void CmHelp();
	void EvClose();
	void SetupWindow();
	void DrawViewPort();
	 void ResetViewData();
	 void ViewFireMsg();
	void VPNorthMsg(Uint);
	void VPSouthMsg(Uint);
	void VPEastMsg(Uint);
	void VPWestMsg(Uint);
	void VPMaxMsg();
	void VPZoomMsg();
	 void ViewInMsg();
	 void ViewOutMsg();

	DECLARE_RESPONSE_TABLE(ViewPortDialog);
public:
	ViewPortDialog(TWindow* AParent, const char* ATitle, RECT rect);
};


class ViewRasterOptions: public TDialog
{
	TCheckBox *FileCheck;
	TStatic *FileName;
	TStatic *MinElev, *MaxElev;
	TStatic *MinHt, *MaxHt;
	TScrollBar *Min, *Max;

	 long hi, lo;
	long BarValue;
	char Value[21];
	char Units[5];
	char *filename;

	bool CanClose();
	void CmCancel();
	void CmHelp();
	void EvClose();
	void SetupWindow();
	bool OpenReadRasterFile();

	DECLARE_RESPONSE_TABLE(ViewRasterOptions);
public:
	ViewRasterOptions(TWindow* AParent, const char* ATitle, bool HaveFileAlready);

	void FileNameMsg();
	void MinHeightMsg(Uint);
	void MaxHeightMsg(Uint);
};


class ZoomDelay: public TDialog
{
	 TUpDown   *delay;
	 TEdit     *sdelay;

	 bool CanClose();
	void CmCancel();
	void EvClose();
	 void SetupWindow();

	DECLARE_RESPONSE_TABLE(ZoomDelay);
public:
	 ZoomDelay(TWindow* AParent, const char* ATitle);
};

*/
struct TransEnvtMap
{
	char FileName[256];
	int one, ten, hund, thou, duff;
	int mfw, temp, humid, solar;
	int x1, x2, x5, x10, x20;
	int vp, nw;
	int eng, met;
	char Mo[8], Dy[8], Hr[8];
	TransEnvtMap();
};

/*
class _USERCLASS EnvtMaps: public TDialog
{
	char FileName[256];
	TRadioButton *one, *ten, *hund, *thou, *duff;
	 TRadioButton *mfw, *temp, *humid, *solar;
	 TRadioButton *x1, *x2, *x5, *x10, *x20;
	 TRadioButton *eng, *met;
	 TCheckBox *vponly, *now;
	 TButton *name;
	 TStatic *sname, *start, *end;
	 TEdit *mo, *dy, *time;
	 double *When, SimTime;

	 bool CanClose();
	 void CmCancel();
	 void CmHelp();
	 void EvClose();
	 void SetupWindow();
	 void DuffMsg();
	 void ThouMsg();
	 void SaveFile();
	 void NowCheck();
	 void EnglishMessage();
	 void MetricMessage();

	 DECLARE_RESPONSE_TABLE(EnvtMaps);
public:
	EnvtMaps(TWindow *Parent, const char *resID, TransEnvtMap &ts, double *When);
};

*/

struct TransCombMap
{
	char FileName[256];
	char Resolution[16];
	int enrate, fuelrate, flame, pm25, pm10, co, co2, ch4;
	int x1, xc;
	int vp;
	TransCombMap();
};

/*
class _USERCLASS CombMaps: public TDialog
{
	char FileName[256];
	 TRadioButton *enrate, *fuelrate, *flame, *pm25, *pm10, *co, *co2, *ch4;
	 TRadioButton *x1, *xc;
	 TCheckBox *vponly;
	 TButton 	*name;
	 TStatic 	*sname;
	 TEdit 	*cres;

	 bool CanClose();
	 void CmCancel();
	 void CmHelp();
	 void EvClose();
	 void CustRes();
	 void SetupWindow();
	 void SaveFile();

	 DECLARE_RESPONSE_TABLE(CombMaps);

public:
	CombMaps(TWindow *Parent, const char *resID, TransCombMap &ts);
};


class _USERCLASS FuelMoisture: public TDialog
{
	 bool 		IMPORT;
	 char 		*FileName, *Desc;
	 char 		TempName[256], TempDesc[512];

	 TButton 		*load, *save, *simdat;
	 TEdit		*desc;
	 TRadioButton 	*pre, *as;
	 TEdit 		*time[3], *elev[3], *slope[3], *asp[3], *cov[3];
	 TStatic		*name;
	FireEnvironment2 *Env;
	 FarInputs		*farinpt;

	 void LoadFile();
	 void SaveFile();
	 void InsertSimDat();
	 bool CanClose();
	 void CmCancel();
	 void CmHelp();
	 void EvClose();
	 void SetupWindow();

	 DECLARE_RESPONSE_TABLE(FuelMoisture);

public:
	FuelMoisture(TWindow *Parent, const char *resID, char *FileName, char *Description, FireEnvironment2 *env, FarInputs *inpt);
};


class _USERCLASS TClickList: public TListBox
{
	void EvLButtonDown(uint, TPoint& pt);

	 DECLARE_RESPONSE_TABLE(TClickList);
public:
	TClickList(TWindow *Parent, int resID);
};


class _USERCLASS EditStop: public TDialog
{
	TEdit 		*xloc, *yloc;
	 TClickList	*stoploc;
	 TButton 		*add, *edit, *del, *enable;

	 void AddStopLoc();
	 void EditStopLoc();
	 void DelStopLoc();
	 void EnableStop();
	 void ListStopLoc();
	 void WriteCoords();
	 bool CanClose();
	 void CmCancel();
	 void CmHelp();
	 void EvClose();
	 void SetupWindow();

	 DECLARE_RESPONSE_TABLE(EditStop);
public:
	EditStop(TWindow *Parent, const char *resID);
};
*/

#endif
