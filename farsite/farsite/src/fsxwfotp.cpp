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
//#include <io.h>
#include <sys/stat.h>
#include <string.h>
#include <algorithm>
#include "Farsite5.h"
#include "FARSITE.h"

using namespace std;

OutputFile::OutputFile(Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
	ld = NULL;
	filepos = 0;
	NumRastAlloc = 0;
	NumRastData = 0;
	ld = new LandscapeData(_pFarsite);
};


OutputFile::~OutputFile()
{
	FreeRastData();
	if(ld)
		delete ld;
	ld = NULL;
};


void OutputFile::SelectOutputs(long OutputFormat)
{
	setHeaderType(OutputFormat) ;

	// save current state of object to map
	SetRastData(x,y,t,f,r,rx,d) ;

	if (OutputFormat == 1 || OutputFormat == 5)
	{
		WriteRastMemFiles();
	}
	else if (OutputFormat > 1)
	{
		FileOutput = OutputFormat;	// specify vector file for optional
		OptionalOutput(false);
	}
}

void OutputFile::SelectMemOutputs(long OutputFormat)
{
	setHeaderType(OutputFormat) ;

	if (OutputFormat == 1 || OutputFormat == 5)
	{
		WriteRastMemFiles();
	}
	else if (OutputFormat > 1)
	{
		FileOutput = OutputFormat;	// specify vector file for optional
		OptionalOutput(true);
	}
}



void OutputFile::WriteRastMemFiles()
{
	WriteFile(RAST_ARRIVALTIME);
	if (pFarsite->GetFileOutputOptions(RAST_FIREINTENSITY))
		WriteFile(RAST_FIREINTENSITY);
	if (pFarsite->GetFileOutputOptions(RAST_FLAMELENGTH))
		WriteFile(RAST_FLAMELENGTH);
	if (pFarsite->GetFileOutputOptions(RAST_SPREADRATE))
		WriteFile(RAST_SPREADRATE);
	if (pFarsite->GetFileOutputOptions(RAST_HEATPERAREA))
		WriteFile(RAST_HEATPERAREA);
	if (pFarsite->GetFileOutputOptions(RAST_REACTIONINTENSITY))
		WriteFile(RAST_REACTIONINTENSITY);
	if (pFarsite->GetFileOutputOptions(RAST_CROWNFIRE))
		WriteFile(RAST_CROWNFIRE);
	if (pFarsite->GetFileOutputOptions(RAST_FIREDIRECTION))
		WriteFile(RAST_FIREDIRECTION);
}



int OutputFile::RastMemFile(long Type)
{
	double OutDoubleData = 0.0;
	long OutLongData = 0;

	// loop over all rows and all columns
	for (long j=0; j < numrows; j++)
	{
		for (long i=0; i < numcols; i++)
		{

			// check to see if this cell has raster data written to it.
			coordinate testxy = std::make_pair(i,j) ;
			RasterMap::iterator data = rd.find(testxy) ;

			// if this cell has no data yet, just print a -1
			if (data == rd.end())
			{
				// output -1 as either a float or a long.
				if(Type == RAST_ARRIVALTIME)
				{
					if(pFarsite->ignitionGrid[j][i] == 1.0)
					{
						if(fprintf(otpfile, "0.0 ") < 0)
							return e_EMS_FILE_WRITE_ERROR;
					}
					else//unburned point
					{
						if(fprintf(otpfile, "-9999.0 ") < 0)
							return e_EMS_FILE_WRITE_ERROR;
					}
				}
				else if (Type == RAST_FIREINTENSITY ||
				 	Type == RAST_HEATPERAREA ||
					Type == RAST_REACTIONINTENSITY ||
				    Type == RAST_FLAMELENGTH ||
				    Type == RAST_SPREADRATE)
				{
					if(fprintf(otpfile, "-9999.0 ") < 0)
							return e_EMS_FILE_WRITE_ERROR;
					//fprintf(otpfile, "%.3lf ", NODATA_VAL);
				}
				else
				{
					if(fprintf(otpfile, "-9999 ") < 0)
							return e_EMS_FILE_WRITE_ERROR;
					//fprintf(otpfile, "%ld ", NODATA_VAL) ;
				}
			}
			else
			{
				/*if (Type != RAST_ARRIVALTIME)
				{
					if ((data->second).Write == false)
						continue;
				}*/
				x = (data->second).x;
				y = (data->second).y;
				t = (data->second).Time;
				f = (data->second).Fli;
				r = (data->second).Ros;
				rx = (data->second).Rcx;
				d = (data->second).Dir;
				switch (Type)
				{
				case RAST_ARRIVALTIME:
					{
						OutDoubleData = t * 60.0;
					}
					break;
				case RAST_FIREINTENSITY:
					OutDoubleData = f / convf1;
					break;
				case RAST_FLAMELENGTH:
					Calcs(FLAME_LENGTH);					// calculate flame length
					OutDoubleData = l;
					break;
				case RAST_SPREADRATE:
					OutDoubleData = r * convf2;
					break;
				case RAST_HEATPERAREA:
					Calcs(HEAT_PER_AREA);   				  // calculate heat/unit area
					OutDoubleData = h;
					break;
				case RAST_REACTIONINTENSITY:
					Calcs(HEAT_PER_AREA);   				  // calculate heat/unit area
					OutDoubleData = rx;
					break;
				case RAST_CROWNFIRE:
					Calcs(CROWNFIRE);
					OutLongData = c;
					break;
				case RAST_FIREDIRECTION:
					OutLongData = d;
					break;
				}
				if(pFarsite->ignitionGrid[j][i] == 1.0)
				{
					if(Type == RAST_ARRIVALTIME)
						OutDoubleData = 0.0;//ignition point
					else
					{
						OutDoubleData = -9999.0;//ignition point
						OutLongData = -9999.0;
					}
				}
				else if(OutDoubleData < 0.0)
				{
					OutDoubleData = NODATA_VAL;
				}
				if(pFarsite->ignitionGrid[j][i] == 0.0)//barrier cludge
				{
					OutDoubleData = -9999.0;
					OutLongData = -9999.0;
				}
				if (Type == RAST_ARRIVALTIME)
				{
					if(fprintf(otpfile, "%09.3lf ", OutDoubleData) < 0)
							return e_EMS_FILE_WRITE_ERROR;
					//(data->second).Write = true;
				}
				else if (Type == RAST_FIREINTENSITY ||
					Type == RAST_HEATPERAREA ||
					Type == RAST_REACTIONINTENSITY)
				{
					if(fprintf(otpfile, "%09.3lf ", OutDoubleData) < 0)
							return e_EMS_FILE_WRITE_ERROR;
				}
				else if (Type == RAST_FLAMELENGTH || Type == RAST_SPREADRATE)
				{
					if(fprintf(otpfile, "%05.2lf ", OutDoubleData) < 0)
							return e_EMS_FILE_WRITE_ERROR;
				}
				else
				{
					if(fprintf(otpfile, "%03ld ", OutLongData) < 0)
							return e_EMS_FILE_WRITE_ERROR;
				}
			}
		}
		if(fprintf(otpfile, "\n") < 0)
			return e_EMS_FILE_WRITE_ERROR;
	}

	return 1;
}

/*
 * \brief encloses the logic to write either a single value to the "optional"
 * output file, or writes the entire accumulated array to the file.  THis
 * does not output in a gridded format, but rather just dumps the accumulated
 * points to the file.
 */
void OutputFile::OptionalOutput(bool FromMemory)
{
	char RasterCopy[256] = "";

	if (FileOutput == 3)
		strcpy(RasterCopy, pFarsite->GetRasterFileName(0));
	else
		strcpy(RasterCopy, pFarsite->GetVectorFileName());
	if (filepos == 0)
	{
		otpfile = fopen(RasterCopy, "w");
		filepos = 1;
	}
	else
		otpfile = fopen(RasterCopy, "a");

	if (otpfile == NULL)
		return;
	if (FromMemory)
	{
		RasterMap::const_iterator data ;
		for (data = rd.begin(); data != rd.end(); ++data)
		{
			x = (data->second).x;
			y = (data->second).y;
			t = (data->second).Time;
			f = (data->second).Fli;
			r = (data->second).Ros;
			rx = (data->second).Rcx;
			d = (data->second).Dir;
			WriteOptionalFile();
			fprintf(otpfile, "\n");
		}
	}
	else
	{
		WriteOptionalFile();
		fprintf(otpfile, "\n");
	}
	fclose(otpfile);
}


/**
 * \brief Writes a single value to the already opened, already positioned
 * output file.
 */
void OutputFile::WriteOptionalFile()
{
	x = pFarsite->ConvertEastingOffsetToUtm(x);
	y = pFarsite->ConvertNorthingOffsetToUtm(y);
	fprintf(otpfile, "%010.4lf %010.4lf %010.4lf", x, y, t);
	if (pFarsite->GetFileOutputOptions(RAST_FIREINTENSITY))
		fprintf(otpfile, " %010.4lf", f / convf1);
	if (pFarsite->GetFileOutputOptions(RAST_FLAMELENGTH))
	{
		Calcs(FLAME_LENGTH);
		fprintf(otpfile, " %010.4lf", l);
	}
	if (pFarsite->GetFileOutputOptions(RAST_SPREADRATE))
		fprintf(otpfile, " %010.4lf", r * convf2);
	if (pFarsite->GetFileOutputOptions(RAST_HEATPERAREA))
	{
		Calcs(HEAT_PER_AREA);
		fprintf(otpfile, " %010.4lf", h);
	}
	if (pFarsite->GetFileOutputOptions(RAST_REACTIONINTENSITY))
	{
		Calcs(HEAT_PER_AREA);
		fprintf(otpfile, " %010.4lf", rx);
	}
	if (pFarsite->GetFileOutputOptions(RAST_CROWNFIRE))
	{
		Calcs(CROWNFIRE);
		fprintf(otpfile, " %ld", c);
	}
	if (pFarsite->GetFileOutputOptions(RAST_FIREDIRECTION))
		fprintf(otpfile, " %ld", d);
}


void OutputFile::Calcs(CalcType TYPE)
{
	long garbage;
	double Io, Ro;
	celldata cd;
	crowndata rd;
	grounddata gd;

	c = 0;

	pFarsite->CellData(x, y, cd, rd, gd, &garbage);
	if (cd.c > 0)
	{
		ld->BaseConvert(rd.b);
		ld->DensityConvert(rd.p);
		c = 1;
		Io = pow(0.010 * ld->ld.base * (460.0 + 25.9 * pFarsite->GetFoliarMC()), 1.5);
		if (f >= Io)
		{
			c = 2;
			if (ld->ld.density > 0.0)
			{
				Ro = 2.8 / ld->ld.density;     // close to the ro+0.9(rac-ro)
				if (r >= Ro)
					c = 3;
			}
		}
	}
	else if (r > 1e-6)
		c = 1;

	switch (TYPE)
	{
	case FLAME_LENGTH:
		if (c > 1)
			l = ((0.2 * pow(f / 3.4613, 2.0 / 3.0)) / 3.2808) * convf2;		// fl for crown fire from Thomas in Rothermel 1991 converted to m
		else
			l = 0.0775 * pow(f, 0.46) * convf2;						// fl from Byram for surface fires
		break;
	case HEAT_PER_AREA:
		if (r > 1e-6)
			h = (60.0 * f / convf2) / (r * convf1);
		else
			h = 0.0;
		break; 	// just hpua
	case FL_AND_HPA:
		if (c > 1)
			l = ((0.2 * pow(f / 3.4613, 2.0 / 3.0)) / 3.2808) * convf2;		// fl for crown fire from Thomas in Rothermel 1991 converted to m
		else
			l = 0.0775 * pow(f, 0.46) * convf2;						// fl from Byram for surface fires
		if (r > 1e-6)
			h = (60.0 * f / convf2) / (r * convf1);
		else
			h = 0.0;
		break;
	case CROWNFIRE:
		/*CellData(x, y, cd, rd, gd, &garbage);
			if(cd.c>0)
			{    ld.BaseConvert(rd.b);
			 	ld.DensityConvert(rd.p);
			 	c=1;
			 	Io=pow(0.010*ld.ld.base* (460.0+25.9*GetFoliarMC()),1.5);
			 	if(f>=Io)
			 	{    c=2;
				 		if(ld.ld.density>0.0)
				 		{    Ro=2.8/ld.ld.density;     // close to the ro+0.9(rac-ro)
					  		if(r>=Ro)
								c=3;
				 		}
				  }
			 }
			 else
			 if(r>1e-6)
				  c=1;
			 */
		break;
	case REACTION_INTENSITY:
		rx /= convf3;
		break;
	default:
	   ; // do nothing
	}
}


void OutputFile::ConvF()
{
	convf1 = convf2 = convf3 = 1.0;
	if (pFarsite->AccessOutputUnits(GETVAL))
	{
		convf1 = 3.4613;
		convf2 = 3.2808;
		convf3 = 0.18189275;
	}
}


void OutputFile::InitRasterFiles(long FileType)
{
	HeaderType = FileType ;
//	FirstFile(FileType, RAST_ARRIVALTIME);
//	if (GetFileOutputOptions(RAST_FIREINTENSITY))
//		FirstFile(FileType, RAST_FIREINTENSITY);
//	if (GetFileOutputOptions(RAST_FLAMELENGTH))
//		FirstFile(FileType, RAST_FLAMELENGTH);
//	if (GetFileOutputOptions(RAST_SPREADRATE))
//		FirstFile(FileType, RAST_SPREADRATE);
//	if (GetFileOutputOptions(RAST_HEATPERAREA))
//		FirstFile(FileType, RAST_HEATPERAREA);
//	if (GetFileOutputOptions(RAST_REACTIONINTENSITY))
//		FirstFile(FileType, RAST_REACTIONINTENSITY);
//	if (GetFileOutputOptions(RAST_CROWNFIRE))
//		FirstFile(FileType, RAST_CROWNFIRE);
//	if (GetFileOutputOptions(RAST_FIREDIRECTION))
//		FirstFile(FileType, RAST_FIREDIRECTION);
	pFarsite->CanSetRasterResolution(0);  	  // false
}

int OutputFile::FarsiteLayerTypeToThemeType(int farsiteType)
{
	switch(farsiteType)
	{
	case RAST_FLAMELENGTH://Flamelength
		return 0;
	case RAST_SPREADRATE://SPREADRATE
		return 1;
	case RAST_FIREINTENSITY://INTENSITY
		return 2;
	case RAST_HEATPERAREA://HEATAREA
		return 3;
	case RAST_CROWNFIRE://CROWNSTATE
		return 4;
	case RAST_FIREDIRECTION://MAXSPREADDIR
		return 10;
	case RAST_REACTIONINTENSITY://RAST_REACTIONINTENSITY
		return RAST_REACTIONINTENSITY;
	}
	return 1;
}
int OutputFile::WriteBinaryHeader(FILE *trg, int outType)
{
	if(!trg)
		return e_EMS_FILE_OPEN_ERROR;

	pFarsite->CanSetRasterResolution(0);  	  // false
	float nullVal = NODATA_VAL;
	//float zeroVal = 0.0;
	double xres, yres;
	long numcells;
	double MetersToKm = 1.0;
		North = pFarsite->GetHiNorth();
		South = pFarsite->GetLoNorth();
		East =pFarsite-> GetHiEast();
		West = pFarsite->GetLoEast();
		//West =pFarsite-> GetHiEast();
		//East = pFarsite->GetLoEast();
	pFarsite->GetRastRes(&xres, &yres);
	//long numNorth = (long) ((North - South) / yres), numEast = (long) ((East - West) / xres);
	float cellResX = xres, westUTM,
		southUTM;
			//fprintf(otpfile, "%s %lf\n", "XLLCORNER",
				westUTM = pFarsite->ConvertEastingOffsetToUtm(West);//*MetersToKm);
			//fprintf(otpfile, "%s %lf\n", "YLLCORNER",
				southUTM = pFarsite->ConvertNorthingOffsetToUtm(South);//*MetersToKm);

	numcols = (long) ((East - West) / xres);//((GetEastUtm()-GetWestUtm())/xres);
	numrows = (long) ((North - South) / yres);//((GetNorthUtm()-GetSouthUtm())/yres);
	numcells = numrows * numcols;
	//memset(RasterCopy, 0x0, sizeof RasterCopy);
	//strcpy(RasterCopy, pFarsite->GetRasterFileName(Type));

	if (pFarsite->CheckCellResUnits() == 2)
		MetersToKm = 0.001;

	char textBuf[64];
	//memset(textBuf, '
	//long layerType = outType;
	switch(outType)
	{
	case RAST_ARRIVALTIME:
		//strcpy(textBuf, "FARSITE_ArrivalTime");
		sprintf(textBuf, "%-32.32s", "FARSITE_ArrivalTime");
		break;
	case RAST_FIREINTENSITY:
		//strcpy(textBuf, "FARSITE_Intensity");
		sprintf(textBuf, "%-32.32s", "FARSITE_Intensity");
		break;
	case RAST_SPREADRATE:
		//strcpy(textBuf, "FARSITE_SpreadRate");
		sprintf(textBuf, "%-32.32s", "FARSITE_SpreadRate");
		break;
	case RAST_FLAMELENGTH:
		//strcpy(textBuf, "FARSITE_FlameLength");
		sprintf(textBuf, "%-32.32s", "FARSITE_FlameLength");
		break;
	case RAST_HEATPERAREA:
		//strcpy(textBuf, "FARSITE_HeatPerArea");
		sprintf(textBuf, "%-32.32s", "FARSITE_HeatPerArea");
		break;
	case RAST_CROWNFIRE:
		//strcpy(textBuf, "FARSITE_CrownFire");
		sprintf(textBuf, "%-32.32s", "FARSITE_CrownFire");
		break;
	case RAST_FIREDIRECTION:
		//strcpy(textBuf, "FARSITE_FireDirection");
		sprintf(textBuf, "%-32.32s", "FARSITE_FireDirection");
		break;
	case RAST_REACTIONINTENSITY:
		//strcpy(textBuf, "FARSITE_ReactionIntensity");
		sprintf(textBuf, "%-32.32s", "FARSITE_ReactionIntensity");
		break;
	default://unknown type
		return 0;
	}
	if(fwrite(textBuf, sizeof(char), 32, trg) != 32)
		return e_EMS_FILE_WRITE_ERROR;
	int32 themeType = FarsiteLayerTypeToThemeType(outType);
	if(fwrite(&themeType, sizeof(int32), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;
	if(fwrite(&numcols, sizeof(numcols), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;
	if(fwrite(&numrows, sizeof(numrows), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;
	if(fwrite(&westUTM, sizeof(westUTM), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;
	if(fwrite(&southUTM, sizeof(southUTM), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;
	if(fwrite(&cellResX, sizeof(cellResX), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;
	if(fwrite(&nullVal, sizeof(nullVal), 1, trg) != 1)
		return e_EMS_FILE_WRITE_ERROR;

	return 1;
}

int OutputFile::WriteBinaryData(FILE *trg, int outType)
{
	if(!trg)
		return e_EMS_FILE_OPEN_ERROR;
	float writeVal, zeroVal = 0.0, nullVal = NODATA_VAL;
	for (long j=0; j < numrows; j++)
	{
		for (long i=0; i < numcols; i++)
		{

			// check to see if this cell has raster data written to it.
			coordinate testxy = std::make_pair(i,j) ;
			//coordinate testxy = std::make_pair(j,i) ;
			RasterMap::iterator data = rd.find(testxy) ;

			// if this cell has no data yet, just print a -1
			if (data == rd.end())
			{
				switch(outType)
				{
				case RAST_ARRIVALTIME:
					if(pFarsite->ignitionGrid[j][i] == 1.0)
					{
						if(fwrite(&zeroVal, sizeof(float), 1, trg) != 1)
							return e_EMS_FILE_WRITE_ERROR;
					}
					else//unburned point
					{
						if(fwrite(&nullVal, sizeof(float), 1, trg) != 1)
							return e_EMS_FILE_WRITE_ERROR;
					}
					break;
				default:
					if(fwrite(&nullVal, sizeof(float), 1, trg) != 1)
							return e_EMS_FILE_WRITE_ERROR;
				}
			}
			else
			{
				x = (data->second).x;
				y = (data->second).y;
				t = (data->second).Time;
				f = (data->second).Fli;
				r = (data->second).Ros;
				rx = (data->second).Rcx;
				d = (data->second).Dir;
				switch(outType)
				{
				case RAST_ARRIVALTIME:
					writeVal = t * 60.0;
					break;
				case RAST_FIREINTENSITY:
					writeVal = f / convf1;
					break;
				case RAST_FLAMELENGTH:
					Calcs(FLAME_LENGTH);					// calculate flame length
					writeVal = l;
					break;
				case RAST_SPREADRATE:
					writeVal = r * convf2;
					break;
				case RAST_HEATPERAREA:
					Calcs(HEAT_PER_AREA);   				  // calculate heat/unit area
					writeVal = h;
					break;
				case RAST_REACTIONINTENSITY:
					Calcs(HEAT_PER_AREA);   				  // calculate heat/unit area
					writeVal = rx;
					break;
				case RAST_CROWNFIRE:
					Calcs(CROWNFIRE);
					writeVal = c;
					break;
				case RAST_FIREDIRECTION:
					writeVal = d;
					break;
				}
				if(writeVal < 0.0)
					writeVal = -9999.0;
				else if(pFarsite->ignitionGrid[j][i] == 1.0)
				{
					if(outType == RAST_ARRIVALTIME)
						writeVal = 0.0;//ignition point
					else
						writeVal = -9999.0;//ignition point
				}
				if(pFarsite->ignitionGrid[j][i] == 0.0)//barrier cludge
					writeVal = -9999.0;
				if(fwrite(&writeVal, sizeof(float), 1, trg) != 1)
					return e_EMS_FILE_WRITE_ERROR;
			}
		}
	}
	return 1;
}

int OutputFile::WriteArrivalTimeBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_ARRIVALTIME);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_ARRIVALTIME);
	fclose(out);
	return ret;
}

int OutputFile::WriteIntensityBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_FIREINTENSITY);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_FIREINTENSITY);
	fclose(out);
	return ret;
}
int OutputFile::WriteFlameLengthBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_FLAMELENGTH);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_FLAMELENGTH);
	fclose(out);
	return ret;
}
int OutputFile::WriteSpreadRateBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_SPREADRATE);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_SPREADRATE);
	fclose(out);
	return ret;
}
int OutputFile::WriteSpreadDirectionBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_FIREDIRECTION);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_FIREDIRECTION);
	fclose(out);
	return ret;
}
int OutputFile::WriteHeatPerUnitAreaBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_HEATPERAREA);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_HEATPERAREA);
	fclose(out);
	return ret;
}
int OutputFile::WriteReactionIntensityBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_REACTIONINTENSITY);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_REACTIONINTENSITY);
	fclose(out);
	return ret;
}
int OutputFile::WriteCrownFireBinary(char *trgName)
{
	FILE *out = fopen(trgName, "wb");
	if(!out)
		return e_EMS_FILE_OPEN_ERROR;
	int ret = WriteBinaryHeader(out, RAST_CROWNFIRE);
	if(ret == 1)
		ret = WriteBinaryData(out, RAST_CROWNFIRE);
	fclose(out);
	return ret;
}



/*int OutputFile::WriteArrivalTimeBinary(char *trgName)
{
	pFarsite->CanSetRasterResolution(0);  	  // false
	float nullVal = NODATA_VAL;
	float zeroVal = 0.0;
	double xres, yres;
	long numcells;
	double MetersToKm = 1.0;
		North = pFarsite->GetHiNorth();
		South = pFarsite->GetLoNorth();
		East =pFarsite-> GetHiEast();
		West = pFarsite->GetLoEast();
		//West =pFarsite-> GetHiEast();
		//East = pFarsite->GetLoEast();
	pFarsite->GetRastRes(&xres, &yres);
	long numNorth = (long) ((North - South) / yres), numEast = (long) ((East - West) / xres);
	float cellResX = xres, cellResY = yres, westUTM,
		southUTM, arrivalTimeVal;
			//fprintf(otpfile, "%s %lf\n", "XLLCORNER",
				westUTM = pFarsite->ConvertEastingOffsetToUtm(West);//MetersToKm);
			//fprintf(otpfile, "%s %lf\n", "YLLCORNER",
				southUTM = pFarsite->ConvertNorthingOffsetToUtm(South);//MetersToKm);

	numcols = (long) ((East - West) / xres);//((GetEastUtm()-GetWestUtm())/xres);
	numrows = (long) ((North - South) / yres);//((GetNorthUtm()-GetSouthUtm())/yres);
	numcells = numrows * numcols;
	//memset(RasterCopy, 0x0, sizeof RasterCopy);
	//strcpy(RasterCopy, pFarsite->GetRasterFileName(Type));

	if (pFarsite->CheckCellResUnits() == 2)
		MetersToKm = 0.001;

	otpfile = fopen(trgName, "wb");
	if (otpfile == NULL)
	{
		return 0;
	}
	char textBuf[64];
	long layerType = RAST_ARRIVALTIME;
	sprintf(textBuf, "%-32.32s", "FARSITE_ArrivalTime");//GetOutputLayerName32char(Layer, textBuf);
	fwrite(textBuf, sizeof(char), 32, otpfile);
	fwrite(&layerType, sizeof(long), 1, otpfile);
	fwrite(&numcols, sizeof(numcols), 1, otpfile);
	fwrite(&numrows, sizeof(numrows), 1, otpfile);
	fwrite(&westUTM, sizeof(westUTM), 1, otpfile);
	fwrite(&southUTM, sizeof(southUTM), 1, otpfile);
	fwrite(&cellResX, sizeof(cellResX), 1, otpfile);
	fwrite(&nullVal, sizeof(nullVal), 1, otpfile);

	for (long j=0; j < numrows; j++)
	{
		for (long i=0; i < numcols; i++)
		{

			// check to see if this cell has raster data written to it.
			coordinate testxy = std::make_pair(i,j) ;
			//coordinate testxy = std::make_pair(j,i) ;
			RasterMap::iterator data = rd.find(testxy) ;

			// if this cell has no data yet, just print a -1
			if (data == rd.end())
			{
				//fwrite(&nullVal, sizeof(float), 1, otpfile);//, "%.3lf ", NODATA_VAL);
				if(pFarsite->ignitionGrid[j][i] == 1.0)
					fwrite(&zeroVal, sizeof(float), 1, otpfile);//, "%.3lf ", NODATA_VAL);
				else//unburned point
					fwrite(&nullVal, sizeof(float), 1, otpfile);//, "%.3lf ", NODATA_VAL);

			}
			else
			{
				arrivalTimeVal = (data->second).Time;
				fwrite(&arrivalTimeVal, sizeof(float), 1, otpfile);//, "%.3lf ", NODATA_VAL);
				//(data->second).Write = true;
			}
		}
		//fprintf(otpfile, "\n");
	}
	fclose(otpfile);
	return 1;
}
*/
int CloseAndReturn(FILE *stream, int returnCode)
{
	int retCode = returnCode;
	if(fclose(stream) != 0)
	{
		if(returnCode != 1)
			retCode = e_EMS_FILE_CLOSE_ERROR;
	}
	stream = NULL;
	return retCode;
}

int OutputFile::WriteFile(long Type)
{
	double xres, yres;
	char RasterCopy[256] = "";
	long numcells;
	double MetersToKm = 1.0;
		North = pFarsite->GetHiNorth();
		South = pFarsite->GetLoNorth();
		East =pFarsite-> GetHiEast();
		West = pFarsite->GetLoEast();

	pFarsite->GetRastRes(&xres, &yres);
	numcols = (long) ((East - West) / xres);//((GetEastUtm()-GetWestUtm())/xres);
	numrows = (long) ((North - South) / yres);//((GetNorthUtm()-GetSouthUtm())/yres);
	numcells = numrows * numcols;
	memset(RasterCopy, 0x0, sizeof RasterCopy);
	strcpy(RasterCopy, pFarsite->GetRasterFileName(Type));

	if (pFarsite->CheckCellResUnits() == 2)
		MetersToKm = 0.001;

	otpfile = fopen(RasterCopy, "w");
	if (otpfile == NULL)
	{
		//otpfile = fopen(RasterCopy, "w");
		return e_EMS_FILE_OPEN_ERROR;//
	}
	switch (HeaderType)
	{
	case 1:
		if (pFarsite->GetNorthUtm() == 0.0 || pFarsite->GetEastUtm() == 0.0)
		{
			if(fprintf(otpfile, "%s    %lf\n", "north:", North) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s    %lf\n", "south:", South) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s     %lf\n", "east:", East) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s     %lf\n", "west:", West) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		}
		else
		{
			if(fprintf(otpfile, "%s    %lf\n", "north:",
				pFarsite->ConvertNorthingOffsetToUtm(North)) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s    %lf\n", "south:",
				pFarsite->ConvertNorthingOffsetToUtm(South)) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s     %lf\n", "east:",
				pFarsite->ConvertEastingOffsetToUtm(East)) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s     %lf\n", "west:",
				pFarsite->ConvertEastingOffsetToUtm(West)) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		}
		if(fprintf(otpfile, "%s     %ld\n", "rows:", numrows) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(otpfile, "%s     %ld\n", "cols:", numcols) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		break;
	case 5:
		if(fprintf(otpfile, "%s %ld\n", "NCOLS", numcols) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(otpfile, "%s %ld\n", "NROWS", numrows) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		if (pFarsite->GetNorthUtm() == 0.0 || pFarsite->GetEastUtm() == 0.0)
		{
			if(fprintf(otpfile, "%s %lf\n", "XLLCORNER", West) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s %lf\n", "YLLCORNER", South) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		}
		else
		{
			if(fprintf(otpfile, "%s %lf\n", "XLLCORNER",
				pFarsite->ConvertEastingOffsetToUtm(West)) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
			if(fprintf(otpfile, "%s %lf\n", "YLLCORNER",
				pFarsite->ConvertNorthingOffsetToUtm(South)) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		}
		if(fprintf(otpfile, "%s %lf\n", "CELLSIZE", xres * MetersToKm) < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		if(fprintf(otpfile, "%s %s\n", "NODATA_VALUE", "-9999.0") < 0)
				return CloseAndReturn(otpfile, e_EMS_FILE_WRITE_ERROR);
		break;
	}

	// write the actual data to the file.
	int retVal = RastMemFile(Type) ;
	fclose(otpfile);
	return retVal;
}

void OutputFile::GetRasterExtent()
{
	if (pFarsite->CanSetRasterResolution(GETVAL) == 0)
		return;

	/*if (SetRasterExtentToViewport(GETVAL))
	{
		North = GetViewNorth();
		South = GetViewSouth();
		East = GetViewEast();
		West = GetViewWest();
	}
	else
	{*/
		North = pFarsite->GetHiNorth();
		South = pFarsite->GetLoNorth();
		East =pFarsite-> GetHiEast();
		West = pFarsite->GetLoEast();
	//}
}






void OutputFile::FreeRastData()
{
	rd.clear() ;
	NumRastAlloc = 0;
	NumRastData = 0;
}

/**
 * \brief Aggregates the provided set of fire behavior characteristics onto
 * the raster.
 *
 * <p>
 * The client code supplies a location (xpt, ypt) expressed in real world
 * coordinates (UTM), and the associated fire behavior characteristics (time,
 * fli, ros, rcx, dir).  This method translates the location into a grid
 * cell index (i,j) and stores the fire behavior data for that cell.
 * </p>
 *
 * <p>
 * In most cases, if the cell already contains a data value, the old value
 * is simply overwritten by the new value.  Time of arrival is an exception
 * to this rule.  The resultant value for the grid cell is the minimum of all
 * the time values assigned.
 * </p>
 */
bool OutputFile::SetRastData()
{
	return true;//SetRastData(x,y,t,f,r,rx,d);
}


bool OutputFile::SetRastData(double xpt, double ypt, double time, double fli,
	double ros, double rcx, long dir)
{

	double xres, yres ;
	coordinate tempxy ;
	RastData temp ;

	pFarsite->GetRastRes(&xres, &yres);

	// compute the grid cell value
	tempxy.first = (long) ((xpt - West) / xres) ;
	tempxy.second = (long) ((North - ypt) / yres) ;

	// check to see if this cell already has an entry
	RasterMap::iterator oldVal = rd.find(tempxy) ;


	if (oldVal != rd.end()) {
		// handle special cases where cell already has value
		temp.Time = min((oldVal->second).Time, time) ;
	} else {
		// handle special cases where cell doesn't have value yet
		temp.Time = time ;
	}

	// handle "standard" case where we blithely replace the value
	temp.x = xpt ;
	temp.y = ypt ;
	temp.Fli = fli;
	temp.Ros = ros;
	temp.Rcx = rcx;
	temp.Dir = dir;

	// add this to the map (if it was already in the map, tell where)
	if (oldVal != rd.end()) {
		rd.insert(oldVal, std::make_pair(tempxy,temp)) ;
	} else {
		rd.insert(std::make_pair(tempxy,temp)) ;
	}

	NumRastData = rd.size() ;

	return true;
}



