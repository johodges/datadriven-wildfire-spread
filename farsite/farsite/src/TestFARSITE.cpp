// TestMTT.cpp : Defines the entry point for the console application.
//

#include "FARSITE.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
bool cancelRequest = false;
int nFarsites = 0;

#ifdef WIN32
#include <direct.h>
#include <conio.h>
#include <process.h>
#include <windows.h>



unsigned int __stdcall  ProgressThread(void *_pFarsite)
{
 	CFarsite *pFarsite = (CFarsite *)_pFarsite;
	//Sleep(1000);
	double progress = 0.0;
	while(!cancelRequest && (progress = pFarsite->GetFarsiteProgress()) < 1.0)
	{
		printf("ProgressThread (10 second notification): Farsite %.2f percent complete.\n",
			progress * 100.0);
		Sleep(10000);
	}
	//Sleep(1000);
	return 1;
}


unsigned int __stdcall  CheckKey( void *_pFarsite )
{
 	CFarsite *pFarsite = (CFarsite *)_pFarsite;
   if(_getch() != 0)
   {
		printf("Farsite terminate request....\n");
		pFarsite->CancelFarsite();    /* _endthread implied */
		cancelRequest = true;
   }
   return 1;
}

unsigned int __stdcall  MultiCheckKey( void *_pFarsite )
{
 	CFarsite *pFarsite = (CFarsite *)_pFarsite;
   if(_getch() != 0)
   {
		printf("Farsite terminate request....\n");
		for(int f = 0; f < nFarsites; f++)
			pFarsite[f].CancelFarsite();    /* _endthread implied */
		cancelRequest = true;
   }
   return 1;
}

unsigned int __stdcall RunFarsiteProc(void *_pFarsite)
{
	CFarsite *pFarsite = (CFarsite *)_pFarsite;
	int ret = pFarsite->LaunchFarsite();
	if(ret != 1)
	{
	   char *a = pFarsite->CommandFileError(ret);
	   printf ("%s\n",a);
	   cancelRequest = true;
	}
	return ret;
}
#else
#include <stdio.h>
#include <stdlib.h>

#endif
bool ParseCommands(char *in, char *lcp, char *inputs, char *ignitPath, char *barrierPath, char *outPath, int *outType)
{
	strcpy(lcp, "");
	strcpy(inputs, "");
	strcpy(outPath, "");
	*outType = 0;
	char seps[] = " ";
	char *p = strtok(in, seps);
	if(p)
		strcpy(lcp, p);
	else
		return false;
	p = strtok(NULL, seps);
	if(p)
		strcpy(inputs, p);
	else
		return false;
	p = strtok(NULL, seps);
	if(p)
		strcpy(ignitPath, p);
	else
		return false;
	p = strtok(NULL, seps);
	if(p)
		strcpy(barrierPath, p);
	else
		return false;
	p = strtok(NULL, seps);
	if(p)
		strcpy(outPath, p);
	else
		return false;
	p = strtok(NULL, seps);
	if(p)
		*outType = atoi(p);
	else
		return false;
	return true;
}
#ifdef WIN32
int regMain(int argc, char* argv[])
{
	char lcpFileName[512], inputsFileName[512], baseOutputsPath[512];
	if(argc < 5)
	{
		printf("TestFARSITE expects four parameters, all required\nTestFARSITE Usage:\n"
			"TestFARSITE [LCPName] [InputsFileName] [outputDirPath] [outputsType]\n"
			"Where:\n\t[LCPName] is the path to the Landscape File\n"
			"\t[InputsFileName] is the path to the FARSITE Inputs File (ASCII Format)\n"
			"\t[outputDirPath] is the path to the output files base name (no extension)\n"
			"\t[outputsType] is the file type for outputs (0 = both, 1 = ASCII grid, 2 = FlamMap binary grid, > 3 = ShapeFile only\n\n");
		exit(1);
	}
	strcpy(lcpFileName, argv[1]);
	strcpy(inputsFileName, argv[2]);
	strcpy(baseOutputsPath, argv[3]);
	int outType = atoi(argv[4]);
	char igFileName[256];
	char barFileName[256];
	char outputName[512];
	CFarsite farsite, farsite2, farsite3;
	if ( !farsite.SetLandscapeFile(lcpFileName))
	{
		printf ("Can't open: %s \n", lcpFileName);
	    return 0;
	}
	int i = farsite.LoadFarsiteInputs(inputsFileName);
	if ( i != 1 )
	{
	   char *a = farsite.CommandFileError(i);
	   printf ("%s\n",a);
	   return 1;
	}
	/*if(!farsite.SetIgnition(igFileName))
	{
		printf ("Error setting ignition using file: %s \n", igFileName);
		_getch();
	    return 0;
	}
	if(!farsite.SetBarriers(barFileName))
	{
		printf ("Error setting barrier using file: %s \n", barFileName);
		_getch();
	    return 0;
	}*/
	printf("\nLaunching Farsite, press any key to terminate.\n");
	HANDLE threadHandles[2];
	HANDLE checkKeyHandles[1];
	unsigned int ThreadID, ThreadID1, ThreadID2, ThreadID3, ThreadID4, ThreadID5;
	printf("Starting 'CheckKey' thread.\n");
	checkKeyHandles[0] =(HANDLE) ::_beginthreadex(NULL, 0, CheckKey, &farsite, NULL, &ThreadID2);
	printf("Starting 'ProgressThread' thread.\n");
    threadHandles[0] =(HANDLE) ::_beginthreadex(NULL, 0, ProgressThread, &farsite, NULL, &ThreadID1);
	//HANDLE hRiskThread;
 	printf("Starting 'RunFarsiteProc' thread.\n");
	threadHandles[1] =(HANDLE) ::_beginthreadex(NULL, 0, RunFarsiteProc, &farsite, NULL, &ThreadID);
  	printf("Waiting for threads.\n");
	WaitForMultipleObjects(2, threadHandles, TRUE, INFINITE);



	/*if(farsite.LaunchFarsite() != 1)
	{
		printf("Error: LaunchFarsite failure.\n");
		_getch();
		return 1;
	}*/
	//if(!cancelRequest)
	//{
		if(outType == 0 || outType == 1)
		{
			sprintf(outputName, "%s_CrownFire.asc", baseOutputsPath);
			farsite.WriteCrownFireGrid(outputName);
			sprintf(outputName, "%s_Intensity.asc", baseOutputsPath);
			farsite.WriteIntensityGrid(outputName);
			sprintf(outputName, "%s_FlameLength.asc", baseOutputsPath);
			farsite.WriteFlameLengthGrid(outputName);
			sprintf(outputName, "%s_SpreadRate.asc", baseOutputsPath);
			farsite.WriteSpreadRateGrid(outputName);
			sprintf(outputName, "%s_SpreadDirection.asc", baseOutputsPath);
			farsite.WriteSpreadDirectionGrid(outputName);
			sprintf(outputName, "%s_HeatPerUnitArea.asc", baseOutputsPath);
			farsite.WriteHeatPerUnitAreaGrid(outputName);
			sprintf(outputName, "%s_ReactionIntensity.asc", baseOutputsPath);
			farsite.WriteReactionIntensityGrid(outputName);
			sprintf(outputName, "%s_Ignitions.asc", baseOutputsPath);
			farsite.WriteIgnitionGrid(outputName);
			sprintf(outputName, "%s_ArrivalTime.asc", baseOutputsPath);
			farsite.WriteArrivalTimeGrid(outputName);
		}
		if(outType == 0 || outType == 2)
		{
			sprintf(outputName, "%s_ArrivalTime.fbg", baseOutputsPath);
			farsite.WriteArrivalTimeGridBinary(outputName);
			sprintf(outputName, "%s_CrownFire.fbg", baseOutputsPath);
			farsite.WriteCrownFireGridBinary(outputName);
			sprintf(outputName, "%s_Intensity.fbg", baseOutputsPath);
			farsite.WriteIntensityGridBinary(outputName);
			sprintf(outputName, "%s_FlameLength.fbg", baseOutputsPath);
			farsite.WriteFlameLengthGridBinary(outputName);
			sprintf(outputName, "%s_SpreadRate.fbg", baseOutputsPath);
			farsite.WriteSpreadRateGridBinary(outputName);
			sprintf(outputName, "%s_SpreadDirection.fbg", baseOutputsPath);
			farsite.WriteSpreadDirectionGridBinary(outputName);
			sprintf(outputName, "%s_HeatPerUnitArea.fbg", baseOutputsPath);
			farsite.WriteHeatPerUnitAreaGridBinary(outputName);
			sprintf(outputName, "%s_ReactionIntensity.fbg", baseOutputsPath);
			farsite.WriteReactionIntensityGridBinary(outputName);
		}
		sprintf(outputName, "%s_Perimeters.shp", baseOutputsPath);
		farsite.WritePerimetersShapeFile(outputName);
		sprintf(outputName, "%s_Timings.txt", baseOutputsPath);
		farsite.WriteTimingsFile(outputName);
		sprintf(outputName, "%s_Spots.shp", baseOutputsPath);
		farsite.WriteSpotShapeFile(outputName);

	//}
	//if(!cancelRequest)
	//{
		printf("Press any key to exit.\n");
		WaitForMultipleObjects(1, checkKeyHandles, TRUE, INFINITE);

	//}
	CloseHandle(checkKeyHandles[0]);
	CloseHandle(threadHandles[0]);
	CloseHandle(threadHandles[1]);
	return 0;
}

int multiMain(int argc, char* argv[])
{
	char lcpFileName[512], inputsFileName[512], ignitName[512], barrierName[512], baseOutputsPath[512];
	if(argc < 2)
	{
		printf("TestFARSITE expects one parameter\nTestFARSITE Usage:\n"
			"TestFARSITE [commandfile]\n"
			"Where:\n\t[commandfile] is the path to the command file.\n");
		printf("The command file contains command lines for multiple Farsite runs, each run's command on a seperate line.\n");
		printf("Each command expects four parameters, all required\n"
			"[LCPName] [InputsFileName] [outputDirPath] [outputsType]\n"
			"Where:\n\t[LCPName] is the path to the Landscape File\n"
			"\t[InputsFileName] is the path to the FARSITE Inputs File (ASCII Format)\n"
			"\t[outputDirPath] is the path to the output files base name (no extension)\n"
			"\t[outputsType] is the file type for outputs (0 = both, 1 = ASCII grid, 2 = FlamMap binary grid, > 3 = ShapeFile only\n\n");
		exit(1);
	}
	FILE *cmd = fopen(argv[1], "rt");
	if(!cmd)
	{
		printf("Error, cannot open %s\n", argv[1]);
		exit(1);
	}
	char outputName[512];
	int outType;// = atoi(argv[4]);
	nFarsites = 0;
	char buf[512];
	while(!feof(cmd))
	{
		fgets(buf, 512, cmd);

		if(ParseCommands(buf, lcpFileName, inputsFileName, ignitName, barrierName, baseOutputsPath, &outType))//assume valid commands
			nFarsites++;
	}
	rewind(cmd);
	char **lcps = new char *[nFarsites];
	char **inputs = new char *[nFarsites];
	char **ignits = new char *[nFarsites];
	char **barriers = new char *[nFarsites];
	char **outs = new char *[nFarsites];
	int *outTypes = new int[nFarsites];
	for(int i = 0; i < nFarsites; i++)
	{
		lcps[i] = new char[MAX_PATH];
		inputs[i] = new char[MAX_PATH];
		ignits[i] = new char[MAX_PATH];
		barriers[i] = new char[MAX_PATH];
		outs[i] = new char[MAX_PATH];
		strcpy(lcps[i], "");
		strcpy(inputs[i], "");
		strcpy(ignits[i], "");
		strcpy(barriers[i], "");
		strcpy(outs[i], "");
	}
	CFarsite *pFarsites = new CFarsite[nFarsites];
	for(int f = 0; f < nFarsites; f++)
	{
		fgets(buf, 512, cmd);
		if(!ParseCommands(buf, lcps[f], inputs[f], ignits[f], barriers[f], outs[f], &outTypes[f]))
			continue;
		if ( !pFarsites[f].SetLandscapeFile(lcps[f]))
		{
			printf ("Can't open: %s \n", lcpFileName);
			_getch();
			return 0;
		}
		int i = pFarsites[f].LoadFarsiteInputs(inputs[f]);
		if ( i != 1 )
		{
		   char *a = pFarsites[f].CommandFileError(i);
		   printf ("%s\n",a);
		   _getch ();
		   return 1;
		}
		pFarsites[f].SetIgnition(ignits[f]);
		pFarsites[f].SetBarriers(barriers[f]);
	}
	fclose(cmd);
	printf("\nLaunching %d Farsites, press any key to terminate.\n", nFarsites);
	HANDLE *threadHandles = new HANDLE[nFarsites + 1];
	HANDLE checkKeyHandles[1];
	unsigned int *ThreadIDs = new unsigned int[nFarsites + 1], ThreadID;
	printf("Starting 'CheckKey' thread.\n");
	checkKeyHandles[0] =(HANDLE) ::_beginthreadex(NULL, 0, MultiCheckKey, &pFarsites[0], NULL, &ThreadID);
	printf("Starting 'ProgressThread' thread.\n");
    threadHandles[0] =(HANDLE) ::_beginthreadex(NULL, 0, ProgressThread, &pFarsites[0], NULL, &ThreadIDs[0]);
	//HANDLE hRiskThread;
 	printf("Starting 'RunFarsiteProc' threads.\n");
	for(int f = 1; f <= nFarsites; f++)
	{
		threadHandles[f] =(HANDLE) ::_beginthreadex(NULL, 0, RunFarsiteProc, &pFarsites[f-1], NULL, &ThreadIDs[f]);
		/*threadHandles[f] =(HANDLE) ::_beginthreadex(NULL, 0, RunFarsiteProc, &pFarsites[f-1], CREATE_SUSPENDED, &ThreadIDs[f]);
		long ProcNum = f;
		while ( ProcNum >= tprocs )
			ProcNum -= tprocs;
		unsigned long Affinity =pow(2.0, (int)ProcNum);
		SetThreadAffinityMask (threadHandles[f], Affinity);
		ResumeThread(threadHandles[f]);*/
	}
 	printf("Waiting for threads.\n");
	WaitForMultipleObjects(nFarsites + 1, threadHandles, TRUE, INFINITE);


	if(!cancelRequest)
	{
		printf("Writing outputs\n");
		for(int f = 0; f < nFarsites; f++)
		{
			if(outTypes[f] == 0 || outTypes[f] == 1)
			{
				sprintf(outputName, "%s_CrownFire.asc", outs[f]);
				pFarsites[f].WriteCrownFireGrid(outputName);
				sprintf(outputName, "%s_Intensity.asc", outs[f]);
				pFarsites[f].WriteIntensityGrid(outputName);
				sprintf(outputName, "%s_FlameLength.asc", outs[f]);
				pFarsites[f].WriteFlameLengthGrid(outputName);
				sprintf(outputName, "%s_SpreadRate.asc", outs[f]);
				pFarsites[f].WriteSpreadRateGrid(outputName);
				sprintf(outputName, "%s_SpreadDirection.asc", outs[f]);
				pFarsites[f].WriteSpreadDirectionGrid(outputName);
				sprintf(outputName, "%s_HeatPerUnitArea.asc", outs[f]);
				pFarsites[f].WriteHeatPerUnitAreaGrid(outputName);
				sprintf(outputName, "%s_ReactionIntensity.asc", outs[f]);
				pFarsites[f].WriteReactionIntensityGrid(outputName);
				sprintf(outputName, "%s_ArrivalTime.asc", outs[f]);
				pFarsites[f].WriteArrivalTimeGrid(outputName);
				sprintf(outputName, "%s_Ignitions.asc", outs[f]);
				pFarsites[f].WriteIgnitionGrid(outputName);
			}
			if(outTypes[f] == 0 || outTypes[f] == 2)
			{
				sprintf(outputName, "%s_CrownFire.fbg", outs[f]);
				pFarsites[f].WriteCrownFireGridBinary(outputName);
				sprintf(outputName, "%s_ArrivalTime.fbg", outs[f]);
				pFarsites[f].WriteArrivalTimeGridBinary(outputName);
				sprintf(outputName, "%s_Intensity.fbg", outs[f]);
				pFarsites[f].WriteIntensityGridBinary(outputName);
				sprintf(outputName, "%s_FlameLength.fbg", outs[f]);
				pFarsites[f].WriteFlameLengthGridBinary(outputName);
				sprintf(outputName, "%s_SpreadRate.fbg", outs[f]);
				pFarsites[f].WriteSpreadRateGridBinary(outputName);
				sprintf(outputName, "%s_SpreadDirection.fbg", outs[f]);
				pFarsites[f].WriteSpreadDirectionGridBinary(outputName);
				sprintf(outputName, "%s_HeatPerUnitArea.fbg", outs[f]);
				pFarsites[f].WriteHeatPerUnitAreaGridBinary(outputName);
				sprintf(outputName, "%s_ReactionIntensity.fbg", outs[f]);
				pFarsites[f].WriteReactionIntensityGridBinary(outputName);
				sprintf(outputName, "%s_ArrivalTime.fbg", outs[f]);
				pFarsites[f].WriteArrivalTimeGridBinary(outputName);
			}
			sprintf(outputName, "%s_Perimeters.shp", outs[f]);
			pFarsites[f].WritePerimetersShapeFile(outputName);
			sprintf(outputName, "%s_Timings.txt", outs[f]);
			pFarsites[f].WriteTimingsFile(outputName);
		}

	}
	if(!cancelRequest)
	{
		printf("Press any key to exit.\n");
		WaitForMultipleObjects(1, checkKeyHandles, TRUE, INFINITE);

	}
	CloseHandle(checkKeyHandles[0]);
	for(int f = 0; f < nFarsites + 1; f++)
		CloseHandle(threadHandles[f]);
	//for(int f = 0; f < tprocs; f++)
	delete[] pFarsites;
	for(int i = 0; i < nFarsites; i++)
	{
		delete[] lcps[i];
		delete[] inputs[i];
		delete[] outs[i];
	}
	delete[] lcps;
	delete[] inputs;
	delete[] outs;
	return 0;
}

int nOmpFarsites = 0;
unsigned int __stdcall  ompProgressThread(void *_pFarsite)
{
 	CFarsite **pFarsite = (CFarsite **)_pFarsite;
	//Sleep(1000);
	double progress = 0.0;
	while(!cancelRequest)// && (progress = pFarsite->GetFarsiteProgress()) < 1.0)
	{
		printf("Progress...\n");// Farsite %.2f percent complete.\n",
		for(int f = 0; f < nOmpFarsites; f++)
		{
			if(pFarsite[f])
			{
				progress = pFarsite[f]->GetFarsiteProgress();
				//if(progress > 0)
					printf("Farsite %d: %.3f complete\n",// Farsite %.2f percent complete.\n",
						f + 1, progress * 100.0);
			}
		}
		Sleep(5000);
	}
	//Sleep(100);
	return 1;
}

unsigned int __stdcall  ompCheckKey( void *_pFarsite )
{
 	CFarsite **pFarsite = (CFarsite **)_pFarsite;
   if(_getch() != 0)
   {
		printf("Farsite terminate request....\n");
		for(int f = 0; f < nOmpFarsites; f++)
		{
			if(pFarsite[f])
				pFarsite[f]->CancelFarsite();    /* _endthread implied */
		}
		cancelRequest = true;
   }
   return 1;
}

typedef CFarsite * LPFARSITE;

int RangedRand( int range_min, int range_max)
{
   // Generate random numbers in the half-closed interval
   // [range_min, range_max). In other words,
   // range_min <= random number < range_max
      int u = (double)rand() / (RAND_MAX + 1) * (range_max - range_min)
            + range_min;
     return u;
}

int openMPMain(int argc, char* argv[])
{
	char lcpFileName[512], inputsFileName[512], ignitName[512], barrierName[512], baseOutputsPath[512];
	if(argc < 2)
	{
		printf("TestFARSITE expects one parameter\nTestFARSITE Usage:\n"
			"TestFARSITE [commandfile]\n"
			"Where:\n\t[commandfile] is the path to the command file.\n");
		printf("The command file contains command lines for multiple Farsite runs, each run's command on a seperate line.\n");
		printf("Each command expects six parameters, all required\n"
			"[LCPName] [InputsFileName] [IgnitionFileName] [BarrierFileName] [outputDirPath] [outputsType]\n"
			"Where:\n\t[LCPName] is the path to the Landscape File\n"
			"\t[InputsFileName] is the path to the FARSITE Inputs File (ASCII Format)\n"
			"\t[IgnitionFileName] is the path to the Ignition shape File\n"
			"\t[BarrierFileName] is the path to the Barrier shape File (0 if no barrier)\n"
			"\t[outputDirPath] is the path to the output files base name (no extension)\n"
			"\t[outputsType] is the file type for outputs (0 = both, 1 = ASCII grid, 2 = FlamMap binary grid\n\n");
		exit(1);
	}
	FILE *cmd = fopen(argv[1], "rt");
	if(!cmd)
	{
		printf("Error, cannot open %s\n", argv[1]);
		exit(1);
	}
	char outputName[512];
	int outType;// = atoi(argv[4]);
	nFarsites = 0;
	char buf[512];
	while(!feof(cmd))
	{
		fgets(buf, 512, cmd);

		if(ParseCommands(buf, lcpFileName, inputsFileName, ignitName, barrierName, baseOutputsPath, &outType))//assume valid commands
			nFarsites++;
	}
	rewind(cmd);
	//SYSTEM_INFO sysInfo;
	//GetSystemInfo(&sysInfo);
	//nOmpFarsites =  sysInfo.dwNumberOfProcessors;

	nOmpFarsites = nFarsites;
	char **lcps = new char *[nFarsites];
	char **inputs = new char *[nFarsites];
	char **ignits = new char *[nFarsites];
	char **barriers = new char *[nFarsites];
	char **outs = new char *[nFarsites];
	int *outTypes = new int[nFarsites];
	CFarsite **pFarsites = new CFarsite*[nOmpFarsites];
	for(int i = 0; i < nOmpFarsites; i++)
		pFarsites[i] = NULL;
	for(int i = 0; i < nFarsites; i++)
	{
		lcps[i] = new char[MAX_PATH];
		inputs[i] = new char[MAX_PATH];
		ignits[i] = new char[MAX_PATH];
		barriers[i] = new char[MAX_PATH];
		outs[i] = new char[MAX_PATH];
		strcpy(lcps[i], "");
		strcpy(inputs[i], "");
		strcpy(ignits[i], "");
		strcpy(barriers[i], "");
		strcpy(outs[i], "");
		fgets(buf, 512, cmd);
		if(!ParseCommands(buf, lcps[i], inputs[i], ignits[i], barriers[i], outs[i], &outTypes[i]))
			continue;
	}

	HANDLE threadHandle[1];
	HANDLE checkKeyHandles[1];
	unsigned int ThreadID1, ThreadID2;
	printf("Starting 'CheckKey' thread.\n");
	checkKeyHandles[0] =(HANDLE) ::_beginthreadex(NULL, 0, ompCheckKey, (void *) pFarsites, NULL, &ThreadID1);
	printf("Starting 'ProgressThread' thread.\n");
    threadHandle[0] =(HANDLE) ::_beginthreadex(NULL, 0, ompProgressThread, &pFarsites[0], NULL, &ThreadID2);
	int ret;
	int f, waitTime;
	timeb t;

	ftime(&t);
	srand(t.time);
#pragma omp parallel for default(shared) private (buf, ret, outputName, waitTime) schedule (dynamic, 1)
	for(f = 0; f < nFarsites; f++)
	{
		#pragma omp critical
		{
			waitTime = f;//RangedRand(5, 2000);//need a wait here so farsite random number generators are unique
			printf("Sleeping %d ms for iteration %d\n", waitTime*f, f);
			Sleep(waitTime*f);
			pFarsites[f] = new CFarsite();
		}
		if(!cancelRequest)
		{
			printf("Loading lcp file for Farsite #%d: %s\n", f + 1, lcps[f]);
			if ( !pFarsites[f]->SetLandscapeFile(lcps[f]))
			{
				printf ("Can't open: %s \n", lcpFileName);
				delete pFarsites[f];
				pFarsites[f] = NULL;
				//_getch();
				continue;
			}
		}
		if(!cancelRequest)
		{
			printf("Loading inputs file for Farsite #%d: %s\n", f + 1, inputs[f]);
			ret = pFarsites[f]->LoadFarsiteInputs(inputs[f]);
			if ( ret != 1 )
			{
			   char *a = pFarsites[f]->CommandFileError(ret);
			   printf ("%s\n",a);
				delete pFarsites[f];
				pFarsites[f] = NULL;
				continue;
			}
		}
		if(!cancelRequest)
		{
			printf("Loading ignition file for Farsite #%d: %s\n", f + 1, ignits[f]);
			pFarsites[f]->SetIgnition(ignits[f]);
			if(strlen(barriers[f]) > 2)
			{
				printf("Loading barrier file for Farsite #%d: %s\n", f + 1, barriers[f]);
				pFarsites[f]->SetBarriers(barriers[f]);
			}
		}
		if(!cancelRequest)
		{
			printf("Launching Farsite #%d\n", f + 1);
			ret = pFarsites[f]->LaunchFarsite();
			if(ret != 1)
			{
			   char *a = pFarsites[f]->CommandFileError(ret);
			   printf ("%s\n",a);
				delete pFarsites[f];
				pFarsites[f] = NULL;
				continue;
			}
		}
		if(!cancelRequest)
		{
			printf("Writing outputs for Farsite #%d to %s\n", f + 1, outs[f]);
			if(outTypes[f] == 0 || outTypes[f] == 1 || outTypes[f] == 4)
			{
				if(outTypes[f] == 4)
				{
					pFarsites[f]->WriteOneHours(outs[f]);
					pFarsites[f]->WriteTenHours(outs[f]);
				}
				sprintf(outputName, "%s_CrownFire.asc", outs[f]);
				ret = pFarsites[f]->WriteCrownFireGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_Intensity.asc", outs[f]);
				ret = pFarsites[f]->WriteIntensityGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_FlameLength.asc", outs[f]);
				ret = pFarsites[f]->WriteFlameLengthGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadRate.asc", outs[f]);
				ret = pFarsites[f]->WriteSpreadRateGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadDirection.asc", outs[f]);
				ret = pFarsites[f]->WriteSpreadDirectionGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_HeatPerUnitArea.asc", outs[f]);
				ret = pFarsites[f]->WriteHeatPerUnitAreaGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ReactionIntensity.asc", outs[f]);
				ret = pFarsites[f]->WriteReactionIntensityGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ArrivalTime.asc", outs[f]);
				ret = pFarsites[f]->WriteArrivalTimeGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
			}
			if(outTypes[f] == 0 || outTypes[f] == 2)
			{
				sprintf(outputName, "%s_CrownFire.fbg", outs[f]);
				ret = pFarsites[f]->WriteCrownFireGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ArrivalTime.fbg", outs[f]);
				ret = pFarsites[f]->WriteArrivalTimeGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_Intensity.fbg", outs[f]);
				ret = pFarsites[f]->WriteIntensityGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_FlameLength.fbg", outs[f]);
				ret = pFarsites[f]->WriteFlameLengthGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadRate.fbg", outs[f]);
				ret = pFarsites[f]->WriteSpreadRateGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadDirection.fbg", outs[f]);
				ret = pFarsites[f]->WriteSpreadDirectionGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_HeatPerUnitArea.fbg", outs[f]);
				ret = pFarsites[f]->WriteHeatPerUnitAreaGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ReactionIntensity.fbg", outs[f]);
				ret = pFarsites[f]->WriteReactionIntensityGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ArrivalTime.fbg", outs[f]);
				ret = pFarsites[f]->WriteArrivalTimeGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
			}
			/*sprintf(outputName, "%s_Moistures.txt", outs[f]);
			ret = pFarsites[f]->WriteMoistData(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}*/
			sprintf(outputName, "%s_Ignitions.asc", outs[f]);
			ret = pFarsites[f]->WriteIgnitionGrid(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Perimeters.shp", outs[f]);
			ret = pFarsites[f]->WritePerimetersShapeFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_SpotGrid.asc", outs[f]);
			ret = pFarsites[f]->WriteSpotGrid(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Spots.csv", outs[f]);
			ret = pFarsites[f]->WriteSpotDataFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Spots.shp", outs[f]);
			ret = pFarsites[f]->WriteSpotShapeFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Timings.txt", outs[f]);
			ret = pFarsites[f]->WriteTimingsFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
		}
		#pragma omp critical
		{
			CFarsite *tFarsite = pFarsites[f];
			pFarsites[f] = NULL;
			delete tFarsite;
		}

	}
	bool complete = !cancelRequest;
	cancelRequest = true;
	WaitForMultipleObjects(1, threadHandle, TRUE, INFINITE);
	if(complete)
	{
		printf("Press any key to exit.\n");
		WaitForMultipleObjects(1, checkKeyHandles, TRUE, INFINITE);

	}
	CloseHandle(checkKeyHandles[0]);
	CloseHandle(threadHandle[0]);
	fclose(cmd);
	delete[] pFarsites;
	for(int i = 0; i < nFarsites; i++)
	{
		delete[] lcps[i];
		delete[] inputs[i];
		delete[] outs[i];
		delete[] ignits[i];
		delete[] barriers[i];
	}
	delete[] lcps;
	delete[] inputs;
	delete[] outs;
	delete[] ignits;
	delete[] barriers;
	delete[] outTypes;
	return 0;
}
#else
#define MAX_PATH 512
int linuxMain(int argc, char* argv[])
{
	char lcpFileName[512], inputsFileName[512], ignitName[512], barrierName[512], baseOutputsPath[512];
	if(argc < 2)
	{
		printf("TestFARSITE expects one parameter\nTestFARSITE Usage:\n"
			"TestFARSITE [commandfile]\n"
			"Where:\n\t[commandfile] is the path to the command file.\n");
		printf("The command file contains command lines for multiple Farsite runs, each run's command on a seperate line.\n");
		printf("Each command expects six parameters, all required\n"
			"[LCPName] [InputsFileName] [IgnitionFileName] [BarrierFileName] [outputDirPath] [outputsType]\n"
			"Where:\n\t[LCPName] is the path to the Landscape File\n"
			"\t[InputsFileName] is the path to the FARSITE Inputs File (ASCII Format)\n"
			"\t[IgnitionFileName] is the path to the Ignition shape File\n"
			"\t[BarrierFileName] is the path to the Barrier shape File (0 if no barrier)\n"
			"\t[outputDirPath] is the path to the output files base name (no extension)\n"
			"\t[outputsType] is the file type for outputs (0 = both, 1 = ASCII grid, 2 = FlamMap binary grid\n\n");
		exit(1);
	}
	FILE *cmd = fopen(argv[1], "rt");
	if(!cmd)
	{
		printf("Error, cannot open %s\n", argv[1]);
		exit(1);
	}
	char outputName[512];
	int outType;// = atoi(argv[4]);
	nFarsites = 0;
	char buf[512];
	while(!feof(cmd))
	{
		fgets(buf, 512, cmd);

		if(ParseCommands(buf, lcpFileName, inputsFileName, ignitName, barrierName, baseOutputsPath, &outType))//assume valid commands
			nFarsites++;
	}
	rewind(cmd);
	//SYSTEM_INFO sysInfo;
	//GetSystemInfo(&sysInfo);
	//nOmpFarsites =  sysInfo.dwNumberOfProcessors;

	//nOmpFarsites = nFarsites;
	char **lcps = new char *[nFarsites];
	char **inputs = new char *[nFarsites];
	char **ignits = new char *[nFarsites];
	char **barriers = new char *[nFarsites];
	char **outs = new char *[nFarsites];
	int *outTypes = new int[nFarsites];
	CFarsite **pFarsites = new CFarsite*[nFarsites];
	for(int i = 0; i < nFarsites; i++)
		pFarsites[i] = NULL;
	for(int i = 0; i < nFarsites; i++)
	{
		lcps[i] = new char[MAX_PATH];
		inputs[i] = new char[MAX_PATH];
		ignits[i] = new char[MAX_PATH];
		barriers[i] = new char[MAX_PATH];
		outs[i] = new char[MAX_PATH];
		strcpy(lcps[i], "");
		strcpy(inputs[i], "");
		strcpy(ignits[i], "");
		strcpy(barriers[i], "");
		strcpy(outs[i], "");
		fgets(buf, 512, cmd);
		if(!ParseCommands(buf, lcps[i], inputs[i], ignits[i], barriers[i], outs[i], &outTypes[i]))
			continue;
	}

	int ret;
	int f;
	timeb t;

	ftime(&t);
	srand(t.time);

	for(f = 0; f < nFarsites; f++)
	{
        pFarsites[f] = new CFarsite();
		if(!cancelRequest)
		{
			printf("Loading lcp file for Farsite #%d: %s\n", f + 1, lcps[f]);
			if ( !pFarsites[f]->SetLandscapeFile(lcps[f]))
			{
				printf ("Can't open: %s \n", lcpFileName);
				delete pFarsites[f];
				pFarsites[f] = NULL;
				//_getch();
				continue;
			}
		}
		if(!cancelRequest)
		{
			printf("Loading inputs file for Farsite #%d: %s\n", f + 1, inputs[f]);
			ret = pFarsites[f]->LoadFarsiteInputs(inputs[f]);
			if ( ret != 1 )
			{
			   char *a = pFarsites[f]->CommandFileError(ret);
			   printf ("%s\n",a);
				delete pFarsites[f];
				pFarsites[f] = NULL;
				continue;
			}
		}
		if(!cancelRequest)
		{
			printf("Loading ignition file for Farsite #%d: %s\n", f + 1, ignits[f]);
			pFarsites[f]->SetIgnition(ignits[f]);
			if(strlen(barriers[f]) > 2)
			{
				printf("Loading barrier file for Farsite #%d: %s\n", f + 1, barriers[f]);
				pFarsites[f]->SetBarriers(barriers[f]);
			}
		}
		if(!cancelRequest)
		{
			printf("Launching Farsite #%d\n", f + 1);
			ret = pFarsites[f]->LaunchFarsite();
			if(ret != 1)
			{
			   char *a = pFarsites[f]->CommandFileError(ret);
			   printf ("%s\n",a);
				delete pFarsites[f];
				pFarsites[f] = NULL;
				continue;
			}
		}
		if(!cancelRequest)
		{
			printf("Writing outputs for Farsite #%d to %s\n", f + 1, outs[f]);
			if(outTypes[f] == 0 || outTypes[f] == 1 || outTypes[f] == 4)
			{
				if(outTypes[f] == 4)
				{
					pFarsites[f]->WriteOneHours(outs[f]);
					pFarsites[f]->WriteTenHours(outs[f]);
				}
				sprintf(outputName, "%s_CrownFire.asc", outs[f]);
				ret = pFarsites[f]->WriteCrownFireGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_Intensity.asc", outs[f]);
				ret = pFarsites[f]->WriteIntensityGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_FlameLength.asc", outs[f]);
				ret = pFarsites[f]->WriteFlameLengthGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadRate.asc", outs[f]);
				ret = pFarsites[f]->WriteSpreadRateGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadDirection.asc", outs[f]);
				ret = pFarsites[f]->WriteSpreadDirectionGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_HeatPerUnitArea.asc", outs[f]);
				ret = pFarsites[f]->WriteHeatPerUnitAreaGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ReactionIntensity.asc", outs[f]);
				ret = pFarsites[f]->WriteReactionIntensityGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ArrivalTime.asc", outs[f]);
				ret = pFarsites[f]->WriteArrivalTimeGrid(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
			}
			if(outTypes[f] == 0 || outTypes[f] == 2)
			{
				sprintf(outputName, "%s_CrownFire.fbg", outs[f]);
				ret = pFarsites[f]->WriteCrownFireGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ArrivalTime.fbg", outs[f]);
				ret = pFarsites[f]->WriteArrivalTimeGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_Intensity.fbg", outs[f]);
				ret = pFarsites[f]->WriteIntensityGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_FlameLength.fbg", outs[f]);
				ret = pFarsites[f]->WriteFlameLengthGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadRate.fbg", outs[f]);
				ret = pFarsites[f]->WriteSpreadRateGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_SpreadDirection.fbg", outs[f]);
				ret = pFarsites[f]->WriteSpreadDirectionGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_HeatPerUnitArea.fbg", outs[f]);
				ret = pFarsites[f]->WriteHeatPerUnitAreaGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ReactionIntensity.fbg", outs[f]);
				ret = pFarsites[f]->WriteReactionIntensityGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
				sprintf(outputName, "%s_ArrivalTime.fbg", outs[f]);
				ret = pFarsites[f]->WriteArrivalTimeGridBinary(outputName);
				if(ret != 1)
				{
					char *a = pFarsites[f]->CommandFileError(ret);
					printf ("%s for %s\n",a, outputName);
				}
			}
			/*sprintf(outputName, "%s_Moistures.txt", outs[f]);
			ret = pFarsites[f]->WriteMoistData(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}*/
			sprintf(outputName, "%s_Ignitions.asc", outs[f]);
			ret = pFarsites[f]->WriteIgnitionGrid(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Perimeters.shp", outs[f]);
			ret = pFarsites[f]->WritePerimetersShapeFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_SpotGrid.asc", outs[f]);
			ret = pFarsites[f]->WriteSpotGrid(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Spots.csv", outs[f]);
			ret = pFarsites[f]->WriteSpotDataFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Spots.shp", outs[f]);
			ret = pFarsites[f]->WriteSpotShapeFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
			sprintf(outputName, "%s_Timings.txt", outs[f]);
			ret = pFarsites[f]->WriteTimingsFile(outputName);
			if(ret != 1)
			{
				char *a = pFarsites[f]->CommandFileError(ret);
				printf ("%s for %s\n",a, outputName);
			}
		}
        delete pFarsites[f];
        pFarsites[f] = NULL;

	}
	fclose(cmd);
	delete[] pFarsites;
	for(int i = 0; i < nFarsites; i++)
	{
		delete[] lcps[i];
		delete[] inputs[i];
		delete[] outs[i];
		delete[] ignits[i];
		delete[] barriers[i];
	}
	delete[] lcps;
	delete[] inputs;
	delete[] outs;
	delete[] ignits;
	delete[] barriers;
	delete[] outTypes;
	return 0;
}
#endif
int main(int argc, char* argv[])
{
	//regMain(argc, argv);
	//multiMain(argc, argv);
#ifdef WIN32
	openMPMain(argc, argv);
#else
	linuxMain(argc, argv);
#endif
}
