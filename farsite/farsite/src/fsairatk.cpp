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
// 	FSAIRATK.CPP	Aerial Suppression Capabilities
//
//
//				Copyright 1997
//				Mark A. Finney, Systems for Environmental Management
//******************************************************************************

//#include "fsairdlg.h"
#include "fsairatk.h"
#include "Farsite5.h"
//#include "fsglbvar.h"
//#include <dir.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cctype>

//static long NumAirCraft = 0;
//AirCraft* aircraft[500];



AirCraft::AirCraft()
{
	memset(AirCraftName, 0x0, sizeof(AirCraftName));
	for (long i = 0; i < 6; i++)
		PatternLength[i] = -1;
	Units = 0;  	 //meters by default
}


//------------------------------------------------------------------------------
//
//  	  AirAttack Functions
//
//------------------------------------------------------------------------------

AirAttack::AirAttack()
{
}


AirAttack::~AirAttack()
{
}


bool AirAttack::CheckEffectiveness(AirAttackData* atk, double TimeStep)
{
	atk->ElapsedTime += TimeStep;
	if (atk->EffectiveDuration > 0.0 &&
		atk->ElapsedTime > atk->EffectiveDuration)
		return false;

	return true;
}



//------------------------------------------------------------------------------
//
//  	  Global Air Attack Support Functions
//
//------------------------------------------------------------------------------

/*static long NumAirAttacks = 0;
static long AirAttackCounter = 0;
static AirAttackData* FirstAirAttack;
static AirAttackData* NextAirAttack;
static AirAttackData* CurAirAttack;
static AirAttackData* LastAirAttack;
static char AirAttackLog[256];



long GetNumAirCraft()
{
	return NumAirCraft;
}

void SetNumAirCraft(long NumCraft)
{
	NumAirCraft = NumCraft;
}


bool LoadAirCraft(char* FileName, bool AppendList)
{
	FILE* AirAttackFile;
	char garbage[256], str[256];
	char ch[2] = "";
	long craftnumber, covlev;
	double patlen;

	if ((AirAttackFile = fopen(FileName, "r")) != NULL)
	{
		long i, j;
		j = GetNumAirCraft();
		if (!AppendList)
		{
			for (i = 0; i < j; i++)
				FreeAirCraft(i);
		}
		ch[0] = getc(AirAttackFile);
		do
		{
			memset(garbage, 0x0, sizeof(garbage));
			fgets(garbage, 255, AirAttackFile);
			if (feof(AirAttackFile))
				break;
			craftnumber = SetNewAirCraft();
			if (craftnumber == -1)
				return false;
			aircraft[craftnumber] = GetAirCraft(craftnumber);
			memset(aircraft[craftnumber]->AirCraftName, 0x0,
				sizeof(aircraft[craftnumber]->AirCraftName));
			strncpy(aircraft[craftnumber]->AirCraftName, garbage,
				strlen(garbage) - 2);

			fgets(garbage, 255, AirAttackFile);
			sscanf(garbage, "%s", str);
			std::transform(str, str+strlen(str), str, toupper);  
			if (!strcmp(str, "METERS"))
				aircraft[craftnumber]->Units = 0;
			else if (!strcmp(str, "FEET"))
				aircraft[craftnumber]->Units = 1;
			else
			{
				//::MessageBox(HWindow, "Units Not 'Feet' or 'Meters'", "Error, Unrecognized Units", MB_OK);
				aircraft[craftnumber]->Units = 0;    // default to meters
			}
			for (i = 0; i < 6; i++)
			{
				fgets(garbage, 255, AirAttackFile);
				sscanf(garbage, "%ld %lf", &covlev, &patlen);
				if (covlev < 99)
				{
					if (covlev > 0 && covlev < 5)
						aircraft[craftnumber]->PatternLength[covlev - 1] = (long)
							patlen;
					else if (covlev == 6)
						aircraft[craftnumber]->PatternLength[4] = (long)
							patlen;
					else if (covlev == 8)
						aircraft[craftnumber]->PatternLength[5] = (long)
							patlen;
				}
				else
					break;
			}
			ch[0] = getc(AirAttackFile);
			if ((toupper(ch[0])=='R') || (toupper(ch[0])=='C'))
			{
				fgets(garbage, 255, AirAttackFile);
				sscanf(garbage, "%s %lf", str, &patlen);
				std::transform(str, str+strlen(str), str, toupper) ; 
				if (!strcmp(str, "ETURN_TIME"))
					aircraft[craftnumber]->ReturnTime = patlen;
				else if (!strcmp(str, "OST"))
					aircraft[craftnumber]->Cost = patlen;
				ch[0] = getc(AirAttackFile);
				if ((toupper(ch[0])=='R') || (toupper(ch[0])=='C'))
				{
					fgets(garbage, 255, AirAttackFile);
					sscanf(garbage, "%s %lf", str, &patlen);
					std::transform(str, str+strlen(str), str, toupper) ; 
					if (!strcmp(str, "ETURN_TIME"))
						aircraft[craftnumber]->ReturnTime = patlen;
					else if (!strcmp(str, "OST"))
						aircraft[craftnumber]->Cost = patlen;
					ch[0] = getc(AirAttackFile);
				}
			}
			else // set return time and cost to unknown values
			{
				aircraft[craftnumber]->ReturnTime = 60.0;
				aircraft[craftnumber]->Cost = 1000.0;
			}
		}
		while (!feof(AirAttackFile));//strncmp(ch, "#", 1));  //(!feof(AirAttackFile));
		fclose(AirAttackFile);

		return true;
	}

	return false;
}

long SetNewAirCraft()
{
	if ((aircraft[NumAirCraft] = (AirCraft *) calloc(1, sizeof(AirCraft))) !=
		NULL)
		return NumAirCraft++;

	return -1;
}

AirCraft* GetAirCraft(long AirCraftNumber)
{
	if (aircraft[AirCraftNumber])
		return aircraft[AirCraftNumber];

	return 0;
}

long SetAirCraft(long AirCraftNumber)
{
	if ((aircraft[AirCraftNumber] = (AirCraft *) calloc(1, sizeof(AirCraft))) !=
		NULL)
		return NumAirCraft++;

	return -1;
}

void FreeAirCraft(long AirCraftNumber)
{
	if (aircraft[AirCraftNumber])
	{
		free(aircraft[AirCraftNumber]);
		NumAirCraft--;
		if (NumAirCraft < 0)
			NumAirCraft = 0;
	}

	aircraft[AirCraftNumber] = 0;
}


long GetNumAirAttacks()
{
	return NumAirAttacks;
}


void SetNumAirAttacks(long NumAtk)
{
	NumAirAttacks = NumAtk;
}

long SetupAirAttack(long AirCraftNumber, long CoverageLevel, long Duration,
	double* startpoint)
{
	long i;

	for (i = 0; i <= NumAirAttacks; i++)
	{
		if (NumAirAttacks == 0)
		{
			if ((FirstAirAttack = (AirAttackData *) calloc(1,
														sizeof(AirAttackData))) !=
				NULL)
			{
				CurAirAttack = FirstAirAttack;
				if (AirAttackCounter == 0)
				{
					memset(AirAttackLog, 0x0, sizeof(AirAttackLog));
					//getcwd(AirAttackLog, 255);
					strcat(AirAttackLog, "airattk.log");
					//DeleteFile(AirAttackLog);
					remove(AirAttackLog);
				}
			}
			else
				return 0;
		}
		else if (i == 0)
			CurAirAttack = FirstAirAttack;
		else
			CurAirAttack = NextAirAttack;
		if (i < NumAirAttacks)
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
	}

	if ((NextAirAttack = (AirAttackData *) calloc(1, sizeof(AirAttackData))) !=
		NULL)
	{
		NumAirAttacks++;
		CurAirAttack->next = (AirAttackData *) NextAirAttack;
		CurAirAttack->AirAttackNumber = ++AirAttackCounter;
		CurAirAttack->EffectiveDuration = Duration;
		CurAirAttack->CoverageLevel = CoverageLevel;
		CurAirAttack->AirCraftNumber = AirCraftNumber;
		CurAirAttack->ElapsedTime = 0.0;

		if (startpoint)
		{
			double InitDist, MinDist, Mult = 1.0, Ratio;
			double NewEndPtX, NewEndPtY, XDist, YDist;
			double xpt, ypt, xmax, ymax, xmin, ymin;
			double DistRes, PerimRes, PtX, PtY;

			DistRes = GetDistRes() / 1.4 * MetricResolutionConvert();
			PerimRes = GetDistRes() / 2.0 * MetricResolutionConvert();//GetPerimRes()/2.0;
			if (aircraft[AirCraftNumber]->Units == 1)
				Mult = 0.3048;  	  // feet to meters
			XDist = startpoint[0] - startpoint[2];
			YDist = startpoint[1] - startpoint[3];
			InitDist = sqrt(pow2(XDist) + pow2(YDist));
			MinDist = aircraft[AirCraftNumber]->PatternLength[CoverageLevel] * Mult * MetricResolutionConvert();
			Ratio = InitDist / MinDist;
			NewEndPtX = startpoint[0] - (XDist) / Ratio;
			NewEndPtY = startpoint[1] - (YDist) / Ratio;

			Ratio = InitDist / (MinDist + PerimRes);
			PtX = startpoint[0] - (XDist) / Ratio;
			PtY = startpoint[1] - (YDist) / Ratio;

			XDist = startpoint[0] - NewEndPtX;
			YDist = startpoint[1] - NewEndPtY;

			CurAirAttack->PatternNumber = GetNewFires();
			AllocPerimeter1(CurAirAttack->PatternNumber, 6);
			SetInout(CurAirAttack->PatternNumber, 3);
			SetNumPoints(CurAirAttack->PatternNumber, 5);

			xpt = xmin = xmax = startpoint[0] + DistRes / MinDist * YDist;
			ypt = ymin = ymax = startpoint[1] - DistRes / MinDist * XDist;
			SetPerimeter1(GetNewFires(), 0, xpt, ypt);
			SetFireChx(GetNewFires(), 0, 0, -1);

			xpt = startpoint[0] - DistRes / MinDist * YDist;
			ypt = startpoint[1] + DistRes / MinDist * XDist;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 1, xpt, ypt);
			SetFireChx(GetNewFires(), 1, 0, -1);

			xpt = NewEndPtX - DistRes / MinDist * YDist;
			ypt = NewEndPtY + DistRes / MinDist * XDist;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 2, xpt, ypt);
			SetFireChx(GetNewFires(), 2, 0, -1);

			xpt = PtX;
			ypt = PtY;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 3, xpt, ypt);
			SetFireChx(GetNewFires(), 3, 0, -1);

			xpt = NewEndPtX + DistRes / MinDist * YDist;
			ypt = NewEndPtY - DistRes / MinDist * XDist;
			if (xpt < xmin)
				xmin = xpt;
			if (ypt < ymin)
				ymin = ypt;
			if (xpt > xmax)
				xmax = xpt;
			if (ypt > ymax)
				ymax = ypt;
			SetPerimeter1(GetNewFires(), 4, xpt, ypt);
			SetFireChx(GetNewFires(), 4, 0, -1);

			SetPerimeter1(GetNewFires(), 5, xmin, xmax);
			SetFireChx(GetNewFires(), 5, ymin, ymax);
			IncNewFires(1);
			IncNumFires(1);
		}
	}
	else
		return 0;

	return AirAttackCounter;
}


void CancelAirAttack(long AirAttackCounter)
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		if (CurAirAttack->AirAttackNumber == AirAttackCounter)
		{
			if (i == 0)
				FirstAirAttack = (AirAttackData *) CurAirAttack->next;
			else
				LastAirAttack->next = (AirAttackData *) NextAirAttack;
			WriteAirAttackLog(CurAirAttack);
			free(CurAirAttack);
			NumAirAttacks--;
			if (NumAirAttacks == 0)
				free(NextAirAttack);
			break;
		}
		else
		{
			LastAirAttack = CurAirAttack;
			CurAirAttack = NextAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
	}
}

void LoadAirAttack(AirAttackData airattackdata)
{
	// function only for loading air attacks from bookmark
	if (NumAirAttacks == 0)
	{
		FirstAirAttack = (AirAttackData *) calloc(1, sizeof(AirAttackData));
		CurAirAttack = FirstAirAttack;
		memcpy(FirstAirAttack, &airattackdata, sizeof(AirAttackData));
	}
	memcpy(CurAirAttack, &airattackdata, sizeof(AirAttackData));
	NextAirAttack = (AirAttackData *) calloc(1, sizeof(AirAttackData));
	CurAirAttack->next = (AirAttackData *) NextAirAttack;
	if (NumAirAttacks == 0)
		FirstAirAttack->next = (AirAttackData *) NextAirAttack;
	NumAirAttacks++;
	CurAirAttack = NextAirAttack;
}

void FreeAllAirAttacks()
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		WriteAirAttackLog(CurAirAttack);
		free(CurAirAttack);
		CurAirAttack = NextAirAttack;
		NextAirAttack = (AirAttackData *) CurAirAttack->next;
	}
	if (NumAirAttacks > 0)
	{
		free(CurAirAttack);
		NumAirAttacks = 0;
	}
	AirAttackCounter = 0;
}

AirAttackData* GetAirAttack(long AirAttackCounter)
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		if (CurAirAttack->AirAttackNumber == AirAttackCounter)
			return CurAirAttack;
		else
		{
			CurAirAttack = NextAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
	}

	return NULL;
}

AirAttackData* GetAirAttackByOrder(long OrdinalAttackNum)
{
	// retrieves indirect attack in order
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
			CurAirAttack = FirstAirAttack;
		else
			CurAirAttack = NextAirAttack;
		NextAirAttack = (AirAttackData *) CurAirAttack->next;
		if (i == OrdinalAttackNum)
			return CurAirAttack;
	}

	return 0;
}


void SetNewFireNumberForAirAttack(long oldnumfire, long newnumfire)
{
	for (long i = 0; i < NumAirAttacks; i++)
	{
		if (i == 0)
		{
			CurAirAttack = FirstAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
		if (CurAirAttack->PatternNumber == oldnumfire)
		{
			CurAirAttack->PatternNumber = newnumfire;

			break;
		}
		else
		{
			CurAirAttack = NextAirAttack;
			NextAirAttack = (AirAttackData *) CurAirAttack->next;
		}
	}
}


void WriteAirAttackLog(AirAttackData* atk)
{
	FILE* airatklog;
	char units[24];
	long covlevel;

	if ((airatklog = fopen(AirAttackLog, "a")) != NULL)
	{
		if (aircraft[atk->AirCraftNumber]->Units == 0)
			sprintf(units, "meters");
		else
			sprintf(units, "feet");

		if (atk->CoverageLevel == 4)
			covlevel = 6;
		else if (atk->CoverageLevel == 5)
			covlevel = 8;
		else
			covlevel = atk->CoverageLevel + 1;

		if (atk->EffectiveDuration > 0.0)
			fprintf(airatklog, "%s, %s %ld, %s %lf %s, %s %ld %s\n",
				aircraft[atk->AirCraftNumber]->AirCraftName,
				"Coverage Level:", covlevel, "Line Length:",
				aircraft[atk->AirCraftNumber]->PatternLength[atk->CoverageLevel],
				units, "Duration:", (long)
				atk->EffectiveDuration, "mins");
		else
			fprintf(airatklog, "%s, %s %ld, %s %lf %s, %s\n",
				aircraft[atk->AirCraftNumber]->AirCraftName,
				"Coverage Level:", covlevel, "Line Length:",
				aircraft[atk->AirCraftNumber]->PatternLength[atk->CoverageLevel],
				units, "Duration: Unlimited");

		fclose(airatklog);
	}
}
*/

//------------------------------------------------------------------------------
//
//  	  Group Air Attack Support Functions
//
//------------------------------------------------------------------------------
/*
static long NumGroupAttacks = 0;
static long GroupAttackCounter = 0;
static GroupAttackData* FirstGroupAttack;
static GroupAttackData* NextGroupAttack;
static GroupAttackData* CurGroupAttack;
static GroupAttackData* LastGroupAttack;
static GroupAttackData* ReAtk;

long SetupGroupAirAttack(long AircraftNumber, long CoverageLevel,
	long Duration, double* line, long FireNum, char* GroupName)
{
	long i;
	GroupAirAttack* gp;

	for (i = 0; i <= NumGroupAttacks; i++)
	{
		if (NumGroupAttacks == 0)
		{
			if ((FirstGroupAttack = (GroupAttackData *)
				calloc(1,
					sizeof(GroupAttackData))) !=
				NULL)
				CurGroupAttack = FirstGroupAttack;
			else
				return 0;
		}
		else if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		else
			CurGroupAttack = NextGroupAttack;
		if (i < NumGroupAttacks)
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
	}

	if ((NextGroupAttack = (GroupAttackData *) calloc(1,
												sizeof(GroupAttackData))) !=
		NULL)
	{
		NumGroupAttacks++;
		CurGroupAttack->next = (GroupAttackData *) NextGroupAttack;
		CurGroupAttack->GroupAttackNumber = ++GroupAttackCounter;
		strcpy(CurGroupAttack->GroupName, GroupName);
		CurGroupAttack->Suspended = 0;	// false
		gp = new GroupAirAttack(CurGroupAttack);
		gp->AllocGroup(GROUPSIZE);
		gp->AddGroupMember(AircraftNumber, CoverageLevel, Duration);
		gp->SetGroupAssignment(line, FireNum, false);
		//gp->ExecuteAttacks(0.0);
		delete gp;
	}

	return GroupAttackCounter;
}


GroupAttackData* ReassignGroupAirAttack(GroupAttackData* atk)
{
	if (atk != 0)
		ReAtk = atk;

	return ReAtk;
}


void CancelGroupAirAttack(long GroupCounter)
{
	long i;

	for (i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
		{
			CurGroupAttack = FirstGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
		if (CurGroupAttack->GroupAttackNumber == GroupCounter)
		{
			if (i == 0)
				FirstGroupAttack = (GroupAttackData *) CurGroupAttack->next;
			else
				LastGroupAttack->next = (GroupAttackData *) NextGroupAttack;
			if (CurGroupAttack->IndirectLine)
				free(CurGroupAttack->IndirectLine);
			free(CurGroupAttack->WaitTime);
			free(CurGroupAttack->CoverageLevel);
			free(CurGroupAttack->EffectiveDuration);
			free(CurGroupAttack->AircraftNumber);
			free(CurGroupAttack);
			NumGroupAttacks--;
			if (NumGroupAttacks == 0)
				free(NextGroupAttack);
			break;
		}
		else
		{
			LastGroupAttack = CurGroupAttack;
			CurGroupAttack = NextGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
	}
}


void LoadGroupAttack(GroupAttackData gatk)
{
	if (gatk.NumCurrentAircraft <= 0)
		return;

	long i;
	for (i = 0; i <= NumGroupAttacks; i++)
	{
		if (NumGroupAttacks == 0)
		{
			if ((FirstGroupAttack = (GroupAttackData *)
				calloc(1,
					sizeof(GroupAttackData))) !=
				NULL)
				CurGroupAttack = FirstGroupAttack;
			else
				return;
		}
		else if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		else
			CurGroupAttack = NextGroupAttack;
		if (i < NumGroupAttacks)
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
	}

	if ((NextGroupAttack = (GroupAttackData *) calloc(1,
												sizeof(GroupAttackData))) !=
		NULL)
	{
		NumGroupAttacks++;
		CurGroupAttack->next = (GroupAttackData *) NextGroupAttack;
		CurGroupAttack->GroupAttackNumber = ++GroupAttackCounter;
		strcpy(CurGroupAttack->GroupName, gatk.GroupName);
		CurGroupAttack->Suspended = 0;	// false
		if (gatk.NumPoints < 0)
		{
			CurGroupAttack->IndirectLine = (double *)
				calloc(4, sizeof(double));
			memcpy(CurGroupAttack->IndirectLine, gatk.IndirectLine,
				4 * sizeof(double));
		}
		else
		{
			CurGroupAttack->IndirectLine = (double *)
				calloc(gatk.NumPoints * 2,
														sizeof(double));
			memcpy(CurGroupAttack->IndirectLine, gatk.IndirectLine,
				gatk.NumPoints * 2 * sizeof(double));
		}
		CurGroupAttack->CoverageLevel = (long *) calloc(gatk.NumTotalAircraft,
													sizeof(long));
		CurGroupAttack->EffectiveDuration = (long *)
			calloc(gatk.NumTotalAircraft,
														sizeof(long));
		CurGroupAttack->AircraftNumber = (long *)
			calloc(gatk.NumTotalAircraft,
													sizeof(long));
		CurGroupAttack->WaitTime = (double *) calloc(gatk.NumTotalAircraft,
												sizeof(double));
		memcpy(CurGroupAttack->CoverageLevel, gatk.CoverageLevel,
			gatk.NumCurrentAircraft * sizeof(long));
		memcpy(CurGroupAttack->EffectiveDuration, gatk.EffectiveDuration,
			gatk.NumCurrentAircraft * sizeof(long));
		memcpy(CurGroupAttack->AircraftNumber, gatk.AircraftNumber,
			gatk.NumCurrentAircraft * sizeof(long));
		memcpy(CurGroupAttack->WaitTime, gatk.WaitTime,
			gatk.NumCurrentAircraft * sizeof(double));
		CurGroupAttack->NumTotalAircraft = gatk.NumTotalAircraft;
		CurGroupAttack->NumCurrentAircraft = gatk.NumCurrentAircraft;
		CurGroupAttack->Suspended = gatk.Suspended;
		memcpy(CurGroupAttack->GroupName, gatk.GroupName,
			256 * sizeof(char));
		CurGroupAttack->NumPoints = gatk.NumPoints;
		CurGroupAttack->FireNumber = gatk.FireNumber;
	}
}


void FreeAllGroupAirAttacks()
{
	long i;

	for (i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
		{
			CurGroupAttack = FirstGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
		if (CurGroupAttack->IndirectLine)
			free(CurGroupAttack->IndirectLine);
		free(CurGroupAttack->CoverageLevel);
		free(CurGroupAttack->EffectiveDuration);
		free(CurGroupAttack->AircraftNumber);
		free(CurGroupAttack->WaitTime);
		free(CurGroupAttack);
		CurGroupAttack = NextGroupAttack;
		NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
	}
	if (NumGroupAttacks > 0)
		free(CurGroupAttack);

	NumGroupAttacks = 0;
	GroupAttackCounter = 0;
}

GroupAttackData* GetGroupAirAttack(long GroupAttackCounter)
{
	for (long i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
		{
			CurGroupAttack = FirstGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
		if (CurGroupAttack->GroupAttackNumber == GroupAttackCounter)
			return CurGroupAttack;
		else
		{
			CurGroupAttack = NextGroupAttack;
			NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		}
	}

	return NULL;
}

GroupAttackData* GetGroupAttackByOrder(long OrdinalAttackNumber)
{
	for (long i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		else
			CurGroupAttack = NextGroupAttack;
		NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		if (i == OrdinalAttackNumber)
			return CurGroupAttack;
	}

	return 0;
}


GroupAttackData* GetGroupAttackForFireNumber(long NumFire,
	long StartAttackNum, long* LastAttackNumber)
{
	for (long i = 0; i < NumGroupAttacks; i++)
	{
		if (i == 0)
			CurGroupAttack = FirstGroupAttack;
		NextGroupAttack = (GroupAttackData *) CurGroupAttack->next;
		if (i >= StartAttackNum)
		{
			if (CurGroupAttack->FireNumber == NumFire)// && CurAttack->Suspended==0)
			{
				*LastAttackNumber = i;

				return CurGroupAttack;
			}
		}
		CurGroupAttack = NextGroupAttack;
	}

	return 0;
}

void SetNewFireNumberForGroupAttack(long OldFireNum, long NewFireNum)
{
	long ThisGroup, NextGroup;

	ThisGroup = 0;
	while (GetGroupAttackForFireNumber(OldFireNum, ThisGroup, &NextGroup))
	{
		ThisGroup = NextGroup + 1;
		CurGroupAttack->FireNumber = NewFireNum;
	}
}

long GetNumGroupAttacks()
{
	return NumGroupAttacks;
}

void SetNumGroupAttacks(long NumGroups)
{
	NumGroupAttacks = NumGroups;
}
*/

//------------------------------------------------------------------------------

GroupAirAttack::GroupAirAttack(Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
}

GroupAirAttack::GroupAirAttack(GroupAttackData* atk, Farsite5 *_pFarsite)
{
	attack = atk;
	pFarsite = _pFarsite;
}

GroupAirAttack::~GroupAirAttack()
{
}


void GroupAirAttack::SetGroup(GroupAttackData* gatk)
{
	attack = gatk;
}


void GroupAirAttack::GetCurrentGroup()
{
	attack = pFarsite->CurGroupAttack;
}

void GroupAirAttack::ExecuteAllIndirectAttacks(double TimeIncrement)
{
	long i;
	double OriginalTimeInc, NextTime;

	OriginalTimeInc = TimeIncrement;
	for (i = 0; i < pFarsite->NumGroupAttacks; i++)
	{
		attack = pFarsite->GetGroupAttackByOrder(i);
		if (CheckSuspendState(GETVAL))
			continue;
		if (attack->FireNumber < 0)// || TimeIncrement>0.0)   // if indirect attack
		{
			do  				  // find smallest attack time
			{
				NextTime = GetNextAttackTime(TimeIncrement);
				if (!ExecuteAttacks(NextTime))
				{
					pFarsite->CancelGroupAirAttack(attack->GroupAttackNumber);
					i--;	// go back one because NumGroupAttacks has been decremented
					break;
				}
				IncrementWaitTimes(NextTime);
				TimeIncrement -= NextTime;
			}
			while (TimeIncrement > 0.0);
			TimeIncrement = OriginalTimeInc;
		}
	}
}


bool GroupAirAttack::AllocGroup(long NewNumber)
{
	if ((attack->CoverageLevel = (long *) calloc(NewNumber, sizeof(long))) ==
		NULL)
		return false;
	if ((attack->EffectiveDuration = (long *) calloc(NewNumber, sizeof(long))) ==
		NULL)
		return false;
	if ((attack->AircraftNumber = (long *) calloc(NewNumber, sizeof(long))) ==
		NULL)
		return false;
	if ((attack->WaitTime = (double *) calloc(NewNumber, sizeof(double))) ==
		NULL)
		return false;
	attack->NumCurrentAircraft = 0;
	attack->NumTotalAircraft = NewNumber;

	return true;
}

bool GroupAirAttack::AddGroupMember(long AircraftNumber, long CoverageLevel,
	long EffectiveDuration)
{
	long numcur, * tempcov, * tempnum, * tempdur;
	double* tempelapsed;

	numcur = attack->NumCurrentAircraft;
	if (numcur == attack->NumTotalAircraft)
	{
		if (numcur > 0)
		{
			if ((tempcov = (long *) calloc(numcur, sizeof(long))) == NULL)
				return false;
			memcpy(tempcov, attack->CoverageLevel, numcur * sizeof(long));
			if ((tempnum = (long *) calloc(numcur, sizeof(long))) == NULL)
				return false;
			memcpy(tempnum, attack->AircraftNumber, numcur * sizeof(long));
			if ((tempdur = (long *) calloc(numcur, sizeof(long))) == NULL)
				return false;
			memcpy(tempdur, attack->EffectiveDuration, sizeof(long));
			if ((tempelapsed = (double *) calloc(numcur, sizeof(double))) ==
				NULL)
				return false;
			memcpy(tempelapsed, attack->WaitTime, numcur * sizeof(double));
			free(attack->CoverageLevel);
			free(attack->EffectiveDuration);
			free(attack->AircraftNumber);
			free(attack->WaitTime);
		}
		attack->NumTotalAircraft += GROUPSIZE;
		if (!AllocGroup(attack->NumTotalAircraft))
			return false;
		if (numcur > 0)
		{
			memcpy(attack->EffectiveDuration, tempdur, numcur * sizeof(long));
			memcpy(attack->AircraftNumber, tempnum, numcur * sizeof(long));
			memcpy(attack->CoverageLevel, tempcov, numcur * sizeof(long));
			memcpy(attack->WaitTime, tempelapsed, numcur * sizeof(double));
			free(tempdur);
			free(tempcov);
			free(tempnum);
			free(tempelapsed);
		}
	}
	attack->EffectiveDuration[numcur] = EffectiveDuration;
	attack->AircraftNumber[numcur] = AircraftNumber;
	attack->CoverageLevel[numcur] = CoverageLevel;
	attack->WaitTime[numcur] = 0.0;
	attack->NumCurrentAircraft++;

	return true;
}


bool GroupAirAttack::SetGroupAssignment(double* line, long FireNum, bool Reset)
{
	long i;
	bool OK = true;

	if (Reset)
	{
		if (attack->IndirectLine)
			free(attack->IndirectLine);
		attack->IndirectLine = 0;
	}

	if (FireNum < 0)		 // negative flag indicating indirect line
	{
		FireNum *= -1;
		if ((attack->IndirectLine = (double *) calloc(FireNum * 2,
												sizeof(double))) ==
			NULL)
		{
			OK = false;

			return OK;
		}
		attack->NumPoints = FireNum;
		for (i = 0; i < FireNum; i++)
		{
			attack->IndirectLine[i * 2] = line[i * 2];
			attack->IndirectLine[i * 2 + 1] = line[i * 2 + 1];
		}
		attack->FireNumber = -1;
	}
	else				  // positive number indicating actual fire number, line indicates 1st pt
	{
		long pt;
		double mindist, dist, x, y;

		if ((attack->IndirectLine = (double *) calloc(4, sizeof(double))) ==
			NULL)
		{
			OK = false;

			return OK;
		}
		attack->FireNumber = FireNum;
		//attack->IndirectLine[2]=line[0];  	// put in secondary positions
		//attack->IndirectLine[3]=line[1];  	// to be transfered in Execute()

		pt = 0;
		for (i = 0; i < pFarsite->GetNumPoints(FireNum); i++)
		{
			x = pFarsite->GetPerimeter1Value(FireNum, i, XCOORD);
			y = pFarsite->GetPerimeter1Value(FireNum, i, YCOORD);
			dist = pow2(x - line[0]) + pow2(y - line[1]);
			if (i == 0)
			{
				mindist = dist;

				continue;
			}
			if (dist < mindist)
			{
				mindist = dist;
				pt = i;
			}
		}
		x = pFarsite->GetPerimeter1Value(FireNum, pt, XCOORD);
		y = pFarsite->GetPerimeter1Value(FireNum, pt, YCOORD);
		attack->IndirectLine[2] = x;	  // put in secondary positions
		attack->IndirectLine[3] = y;	  // to be transfered in Execute()
		pt -= 1;
		if (pt < 0)
			pt += pFarsite->GetNumPoints(FireNum);
		x = pFarsite->GetPerimeter1Value(FireNum, pt, XCOORD);
		y = pFarsite->GetPerimeter1Value(FireNum, pt, YCOORD);
		dist = pow2(x - line[2]) + pow2(y - line[3]);
		pt += 2;
		if (pt > pFarsite->GetNumPoints(FireNum) - 1)
			pt -= pFarsite->GetNumPoints(FireNum);
		x = pFarsite->GetPerimeter1Value(FireNum, pt, XCOORD);
		y = pFarsite->GetPerimeter1Value(FireNum, pt, YCOORD);
		mindist = pow2(x - line[2]) + pow2(y - line[3]);
		if (mindist <= dist)
			attack->Direction = 0;
		else
			attack->Direction = 1;
		attack->NumPoints = -1;
	}

	return OK;
}


bool GroupAirAttack::RemoveGroupMember(long membernumber)
{
	if (membernumber < attack->NumCurrentAircraft)
	{
		attack->AircraftNumber[membernumber] = -1;
		ReorderGroupList(membernumber);
	}
	else
		return false;

	return true;
}


void GroupAirAttack::ReorderGroupList(long Start)
{
	long i;

	if (Start < 0)
	{
		for (i = 0; i < attack->NumCurrentAircraft; i++)
		{
			if (attack->AircraftNumber[i] < 0)
			{
				Start = i;
				break;
			}
		}
	}
	for (i = Start; i < attack->NumCurrentAircraft - 1; i++)
	{
		attack->CoverageLevel[i] = attack->CoverageLevel[i + 1];
		attack->EffectiveDuration[i] = attack->EffectiveDuration[i + 1];
		attack->AircraftNumber[i] = attack->AircraftNumber[i + 1];
		attack->WaitTime[i] = attack->WaitTime[i + 1];
	}
	attack->NumCurrentAircraft--;
}


bool GroupAirAttack::ExecuteAttacks(double TimeIncrement)
{
	long i, j;
	double AttackLine[4];
	long CurPt, NextPt;
	double InitDist, MinDist, TestDist, Mult = 1.0, Ratio;
	double DistRes, NewEndPtX, NewEndPtY, XDist, YDist;

	for (i = 0; i < attack->NumCurrentAircraft; i++)
	{
		if (fabs(attack->WaitTime[i]) < 1e-9) // <=TimeIncrement// prosecute attack
		{
			if (pFarsite->aircraft[attack->AircraftNumber[i]]->Units == 1)
				Mult = 0.3048;  	  // feet to meters

			XDist = YDist = 0.0;
			MinDist = pFarsite->aircraft[attack->AircraftNumber[i]]->PatternLength[attack->CoverageLevel[i]] * Mult * pFarsite->MetricResolutionConvert();
			if (attack->FireNumber < 0) 	 // indirect attack
			{
				CurPt = NextPt = (attack->FireNumber + 1) * -1;
				if (CurPt == attack->NumPoints) 		   // nothing to do
					return false;
				for (j = CurPt + 1; j < attack->NumPoints; j++)
				{
					XDist += attack->IndirectLine[CurPt * 2] -
						attack->IndirectLine[j * 2];
					YDist += attack->IndirectLine[CurPt * 2 + 1] -
						attack->IndirectLine[j * 2 + 1];
					InitDist = sqrt(pow2(XDist) + pow2(YDist));
					NextPt = j - 1; 					  // sub one
					if (InitDist >= MinDist)
						break;
				}
				if (InitDist < MinDist)
					return false;
				AttackLine[0] = attack->IndirectLine[CurPt * 2];
				AttackLine[1] = attack->IndirectLine[CurPt * 2 + 1];

				Ratio = InitDist / MinDist;
				NewEndPtX = attack->IndirectLine[CurPt * 2] - (XDist) / Ratio;
				NewEndPtY = attack->IndirectLine[CurPt * 2 + 1] -
					(YDist) /
					Ratio;
				attack->IndirectLine[NextPt * 2] = NewEndPtX;   	// set up for next attack
				attack->IndirectLine[NextPt * 2 + 1] = NewEndPtY;
				attack->FireNumber = -(NextPt + 1);

				AttackLine[2] = attack->IndirectLine[NextPt * 2];
				AttackLine[3] = attack->IndirectLine[NextPt * 2 + 1];
				attack->WaitTime[i] += pFarsite->aircraft[attack->AircraftNumber[i]]->ReturnTime;
				if (fabs(attack->WaitTime[i] -
						pFarsite->aircraft[attack->AircraftNumber[i]]->ReturnTime) <=
					1e-3)
					attack->WaitTime[i] = pFarsite->aircraft[attack->AircraftNumber[i]]->ReturnTime;
			}
			else				  // direct attack, get current position on fire
			{
				DistRes = pFarsite->GetDistRes();

				attack->IndirectLine[0] = attack->IndirectLine[2];
				attack->IndirectLine[1] = attack->IndirectLine[3];
				CurPt = 0;
				for (j = 0; j < pFarsite->GetNumPoints(attack->FireNumber); j++)
				{
					InitDist = pow2(attack->IndirectLine[0] -
								pFarsite->GetPerimeter1Value(attack->FireNumber, j,
									XCOORD)) +
						pow2(attack->IndirectLine[1] -
							pFarsite->GetPerimeter1Value(attack->FireNumber,
								j,
								YCOORD));
					if (j == 0)
						TestDist = InitDist;
					else if (InitDist < TestDist)
					{
						TestDist = InitDist;
						CurPt = j;
						if (TestDist == 0.0)
							break;
					}
				}
				for (j = 1; j <= pFarsite->GetNumPoints(attack->FireNumber); j++)
				{
					if (attack->Direction == 0)
					{
						NextPt = CurPt + j;
						if (NextPt > pFarsite->GetNumPoints(attack->FireNumber) - 1)
							NextPt -= pFarsite->GetNumPoints(attack->FireNumber);
					}
					else
					{
						NextPt = CurPt - j;
						if (NextPt < 0)
							NextPt += pFarsite->GetNumPoints(attack->FireNumber);
					}
					InitDist = sqrt(pow2(attack->IndirectLine[0] -
										pFarsite->GetPerimeter1Value(attack->FireNumber,
											NextPt, XCOORD)) +
								pow2(attack->IndirectLine[1] -
									pFarsite->GetPerimeter1Value(attack->FireNumber,
										NextPt, YCOORD)));
					if (InitDist > MinDist)
						break;
				}
				AttackLine[2] = pFarsite->GetPerimeter1Value(attack->FireNumber, NextPt,
									XCOORD);
				AttackLine[3] = pFarsite->GetPerimeter1Value(attack->FireNumber, NextPt,
									YCOORD);
				XDist = attack->IndirectLine[0] - AttackLine[2];
				YDist = attack->IndirectLine[1] - AttackLine[3];
				Ratio = MinDist / InitDist;
				attack->IndirectLine[2] = attack->IndirectLine[0] -
					XDist * Ratio;
				attack->IndirectLine[3] = attack->IndirectLine[1] -
					YDist * Ratio;
				AttackLine[0] = attack->IndirectLine[0];
				AttackLine[1] = attack->IndirectLine[1];

				// move drop line out from existing fire front by DistRes
				Ratio = DistRes / InitDist;
				if (attack->Direction == 0)		// if following original fire direction
				{
					AttackLine[0] = AttackLine[0] - YDist * Ratio;
					AttackLine[1] = AttackLine[1] + XDist * Ratio;
					AttackLine[2] = AttackLine[2] - YDist * Ratio;
					AttackLine[3] = AttackLine[3] + XDist * Ratio;
				}
				else
				{
					AttackLine[0] = AttackLine[0] + YDist * Ratio;
					AttackLine[1] = AttackLine[1] - XDist * Ratio;
					AttackLine[2] = AttackLine[2] + YDist * Ratio;
					AttackLine[3] = AttackLine[3] - XDist * Ratio;
				}
			}
			pFarsite->SetupAirAttack(attack->AircraftNumber[i],
				attack->CoverageLevel[i], attack->EffectiveDuration[i],
				AttackLine);
			attack->WaitTime[i] += pFarsite->aircraft[attack->AircraftNumber[i]]->ReturnTime;
			if (fabs(attack->WaitTime[i] -
					pFarsite->aircraft[attack->AircraftNumber[i]]->ReturnTime) <= 1e-3)
				attack->WaitTime[i] = pFarsite->aircraft[attack->AircraftNumber[i]]->ReturnTime;
		}
	}
	//IncrementWaitTimes(TimeIncrement);

	return true;
}


double GroupAirAttack::GetNextAttackTime(double TimeIncrement)
{
	long i;

	for (i = 0; i < attack->NumCurrentAircraft; i++)
	{
		if (attack->WaitTime[i] < TimeIncrement)
			TimeIncrement = attack->WaitTime[i];
	}

	return TimeIncrement;
}


void GroupAirAttack::IncrementWaitTimes(double TimeIncrement)
{
	long i;

	for (i = 0; i < attack->NumCurrentAircraft; i++)
	{
		attack->WaitTime[i] -= TimeIncrement;
		if (attack->WaitTime[i] <= 0.0)
			attack->WaitTime[i] = 0.0;
	}
}


long GroupAirAttack::CheckSuspendState(long OnOff)
{
	if (OnOff >= 0)
		attack->Suspended = OnOff;

	return attack->Suspended;
}




//------------------------------------------------------------------------------
//
//   Air Resources Dialog
//
//------------------------------------------------------------------------------
/*

TTransAirData::TTransAirData()
{
	ResetData();
}


void TTransAirData::ResetData()
{
	memset(Dat.AirFileName, 0x0, sizeof(Dat.AirFileName));
	Dat.m = true;
	Dat.ft = false;
}

//------------------------------------------------------------------------------
//
//   Air Attack Dialog
//
//------------------------------------------------------------------------------



TTransAirAttackData::TTransAirAttackData()
{
	Dat.l1 = true;
	Dat.l2 = false;
	Dat.l3 = false;
	Dat.l4 = false;
	Dat.l5 = false;
	Dat.l6 = false;
	memset(Dat.Duration, 0x0, sizeof(Dat.Duration));
	sprintf(Dat.Duration, "4");
	Dat.gc1 = true;
	Dat.gc2 = false;
	Dat.gc3 = false;
	Dat.gc4 = false;
}

//------------------------------------------------------------------------------
//
//     Group Air Revisions
//
//------------------------------------------------------------------------------


static char Buffer[512];
*/









