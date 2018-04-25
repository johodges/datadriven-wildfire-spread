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
#include "Farsite5.h"

Crown::Crown(Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
}

Crown::~Crown()
{
}

void Crown::CrownIgnite(double height, double base, double density)
	// Determines if surface fire will become crownfire
{
	double temp;

	HLCB = base;//GetCrownBase();
	if (HLCB == 0.0)
		HLCB = 0.01;
	FoliarMC = pFarsite->GetFoliarMC();
	CrownBulkDensity = density;//GetCrownBD();
	if (CrownBulkDensity == 0.0)
		CrownBulkDensity = 0.01;
	TreeHeight = height;//GetCrownHeight();
	if (TreeHeight <= HLCB)
		TreeHeight = HLCB + 1.0;	// 1 meter higher
	temp = 0.010 * HLCB * (460.0 + 25.9 * FoliarMC);
	Io = temp * sqrt(temp);   // Alexander 1988 from Van Wagner 1977 & 1993,
	//	Io=pow(.010*HLCB*(460.0+25.9*FoliarMC),1.5);   // Alexander 1988 from Van Wagner 1977 & 1993,
}



double Crown::CrownSpread(double avgros, double R10)		 // R10 is from Rothermel 1991
{
	double CrownROS = 0.0;


	if (CrownBulkDensity > 0.0)
	{
		/*
		double CritCFB;
		if(R10>avgros)
		{    CritCFB=(Ro+0.8*(CritRos-Ro)-avgros)/(R10-avgros);
		if(CrownFractionBurned>=CritCFB)
		CrownROS=avgros+CrownFractionBurned*(R10-avgros);	// Van Wagner's 2-equation crown fire model
		}
		*/
		// joe's method for calculating cfb and crown spread rate
		/*
		double a, CFB;
		a=-log(0.1)/(0.9*(CritRos-Ro));
		do
		{    CFB=CrownFractionBurned;
		CrownROS=avgros+CFB*(R10-avgros);
		CrownFractionBurned=1.0-exp(-a*(CrownROS-Ro));
		} while(fabs(CFB-CrownFractionBurned)>1e-6);
		if(CFB<0.9)
		CrownROS=0.0;
		*/
		CrownROS = avgros + CrownFractionBurned * (R10 - avgros);	// Van Wagner's 2-equation crown fire model
		if(pFarsite->GetCrownFireCalculation()==0)
		{
			if (CrownROS < CritRos)						 // CrownROS must be > CritRoss to be ACTIVE crown fire
				CrownROS = 0.0; 				  // otherwise PASSIVE crown fire and no increase in ROS
		}
	}

	return CrownROS;
	//CrownROS=avgros*(1.0+CrownFractionBurned*(1.285*FoliarMoistureEffect-1.0));    // Van Wagner's 1-equation crown fire model
}



void Crown::CrownIntensity(double SpreadRate, double* fli)
{
	/*  	double UnitEnergy;
		 switch(fuel)   /// from Rothermel 1991
		{    case 2: UnitEnergy=760; break;
			case 8: UnitEnergy=580; break;
			case 9: UnitEnergy=1050; break;
			case 10: UnitEnergy=1325; break;
			case 12: UnitEnergy=3430; break;
		}
	*/

	//	UnitEnergy=1586.01*SFC;			 // 1586.01 BTU/FT2 per kg/m2, assum 18000 kJ/kg;
	SpreadRate /= (0.3048 * 60.0);  	   // m/m to ft/sec
	*fli = fabs(SpreadRate * (SFC + CrownLoadingBurned) * 1586.01);  // from Rothermel 1991 pp10,11
	FlameLength = (0.2 * pow(*fli, 2.0 / 3.0)) / 3.281;		// fl in feet, from Thomas in Rothermel 1991 converted to m
	*fli *= 3.4613;					   // kW/m from btu/ft/s
}


void Crown::CrownBurn(double avgros, double fli, double AccelConstant)
{
	SFC = fli / (300.0 * avgros);   	 // calculate surface fuel consumption from Van Wagner 1990, kg/m2;
	CritRos = 3.0 / CrownBulkDensity;

	//-------------
	double a;

	Ro = Io / (300.0 * SFC);
	if (CritRos > Ro)
	{
		a = -log(0.1) / (0.9 * (CritRos - Ro));
		CrownFractionBurned = 1.0 - exp(-a * (fli - Io) * (avgros / fli));
		if (CrownFractionBurned <= 0.0)	// debugging
			CrownFractionBurned = 0.01;
		else if (CrownFractionBurned > 1.0)
			CrownFractionBurned = 1.0;
	}
	else
		CrownFractionBurned = 1.0;

	//CritRos=Ro+0.9*(CritRos-Ro);
	//-------------*/

	/*/--------------
	double a;
	double IgnitionHeight, HeightDiff;
	double NewBaseHeight, CrownLoad;
	NewBaseHeight=HLCB;
	do
	{    IgnitionHeight=pow(fli, 2.0/3.0)/(4.60+0.256*FoliarMC);
		 HeightDiff=IgnitionHeight-NewBaseHeight;
		 CrownLoad=HeightDiff*CrownBulkDensity;
		 fli+=CrownLoad*333.0*avgros;  // kW/m
		 NewBaseHeight=IgnitionHeight;
	} while(HeightDiff>0.01);
	a=-log(0.1)/(0.7*(TreeHeight-HLCB));
	CrownFractionBurned=1.0-exp(-a*(IgnitionHeight-HLCB));
	//---------------*/

	//CrownFractionBurned=1.0-exp(-.238*((fli-Io)/(300.0*SFC)));	 // Van Wagner 1990,
	CrownLoadingBurned = CrownFractionBurned * CrownBulkDensity * fabs(TreeHeight -
																	HLCB);   // calc loading (kg/m2)
	A = AccelConstant -
		18.8 * pow2(CrownFractionBurned) * sqrt(CrownFractionBurned) * exp(-8.0 * CrownFractionBurned);		// CFFBPS 1992, equation 72, page 41, subst .3 for .115
	//A=AccelConstant-18.8*pow(CrownFractionBurned,2.5)*exp(-8.0*CrownFractionBurned);		// CFFBPS 1992, equation 72, page 41, subst .3 for .115
}

double Crown::CrownBurn2(double avgros, double fli, double AccelConstant, void *vt)
{
	// uses crown fire calculations from Scott and Reinhardt 2001
     double R10, Rratio, Eratio, Ractual, Rsa;//a,
     double cw1, cw2, cs, cpw1, cpw2, X, inc, CritWind;
     double sw1, sw2, ss, spw1, spw2;//, srateo;
     BurnThread *bt;

     bt=(BurnThread *) vt;


	SFC=fli/(300.0*avgros);        // calculate surface fuel consumption from Van Wagner 1990, kg/m2;
     CritRos=3.0/CrownBulkDensity;

     // get surface fire behavior parameters
     bt->GetEquationTerms(&sw1, &ss, &spw1, &spw2);
     Eratio=avgros/(bt->fros*(1.0+sw1+ss));		// ratio of actual to heading spread rate of surface fire

     // now calculate crown fire behavior
     Ro=Io/(300.0*SFC);            // class variable
     R10=bt->spreadrate(bt->ld.slope, bt->m_twindspd*0.4, 10);
	 if(R10 <= 0.0)
	 {
		 CrownFractionBurned = CrownLoadingBurned = 0.0;
		 A = AccelConstant;
		 return avgros;
	 }
     if(Ro>CritRos)
     	CrownFractionBurned=1.0;
     else
     {	bt->GetEquationTerms(&cw1, &cs, &cpw1, &cpw2);
     	Ractual=3.34*R10*(1.0+cs+cw1);		// with wind and slope
	     Rratio=(Ractual*Eratio)/CritRos;
          if(Rratio<1.0)
		{	cw2=cw1;
	          inc=1.0;
     		do
	     	{    X=Rratio*(1.0+cs+cw2)-(1.0+cs);		// iterate to solve for phiW
          		if(X>cw1)
	               {	cw2-=inc;
     	          	inc/=2.0;
          	     }
               	cw2+=inc;
	     	}while(fabs(cw1-X)>1e-6);

	    		CritWind=pow(cw2/cpw2, 1.0/cpw1);
     	     CritWind/=0.4;
			 if(bt->m_twindspd > 0)
          		sw2=pow(CritWind*bt->m_windspd/bt->m_twindspd, spw1)*spw2;
			 else
				 sw2 = 0;
	          Rsa=bt->fros*(1.0+ss+sw2);
               if(Rsa>CritRos)
               	Rsa=CritRos;
     	     CrownFractionBurned=(avgros-Ro)/(Rsa-Ro);	// maxcrownfraction burned in heading fire
               if(CrownFractionBurned>1.0)
               	CrownFractionBurned=1.0;
          }
          else
          	CrownFractionBurned=1.0;
     }

	CrownLoadingBurned=CrownFractionBurned*CrownBulkDensity*fabs(TreeHeight-HLCB);   // calc loading (kg/m2)
     A=AccelConstant-18.8*pow2(CrownFractionBurned)*sqrt(CrownFractionBurned)*exp(-8.0*CrownFractionBurned);		// CFFBPS 1992, equation 72, page 41, subst .3 for .115

	return R10;
}
