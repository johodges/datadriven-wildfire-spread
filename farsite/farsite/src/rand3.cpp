//------------------------------------------------------------------------------
//
//  Random Number Generator from Recipies in C, page 282
//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//#include "stdafx.h"
#include "rand3.h"

//#define NTAB 32  // included in .cpp file
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


Random::Random()
{
     idum2=123456789;
     iy=0;
     memset(iv, 0x0, 32*sizeof(long));;

	ResetRandomSeed();
     //	printf("random numbers initialized\n"); // initialize random num gen
     //rand2(&idum);
	//fixedOut = NULL;
}

Random::~Random()
{
	//if(fixedOut)
	//	fclose(fixedOut);
}

void Random::SetFixedSeed(long Number)
{
//	if(fixedOut)
//	{
//		fclose(fixedOut);
//		fixedOut = NULL;
//	}
	char name[256];
	sprintf(name, "seed_%ld_output.txt", Number);
	if(Number>0)
		Number*=-1;
	idum=Number;
	//fixedOut = fopen(name, "wt");
	//fprintf(fixedOut, "Seed: %ld\n", Number);
	//rand2(&idum);
}


void Random::ResetRandomSeed()
{
	ftime(&t);
	srand(t.time+t.millitm);
     idum=-((rand()%10000)+1);
}
	


long Random::GetCurrentSeed()
{
	return idum;
}


double Random::rand3()
{
	return rand2(&idum);
}


double Random::rand2(long *idum)
{
	long j, k;
     //static long idum2=123456789;
     //static long iy=0;
     //static long iv[NTAB];
     double temp;

     if(*idum<=0)
     {	if(-(*idum)<1)
     		*idum=1;
     	else
          	*idum=-(*idum);
		idum2=(*idum);

          for(j=NTAB+7; j>=0; j--)
          {	k=(*idum)/IQ1;
          	*idum=IA1*(*idum-k*IQ1)-k*IR1;
               if(*idum<0)
               	*idum+=IM1;
               if(j<NTAB)
               	iv[j]=*idum;

          }
          iy=iv[0];
     }
     k=(*idum)/IQ1;
     *idum=IA1*(*idum-k*IQ1)-k*IR1;
     if(*idum<0)
     	*idum+=IM1;
	k=idum2/IQ2;
     idum2=IA2*(idum2-k*IQ2)-k*IR2;
     if(idum2<0)
     	idum2+=IM2;
     j=iy/NDIV;
     if(j<0)
          j=labs(j);
     iy=iv[j]-idum2;
	iv[j]=*idum;
     if(iy<1)
     	iy+=IMM1;
     if((temp=AM*iy)>RNMX)
	 {
//		 if(fixedOut)
//			 fprintf(fixedOut, "%lf\n", RNMX);
     	return RNMX;
	 }

	if(temp<0.0)
     	temp*=-1.0;

//		 if(fixedOut)
//			 fprintf(fixedOut, "%lf\n", temp);
     return temp;
}







