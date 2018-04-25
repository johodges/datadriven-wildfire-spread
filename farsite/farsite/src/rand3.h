#include <math.h>
#include <sys/timeb.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

#ifndef RANDOM_GENERATOR
#define RANDOM_GENERATOR

#define NTAB 32 


class Random
{
	timeb t;
     long idum;
     long idum2;//=123456789;
     long iy;//=0;
     long iv[NTAB];
     
public:
	Random();
	~Random();
	void SetFixedSeed(long Number);
	void ResetRandomSeed();
	long GetCurrentSeed();
	double rand2(long *idum);
     double rand3();
	 //FILE *fixedOut;
};

#endif // RANDOM_GENERATOR
