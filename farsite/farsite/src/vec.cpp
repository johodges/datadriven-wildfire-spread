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
#include "vec.h"

vec operator+(vec& u, vec& v)
{
	return vec(u.x + v.x, u.y + v.y);
}

vec operator-(vec& u, vec& v)
{
	return vec(u.x - v.x, u.y - v.y);
}

vec operator*(double c, vec& v)
{
	return vec(c * v.x, c * v.y);
}

vec& operator+=(vec& u, vec& v)
{
	u.x += v.x;
	u.y += v.y;

	return u;
}

vec& operator-=(vec& u, vec& v)
{
	u.x -= v.x;
	u.y -= v.y;

	return u;
}
                                                   
vec& operator*=(vec& v, double c)
{
	v.x *= c;
	v.y *= c;

	return v;
}

int Orientation(vec& P, vec& Q, vec& R)
	//int Orientation(double x1, double y1, double x2, double y2, double x3, double y3);
{
	const double EPS = 1e-6;
	//	vec P(x1, y1);
	//	vec Q(x2, y2);		// first
	//	vec R(x3, y3);		// second
	vec A, B;
	double determinant;

	A = Q - P;
	B = R - P;
	determinant = A.x * B.y - A.y * B.x;

	return (determinant <-EPS ? -1 : determinant> EPS);
}

int Inside(vec& P, vec& A, vec& B, vec& C)
{
	// P lies in side triangle ABC
	int OrientABP = Orientation(A, B, P);

	return (Orientation(B, C, P) == OrientABP &&
		Orientation(C, A, P) == OrientABP);
}



