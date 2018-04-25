/********************************************************************************
* BSG  - Burn Spot Grid
*        Used to track burned and ember spot locations
*
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include "Far_BSG.h"


/*************************************************************************
* Name: Init
* Desc: Allocate the Burn Spot grid and init it to 0
* Note-1  this loop will make sure we get the
*         number of cells we need, allowing for the 0,0 position
*         and partial resolution counts.
* Note-2 If we should get fh == NULL, error opening file, the other functions
*         check for NULL and won't try to write to the Test file.
*   In: low/high east/north
*       res...cell resolution to set BSG
*       cr_FN....use ""
*                if a file path/name is send in a Test file will be opened
*                and outputed to - see the BSG.Set() function
*                file is closed in the ~BSG() destructor
*************************************************************************/
 int BSG::Init (double loeast, double hieast, double lonorth, double hinorth,
                double res)//, char cr_FN[])
{
long e,n,r;
long l;

 // Save these in the class object
   l_loeast  = (long) loeast;
   l_hieast  = (long) hieast;
   l_lonorth = (long) lonorth;
   l_hinorth = (long) hinorth;
   l_res     = (long) res;

// Determine number of cells across east - west - see Note-1 above
   lN_East = 0;
   for ( l = l_loeast; l <= l_hieast; l = l + l_res)
      lN_East++;

// Determine number of cell going north south - see Note-1 above
   lN_North = 0;
   for ( l = l_lonorth; l <= l_hinorth; l = l + l_res)
      lN_North++;

// Allocate memory for 2 dimension grid
   Grid = new d_GridType *[lN_North];
   for ( r  = 0; r < lN_North; r++ )
     Grid [r] = new d_GridType [lN_East];

   for ( n = 0; n < lN_North; n++ )
   {
    for ( e = 0; e < lN_East; e++ )
	{
      Grid[n][e] = 0;
//  printf ("[%d] [%d] = %d \n", e, n, Grid[e][n]);
     }
   }

  iN_Hits = 0;
  return 1;
}

/*************************************************************************
* Name: Set
* Desc: Set a point in the grid
*       Mark the point by incrementing it,
*       Set Get() function below
*
*   In: cr......used to help test, see callers they'll send in an
*               "F" - flaming/burn is doing the set
*               "S" - spotting is doing the set
*************************************************************************/
void BSG::Set (double d_x, double d_y, char cr[])
{
long Row, Col;

	if(!Grid)
		return;
   GetRowCol (d_x, d_y, &Row, &Col);
   if(Grid[Row][Col] == 0 || cr[0] == 'S')
		Grid[Row][Col]++;

 // if ( fh != NULL ) {
 //   if ( !strcmpi (cr,"S")) {  // Spotting is setting the cell as a hit
 //   fprintf (fh, "%s    %f  %f %d \n", cr, d_x, d_y, Grid[Row][Col]);
  //}}

}

/*************************************************************************
* Name: Get
* Desc: see if the point on the grid has been set,
* Note-1: This basically counts how many total spot embers won't get created
*         because when the cell count is greater than 0 it means no
*         ember will get created.
* Note-2: Test file, tells how many times the cell was 'set' because a fire
*          was burning it, and/or a spot was going to land on it.
*  Ret: 0 = point has not been previously set
*       N = number of times the cell on grid has been set,
*
*************************************************************************/
int  BSG::Get (double d_x, double d_y)
{
long Row, Col;
	if(!Grid)
		return 0;

   GetRowCol (d_x, d_y, &Row, &Col);

 // if ( fh != NULL )   // See Note-2 above
 //    fprintf (fh, "x: %f   y: %f =  %d\n", d_x, d_y, (int) Grid[Row][Col]);

  if ( Grid[Row][Col] > 0 )  // See Note-1 above
  {
		Grid[Row][Col]++;
		iN_Hits++;
  }

  return (int) Grid[Row][Col];

}


/*************************************************************************
* Name: GetRowCol
* Desc: get the row and col grid indexes for the x y coordinates
*************************************************************************/
int BSG::GetRowCol (double d_x, double d_y, long *Row, long *Col)
{
//long l ;
long x,y, X, Y;//,r;

  X = (long)d_x;
  Y = (long)d_y;

  if ( X > l_hieast )
    x = lN_East - 1;
  else if ( X < l_loeast )
    x = 0;
  else {
    x = l_loeast - X;
    x = x / l_res;
    x = _Pos (x); }

  if ( Y > l_hinorth )
    y = lN_North - 1;
  else if ( Y < l_lonorth )
    y = 0;
  else {
    y = l_lonorth - Y;
    y = y / l_res;
    y = _Pos (y);  }

//printf (" x = %d, y = %d \n", x,y);


  *Row = y;//x;
  *Col = x;//y;

  return 1;

}

/*************************************************************************
* Return a positive number
**************************************************************************/
long  BSG::_Pos (long l)
{
double m;

  m = l;

  if ( l < 0 )
    m = l * -1.0;
  return m;

}


/*************************************************************************
* Classs Destructor - free up the grid
*
*************************************************************************/
BSG::BSG()
{
 // fh = NULL;
  Grid = NULL;
  iN_Hits = 0;
}

BSG::~BSG()
{
	long r;

	//  this->Display("c:\\larry\\bsg-grid.txt");

	//if ( fh != NULL ){ /* For testing */
	//	fprintf (fh, "Redundant Spot Hits: %ld \n", this->iN_Hits);
	//	fclose (fh); }

	if ( Grid == NULL )
		return;

	for ( r = 0; r < lN_North; r++ )
	{
		if ( Grid[r] != NULL )
		{
			delete[] Grid[r];
		}
	}

	if ( Grid != NULL )
		delete []  Grid;

}


// ************************************************************************
// *************************************************************************
//

/*************************************************************************
* For Testing
*
*************************************************************************/
int BSG::TestReport ()
{
long r,c, N, X, Y;
   N = 0;

   X = l_loeast - l_res ;

   for ( r = 0; r < lN_East; r++ ) {
     X += l_res;
     Y = l_lonorth - l_res ;

     for ( c = 0; c < lN_North; c++ ) {
       Y += l_res;

       if ( Grid[r][c] == 0 )
         continue;

       printf ("east-X: %ld  north-Y: %ld   Hits: %d\n", X,Y, Grid[r][c]);
       N++;

   }}
   printf ("Active Cells: %ld\n",N);
   return N;
}



/*************************************************************************
* Display - for testing
*
*************************************************************************/
int BSG::Display (char cr_FN[])
{
long e,n, N;
FILE *fh;

   fh = fopen (cr_FN,"w");


   N = 0;
// for ( r = 0; r < lN_East; r++ ) {

  for ( n = lN_North - 1; n >= 0;  n-- ) {
   for ( e = 0; e < lN_East; e++ ) {
    if ( Grid[e][n] == 0 )
       fprintf (fh,".");
    else {
      fprintf (fh,"X");
//     N++; }
     }
   }
     fprintf (fh, "\n");
  }

 fclose (fh);
   return N;
}

