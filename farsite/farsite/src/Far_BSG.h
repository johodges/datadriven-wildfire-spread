
//-------------------------------------------------------------
//                     Burn Spot Grid
// Class is used to keep track of burn cells and landing embers
//
// Each cell in the grid is incremented everytime a fire burns it
// of a spot lands on it, 0 means it hasn't burned and no spot has
// landed it on it.
// once the cell values is 1 or above there is no need to create
// another ember/spot on that cell.
// The cell uses a counter rather than a bool mostly for test purposes
// so that you can see how many times it is being hit.
class BSG {

public:

typedef  int  d_GridType;
   d_GridType **Grid;
   long  l_loeast;
   long  l_hieast;
   long  l_lonorth;
   long  l_hinorth;
   long  l_res;
   long  lN_East, lN_North;   // # of rows & cols in the Grid

   //FILE *fh;
   long  iN_Hits;   /* Just used to count multiple spot hits for testing */

//.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
   int Init (double loeast, double hieast, double lonorth, double hinorht,
             double res);//, char cr_FN[]);

   int GetRowCol (double d_x, double d_y, long *Row, long *Col);
   void Set (double d_x, double d_y, char cr[]);
   int Get (double d_x, double d_y);
   int TestReport ();
   int Display (char cr_FN[]);
   BSG();
   ~BSG();
   long  _Pos (long l);

};

