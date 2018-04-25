//#include <stdexcept>
//#include <cstdlib>
//#include <iostream>
//#include "arc.hpp"
//#include "boost/filesystem/fstream.hpp"
#include "barriers.hpp"
#include "fsxw.hpp"
#include "fsxwbar.h"
//#include "fsxwshap.h"
#include "Farsite5.h"

ShapefileBarrier::ShapefileBarrier(Farsite5 *_pFarsite)
{
	pFarsite = _pFarsite;
}

void ShapefileBarrier::setFileName(char *name)
{
	if(fname)
		strcpy(fname, name);
}

	/**
	 * \brief Loads a barrier from a shapefile.
	 * 
	 * The shapefile's filename must already be set, and must contain a 
	 * line or polygon geometry type.  If the geometry is "lines", then the 
	 * barrier is converted to a polygon by buffering to the current value 
	 * of the distance resolution.  
	 * 
	 * Mark does do some resampling of the perimeter (e.g. to place points 
	 * around the perimeter in a regular fashion) somewhere in the code,
	 * but I do not believe that it is performed within this function.
	 */
	void ShapefileBarrier::load()
	{//IgnitionFile &ign){ 
		ShapeFileUtils shape(pFarsite) ; 
		VectorBarrier vectorbarrier(pFarsite); 
		long ShapeType;

        long NewFires = pFarsite->GetNewFires() ; 
		// load barrier
	    ShapeType=shape.ImportShapeData(fname, 3);	
	    if(ShapeType==-1) { 
	    	//throw std::logic_error(
	    	//     "Can't Import Point or MultiPoint Barriers") ; 
	    } else //if(ShapeType==1)     // must buffer lines to polygons using dist res
	    {   
	    	double xpt, ypt;
            long TempFire;
           	for(long count=NewFires; count<pFarsite->GetNewFires(); count++)
            {    
            	vectorbarrier.AllocBarrier(pFarsite->GetNumPoints(count));
                for(long count1=0; count1<pFarsite->GetNumPoints(count); count1++) {  	
                	xpt=pFarsite->GetPerimeter1Value(count, count1, XCOORD);
					ypt=pFarsite->GetPerimeter1Value(count, count1, YCOORD);
					vectorbarrier.SetBarrierVertex(count1, xpt, ypt);
				}
                pFarsite->FreePerimeter1(count);
     			vectorbarrier.BufferBarrier(1.2);
     			vectorbarrier.TransferBarrier(count);
     			
     			// set the bounding box of the ignition
                TempFire=pFarsite->GetNewFires();
                pFarsite->SetNewFires(count+1);
                pFarsite->Ignition.BoundingBox(count);
                pFarsite->SetNewFires(TempFire);
            }
	    }		
	}
	
	/*void ArcBarrier::load(IgnitionFile &ign) const { 
		
		// set the file name
		ArcVectorFile arcFile(getFileName()) ; 
		
		// load data from the file.
		arcFile.load() ; 
		
		// transfer data to the internal memory arrays.
		ArcVectorFile::SequenceList fileContents = arcFile.getContents() ; 
		ArcVectorFile::SequenceList::const_iterator seq ; 
		int vertex = 0; 
		
		// loop over all the point sequences
		for (seq=fileContents.begin(); seq!=fileContents.end(); ++seq) {
			ArcVectorFile::PointSequence::const_iterator points ; 
			VectorBarrier vectorbarrier ; 
			vectorbarrier.AllocBarrier(seq->size()) ;
			vertex = 0 ; 
			
			// loop over all the points
			for (points=seq->begin() ; points!=seq->end(); ++points) {
				ArcVectorFile::Point point = *points ; 
				point.first = ConvertUtmToEastingOffset(point.first) ; 
				point.second = ConvertUtmToNorthingOffset(point.second) ; 
				vectorbarrier.SetBarrierVertex(vertex, point.first, point.second);
				vertex ++ ; 
			}
			
			// transfer this point sequence to FARSITE data arrays
			vectorbarrier.BufferBarrier(1.2) ; 
			vectorbarrier.TransferBarrier(GetNewFires()) ; 
			IncNewFires(1) ; 
		}
	}	*/
	
