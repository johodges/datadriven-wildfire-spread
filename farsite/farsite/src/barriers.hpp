#ifndef BARRIERS_HPP_
#define BARRIERS_HPP_

//#include "boost/filesystem/path.hpp"
#include <string>
#include <iostream>
#include "fsxwignt.h"

class Farsite5;
	
/*class BarrierFile 
	{
		char fname[256];
		public :
		BarrierFile();// {}
		//inline BarrierFile(char *name) {fname=name;}
		void setFileName(char *name);
		//inline boost::filesystem::path &getFileName() { 
		//	return fname ; 
		//}
		//inline const boost::filesystem::path &getFileName() const { 
		//	return fname ; 
		//}
        //inline std::string getNativeFileName() const { 
        //    return fname.native_file_string() ; 
        //}
		
		// a do nothing function to be overridden
		virtual void load(IgnitionFile &) const = 0 ; 
		
		private: 
		//boost::filesystem::path fname ;
	} ; */

	class ShapefileBarrier// : public BarrierFile  
	{
		char fname[256];
		Farsite5 *pFarsite;
		public : 
		ShapefileBarrier(Farsite5 *_pFarsite);
		void setFileName(char *name);
		void load();//IgnitionFile &) const ; 
	} ; 
	
	/*class ArcBarrier : public BarrierFile  {
		public : 
		inline ArcBarrier() {} 
		inline ArcBarrier(const boost::filesystem::path name) :
			BarrierFile(name) {} 
		void load(IgnitionFile &) const ; 
		
	} ; */
	
#endif /*BARRIERS_HPP_*/
