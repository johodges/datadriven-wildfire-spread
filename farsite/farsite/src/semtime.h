//------------------------------------------------------------------------------
/*! \file semtime.h
    \brief SemTime class interface and declarations.
    \author Copyright (C) 2005 by Collin D. Bevins.
    \license This file is released under the GNU General Public License (GPL).
    \version 1.0 
 */

#ifndef SEM_TIME_H_
#define SEM_TIME_H_ 1

// Standard include files
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
using std::istream;
using std::ostream;
using std::setfill;
using std::setw;

// Custom include files
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


//------------------------------------------------------------------------------
/*! \class SemTime semtime.h
    \brief Date and time class.
  
    SemTime is a fairly light-weight date and time class that supports Julian
    date conversion and arithmetic.  It was originally designed for use with
    the DeadFuelMoisture class.

    SemTime creates first class objects; i.e., support is provided for
    assignment, copy construction, equality testing, sequential i/o,
    and serialization. Access methods are provided for all attributes.
  
    SemTime instances can be created and initialized
    -- with the current system date and time via SemTime(),
    -- with a specified Gregorian date and time via SemTime( int y, int mo,
        int d, int h, int mn , int s, int ms), or
    -- with a specified Julian date via SemTime(double julian).
  
    Once created, an instance's date and time fields may be modified via
    set(), setTime(), setNow(), and setJulian().
  
    Simple date-time arithmetic is provided by the addDays(), addHours(), and
    addMinutes() methods.
  
    The julianDate() return the Julian date for the SemTime instance.
    The Julian day number definition is the astronomical Julian date, where
    Day 0.0 is -4712 January 01 12:00:00 UT. There is a year 0 between year
    -1 and year 1. Years greater than 0 thus correspond to common era
    Julian/Gregorian calendar dates. 04 October 1582 Julian (JD 2299160)
    is followed by 15 October 1582 Gregorian (JD 2299161).
  
    \cite meeus1982
 */

class SemTime
{
// Friends
public:
    friend ostream &operator<<( ostream& output, const SemTime& dt );
    friend istream &operator>>( istream& input, SemTime& dt );

// 'tors
public:
    // Constructors
    SemTime( void ) ;
    SemTime( int year, int month=1, int day=1, int hour=0, int minute=0,
        int second=0, int millisecond=0 ) ;
    SemTime( const double julianDate ) ;
    // Virtual destructor
    virtual ~SemTime( void ) ;
    // Copy constructor
    SemTime( const SemTime& rhs ) ;
    // Assignment operator
    const SemTime& operator=( const SemTime& rhs ) ;
    // Equality operator
    bool operator==( const SemTime& rhs ) const ;
    bool operator!=( const SemTime& rhs ) const ;
    bool operator<( const SemTime& rhs ) const ;
    bool operator<=( const SemTime& rhs ) const ;
    bool operator>( const SemTime& rhs ) const ;
    bool operator>=( const SemTime& rhs ) const ;

// Data access methods
public:
    const char* className( void ) const { return( "SemTime" ); }
    long    dayOfMonth( void ) const ;
    double  dayOfMonthFraction( void ) const ;
    long    dayOfWeek( void ) const ;
    long    dayOfYear( void ) const ;
    long    hourOfDay( void ) const ;
    double  hourOfDayFraction( void ) const ;
    long    hourOfMonth( void ) const ;
    long    hourOfYear( void ) const ;
    bool    isLeap( void ) const ;
    double  julianDate( void ) const ;
    long    julianDay( void ) const ;
    long    millisecondOfSecond( void ) const ;
    long    millisecondOfMinute( void ) const ;
    long    millisecondOfHour( void ) const ;
    long    millisecondOfDay( void ) const ;
    unsigned long millisecondOfMonth( void ) const ;
    long    minuteOfHour( void ) const ;
    double  minuteOfHourFraction( void ) const ;
    long    minuteOfDay( void ) const ;
    long    minuteOfMonth( void ) const ;
    long    minuteOfYear( void ) const ;
    double  modifiedJulianDate( void ) const ;
    long    monthOfYear( void ) const ;
    long    secondOfMinute( void ) const ;
    double  secondOfMinuteFraction( void ) const ;
    long    secondOfHour( void ) const ;
    long    secondOfDay( void ) const ;
    long    secondOfMonth( void ) const ;
    long    secondOfYear( void ) const ;
    long    year( void ) const ;

// Data update methods
public:
    void addDays( double days ) ;
    void addHours( double hours ) ;
    void addMinutes( double minutes ) ;
    void set( int year, int month=1, int day=1, int hour=0, int minute=0,
            int second=0, int millisecond=0 ) ;
    void setJulian( double julianDay ) ;
    void setNow( void ) ;
    void setTime( int hour=0, int minute=0, int second=0, int millisecond=0 ) ;

// Protected data
protected:
    int m_year;     //!< Year [-4712 .. 9999].  There is a Year 0, which is -1 B.C.
    int m_month;    //!< Month of the year [1..12].
    int m_day;      //!< Day of the month [1..31].
    int m_hour;     //!< Hour of the day [0..23].
    int m_minute;   //!< Minute of the hour [0..59];
    int m_second;   //!< Second of the minute [0..59];
    int m_ms;       //!< Millisecond of the second [0..999]
   // mutable QMutex mutex;
};



#endif

//------------------------------------------------------------------------------
//  End of semtime.h
//------------------------------------------------------------------------------

