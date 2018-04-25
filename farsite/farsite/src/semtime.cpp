//------------------------------------------------------------------------------
/*! \file semtime.cpp
    \brief SemTime class implementation and definition.
    \author Copyright (C) 2005 by Collin D. Bevins.
    \license This file is released under the GNU General Public License (GPL).
    \version 1.0 
 */

// Custom include files
// #include "stdafx.h"

/* Get rid of 'deprecate' compiler warnings.. */
#define _CRT_SECURE_NO_WARNINGS

#include "semtime.h"



//------------------------------------------------------------------------------
/*! \brief Default class constructor.

    Creates a SemTime instance with the current system date and time.
 */

SemTime::SemTime( void )
{
    setNow();
    return;
}

//------------------------------------------------------------------------------
/*! \brief Custom class constructor.

    Creates a SemTime instance with the specified Gregorian date and time.
  
    \param[in] year         Year [-4712 .. 9999].
    \param[in] month        Month of the year [1..12].
    \param[in] day          Day of the month [1..31].
    \param[in] hour         Hour of the day [0..23].
    \param[in] minute       Minute of the hour [0..59].
    \param[in] second       Second of the minute [0..59].
    \param[in] millisecond  Millisecond of the second [0..999].
 */

SemTime::SemTime( int year, int month, int day, int hour, int minute,
        int second, int millisecond ) :
    m_year( year ),
    m_month( month ),
    m_day( day ),
    m_hour( hour ),
    m_minute( minute ),
    m_second( second ),
    m_ms( millisecond )
{
    return;
}

//------------------------------------------------------------------------------
/*! \brief Custom class constructor.

    Creates a SemTime with the specified Julian date and time.
  
    \param[in] julian Julian decimal date.
 */

SemTime::SemTime( double julian ) :
    m_year( 2000 ),
    m_month( 1 ),
    m_day( 1 ),
    m_hour( 0 ),
    m_minute( 0 ),
    m_second( 0 ),
    m_ms( 0 )
{
    setJulian( julian );
    return;
}

//------------------------------------------------------------------------------
/*! \brief Virtual class destructor.
 */

SemTime::~SemTime( void )
{
    return;
}

//------------------------------------------------------------------------------
/*! \brief Copy constructor.
  
    \param[in] rhs Reference to the SemTime from which to copy.
 */

SemTime::SemTime( const SemTime& rhs )
{
//    //QMutexLocker locker( &mutex );
    m_year   = rhs.m_year;
    m_month  = rhs.m_month;
    m_day    = rhs.m_day;
    m_hour   = rhs.m_hour;
    m_minute = rhs.m_minute;
    m_second = rhs.m_second;
    m_ms     = rhs.m_ms;
    return;
}

//------------------------------------------------------------------------------
/*! \brief Assignment operator.
  
    \param[in] rhs Reference to the SemTime instance from which to copy.
 */

const SemTime& SemTime::operator=( const SemTime& rhs )
{
    if ( this != &rhs )
    {
       // //QMutexLocker locker( &mutex );
        m_year   = rhs.m_year;
        m_month  = rhs.m_month;
        m_day    = rhs.m_day;
        m_hour   = rhs.m_hour;
        m_minute = rhs.m_minute;
        m_second = rhs.m_second;
        m_ms     = rhs.m_ms;
    }
    return( *this );
}

//------------------------------------------------------------------------------
/*! \brief Equality operator==().
  
    \param[in] rhs Reference to the right-hand-side instance.
 */

bool SemTime::operator==( const SemTime& rhs ) const
{
//    //QMutexLocker locker( &mutex );
    return( m_year   == rhs.m_year
         && m_month  == rhs.m_month
         && m_day    == rhs.m_day
         && m_hour   == rhs.m_hour
         && m_minute == rhs.m_minute
         && m_second == rhs.m_second
         && m_ms     == rhs.m_ms
    );
}

//------------------------------------------------------------------------------
/*! \brief Inequality operator!=().
  
    \param[in] rhs Reference to the right-hand-side instance.
 */

bool SemTime::operator!=( const SemTime& rhs ) const
{
    return( ! ( *this == rhs ) );
}

//------------------------------------------------------------------------------
/*! \brief Less than operator<().
  
    \param[in] rhs Reference to the right-hand-side instance.
 */

bool SemTime::operator<( const SemTime& rhs ) const
{
  //  //QMutexLocker locker( &mutex );
    if ( m_year   != rhs.m_year )   return( m_year   < rhs.m_year );
    if ( m_month  != rhs.m_month )  return( m_month  < rhs.m_month );
    if ( m_day    != rhs.m_day )    return( m_day    < rhs.m_day );
    if ( m_hour   != rhs.m_hour )   return( m_hour   < rhs.m_hour );
    if ( m_minute != rhs.m_minute ) return( m_minute < rhs.m_minute );
    if ( m_second != rhs.m_second ) return( m_second < rhs.m_second );
    return( m_ms  < rhs.m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Less than or equal to operator<=().
  
    \param[in] rhs Reference to the right-hand-side instance.
 */

bool SemTime::operator<=( const SemTime& rhs ) const
{
    //QMutexLocker locker( &mutex );
    if ( m_year   != rhs.m_year )   return( m_year   < rhs.m_year );
    if ( m_month  != rhs.m_month )  return( m_month  < rhs.m_month );
    if ( m_day    != rhs.m_day )    return( m_day    < rhs.m_day );
    if ( m_hour   != rhs.m_hour )   return( m_hour   < rhs.m_hour );
    if ( m_minute != rhs.m_minute ) return( m_minute < rhs.m_minute );
    if ( m_second != rhs.m_second ) return( m_second < rhs.m_second );
    return( m_ms  <= rhs.m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Greater than operator>().
  
    \param[in] rhs Reference to the right-hand-side instance.
 */

bool SemTime::operator>( const SemTime& rhs ) const
{
    //QMutexLocker locker( &mutex );
    if ( m_year   != rhs.m_year )   return( m_year   > rhs.m_year );
    if ( m_month  != rhs.m_month )  return( m_month  > rhs.m_month );
    if ( m_day    != rhs.m_day )    return( m_day    > rhs.m_day );
    if ( m_hour   != rhs.m_hour )   return( m_hour   > rhs.m_hour );
    if ( m_minute != rhs.m_minute ) return( m_minute > rhs.m_minute );
    if ( m_second != rhs.m_second ) return( m_second > rhs.m_second );
    return( m_ms  > rhs.m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Greater than or equal to operator>=().
  
    \param[in] rhs Reference to the right-hand-side instance.
 */

bool SemTime::operator>=( const SemTime& rhs ) const
{
    //QMutexLocker locker( &mutex );
    if ( m_year   != rhs.m_year )   return( m_year   > rhs.m_year );
    if ( m_month  != rhs.m_month )  return( m_month  > rhs.m_month );
    if ( m_day    != rhs.m_day )    return( m_day    > rhs.m_day );
    if ( m_hour   != rhs.m_hour )   return( m_hour   > rhs.m_hour );
    if ( m_minute != rhs.m_minute ) return( m_minute > rhs.m_minute );
    if ( m_second != rhs.m_second ) return( m_second > rhs.m_second );
    return( m_ms  >= rhs.m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Adds a specified number of days to the SemTime.
  
    \param[in] days Number of days to add; may be negative or positive.
 */

void SemTime::addDays( double days )
{
    setJulian( julianDate() + days );
    return;
}

//------------------------------------------------------------------------------
/*! \brief Adds a specified number of hours to the SemTime.
  
    \param[in] hours Number of hours to add; may be negative or positive.
 */

void SemTime::addHours( double hours )
{
    setJulian( julianDate() + ( hours / 24. ) );
    return;
}

//------------------------------------------------------------------------------
/*! \brief Adds a specified number of minutes to the SemTime.
  
    \param[in] minutes Number of minutes to add; may be negative or positive.
 */

void SemTime::addMinutes( double minutes )
{
    setJulian( julianDate() + ( minutes / ( 24. * 60. ) ) );
    return;
}

//------------------------------------------------------------------------------
/*! \brief Access to the current day of the month.
  
    \return Current SemTime day of the month [1..31].
 */

long SemTime::dayOfMonth( void ) const
{
    return( m_day );
}
    
//------------------------------------------------------------------------------
/*! \brief Access to the current decimal day fraction of the month.
  
    The day of the month fraction for 1952/09/04 12:34:56.789 is 4.524268391.
  
    \return Current SemTime day of the month as a decimal fraction.
 */

double SemTime::dayOfMonthFraction( void ) const
{
    double f = (double) m_day
             + ( (double) m_hour / 24.0 )
             + ( (double) m_minute / 1440.0 )
             + ( (double) m_second / 86400.0 )
             + ( (double) m_ms / 86400000.0 );
    return( f );
}

//------------------------------------------------------------------------------
/*! \brief Calculates the day-of-the-week from current SemTime field values.
  
    \cite meeus1982
  
    \retval 0 Sunday
    \retval 1 Monday 
    \retval 2 Tuesday 
    \retval 3 Wednesday 
    \retval 4 Thursday 
    \retval 5 Friday 
    \retval 6 Saturday
 */

long SemTime::dayOfWeek( void ) const
{
    // Julian decimal date
    double jd = julianDate();

    // Julian day number
    long jdn = (long) ( jd + 0.5 );

    // Day of the week
    int dow = ( jdn + 1L) % 7L;
    return( dow );
}

/*----------------------------------------------------------------------------*/
/*! \brief Calculates the day-of-the-year from current SemTime field values.
  
    \cite meeus1982
  
    \return Day of the year [1..366].
 */

long SemTime::dayOfYear( void ) const
{
    long doy = (int) ( 275. * (double) m_month / 9. )
             - (int) ( ( (double) m_month + 9. ) / 12. )
             + m_day - 30;
    if ( ! isLeap() )
    {
        doy -= (int) ( ( (double) m_month + 9. ) / 12. );
    }
    return( doy );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current hour of the day.
  
    \return Current SemTime hour of the day [0..23].
 */

long SemTime::hourOfDay( void ) const
{
    return( m_hour );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current hour fraction of the day.
  
    The hour fraction for 1952/09/04 12:34:56.789 is 12.582441389.
  
    \return Current SemTime hour of the day as a decimal fraction.
 */

double SemTime::hourOfDayFraction( void ) const
{
    double f = (double) m_hour
             + ( (double) m_minute / 60. )
             + ( (double) m_second / 3600. )
             + ( (double) m_ms / 3600000. );
    return( f );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current hour of the month.
  
    \return Current SemTime hour of the month [0..743].
 */

long SemTime::hourOfMonth( void ) const
{
    return( 24 * ( m_day - 1 ) + m_hour );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current hour of the year.
  
    \return Current SemTime hour of the year [0..8784].
 */

long SemTime::hourOfYear( void ) const
{
    return( 24 * ( dayOfYear() - 1 ) + m_hour );
}

//------------------------------------------------------------------------------
/*! \brief Determines if the current SemTime year is a leap year.
  
    \retval TRUE if the current SemTime \a m_year is a leap year.
    \retval FALSE if the current SemTime \a m_year is a <b>not</b>leap year.
 */

bool SemTime::isLeap( void ) const
{
    // If its not divisible by 4, its not a leap year.
    if ( m_year % 4 != 0 )
    {
        return( false );
    }
    // All years divisible by 4 prior to 1582 were leap years.
    else if ( m_year < 1582 )
    {
        return( true );
    }
    // If divisible by 4, but not by 100, its a leap year.
    else if ( m_year % 100 != 0 )
    {
        return( true );
    }
    // If divisible by 100, but not by 400, its not a leap year.
    else if ( m_year % 400 != 0 )
    {
        return( false );
    }
    // If divisible by 400, its a leap year.
    else
    {
        return( true );
    }
}

//------------------------------------------------------------------------------
/*! \brief Determines the decimal Julian date from current SemTime field values.
  
    Computes the Julian decimal day number from the current Gregorian date
    and time attributes stored in the SemTime instance.
    For astronomical purposes, The Gregorian calendar reform occurred
    on 15 Oct. 1582.  This is 05 Oct 1582 by the Julian calendar.
  
    The Julian date for 1952/09/04 12:34:56.789 is 2434260.02426839.
  
    Each millisecond is 1.2E-8 days (0.000000012), or 9 decimal places precision.
    With 6 decimal places, precision is 1/1000000 of a day, or 0.0864 second.
  
    \cite meeus1982
  
    \return The decimal Julian Day number.
 */

double SemTime::julianDate( void ) const
{
    // Decimal day fraction
    double frac = ( (double) m_hour / 24.0 )
                + ( (double) m_minute / 1440.0 )
                + ( (double) m_second / 86400.0 )
                + ( (double) m_ms / 86400000.0 );

    // Convert date to format YYYY.MMDDdd
    double gyr  = (double) m_year
                + ( 0.01 * (double) m_month)
                + ( 0.0001 * (double) m_day)
                + ( 0.0001 * frac ) + 1.0e-9;

    // Conversion factors
    long iy0 = m_year;
    long im0 = m_month;
    if ( m_month <= 2 )
    {
        iy0 = m_year - 1L;
        im0 = m_month + 12;
    }

    // Adjust for negative years
    double adjust = ( m_year <= 0L ) ? 0.75 : 0.0;

    // Julian day number
    long jdn = (long) ( (365.25 * (double) iy0 ) - adjust )
             + (long) ( 30.6001 * (double) (im0 + 1L) )
             + (long) m_day + 1720994L;

    // Adjust for the Gregorian correction (on or after 15 Oct 1582)
    if ( gyr >= 1582.1015 )
    {
        long ia = iy0 / 100L;
        long ib = 2L - ia + ( ia >> 2 );
        jdn += ib;
    }

    // Decimal Julian date
    double djd = (double) jdn + frac + 0.5;
    return( djd );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current whole Julian day number.
  
    \return Current whole Julian day number.
 */

long SemTime::julianDay( void ) const
{
    return( (long) julianDate() );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current millisecond of the day.
  
    \return Current SemTime millisecond of the day [0 .. 86,399,999].
 */

long SemTime::millisecondOfDay( void ) const
{
    return( 1000 * secondOfDay() + m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current millisecond of the hour.
  
    \return Current SemTime millisecond of the hour [0 .. 3,599,999].
 */

long SemTime::millisecondOfHour( void ) const
{
    return( 1000 * secondOfHour() + m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current millisecond of the minute.
  
    \return Current SemTime millisecond of the hour [0 .. 59,999].
 */

long SemTime::millisecondOfMinute( void ) const
{
    return( 1000 * m_second + m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current millisecond of the second
  
    \return Current SemTime millisecond of the hour [0 .. 999].
 */

long SemTime::millisecondOfSecond( void ) const
{
    return( m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current millisecond of the month.
  
    \return Current SemTime millisecond of the month [0 .. 2,678,399,999].
 */

unsigned long SemTime::millisecondOfMonth( void ) const
{
    return( 1000 * secondOfMonth() + m_ms );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current minute of the day.
  
    \return Current SemTime minute of the day [0..1339].
 */

long SemTime::minuteOfDay( void ) const
{
    return( 60 * m_hour + m_minute );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current minute of the hour.
  
    \return Current SemTime minute of the hour [0..59].
 */

long SemTime::minuteOfHour( void ) const
{
    return( m_minute );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current minute fraction of the hour.
  
    The minute of the hour fraction for 1952/09/04 12:34:56.789 is 34.946483333.
  
    \return Current SemTime minute of the hour as a decimal fraction.
 */

double SemTime::minuteOfHourFraction( void ) const
{
    double f = (double) m_minute
             + ( (double) m_second / 60. )
             + ( (double) m_ms / 60000. );
    return( f );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current minute of the month.
  
    \return Current SemTime minute of the month [0..44639].
 */

long SemTime::minuteOfMonth( void ) const
{
    return( 60 * hourOfMonth() + m_minute );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current minute of the year.
  
    \return Current SemTime minute of the year.
 */

long SemTime::minuteOfYear( void ) const
{
    return( 60 * hourOfYear() + m_minute );
}

//------------------------------------------------------------------------------
/*! \brief Determines the modified Julian date.
  
    The modified Julian Date begins at 1858 Nov 17 00:00:00,
    while the Julian date begins on -4712 Jan 1 12:00:00.
  
    \return The modified Julian date for the current date and time fields.
 */

double SemTime::modifiedJulianDate( void ) const
{
    return( julianDate() - 2400000.5 );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current month of the year.
  
    \return Current SemTime month of the year [1..12].
 */

long SemTime::monthOfYear( void ) const
{
    return( m_month );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current second of the day.
  
    \return Current SemTime second of the day [0..86399].
 */

long SemTime::secondOfDay( void ) const
{
    return( 60 * minuteOfDay() + m_second );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current second of the hour.
  
    \return Current SemTime second of the hour [0..3599].
 */

long SemTime::secondOfHour( void ) const
{
    return( 60 * m_minute + m_second );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current second of the minute.
  
    \return Current SemTime second of the hour [0..59].
 */

long SemTime::secondOfMinute( void ) const
{
    return( m_second );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current seconds fraction of the minute.
  
    The second of the minute fraction for 1952/09/04 12:34:56.789 is 56.789.
  
    \return Current SemTime seconds of the minute as a decimal fraction.
 */

double SemTime::secondOfMinuteFraction( void ) const
{
    double f = (double) m_second + ( (double) m_ms / 1000. );
    return( f );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current second of the month.
  
    \return Current SemTime second of the month [0..2,678,399].
 */

long SemTime::secondOfMonth( void ) const
{
    return( 60 * minuteOfMonth() + m_second );
}

//------------------------------------------------------------------------------
/*! \brief Access to the current second of the year.
  
    \return Current SemTime second of the year [0..31,622,399].
 */

long SemTime::secondOfYear( void ) const
{
    return( 60 * minuteOfYear() + m_second );
}

//------------------------------------------------------------------------------
/*! \brief Sets the SemTime fields to the passed Gregorian date and time.
  
    \param[in] year         Year [-4712 .. 9999].
    \param[in] month        Month of the year [1..12].
    \param[in] day          Day of the month [1..31].
    \param[in] hour         Hour of the day [0..23].
    \param[in] minute       Minute of the hour [0..59].
    \param[in] second       Second of the minute [0..59].
    \param[in] millisecond  Millisecond of the second [0..999].
 */

void SemTime::set( int year, int month, int day, int hour, int minute,
        int second, int millisecond )
{
    //QMutexLocker locker( &mutex );
    m_year   = year;
    m_month  = month;
    m_day    = day;
    m_hour   = hour;
    m_minute = minute;
    m_second = second;
    m_ms     = millisecond;
    return;
}

/*----------------------------------------------------------------------------*/
/*! \brief Sets the SemTime fields from the passed Julian date.
  
    For astronomical purposes, The Gregorian calendar reform occurred
    on 15 Oct. 1582.  This is 05 Oct 1582 by the julian calendar.
  
    \param[in] julian Julian date (0 == 01 Jan -4712 12 HR UT ).
  
    \cite meeus1982
 */

void SemTime::setJulian( double julian )
{
    //QMutexLocker locker( &mutex );
    // Julian day number
    long jd = (long) (julian + 0.5);
    
    // Day fraction
    double frac = julian + 0.5 - (double) jd + 1.0e-10;

    // Constants
    long ka = (long) jd;
    if ( jd >= 2299161L )
    {
        long ialp = (long int) ( ( (double) jd - 1867216.25 ) / 36524.25 );
        ka = jd + 1L + ialp - ( ialp >> 2 );
    }
    long kb = ka + 1524L;
    long kc = (long int) ( ( (double) kb - 122.1 ) / 365.25 );
    long kd = (long int) ( (double) kc * 365.25 );
    long ke = (long int) ( (double) ( kb - kd ) / 30.6001 );

    // Day of the month
    m_day = kb - kd - ((long) ( (double) ke * 30.6001 ));

    // Month of the year
    if ( ke > 13L )
    {
        m_month = ke - 13L;
    }
    else
    {
        m_month = ke - 1L;
    }
    if ( m_month == 2 && m_day > 28 )
    {
        m_day = 29;
    }
    // Year
    if ( m_month == 2 && m_day == 29 && ke == 3L )
    {
        m_year = kc - 4716L;
    }
    else if ( m_month > 2 )
    {
        m_year = kc - 4716L;
    }
    else
    {
        m_year = kc - 4715L;
    }
    // Hour of the day
    double d_hour = frac * 24.0;
    m_hour = (int) d_hour;
    // Minute of the day
    double d_minute = ( d_hour - (double) m_hour ) * 60.0;
    m_minute = (int) d_minute;
    // Second of the minute
    double d_second = ( d_minute - (double) m_minute ) * 60.0;
    m_second = (int) d_second;
    // Millisecond of the second
    double d_ms = ( d_second - (double) m_second ) * 1000;
    m_ms = (int) d_ms;
    return;
}

//------------------------------------------------------------------------------
/*! \brief Sets the SemTime fields to the current system time.
 */

void SemTime::setNow( void )
{
    //QMutexLocker locker( &mutex );
    time_t now_time;
    time( &now_time );

    struct tm *t = localtime( &now_time );
    m_year   = t->tm_year + 1900;;
    m_month  = t->tm_mon + 1;
    m_day    = t->tm_mday;
    m_hour   = t->tm_hour;
    m_minute = t->tm_min;
    m_second = t->tm_sec;
    m_ms     = 0;
    return;
}

//------------------------------------------------------------------------------
/*! \brief Sets the SemTime time fields to the passed time arguments.
  
    \param[in] hour         Hour of the day [0..23].
    \param[in] minute       Minute of the hour [0..59].
    \param[in] second       Second of the minute [0..59].
    \param[in] millisecond  Millisecond of the second [0..999].
 */

void SemTime::setTime( int hour, int minute, int second, int millisecond )
{
    //QMutexLocker locker( &mutex );
    m_hour   = hour;
    m_minute = minute;
    m_second = second;
    m_ms     = millisecond;
    return;
}

//------------------------------------------------------------------------------
/*! \brief Access to the current SemTime year.
  
    \return Current SemTime year [-4712 .. 9999].
 */

long SemTime::year( void ) const
{
    return( m_year );
}

//------------------------------------------------------------------------------
/*! \brief SemTime stream insertion operator.
  
    The SemTime output format is "yyyy/mm/dd hh:mm:ss.sss".
  
    \param[in] output   Reference to the output stream.
    \param[in] dt       Reference to the SemTime instance.
  
    \relates SemTime
  
    \return Reference to the output stream.
 */

ostream& operator<<( ostream& output, const SemTime& dt )
{
    output << setw(4) << setfill('0') << dt.m_year   << "/"
           << setw(2) << setfill('0') << dt.m_month  << "/" 
           << setw(2) << setfill('0') << dt.m_day    << " " 
           << setw(2) << setfill('0') << dt.m_hour   << "/" 
           << setw(2) << setfill('0') << dt.m_minute << "/" 
           << setw(2) << setfill('0') << dt.m_second << "." 
           << setw(2) << setfill('0') << dt.m_ms;
    return( output );
}

//------------------------------------------------------------------------------
/*! \brief SemTime stream extraction operator.
  
    The SemTime input format must be "yyyy/mm/dd hh:mm:ss.sss".
  
    \param[in] input    Reference to the input stream.
    \param[in] dt       Reference to the SemTime instance.
  
    \relates SemTime
  
    \return Reference to the input stream.
 */

istream& operator>>( istream& input, SemTime& dt )
{
    input >> setw( 4 ) >> dt.m_year;
    input.ignore();
    input >> setw( 2 ) >> dt.m_month;
    input.ignore();
    input >> setw( 2 ) >> dt.m_day;
    input.ignore();
    input >> setw( 2 ) >> dt.m_hour;
    input.ignore();
    input >> setw( 2 ) >> dt.m_minute;
    input.ignore();
    input >> setw( 2 ) >> dt.m_second;
    input.ignore();
    input >> setw( 2 ) >> dt.m_ms;
    return( input );
}



//------------------------------------------------------------------------------
//  End of semtime.cpp
//------------------------------------------------------------------------------

