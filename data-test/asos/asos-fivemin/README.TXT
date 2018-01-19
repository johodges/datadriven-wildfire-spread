This directory contains the ASOS five minute (temporal resolution)
data organized in subdirectories named by data set name and 4 digit year. 

DSI 6401 files: Each data file contains data for one station-month.
The files names begin with 64010 (for data set name), followed 
by the 4 character call letter identifier (e.g. KNYC = New York Central Park, NY), 
the 4 digit year and two digit month. The file extensions are ".dat".
The format documentation for these files are available in this directory in file
td6401-1.txt and.td6401b.txt.

Anonymous ftp instructions:

   1. Enter: open ftp.ncdc.noaa.gov
   2. Login is: ftp or anonymous
   3. Password is: your email address
   4. You are now logged onto a workstation. Enter help if you'd like a list of available commands.
   5. To move to the correct subdirectory, enter: cd /pub/data/asos-fivemin/.
   6. To get a copy of the file descriptions, enter: get README.TXT destination (destination is your output location and name)...e.g.--get README.TXT c:README.TXT - copies to hard drive c: Note that file names are in all CAPITAL letters/numbers.
   7. To logoff the system when finished, enter: bye

For inquiries please contact:
Blake Lasher at blake.l.lasher@ncdc.noaa.gov (828-271-4460)
or
Dan Dellinger at dan.dellinger@ncdc.noaa.gov (828-271-4290)
