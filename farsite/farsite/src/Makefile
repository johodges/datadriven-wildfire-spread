BASE=..


CC=g++
INC=
CFLAGS=-c -g -Wall -DUNIX -Wno-deprecated

SOURCES=barriers.cpp burnupw.cpp cdtlib.cpp \
	crossthread.cpp dbfopen.cpp Far_BSG.cpp Far_Cond.cpp Far_FPC.cpp \
	FARSITE.cpp fsairatk.cpp fsxpfront.cpp fsxwaccl.cpp \
	fsxwatm.cpp fsxwattk.cpp fsxwbar.cpp fsxwburn4.cpp fsxwcrwn.cpp \
	fsxwenvt2.cpp fsxwfms2.cpp fsxwfotp.cpp fsxwignt.cpp fsxwmech.cpp \
	fsxwrast.cpp fsxwshap.cpp fsxwspot.cpp fsxwutil.cpp fsxwvect.cpp \
	gridthem.cpp newclip.cpp newfms.cpp nn_cls.cpp PerimeterData.cpp \
	polygon.cpp shpopen.cpp vec.cpp icf_input.cpp icf_chk.cpp \
	deadfuelmoisture.cpp FMC_CFMC.cpp Farsite5.cpp\
	FMC_CI.cpp FMC_CI2.cpp FMC_CI3.cpp FMC_CI4.cpp FMC_FE2.cpp \
	FMC_FE22.cpp FMC_FE3.cpp semtime.cpp rand3.cpp TestFARSITE.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=TestFARSITE


all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@
#	$(CC) TestFARSITE.o $(LD_EXE) -o $@

.cpp.o:
	$(CC) $(INC) $(CFLAGS) $< -o $@
clean:
	rm -f *.o $(EXECUTABLE)
