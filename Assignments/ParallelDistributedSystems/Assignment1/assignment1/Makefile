CXX                = g++ -std=c++17
OPTFLAGS	   = -O3
AUTOFLAGS          = -march=native -ffast-math -fopenmp-simd #-fprofile-generate -fprofile-use -fopt-info-vec-all 
AVXFLAGS           = -mavx -ffast-math -fopenmp-simd -mfma #-fprofile-generate -fprofile-use -fopt-info-vec-all
# WIP optimized: AVXFLAGS	=	-mavx2 -mfma -march=native -falign-loops=32
#AVXFLAGS = -mavx2 -mfma -march=native -falign-loops=32 -falign-functions=32
CXXFLAGS          += -Wall 
INCLUDES	   = -I. -I./include
LIBS               = #-pthread -fopenmp
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# For files with _avx in their name, append flags to CXXFLAGS
%_avx: CXXFLAGS += ${AVXFLAGS}

# For files with _auto in their name, append flags to CXXFLAGS
%_auto: CXXFLAGS += ${AUTOFLAGS}

all: $(TARGET)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
