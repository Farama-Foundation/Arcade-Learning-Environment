# Modified from the sharedLibraryInterface's makefile.
# Video recording requires SDL. If you do not have SDL installed, you will not be able to
# compile this example.
#
# > make recordingAgent 

USE_SDL := 1

# This will likely need to be changed to suit your installation.
ALE := ../..

FLAGS := -I$(ALE)/src -I$(ALE)/src/controllers -I$(ALE)/src/os_dependent -I$(ALE)/src/environment -I$(ALE)/src/external -L$(ALE)
CXX := g++
FILE := videoRecordingExample
LDFLAGS := -lale -lz

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    FLAGS += -Wl,-rpath=$(ALE)
endif
ifeq ($(UNAME_S),Darwin)
    FLAGS += -framework Cocoa
endif

ifeq ($(strip $(USE_SDL)), 1)
  DEFINES += -D__USE_SDL -DSOUND_SUPPORT
  FLAGS += $(shell sdl-config --cflags)
  LDFLAGS += $(shell sdl-config --libs)
endif

all: videoRecordingExample

videoRecordingExample:
	$(CXX) $(DEFINES) $(FLAGS) $(FILE).cpp $(LDFLAGS) -o $(FILE)

clean:
	rm -rf videoRecordingExample *.o
