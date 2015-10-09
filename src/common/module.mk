MODULE := src/common

MODULE_OBJS := \
	src/common/SoundNull.o \
	src/common/SoundSDL.o \
    src/common/SoundExporter.o \
	src/common/display_screen.o \
	src/common/ColourPalette.o \
	src/common/ScreenExporter.o \
	src/common/Constants.o \
    src/common/Log.o

MODULE_DIRS += \
	src/common

# Include common rules 
include $(srcdir)/common.rules
