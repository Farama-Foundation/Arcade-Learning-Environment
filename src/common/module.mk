MODULE := src/common

MODULE_OBJS := \
	src/common/SoundNull.o \
	src/common/export_screen.o \
	src/common/display_screen.o \
	src/common/Constants.o \
	src/common/Defaults.o \

MODULE_DIRS += \
	src/common

# Include common rules 
include $(srcdir)/common.rules
