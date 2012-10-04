MODULE := src/os_dependent

MODULE_OBJS := \
	src/os_dependent/FSNodePOSIX.o \
	src/os_dependent/OSystemUNIX.o \
	src/os_dependent/SettingsUNIX.o \
	
MODULE_DIRS += \
	src/os_dependent

# Include common rules 
include $(srcdir)/common.rules
