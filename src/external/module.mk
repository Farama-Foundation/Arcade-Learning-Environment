MODULE := src/external

MODULE_OBJS := \
	src/external/TinyMT/tinymt32.o \

MODULE_DIRS += \
	src/external/TinyMT

# Include common rules
include $(srcdir)/common.rules
