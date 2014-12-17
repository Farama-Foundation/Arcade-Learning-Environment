MODULE := src/

MODULE_OBJS := \
	src/main.o \
	src/ale_interface.o

MODULE_DIRS += \
	src/

# Include common rules
include $(srcdir)/common.rules
