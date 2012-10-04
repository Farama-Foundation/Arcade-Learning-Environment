MODULE := src/

MODULE_OBJS := \
	src/main.o 
	
MODULE_DIRS += \
	src/

# Include common rules 
include $(srcdir)/common.rules
