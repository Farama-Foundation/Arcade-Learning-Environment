MODULE := src/environment 

MODULE_OBJS := \
	src/environment/ale_state.o \
	src/environment/stella_environment.o \
	src/environment/phosphor_blend.o \
	
MODULE_DIRS += \
	src/environment

# Include common rules 
include $(srcdir)/common.rules
