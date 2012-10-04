MODULE := src/control

MODULE_OBJS := \
	src/control/ALEState.o \
	src/control/fifo_controller.o \
	src/control/game_controller.o \
	src/control/internal_controller.o \
	
MODULE_DIRS += \
	src/control

# Include common rules 
include $(srcdir)/common.rules
