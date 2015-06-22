MODULE := src/controllers 

MODULE_OBJS := \
	src/controllers/ale_controller.o \
	src/controllers/fifo_controller.o \
	src/controllers/rlglue_controller.o \
	
MODULE_DIRS += \
	src/controllers

# Include common rules 
include $(srcdir)/common.rules
