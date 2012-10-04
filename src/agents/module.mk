MODULE := src/agents

MODULE_OBJS := \
	src/agents/PlayerAgent.o \
	src/agents/RandomAgent.o \
	src/agents/SingleActionAgent.o \

MODULE_DIRS += \
	src/agents

# Include common rules 
include $(srcdir)/common.rules
