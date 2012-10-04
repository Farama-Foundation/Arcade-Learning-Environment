MODULE := src/emucore/m6502

MODULE_OBJS := \
	src/emucore/m6502/src/Device.o \
	src/emucore/m6502/src/M6502.o \
	src/emucore/m6502/src/M6502Low.o \
	src/emucore/m6502/src/M6502Hi.o \
	src/emucore/m6502/src/NullDev.o \
	src/emucore/m6502/src/System.o

MODULE_DIRS += \
	src/emucore/m6502/src

# Include common rules 
include $(srcdir)/common.rules
