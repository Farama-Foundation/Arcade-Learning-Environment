MODULE := src/emucore/rsynth

MODULE_OBJS := \
        src/emucore/rsynth/darray.o \
        src/emucore/rsynth/elements.o \
        src/emucore/rsynth/holmes.o \
        src/emucore/rsynth/opsynth.o \
        src/emucore/rsynth/phones.o \
        src/emucore/rsynth/phtoelm.o \
        src/emucore/rsynth/trie.o

MODULE_DIRS += \
        src/emucore/rsynth

# Include common rules
include $(srcdir)/common.rules
