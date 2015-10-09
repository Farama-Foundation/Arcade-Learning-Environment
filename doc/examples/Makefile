# We do not automatically build the recording agent, which requires SDL. To build it, run
#
# > make recordingAgent
all: sharedLibraryAgent rlglueAgent fifoAgent

sharedLibraryAgent: 
	make -f Makefile.sharedlibrary

rlglueAgent:
	make -f Makefile.rlglue

fifoAgent: 
	make -f Makefile.fifo

recordingAgent: 
	make -f Makefile.recording

clean:
	make -f Makefile.rlglue clean
	make -f Makefile.sharedlibrary clean
	make -f Makefile.fifo clean
	make -f Makefile.recording clean
