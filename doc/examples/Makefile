all: sharedLibraryAgent rlglueAgent fifoAgent

sharedLibraryAgent: 
	make -f Makefile.sharedlibrary

rlglueAgent:
	make -f Makefile.rlglue

fifoAgent: 
	make -f Makefile.fifo

clean:
	make -f Makefile.rlglue clean
	make -f Makefile.sharedlibrary clean
	make -f Makefile.fifo clean
