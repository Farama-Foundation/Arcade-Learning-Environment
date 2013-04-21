all: sharedLibraryAgent rlglueAgent

sharedLibraryAgent: 
	make -f Makefile.sharedlibrary

rlglueAgent:
	make -f Makefile.rlglue

clean:
	make -f Makefile.rlglue clean
	make -f Makefile.sharedlibrary clean
