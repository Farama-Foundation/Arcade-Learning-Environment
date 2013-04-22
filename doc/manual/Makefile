FILE=manual

all: $(FILE).pdf

%.pdf: $(FILE).tex
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

clean:
	rm -f *.{aux,log,toc,out}
