FILE=java-agent

all: $(FILE).pdf

%.pdf: $(FILE).tex
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

clean:
	rm -f *.{aux,log}
