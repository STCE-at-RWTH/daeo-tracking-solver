all: slides.pdf

%.pdf: %.tex 
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

clean: 
	rm -f *~ *bak *.vrb *.snm *.out *.nav *.toc *.aux *.ps *.log *.lof *.bbl *.blg *.dvi slides.pdf *.vrb _minted-slides -fr

.PHONY: all clean

slides.pdf : $(wildcard *.tex)
