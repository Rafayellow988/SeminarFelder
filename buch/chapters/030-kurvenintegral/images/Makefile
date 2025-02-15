#
# Makefile
#
# (c) 2024 Prof Dr Andreas Müller
#
all:	1koordinaten.pdf						\
	loxodrome.pdf							\
	wege.pdf							\
	integrand.pdf							\
	mercator.pdf							\
	zerlegung.pdf

1koordinaten.pdf:	1koordinaten.tex
	pdflatex 1koordinaten.tex

loxodrome.png:	loxodrome.pov
	povray +A0.1 -W1920 -H1080 -Oloxodrome.png loxodrome.pov

loxodrome.jpg:	loxodrome.png Makefile
	magick -extract 940x1068+490+6 loxodrome.png loxodrome.jpg

loxodrome.pdf:	loxodrome.tex loxodrome.jpg
	pdflatex loxodrome.tex

wege.pdf:	wege.tex
	pdflatex wege.tex

integrand.pdf:	integrand.tex
	pdflatex integrand.tex

mercator.pdf:	mercator.tex
	pdflatex mercator.tex

zerlegungpath.tex:	zerlegung.m
	octave zerlegung.m

zerlegung.pdf:	zerlegung.tex zerlegungpath.tex
	pdflatex zerlegung.tex

