#
# Makefile
#
# (c) 2024 Prof Dr Andreas Müller
#
all:	1koordinaten.pdf

1koordinaten.pdf:	1koordinaten.tex
	pdflatex 1koordinaten.tex

loxodrome.png:	loxodrome.pov
	povray +A0.1 -W1920 -H1080 -Oloxodrome.png loxodrome.pov

