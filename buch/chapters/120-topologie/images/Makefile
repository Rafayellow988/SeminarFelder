#
# Makefile -- build topology images
#
# (c) 2025 Prof Dr Andreas Müller
#

all:	unterteilung.pdf						\
	sphaere.jpg

sphaere.png:	sphaere.pov
	povray +A0.1 +W1920 -H1080 -Osphaere.png sphaere.pov

sphaere.jpg:	sphaere.png Makefile
	magick -extract 1080x1080+420+0 sphaere.png sphaere.jpg

unterteilung.pdf:	unterteilung.tex
	pdflatex unterteilung.tex
