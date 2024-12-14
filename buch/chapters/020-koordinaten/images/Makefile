#
# Makefile
#
# (c) 2024 Prof Dr Andreas Müller
#
all:	kartpolar.pdf							\
	kartkugel.pdf							\
	koordinaten.pdf

kartpolar.pdf:	kartpolar.tex
	pdflatex kartpolar.tex

kartkugel.png:	kartkugel.pov
	povray +A0.1 -W1920 -H1080 -Okartkugel.png kartkugel.pov

kartkugel.jpg:	kartkugel.png Makefile
	magick -extract 1150x1080+500+0 kartkugel.png kartkugel.jpg

kartkugel.pdf:	kartkugel.tex kartkugel.jpg
	pdflatex kartkugel.tex

koordinaten.pdf:	koordinaten.tex
	pdflatex koordinaten.tex
