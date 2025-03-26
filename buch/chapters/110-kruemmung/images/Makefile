#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	exzess.pdf

exzess.jpg:	exzess.png
	magick exzess.png exzess.jpg

exzess.png:	exzess.pov common.inc
	povray +A0.1 -W1920 -H1080 -Oexzess.png exzess.pov

auswahl = -extract 1120x1080+440+0

exzess0.png:	exzess0.pov common.inc
	povray +A0.1 -W1920 -H1080 -Oexzess0.png exzess0.pov
exzess0.jpg:	exzess0.png Makefile
	magick $(auswahl) exzess0.png exzess0.jpg

exzess1.png:	exzess1.pov common.inc
	povray +A0.1 -W1920 -H1080 -Oexzess1.png exzess1.pov
exzess1.jpg:	exzess1.png Makefile
	magick $(auswahl) exzess1.png exzess1.jpg

exzess2.png:	exzess2.pov common.inc
	povray +A0.1 -W1920 -H1080 -Oexzess2.png exzess2.pov
exzess2.jpg:	exzess2.png Makefile
	magick $(auswahl) exzess2.png exzess2.jpg

exzess.pdf:	exzess.tex exzess0.jpg exzess1.jpg exzess2.jpg
	pdflatex exzess.tex
