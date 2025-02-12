#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	kuh.pdf kont.pdf gaussrand.pdf

# kuh image
kuh.pdf:	kuh.png kuh.tex
	pdflatex kuh.tex

# kont image
kont.png:	kont.pov
	povray +A0.1 -W1920 -H1080 -Okont.png kont.pov
kont.jpg:	kont.png Makefile
	magick -extract 1095x1070+455+6 kont.png kont.jpg
kont.pdf:	kont.tex kont.jpg
	pdflatex kont.tex

# gaussrand image
gaussrand.png:	gaussrand.pov
	povray +A0.1 -W1920 -H1080 -Ogaussrand.png gaussrand.pov
gaussrand.jpg:	gaussrand.png Makefile
	magick -extract 1724x1072+94+8 gaussrand.png gaussrand.jpg
gaussrand.pdf:	gaussrand.tex gaussrand.jpg
	pdflatex gaussrand.tex

