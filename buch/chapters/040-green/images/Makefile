#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	parallelogramm.pdf lorentz.pdf randkarten.pdf

#
# Parallelogramm-Fäche
#
parallelogramm.pdf:	parallelogramm.tex
	pdflatex parallelogramm.tex

#
# Lorentz-Kraft
#
lorentz.png:	lorentz.pov
	povray +A0.1 -W1920 -H1080 -Olorentz.png lorentz.pov

lorentz.jpg:	lorentz.png Makefile
	magick -extract 920x1080+420+0 lorentz.png lorentz.jpg

lorentz.pdf:	lorentz.tex lorentz.jpg
	pdflatex lorentz.tex

#
# Randkarten
#
randkarten.png:	randkarten.pov	
	povray +A0.1 -W1920 -H1080 -Orandkarten.png randkarten.pov

randkarten.jpg:	randkarten.png Makefile
	magick -extract 1574x1040+173+12 randkarten.png randkarten.jpg

randkarten.pdf:	randkarten.tex randkarten.jpg
	pdflatex randkarten.tex


