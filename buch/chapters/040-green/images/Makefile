#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	parallelogramm.pdf lorentz.pdf

parallelogramm.pdf:	parallelogramm.tex
	pdflatex parallelogramm.tex

lorentz.png:	lorentz.pov
	povray +A0.1 -W1920 -H1080 -Olorentz.png lorentz.pov

lorentz.jpg:	lorentz.png Makefile
	magick -extract 920x1080+420+0 lorentz.png lorentz.jpg

lorentz.pdf:	lorentz.tex lorentz.jpg
	pdflatex lorentz.tex

