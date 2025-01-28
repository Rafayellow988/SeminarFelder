#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	kruemmung.pdf

kruemmung.png:	kruemmung.pov
	povray +A0.1 -W1920 -H1080 -Okruemmung.png kruemmung.pov

kruemmung.jpg:	kruemmung.png Makefile
	magick -extract 1440x1080+242+0 kruemmung.png kruemmung.jpg

kruemmung.pdf:	kruemmung.tex kruemmung.jpg
	pdflatex kruemmung.tex

