#
# Makefile to build numbering script
#
# (c) 2022 Prof Dr Andreas Müller, OST Ostschweizer Fachhochschule
#
all:	numberpages lncover.jpg

numberpages:	numberpages.cpp
	c++ -std=c++11 -g -Wall -o numberpages numberpages.cpp

test:	numberpages
	./numberpages --debug --section "Komplexe Funktionentheorie" \
		--force \
		"../MSE/1 - Komplexe Funktionen.pdf" blubb.pdf

lncover.pdf:	lncover.tex
	pdflatex lncover.tex
xoffset = 84
yoffset = 94
lncover.jpg:	lncover.pdf Makefile
	convert -density 300 -extract					\
		`expr 2008 - $(xoffset)`x`expr 2835 - $(yoffset)`+$(xoffset)+$(yoffset) \
		-background white -flatten lncover.pdf lncover.jpg

