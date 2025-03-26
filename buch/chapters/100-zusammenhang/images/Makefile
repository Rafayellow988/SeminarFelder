#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	kugeldreieck.pdf						\
	polargeodaete.pdf						\
	transport.pdf

kugeldreieck.png:	kugeldreieck.pov
	povray +A0.1 -W1920 -H1080 -Okugeldreieck.png kugeldreieck.pov

kugeldreieck.jpg:	kugeldreieck.png Makefile
	magick -extract 1300x1080+310+0 kugeldreieck.png kugeldreieck.jpg

kugeldreieck.pdf:	kugeldreieck.tex kugeldreieck.jpg
	pdflatex kugeldreieck.tex

polargeodaete.pdf:	polargeodaete.tex
	pdflatex polargeodaete.tex

transport.png:	transport.pov transportvektoren.inc
	povray +A0.1 -W1920 -H1080 -Otransport.png transport.pov

transport.jpg:	transport.png Makefile
	magick -extract 1420x1080+250+0 transport.png transport.jpg

transportvektoren.inc:	transport.m
	octave transport.m

transport.pdf:	transport.tex transport.jpg
	pdflatex transport.tex
