#
# Makefile
#
# (c) 2025 Prof Dr Andreas Müller
#
all:	wirbelring.jpg

wirbelring.png:	wirbelring.pov
	povray -W1920 -H1080 -Owirbelring.png wirbelring.pov

wirbelring.jpg:	wirbelring.png Makefile
	magick -extract 1920x920+0+80 wirbelring.png wirbelring.jpg

