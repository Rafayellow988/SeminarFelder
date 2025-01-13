#
# Makefile.inc
#
# (c) 2022 Prof Dr Andreas Müller, OST Ostschweizer Fachhochschule
#

CHAPTERFILES = 								\
	chapters/part1.tex						\
	chapters/vorwort.tex

include chapters/000-einleitung/Makefile.inc
include chapters/010-fallstudie/Makefile.inc
include chapters/020-koordinaten/Makefile.inc
include chapters/030-kurvenintegral/Makefile.inc
include chapters/040-green/Makefile.inc
