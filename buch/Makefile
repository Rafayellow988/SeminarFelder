#
# Makefile zum Skript ueber Felder
#
# (c) 2022 Prof Dr Andreas Mueller, OST Ostschweizer Fachhochschule
#
seminar = SeminarFelder

all:	buch.pdf

buch:	$(seminar).pdf

papers/common/Makefile.inc:	papers/common/paperlist

include common/Makefile.inc
include chapters/Makefile.inc
include papers/common/Makefile.inc
include papers/common/includes.inc

ALLTEXFILES = $(TEXFILES) $(CHAPTERFILES)

#
# Buchblock für Druckerei
#
$(workdir):
	mkdir -p $(workdir)
buch.pdf:	$(workdir)/buch.pdf
	cp $(workdir)/buch.pdf buch.pdf

$(workdir)/buch.pdf:	$(workdir) buch.tex $(TEXFILES) $(workdir)/buch.ind $(BLXFILES)
	$(pdflatex) buch.tex
	cd $(workdir); $(bibtex) buch

$(workdir)/buch.ind:	$(workdir) $(workdir)/buch.idx
	cd $(workdir); $(makeindex) buch.idx

$(workdir)/buch.idx:	$(workdir) buch.tex $(TEXFILES) images
	$(touch) $(workdir)/buch.ind
	$(pdflatex) buch.tex

#
# Papers in einzelne PDF-Files separieren für digitales Feedback
#
separate:	build/buch.aux buch.pdf
	bash tools/splitpapers

#
# Version mit Buchdeckeln für Release/Website
#
buch = $(seminar)
$(buch).pdf:	$(workdir)/$(buch).pdf
	cp $(workdir)/$(buch).pdf $(buch).pdf

$(workdir)/$(seminar).pdf:	$(buch).tex $(TEXFILES) $(workdir)/$(buch).ind $(SBLXFILES)
	$(pdflatex) $(buch).tex
	cd $(workdir); $(bibtex) $(buch)

$(workdir)/$(buch).idx:	$(workdir) $(buch).tex $(TEXFILES) images
	$(touch) $(workdir)/$(buch).ind
	$(pdflatex) $(buch).tex

$(workdir)/$(buch).ind:	$(workdir)/$(buch).idx
	cd $(workdir); $(makeindex) $(buch).idx

#
# Einzelnes Paper
#
paper:	paper.pdf
paper.pdf:	$(workdir) $(workdir)/paper.pdf
	cp $(workdir)/paper.pdf .

$(workdir)/paper.pdf:	paper.tex papername.tex $(TEXFILES)
	$(pdflatex) paper.tex

papername.tex:
	@echo "FEHLER: Das File papername.tex wurde nicht gefunden. Es muss von Hand "
	@echo "FEHLER: angelegt werden. Es enthält den Namen des Kapitels, das vom "
	@echo "FEHLER: File paper.tex als einzelnes Kapitel gesetzt werden soll. "
	@echo "FEHLER: Der Name muss in der Form"
	@echo "FEHLER: "
	@echo "FEHLER: \\def\\papername{diffortho}"
	@echo "FEHLER: "
	@echo "FEHLER: eingegeben werden, damit das Paper 'diffortho' gebaut wird."
	@echo "FEHLER: Konsultieren Sie die Anleitung zum Seminar, Abschnitt 4.2,"
	@echo "FEHLER: für Details."
	@echo "FEHLER: Achtung: Ihr persöhnliches papername.tex darf nicht zum "
	@echo "FEHLER:          Repo hinzugefügt werden!"
	@exit 1

#
# This Makefile can also construct the short tests
#
include test/Makefile.inc

#
# Errata
#
errata.pdf:	errata.tex
	$(pdflatex) errata.tex

#
# publish
#
publish:	$(seminar).pdf
	bash publish
