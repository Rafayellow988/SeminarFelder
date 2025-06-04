import matplotlib # version <= 3.7.1 needed for proper pgf handling
def enable_export():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        # 'font.size' : 8,
        'text.usetex': True,
        'pgf.rcfonts': False,
        'axes.unicode_minus' : False,
        "pgf.preamble": "\\usepackage{bm}\n\\usepackage{amsmath}\n\\usepackage{xcolor}\n\\usepackage{tgtermes}",
    })
