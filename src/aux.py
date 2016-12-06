import os, sys

from itertools import izip, count

from math import floor

# Plotting

colours = [
    (0.07, 0.40, 0.69), # blue
    (0.92, 0.16, 0.16), # red
    (0.09, 0.65, 0.28), # green
    (0.99, 0.96, 0.21), # yellow
    (0.55, 0.19, 0.52), # purple
    (0.96, 0.51, 0.20), # orange
    (0.16, 0.72, 0.70), # blue--green
    (0.70, 0.19, 0.43), # red--purple
    (0.46, 0.76, 0.27), # green--yellow
    (0.98, 0.72, 0.22), # yellow--orange
    (0.24, 0.22, 0.55), # purple--blue
    (0.94, 0.35, 0.18), # orange--red
]
# From: http://www.perceptualedge.com/articles/b-eye/choosing_colors.pdf

# Default paths for input and output

def script_directory():
    os.chdir(sys.path[0])

def savepath(path=None):
	def format_function(file_path = "", base_path = path):
		if not base_path.endswith(os.sep):
			base_path = base_path + os.sep
		file_name = os.path.realpath(base_path + file_path)
		path, name = os.path.split(file_name)
		try:
			if not os.path.exists(path):
				os.makedirs(path)
		except:
			print('ERROR:', 'Failed to create path {}'.format(path))
			path = './'
		return path + os.sep + name

	return format_function

figures_path  = savepath(path='../fig')
data_path = savepath(path='../data')
preprocessed_path = savepath(path='../data/preprocessed')
models_path = savepath(path='../data/models')
result_path = savepath(path='../data/results')

# Math

math_functions = ["log", "exp", "sin", "cos", "tan", "tanh"]
greek_letters = [
    "alpha", "beta", "gamma", "Gamma", "delta", "Delta", "epsilon", "zeta", 
    "eta", "theta", "Theta", "iota", "kappa", "lambda", "Lambda", "mu", 
    "nu", "xi", "Xi", "omicron", "pi", "Pi", "rho", "sigma", "Sigma", "tau", 
    "upsilon", "phi", "Phi", "chi", "psi", "Psi", "omega", "Omega"
]

def labelWithDefaultSymbol(default_symbol):
    def label_function(symbol = None):
        if not symbol:
            return "${}$".format(default_symbol)
        elif symbol == "mean":
            return "$\\bar{}$".format(default_symbol)
        else:
            symbol_parts = symbol.split("_")
            symbol_string = "$"
            for symbol_part in symbol_parts:
                if symbol_part in math_functions or symbol_part in greek_letters:
                    symbol_string += "\\" + symbol_part
                else:
                    symbol_string += symbol_part
            symbol_string += "$"
            return symbol_string
    
    return label_function

# Helper functions
def enumerate_reversed(a_list, start = -1):
    return izip(count(len(a_list) + start, -1), reversed(a_list))

def convertTimeToString(seconds):
    if seconds < 1:
        return "{:.0f} ms".format(1000 * seconds)
    elif seconds < 60:
        return "{:.3g} s".format(seconds)
    elif seconds < 60 * 60:
        minutes = floor(seconds / 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        return "{:.0f}m {:.0f}s".format(minutes, seconds)
    else:
        hours = floor(seconds / 60 / 60)
        minutes = floor((seconds / 60) % 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
        return "{:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds)

# Shell output

RESETFORMAT = "\033[0m"
BOLD = "\033[1m"

def bold(string):
    """Convert to bold type."""
    return BOLD + string + RESETFORMAT

def underline(string, character="="):
    """Convert string to header marks"""
    return character * len(string)

def title(string):
    """Display a coloured title."""
    print("{}\n{}\n".format(bold(string), underline(string, "=")))

def subtitle(string):
    """Display a coloured subtitle."""
    print("{}\n{}\n".format(bold(string), underline(string, "-")))
