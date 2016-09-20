import os, sys

from itertools import izip, count

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
    def format_function(fname='fname', path=path):
        if not path.endswith('/'):
            path = path + '/'
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except:
            print('ERROR:', 'Failed to create path {}'.format(path))
            path = './'
        return (path + '{}').format(fname)

    return format_function

figure_path  = savepath(path='../fig')
data_path = savepath(path='../data')

# Helper functions
def enumerate_reversed(a_list, start = -1):
    return izip(count(len(a_list) + start, -1), reversed(a_list))

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
