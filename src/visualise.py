#!/usr/bin/env python3

import pickle
from numpy import power, exp, linspace
from scipy.misc import factorial
from matplotlib import pyplot
import seaborn

base_name = "Poisson"
extension = ".png"

with open(data_path("DGE_matrix_counts_sparse.pkl"), "rb") as DGE_file:
    data = pickle.load(DGE_file)

M, N = data.shape

counts = data[45].todense()

lambda_ = counts.mean()
k_min = counts.min()
k_max = counts.max()

k = linspace(k_min, k_max, k_max + 1)
P = N * power(lambda_, k) * exp(-lambda_) / factorial(k)

figure = pyplot.figure()
axis = figure.add_subplot(1, 1, 1)

axis.hist(counts.T, bins = k_max)
axis.plot(k, P)

plot_name = base_name + "_histogram" + extension
figure.savefig(plot_name)

figure = pyplot.figure()
axis = figure.add_subplot(1, 1, 1)

axis.hist(counts.T, bins = k_max - 1, range = (k_min + 1, k_max))
axis.plot(k, P)

axis.set_xlim(k_min + 1, k_max)
axis.set_ylim(0, 1000)

plot_name = base_name + "_histogram_no_zeros" + extension
figure.savefig(figure_path(plot_name))
