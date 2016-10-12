#!/usr/bin/env python2

import pickle
from numpy import power, exp, linspace, append, array
from scipy.misc import factorial
from matplotlib import pyplot
import seaborn
from aux import data_path, figure_path
import gzip

# base_name = "Poisson"
extension = ".png"

with gzip.open(data_path("DGE_matrix_counts_sparse.pkl.gz"), "rb") as DGE_file:
    data = pickle.load(DGE_file)

M, N = data.shape

counts = data.T.todense()

figure = pyplot.figure()
axis = figure.add_subplot(1, 1, 1)

x = linspace(0, M, M)
axis.bar(x, counts.sum(axis = 0).T)
axis.set_yscale("log")

plot_name = "total_profile_1d" + extension
figure.savefig(figure_path(plot_name))

figure = pyplot.figure()
axis = figure.add_subplot(1, 1, 1)

seaborn.heatmap(counts.T)

plot_name = "total_profile_2d" + extension
figure.savefig(figure_path(plot_name))

# for i in range(5):
#     cell = array(counts[i].T).flatten()
#
#     figure = pyplot.figure()
#     axis = figure.add_subplot(1, 1, 1)
#
#     x = linspace(0, M, M)
#     axis.bar(x, cell)
#
#     plot_name = str(i) + "_profile_1d" + extension
#     figure.savefig(figure_path(plot_name))
#
#     figure = pyplot.figure()
#     axis = figure.add_subplot(1, 1, 1)
#
#     cell_map = append(cell, [0, 0]).reshape((-1, 137))
#     seaborn.heatmap(cell_map.T)
#
#     plot_name = str(i) + "_profile_2d" + extension
#     figure.savefig(figure_path(plot_name))

# lambda_ = counts.mean()
# k_min = counts.min()
# k_max = counts.max()
#
# k = linspace(k_min, k_max, k_max + 1)
# P = N * power(lambda_, k) * exp(-lambda_) / factorial(k)
#
# figure = pyplot.figure()
# axis = figure.add_subplot(1, 1, 1)
#
# axis.hist(counts.T, bins = k_max)
# axis.plot(k, P)
#
# plot_name = base_name + "_histogram" + extension
# figure.savefig(plot_name)
#
# figure = pyplot.figure()
# axis = figure.add_subplot(1, 1, 1)
#
# axis.hist(counts.T, bins = k_max - 1, range = (k_min + 1, k_max))
# axis.plot(k, P)
#
# axis.set_xlim(k_min + 1, k_max)
# axis.set_ylim(0, 1000)
#
# plot_name = base_name + "_histogram_no_zeros" + extension
# figure.savefig(figure_path(plot_name))
