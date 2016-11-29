#!/usr/bin/env python

import data

from matplotlib import pyplot
import seaborn

from numpy import linspace, random, nonzero, where, inf, log, exp
from sklearn.decomposition import PCA

palette = seaborn.color_palette('Set2', 8)
seaborn.set(style='ticks', palette = palette)

pyplot.rcParams.update({'figure.max_open_warning': 0})

def analyseData(data_set, name = "data", intensive_calculations = False):
    
    N, D = data_set.shape
    
    if intensive_calculations:
        plotHeatMap(data_set, name)
    
    average_cell = data_set.mean(axis = 0)
    average_cell_name = name + "_mean"
    plotProfile(average_cell, average_cell_name)
    
    subset = random.randint(N, size = 10)
    for j, i in enumerate(subset):
        cell = data_set[i]
        cell_name = name + "_cell_{}".format(j)
        plotProfile(cell, cell_name)
    
    # average_genes_per_cell = data_set.sum(axis = 1)
    # print(average_genes_per_cell.std() / average_genes_per_cell.mean())

def analyseModel(model, name = "model"):
    
    plotLearningCurves(model.learning_curves, name)

def analyseResults(x_test, x_test_recon, x_test_headers, clusters, latent_set,
    x_sample, name = "results", intensive_calculations = False):
    
    N, D = x_test.shape
    
    data_sets = [
        {"data_set": x_test, "name": "Test", "tolerance": 0.5},
        {"data_set": x_test_recon["mean"], "name": "Reconstructed",
         "tolerance": 0.5},
        {"data_set": x_test_recon["p"], "name": "p"},
        {"data_set": exp(x_test_recon["log_r"]), "name": "r"}
    ]
    printSummaryStatistics([statistics(**data_set) for data_set in data_sets])
    
    if intensive_calculations:
        print("Creating heat maps.")
        
        test_set_name = name + "_test"
        plotHeatMap(x_test, test_set_name)
        
        reconstructed_test_set_name = name + "_test_recon"
        plotHeatMap(x_test_recon["mean"], reconstructed_test_set_name)
        
        difference_name = name + "_test_difference"
        plotHeatMap(x_test - x_test_recon["mean"], difference_name)
        
        log_ratio_name = name + "_test_log_ratio"
        plotHeatMap(log(x_test / x_test_recon["mean"] + 1), log_ratio_name)
    
    print("Creating latent space scatter plot.")
    plotLatentSpace(latent_set, x_test_headers, clusters, name)
    
    subset = random.randint(N, size = 10)
    for j, i in enumerate(subset):
        
        print("Creating profiles for cell {}.".format(x_test_headers["cells"][i]))
        
        cell_test = x_test[i]
        cell_test_name = name + "_cell_{}_test".format(j)
        plotProfile(cell_test, cell_test_name)
        
        cell_recon = x_test_recon["mean"][i]
        cell_recon_name = name + "_cell_{}_recon_mean".format(j)
        plotProfile(cell_recon, cell_recon_name)
        
        cell_recon_p = x_test_recon["p"][i]
        cell_recon_p_name = name + "_cell_{}_recon_p".format(j)
        plotProfile(cell_recon_p, cell_recon_p_name)
        
        cell_recon_log_r = x_test_recon["log_r"][i]
        cell_recon_log_r_name = name + "_cell_{}_recon_log_r".format(j)
        plotProfile(cell_recon_log_r, cell_recon_log_r_name)
        
        cell_diff = cell_test - cell_recon
        cell_diff_name = name + "_cell_{}_diff".format(j)
        plotProfile(cell_diff, cell_diff_name)

def statistics(data_set, name = "", tolerance = 1e-3):
    
    statistics = {
        "name": name,
        "mean": data_set.mean(),
        "std": data_set.std(),
        "min": data_set.min(),
        "minimums": (data_set < data_set.min() + tolerance).sum(),
        "max": data_set.max(),
        "maximums": (data_set >= data_set.max() - tolerance).sum(),
        "sparsity": float((data_set < tolerance).sum()) / float(data_set.size)
    }
    
    return statistics

def printSummaryStatistics(statistics_sets):
    
    if type(statistics_sets) != list:
        statistics_sets = [statistics_sets]
    
    name_width = 0
    
    for statistics_set in statistics_sets:
        name_width = max(len(statistics_set["name"]), name_width)
    
    print("Statistics:")
    print("  ".join(["{:{}}".format("Data set", name_width), "mean", "std ", " minimum ", "n_minimum", " maximum ", "n_maximum", "sparsity"]))
    
    for statistics_set in statistics_sets:
        string_parts = [
            "{:{}}".format(statistics_set["name"], name_width),
            "{:<4.2f}".format(statistics_set["mean"]),
            "{:<4.2g}".format(statistics_set["std"]),
            "{:<9.3g}".format(statistics_set["min"]),
            "{:>7d}".format(statistics_set["minimums"]),
            "{:<9.3g}".format(statistics_set["max"]),
            "{:>7d}".format(statistics_set["maximums"]),
            "{:<7.5g}".format(statistics_set["sparsity"]),
        ]
        
        print("  ".join(string_parts))

def plotProfile(cell, name):
    
    D = cell.shape[0]
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    x = linspace(0, D, D)
    # axis.bar(x, cell)
    axis.plot(x, cell)
    
    figure_name = name + "_profile"
    data.saveFigure(figure, figure_name)

def plotHeatMap(data_set, name):
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.heatmap(data_set.T, xticklabels = False, yticklabels = False, cbar = True, square = True, ax = axis)
    
    figure_name = name + "_heat_map"
    data.saveFigure(figure, figure_name, no_spine = False)

def plotLearningCurves(curves, name):
    
    figure_1 = pyplot.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    
    figure_2 = pyplot.figure()
    axis_2 = figure_2.add_subplot(1, 1, 1)
    
    for i, (curve_set_name, curve_set) in enumerate(sorted(curves.items())):
        
        colour = palette[i]
        
        for curve_name, curve in sorted(curve_set.items()):
            if curve_name == "lower bound":
                line_style = "solid"
                curve_name = curve_name.capitalize()
                axis = axis_1
            elif curve_name == "log p(x|z)":
                line_style = "dashed"
                axis = axis_1
            elif curve_name == "KL divergence":
                line_style = "dotted"
                axis = axis_2
            label = curve_name + " ({} set)".format(curve_set_name)
            axis.plot(curve, color = colour, linestyle = line_style, label = label)
    
    axis_1.legend(loc = "best")
    
    figure_1_name = name + "_learning_curves"
    data.saveFigure(figure_1, figure_1_name)
    
    figure_2_name = name + "_learning_curves_KL"
    data.saveFigure(figure_2, figure_2_name)

def plotLatentSpace(latent_set, x_test_headers, clusters, name):
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    pca = PCA(n_components = 2)
    pca.fit(latent_set)
    latent_set_pc = pca.transform(latent_set)
    
    for cluster_id in clusters:
        
        cluster = clusters[cluster_id]
        subset = []
    
        for cell in cluster:
            index = where(x_test_headers["cells"] == cell)[0]
            # print(index) # cells in headers, and hence in test set, are clustered
            if len(index) == 0:
                continue
            subset.append(int(index))
    
        axis.scatter(latent_set_pc[subset, 0], latent_set_pc[subset, 1],
            c = data.cluster_colours[cluster_id], edgecolors = None)
    
    figure_name = name + "_latent_space"
    data.saveFigure(figure, figure_name)

if __name__ == '__main__':
    random.seed(1234)
    data_set = data.createSampleData(1000, 500, p = 0.95)
    analyseData(data_set, name = "sample", intensive_calculations = True)
