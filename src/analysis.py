#!/usr/bin/env python

import data

from matplotlib import pyplot
import seaborn

from numpy import linspace, random, nonzero

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
    
    # plotLearningCurves(model.training_learning_curve, name + "_training")
    # plotLearningCurves(model.validation_learning_curve, name + "_validation")
    plotLearningCurves([{"name": "Training set", "values": model.training_learning_curve}, {"name": "Validation set", "values": model.validation_learning_curve}], name)

def analyseResults(x_test, x_test_recon, x_sample, name = "results",
    intensive_calculations = False):
    
    N, D = x_test.shape
    
    if intensive_calculations:
        test_set_name = name + "_test"
        plotHeatMap(x_test, test_set_name)
    
        reconstructed_test_set_name = name + "_test_recon"
        plotHeatMap(x_test_recon, reconstructed_test_set_name)
    
        difference_name = name + "_test_difference"
        plotHeatMap(x_test - x_test_recon, difference_name)
    
    subset = random.randint(N, size = 10)
    for j, i in enumerate(subset):
        cell_test = x_test[i]
        cell_test_name = name + "_cell_{}_test".format(j)
        plotProfile(cell_test, cell_test_name)
        cell_recon = x_test_recon[i]
        cell_recon_name = name + "_cell_{}_recon".format(j)
        plotProfile(cell_recon, cell_recon_name)
        cell_diff = cell_test - cell_recon
        cell_diff_name = name + "_cell_{}_diff".format(j)
        plotProfile(cell_diff, cell_diff_name)
    

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
    
    seaborn.heatmap(data_set.T, xticklabels = False, yticklabels = False, cbar = False, square = True, ax = axis)
    
    figure_name = name + "_heat_map"
    data.saveFigure(figure, figure_name, no_spine = False)

def plotLearningCurves(curves, name):
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for curve in curves:
        axis.plot(curve["values"], label = curve["name"])
    axis.legend(loc = "best")
    figure_name = name + "_learning_curves"
    
    data.saveFigure(figure, figure_name)

if __name__ == '__main__':
    random.seed(1234)
    data_set = data.createSampleData(1000, 500, p = 0.95)
    analyseData(data_set, name = "sample", intensive_calculations = True)
