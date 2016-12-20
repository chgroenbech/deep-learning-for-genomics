#!/usr/bin/env python

import data

from matplotlib import pyplot
import seaborn

from numpy import linspace, random, nonzero, where, inf, log, exp, empty, arange, concatenate, sort
from sklearn.decomposition import PCA

from aux import labelWithDefaultSymbol

palette = seaborn.color_palette('Set2', 8)
seaborn.set(style='ticks', palette = palette)

pyplot.rcParams.update({'figure.max_open_warning': 0})

def analyseData(data_sets, name = "data", intensive_calculations = False):
    
    base_name = "Data/" + name +"/"
    
    if type(data_sets) == dict:
        united_data_set = concatenate([data_sets[d] for d in data_sets], axis = 0)
        data_sets["united"] = united_data_set
    else:
        data_sets = {name: data_sets}
    
    label = labelWithDefaultSymbol("x")
    statistics_set = []
    normed = False
    # normed = True
    
    for data_set_name, data_set in data_sets.items():
        
        M, F = data_set.shape

        print(data_set_name.title() + " ({} examples, {} features):".format(M,F))
        
        plot_name = base_name + data_set_name
        
        plotCountHistogram(data_set, k_min = 1, k_max = 10,
            name = plot_name)
        
        plotHistogram(data_set.flatten(), "Counts", "log", normed=normed, name = plot_name)
        
        if M < 14000 and F <= 5000:
            plotHeatMap(data_set, name = plot_name)
        
        series_set = [
            {"name": plot_name + "_genes", "label": "Total counts per gene",
             "per": "Genes", "values": data_set.sum(axis = 0)},
            {"name": plot_name + "_cells", "label": "Total counts per cell",
             "per": "Cells", "values": data_set.sum(axis = 1)},
            {"name": plot_name + "_genes_expressed",
             "label": "Genes expressed per cell",
             "per": "Cells", "values": (data_set != 0).sum(axis = 1)},
            {"name": plot_name + "_genes_mean",
             "label": "Mean counts per gene",
             "per": "Genes", "values": data_set.mean(axis = 0)},
            {"name": plot_name + "_genes_variance_sorted",
             "label": "Count variance per gene",
             "per": "Sorted genes by count variances", "values": sort(data_set.var(axis = 0))[::-1]}
        ]
        
        for series in series_set:
            plotProfile(series["values"], series["per"], series["label"], "log",
                bar = True, name = series["name"])
            plotHistogram(series["values"], series["label"], "log", normed=normed, name = series["name"])
            print(series["label"] + ": " + 
                "mean: {}, std: {}.".format(series["values"].mean(),
                    series["values"].std()))
        
        print("")
        
        statistics_set.append(statistics(data_set, name = data_set_name,
            tolerance = 0.5))
    
    printSummaryStatistics(statistics_set)

def analyseModel(model, name = "model"):
    
    plotLearningCurves(model.learning_curves, name)
    
    for i, (curve_set_name, curve_set) in enumerate(sorted(model.learning_curves.items())):
        
        string = curve_set_name + ": "
        
        for curve_name, curve in sorted(curve_set.items()):
            string += curve_name + ": {}, ".format(curve[-1])
        
        print(string)
    

def analyseResults(x_test, x_test_recon, x_test_headers, clusters, latent_set,
    x_sample, name = "results", intensive_calculations = False):
    
    N, D = x_test.shape
    
    data_sets = [
        {"data_set": x_test, "name": "Test", "tolerance": 0.5},
        {"data_set": x_test_recon["mean"], "name": "Reconstructed",
         "tolerance": 0.5},
    ]
    
    for variable_name in x_test_recon:
        
        if variable_name == "mean":
            continue
        
        if "log" in variable_name:
            variable = exp(x_test_recon[variable_name])
        else:
            variable = x_test_recon[variable_name]
        
        variable_name = variable_name.replace("log_", "")
        
        data_set = {"data_set": variable, "name": variable_name}
        data_sets.append(data_set)
    
    printSummaryStatistics([statistics(**data_set) for data_set in data_sets])
    
    print("")
    
    x_diff = x_test - x_test_recon["mean"]
    x_log_ratio = log((x_test + 1) / (x_test_recon["mean"] + 1))
    
    if intensive_calculations:
        print("Creating heat maps.")
        
        test_set_name = name + "/test"
        plotHeatMap(x_test, x_test_headers, clusters, name = test_set_name)
        
        reconstructed_test_set_name = name + "/test_recon"
        plotHeatMap(x_test_recon["mean"], x_test_headers, clusters,
            name = reconstructed_test_set_name)
        
        difference_name = name + "/test_difference"
        plotHeatMap(x_diff, x_test_headers, clusters,
            center = 0, name = difference_name)
        
        log_ratio_name = name + "/test_log_ratio"
        plotHeatMap(x_log_ratio, x_test_headers, clusters,
            center = 0, name = log_ratio_name)
    
    print("Creating latent space scatter plot.")
    plotLatentSpace(latent_set, x_test_headers, clusters, name)
    
    subset = random.randint(N, size = 10)
    for j, i in enumerate(subset):
        
        print("Creating profiles for cell {}.".format(x_test_headers["cells"][i]))
        
        label = labelWithDefaultSymbol("x")
        
        cell_test = x_test[i]
        cell_test_name = name + "/cell_{}_test".format(j)
        plotProfile(cell_test, "Cell", label(), name = cell_test_name)
        
        for variable_name in x_test_recon:
            
            cell_recon = x_test_recon[variable_name][i]
            cell_recon_name = name + "/cell_{}_recon_{}".format(j, variable_name)
            plotProfile(cell_recon, "Cell", label(variable_name), name = cell_recon_name)
            
            if variable_name == "mean":
                cell_diff = x_diff[i]
                cell_diff_name = name + "/cell_{}_diff".format(j)
                plotProfile(cell_diff, "Cell", label() + "$-$" + label(variable_name),
                    name = cell_diff_name)
                    
                cell_log_ratio = x_log_ratio[i]
                cell_log_ratio_name = name + "/cell_{}_log_ratio".format(j)
                plotProfile(cell_log_ratio, "Cell",
                    "$\\log (($" + label() + "$+1)($" + label(variable_name) + "$+1))$",
                    name = cell_log_ratio_name)
    
    print("")
    
    print("Differences: sum: {}, mean: {}, std: {}".format(x_diff.sum(), x_diff.mean(), x_diff.std()))
    print("log-ratios: sum: {}, mean: {}, std: {}".format(x_log_ratio.sum(), x_log_ratio.mean(), x_log_ratio.std()))

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
            "{:<7.3g}".format(statistics_set["minimums"]),
            "{:<9.3g}".format(statistics_set["max"]),
            "{:<7.3g}".format(statistics_set["maximums"]),
            "{:<7.5g}".format(statistics_set["sparsity"]),
        ]
        
        print("  ".join(string_parts))

def plotCountHistogram(data_set, k_min, k_max, name = None):
    
    figure_name = "count_histogram"
    
    if name:
        figure_name = name + "_" + figure_name
    
    k = linspace(0, k_max, k_max + 1)
    C = empty(k_max + 1)
    
    for i in range(k_max + 1):
        if k[i] < k_max:
            c = (data_set == k[i]).sum()
        if k[i] == k_max:
            c = (data_set >= k_max).sum()
        C[i] = c
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    axis.bar(k, C)
    
    axis.set_yscale("log")
    
    axis.set_xlabel("Counts")
    axis.set_ylabel("Number of counts")
    
    data.saveFigure(figure, figure_name)
    
    for i in reversed(range(k_min, k_max + 1)):
        
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        
        if i < k_max:
            C[i] += C[i + 1]
            C[i + 1] = 0

        axis.bar(k[:i+1], C[:i+1])
    
        axis.set_yscale("log")
    
        axis.set_xlabel("Counts")
        axis.set_ylabel("Number of counts")
        
        data.saveFigure(figure, figure_name + "_" + str(i))

def plotProfile(series, x_label, y_label, scale = "linear", bar = False, name = None):
    
    figure_name = "profile"
    
    if name:
        figure_name = name + "_" + figure_name
    
    D = series.shape[0]
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    x = linspace(0, D, D)
    if bar:
        axis.bar(x, series)
    else:
        axis.plot(x, series)
    
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    data.saveFigure(figure, figure_name)

def plotHistogram(series, x_label, scale = "linear", normed=False, name = None):
    
    figure_name = "histogram"
    
    if name:
        figure_name = name + "_" + figure_name
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.distplot(series, kde = False, norm_hist=normed,ax = axis)
    
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_label)
    # axis.set_ylabel(y_label)
    
    data.saveFigure(figure, figure_name)

def plotHeatMap(data_set, data_set_headers = None, clusters = None, center = None,
    simple = False, name = None):
    
    figure_name = "heat_map"
    
    if name:
        figure_name = name + "_" + figure_name
    
    if data_set_headers and clusters:
    
        sorted_data_set = empty(data_set.shape)
    
        N_seen = 0
        for cluster_id, cluster in sorted(clusters.items()):
            
            subset = []
            
            for cell in cluster:
                index = where(data_set_headers["cells"] == cell)[0]
                if len(index) == 0:
                    continue
                subset.append(int(index))
            
            N_subset = len(subset)
            
            if N_subset == 0:
                continue
            
            sorted_data_set[N_seen:(N_seen + N_subset)] = data_set[subset]
        
            N_seen += N_subset
        
        data_set = sorted_data_set[:N_seen]
        figure_name += "_sorted"
    
    N, M = data_set.shape
    
    # figure = pyplot.figure(figsize = (N/500, M/500))
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.heatmap(data_set.T, xticklabels = False, yticklabels = False,
        cbar = True, square = True, center = center, ax = axis)
    
    axis.set_xlabel("Cell")
    axis.set_ylabel("Gene")
    
    data.saveFigure(figure, figure_name, no_spine = False)

def plotLearningCurves(curves, name = None):
    
    print("Plotting learning curves.")
    
    figure_name = "learning_curves"
    
    if name:
        figure_name = name + "/" + figure_name
    
    figure, (axis_1, axis_2) = pyplot.subplots(2, sharex = True, figsize = (6.4, 9.6))
    
    for i, (curve_set_name, curve_set) in enumerate(sorted(curves.items())):
        
        colour = palette[i]
        
        for curve_name, curve in sorted(curve_set.items()):
            if curve_name == "LB":
                curve_name = "$\\mathcal{L}$"
                line_style = "solid"
                axis = axis_1
            elif curve_name == "ENRE":
                curve_name = "$\\log p(x|z)$"
                line_style = "dashed"
                axis = axis_1
            elif curve_name == "KL":
                line_style = "dashed"
                curve_name = "$KL(p||q)$"
                axis = axis_2
            epochs = arange(len(curve)) + 1
            label = curve_name + " ({} set)".format(curve_set_name)
            axis.plot(curve, color = colour, linestyle = line_style, label = label)
    
    handles, labels = axis_1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    axis_1.legend(handles, labels, loc = "best")
    axis_2.legend(loc = "best")
    
    # axis_1.set_xlabel("Epoch")
    axis_2.set_xlabel("Epoch")
    
    data.saveFigure(figure, figure_name)

def plotLatentSpace(latent_set, latent_set_headers = None, clusters = None, name = None):
    
    figure_name = "latent_space"
    
    if name:
        figure_name = name + "/" + figure_name
    
    N, M = latent_set.shape
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if M > 2:
        # TODO Try t-SNE.
        pca = PCA(n_components = 2)
        pca.fit(latent_set)
        latent_set = pca.transform(latent_set)
        
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
    else:
        axis.set_xlabel("z_1")
        axis.set_ylabel("z_2")
    
    for cluster_id in clusters:
        
        cluster = clusters[cluster_id]
        subset = []
    
        for cell in cluster:
            index = where(latent_set_headers["cells"] == cell)[0]
            if len(index) == 0:
                continue
            subset.append(int(index))
        
        if len(subset) == 0:
            continue
        
        axis.scatter(latent_set[subset, 0], latent_set[subset, 1],
            c = data.cluster_colours[cluster_id], edgecolors = None,
            label = cluster_id)
    
    # axis.legend(loc="best")
    
    data.saveFigure(figure, figure_name)

if __name__ == '__main__':
    random.seed(1234)
    
    # data_set = data.createSampleData(1000, 500, p = 0.95)
    # analyseData(data_set, name = "sample")
    
    # data_set, _ = data.loadDataSet("GSE63472_P14Retina_merged_digital_expression")
    # analyseData(data_set, name = "All")
    
    data_name = "GSE63472_P14Retina_merged_digital_expression"
    splitting_method = "Macosko"
    splitting_fraction = 0.8
    feature_selection = None
    feature_size = None
    filtering_method = None
    clusters = None
    
    (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers) = data.loadCountData(data_name,
        splitting_method, splitting_fraction, feature_selection, feature_size,
        filtering_method, clusters)
    
    print("")
    
    data_set_base_name = data.dataSetBaseName(splitting_method, splitting_fraction,
        filtering_method, feature_selection, feature_size)
    
    data_sets = {"training": training_set,
                 "validation": validation_set,
                 "test": test_set}
    
    analyseData(data_sets, name = data_set_base_name)
    