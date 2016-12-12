#!/usr/bin/env python

import gzip
import pickle
import os

from pandas import read_csv
from numpy import random, array, zeros, nonzero, sort, argsort, where
from scipy.sparse import csr_matrix

from seaborn import despine
figure_extension = ".png"

from aux import (
    script_directory, data_path, preprocessed_path, models_path, figures_path
)

text_extension = ".txt"
zipped_text_extension = text_extension + ".gz"
zipped_pickle_extension = ".pkl.gz"

script_directory()

def loadCountData(name, filtering_method = None, clusters = None, feature_selection = None,
    feature_size = None, splitting_method = "random", splitting_fraction = 0.8):
    
    if filtering_method:
        if filtering_method[0] == splitting_method:
            raise Error("Splitting and filtering method cannot be the same.")
        if filtering_method[0] == "clusters":
            filtering_method[1:] = [clusters[int(c)] for c in filtering_method[1:]]
    
    if name == "sample":
        data_set = createSampleData()
        if filtering_method:
            data_set = filterExamples(data_set, filtering_method = filtering_method)
        if feature_selection:
            data_set = selectFeatures(data_set, feature_selection = feature_selection,
                feature_size = feature_size)
        training_set, validation_set, test_set = splitDataSet(data_set,
            splitting_method = splitting_method, splitting_fraction = splitting_fraction)
        
        training_headers, validation_headers, test_headers = None, None, None
    
    else:
        
        (training_set, training_headers), (validation_set, validation_headers), \
            (test_set, test_headers) = loadSplitDataSets(name, filtering_method,
            feature_selection, feature_size, splitting_method, splitting_fraction)
    
    return (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers)

def loadClusterData(name):
    
    cluster_path = data_path(name + text_extension)
    
    clusters = {}
    # cluster_ids = {}
    
    print("Loading cluster data from {}.".format(cluster_path))
    
    with open(cluster_path, "r") as cluster_data:
        for line in cluster_data.read().split("\n"):
            
            if line == "":
                continue
                
            cell, cluster_id = line.split("\t")
            # cluster_ids[cell] = cluster_id
            
            cluster_id = int(cluster_id)
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append(cell)
    
    print("Cluster data loaded.")
    
    return clusters

def loadSplitDataSets(name, filtering_method, feature_selection, feature_size,
    splitting_method, splitting_fraction):
    
    split_data_sets_name = name
    if filtering_method:
        split_data_sets_name += "_f_" + filtering_method[0].replace(" ", "_")
    if feature_selection:
        split_data_sets_name += "_fs_" + feature_selection.replace(" ", "_") + "_" + str(feature_size)
    split_data_sets_name += "_s_" + splitting_method.replace(" ", "_") + "_" + str(splitting_fraction)
    split_data_sets_path = preprocessed_path(split_data_sets_name +
        zipped_pickle_extension)
    
    if os.path.isfile(split_data_sets_path):
        print("Loading split data sets from {}.".format(split_data_sets_path))
        (training_set, validation_set, test_set), \
            (training_headers, validation_headers, test_headers) = \
            loadSparseData(split_data_sets_path)
        print("Split data sets loaded.")
    else:
        if feature_selection:
            data_set, data_headers = loadFeatureSelectedDataSet(name, filtering_method,
                feature_selection, feature_size)
        elif filtering_method:
            data_set, data_headers = loadFilteredDataSet(name, filtering_method)
        else:
            data_set, data_headers = loadDataSet(name)
        (training_set, training_headers), (validation_set, validation_headers), \
            (test_set, test_headers) = splitDataSet(data_set, data_headers,
            splitting_method, splitting_fraction)
        
        print("Saving split data.")
        saveSparseData([training_set, validation_set, test_set],
            [training_headers, validation_headers, test_headers], split_data_sets_path)
        print("Saved split data sets as {}.".format(split_data_sets_path))
    
    return (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers)

def loadFeatureSelectedDataSet(name, filtering_method, feature_selection, feature_size):
    
    feature_selected_data_set_name = name
    if filtering_method:
        feature_selected_data_set_name += "_f_" + filtering_method[0].replace(" ", "_")
    feature_selected_data_set_name += "_fs_" + feature_selection.replace(" ", "_") + "_" + \
        str(feature_size)
    feature_selected_data_set_path = \
        preprocessed_path(feature_selected_data_set_name + zipped_pickle_extension)
    
    if os.path.isfile(feature_selected_data_set_path):
        print("Loading feature selected data sets from {}.".format(feature_selected_data_set_path))
        feature_selected_data_set, feature_selected_data_headers = \
            loadSparseData(feature_selected_data_set_path)
        print("Feature selected data set loaded.")
    else:
        if filtering_method:
            data_set, data_headers = loadFilteredDataSet(name, filtering_method)
        else:
            data_set, data_headers = loadDataSet(name)
        feature_selected_data_set, feature_selected_data_headers = \
            selectFeatures(data_set, data_headers, feature_selection, feature_size)
        print("Saving feature selected data set.")
        saveSparseData(feature_selected_data_set, feature_selected_data_headers,
            feature_selected_data_set_path)
        print("Saved feature selected data set as {}.".format(feature_selected_data_set_path))
    
    return feature_selected_data_set, feature_selected_data_headers

def loadFilteredDataSet(name, filtering_method):
    
    filtered_data_set_name = name + "_f_" + filtering_method[0].replace(" ", "_")
    filtered_data_set_path = preprocessed_path(filtered_data_set_name +
        zipped_pickle_extension)
    
    if os.path.isfile(filtered_data_set_path):
        print("Loading filtered data set from {}.".format(filtered_data_set_path))
        filtered_data_set, filtered_data_headers = \
            loadSparseData(filtered_data_set_path)
        print("Filtered data set loaded.")
    else:
        data_set, data_headers = loadDataSet(name)
        filtered_data_set, filtered_data_headers = filterExamples(data_set,
            data_headers, filtering_method)
        print("Saving filtered data set.")
        saveSparseData(filtered_data_set, filtered_data_headers,
            filtered_data_set_path)
        print("Saved filtered data set as {}.".format(filtered_data_set_path))
    
    return filtered_data_set, filtered_data_headers

def loadDataSet(name):
    
    original_data_path = data_path(name + zipped_text_extension)
    sparse_data_path = preprocessed_path(name + "_sparse" +
        zipped_pickle_extension)
    
    if os.path.isfile(sparse_data_path):
        print("Loading original data set in sparse representation from {}.".format(sparse_data_path))
        data_set, data_headers = loadSparseData(sparse_data_path)
        print("Original data set loaded.")
    else:
        print("Loading original data set from {}.".format(original_data_path))
        data_set, data_headers = loadOriginalData(original_data_path)
        print("Original data set loaded.")
        print("Saving original data set in sparse representation as {}.".format(sparse_data_path))
        saveSparseData(data_set, data_headers, sparse_data_path)
        print("Original data set in sparse representation saved.")
    
    return data_set, data_headers

def createSampleData(m = 100, n = 20, scale = 2, p = 0.5):
    
    print("Creating sample data.")
    
    data = zeros((m, n))
    
    row = scale * random.rand(n)
    k = 0
    for i in range(m):
        u = random.rand()
        if u > p:
            row = scale * random.rand(n)
            k += 1
        data[i] = row
    
    random.shuffle(data)
    
    for i in range(m):
        for j in range(n):
            data[i, j] = random.poisson(data[i, j])
    
    print("Sample data created with {} different cell types.".format(k))
    
    return data

def filterExamples(data, headers = None, filtering_method = None):
    
    print("Filtering examples.")
    
    N = data.shape[0]
    
    if filtering_method[0] == "Macosko":
        
        minimum_genes_expressed = 900
        
        N_non_zero_elements = (data != 0).sum(axis = 1)
        
        index_examples = nonzero(N_non_zero_elements > minimum_genes_expressed)[0]
    
    elif filtering_method[0] == "clusters":
        
        index_examples = []
            
        for cluster in filtering_method[1:]:
            
            for cell in cluster:
                index = where(headers["cells"] == cell)[0]
                if len(index) == 0:
                    continue
                index_examples.append(int(index))
    
    elif filtering_method is None:
        index_examples = slice(N)
    
    print("Filtered examples with {} remaining.".format(len(index_examples)))
    
    if headers:
        headers["cells"] = headers["cells"][index_examples]
        return data[index_examples], headers
    else:
        return data[index_examples]

def selectFeatures(data, headers = None, feature_selection = None, feature_size = None):
    
    print("Selecting features.")
    
    D = data.shape[1]
    
    if feature_size is None:
        feature_size = D
    
    if feature_selection == "high_variance":
        
        data_variance = data.var(axis = 0)
        index_feature_variance_sorted = argsort(data_variance)
        
        index_feature = index_feature_variance_sorted[-feature_size:]
        
        index_feature = sort(index_feature)
    
    elif feature_selection is None:
        index_feature = slice(D)
    
    print("Filtered features with {} remaining.".format(len(index_feature)))
    
    if headers:
        headers["genes"] = headers["genes"][index_feature]
        return data[:, index_feature], headers
    else:
        return data[:, index_feature]

def splitDataSet(data, headers = None, splitting_method = "random",
    splitting_fraction = 0.8):
    
    print("Splitting data set.")
    
    N, D = data.shape
    
    if splitting_method == "random":
        
        V = int(splitting_fraction * N)
        T = int(splitting_fraction * V)
        
        random.shuffle(data)
        
        index_train = slice(T)
        index_valid = slice(T, V)
        index_test = slice(V, N)
        
    # Combine training set of cells (rows, i) expressing more than 900 genes.
    elif splitting_method == "Macosko":
        
        minimum_genes_expressed = 900
        
        N_non_zero_elements = (data != 0).sum(axis = 1)
        
        index_train = nonzero(N_non_zero_elements > minimum_genes_expressed)[0]
        
        index_test_valid = nonzero(N_non_zero_elements <= minimum_genes_expressed)[0]
        
        random.shuffle(index_test_valid)
        
        N_index_test_valid = len(index_test_valid)
        V = int((1 - splitting_fraction) * N_index_test_valid)
        
        index_valid = index_test_valid[:V]
        index_test = index_test_valid[V:]
    
    print(splitting_method)
    
    training_set = data[index_train]
    validation_set = data[index_valid]
    test_set = data[index_test]
    
    print("Data split into training ({} examples), validation ({} examples), and test ({} examples) sets.".format(training_set.shape[0], validation_set.shape[0], test_set.shape[0]))
    
    if headers:
        training_headers = {"cells": headers["cells"][index_train],
            "genes": headers["genes"]}
        validation_headers = {"cells": headers["cells"][index_valid],
            "genes": headers["genes"]}
        test_headers = {"cells": headers["cells"][index_test],
            "genes": headers["genes"]}
        return (training_set, training_headers), (validation_set, validation_headers), \
            (test_set, test_headers)
    else:
        return training_set, validation_set, test_set

def loadOriginalData(file_path):
    
    data = read_csv(file_path, sep='\s+', index_col = 0,
        compression = "gzip", engine = "python"
    )
    
    data_set = data.values.T
    
    cell_headers = array(data.columns.tolist())
    gene_headers = array(data.index.tolist())
    
    data_headers = {"cells": cell_headers, "genes": gene_headers}
    
    return data_set, data_headers

def loadSparseData(file_path):
    
    converter = lambda sparse_data: sparse_data.todense().A
    
    with gzip.open(file_path, 'rb') as data_file:
        sparse_data = pickle.load(data_file)
        headers = pickle.load(data_file)
    
    if type(sparse_data) == list:
        data = []
        for sparse_data_set in sparse_data:
            data_set = converter(sparse_data_set)
            data.append(data_set)
    else:
        data = converter(sparse_data)
    
    return data, headers

def saveSparseData(data, headers, file_path):
    
    converter = lambda data: csr_matrix(data)
    
    if type(data) != list:
        sparse_data = converter(data)
    else:
        sparse_data = []
        for data_set in data:
            sparse_data_set = converter(data_set)
            sparse_data.append(sparse_data_set)
    
    with gzip.open(file_path, "wb") as data_file:
        pickle.dump(sparse_data, data_file)
        pickle.dump(headers, data_file)

def modelName(base_name, filtering_method, feature_selection, feature_size,
    splitting_method, splitting_fraction, reconstruction_distribution, latent_size,
    hidden_structure, learning_rate, batch_size, number_of_epochs):
    
    model_name = base_name
    
    if filtering_method:
        model_name += "_f_" + filtering_method[0].replace(" ", "_")
    
    if feature_selection:
        model_name += "_fs_" + feature_selection.replace(" ", "_") + "_" \
            + str(feature_size)
    
    model_name += "_s_" + splitting_method.replace(" ", "_") + "_" \
        + str(splitting_fraction)
    
    model_name += "_r_" + reconstruction_distribution.replace(" ", "_")
    
    model_name += "_l_" + str(latent_size) + "_h_" + "_".join(map(str,
        hidden_structure))
    
    model_name += "_lr_{:.1g}".format(learning_rate) 
    model_name += "_b_" + str(batch_size) + "_e_" + str(number_of_epochs)
    
    return model_name

def saveModel(model, model_name):
    
    model_path = models_path(model_name + zipped_pickle_extension)
    
    print("Saving model parameters and metadata.")
    
    with gzip.open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    
    print("Model parameters and metadata saved in {}.".format(model_path))

def loadModel(model_name):
    
    epoch = model_name.split("_e_")[-1]
    
    model_path = models_path(model_name + zipped_pickle_extension)
    
    print("Loading model parameters for epoch {} and metadata.".format(epoch))
    
    with gzip.open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    
    print("Model parameters and metadata loaded.")
    
    return model

def findPreviouslyTrainedModel(model_name):
    
    model_path = models_path(model_name + zipped_pickle_extension)
    
    if os.path.isfile(model_path):
        return model_name, 0
    
    epoch_string = "_e_"
    
    base_name, number_of_epochs = model_name.split(epoch_string)
    number_of_epochs = int(number_of_epochs)
    
    previous_model_names = [m.replace(zipped_pickle_extension, "") for m in
        os.listdir(models_path()) if base_name in m]
    previous_numbers_of_epochs = [int(m.split(epoch_string)[-1]) for m in
        previous_model_names]
    previous_numbers_of_epochs = [e for e in previous_numbers_of_epochs if e <= number_of_epochs]
    previous_model_names = previous_model_names[:len(previous_numbers_of_epochs)]
    
    if len(previous_model_names) == 0:
        return None, number_of_epochs
    
    previous_model_epochs_trained, previous_model_name = \
        sorted(zip(previous_numbers_of_epochs, previous_model_names))[-1]
    
    epochs_still_to_train = number_of_epochs - previous_model_epochs_trained
    
    return previous_model_name, epochs_still_to_train

def modelTrained(model_name):
    
    model_path = models_path(model_name + zipped_pickle_extension)
    
    return os.path.isfile(model_path)

def saveFigure(figure, figure_name, no_spine = True):
    
    if no_spine:
        despine()
    figure.savefig(figures_path(figure_name + figure_extension))

cluster_colours = {
     1: (0.92, 0.24, 0.10),
     2: (0.89, 0.60, 0.14),
     3: (0.78, 0.71, 0.18),
     4: (0.80, 0.74, 0.16),
     5: (0.79, 0.76, 0.16),
     6: (0.81, 0.80, 0.18),
     7: (0.77, 0.79, 0.11),
     8: (0.77, 0.80, 0.16),
     9: (0.73, 0.78, 0.14),
    10: (0.71, 0.79, 0.15),
    11: (0.68, 0.78, 0.20),
    12: (0.65, 0.78, 0.15),
    13: (0.63, 0.79, 0.12),
    14: (0.63, 0.80, 0.17),
    15: (0.61, 0.78, 0.16),
    16: (0.57, 0.78, 0.14),
    17: (0.55, 0.78, 0.16),
    18: (0.53, 0.79, 0.14),
    19: (0.52, 0.80, 0.16),
    20: (0.47, 0.80, 0.17),
    21: (0.44, 0.80, 0.13),
    22: (0.42, 0.80, 0.16),
    23: (0.42, 0.79, 0.13),
    24: (0.12, 0.79, 0.72),
    25: (0.13, 0.64, 0.79),
    26: (0.00, 0.23, 0.88),
    27: (0.00, 0.24, 0.90),
    28: (0.13, 0.23, 0.89),
    29: (0.22, 0.23, 0.90),
    30: (0.33, 0.22, 0.87),
    31: (0.42, 0.23, 0.89),
    32: (0.53, 0.22, 0.87),
    33: (0.59, 0.24, 0.93),
    34: (0.74, 0.14, 0.67),
    35: (0.71, 0.13, 0.62),
    36: (0.74, 0.09, 0.55),
    37: (0.74, 0.08, 0.50),
    38: (0.73, 0.06, 0.44),
    39: (0.74, 0.06, 0.38),
}

if __name__ == '__main__':
    script_directory()
