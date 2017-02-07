#!/usr/bin/env python

import gzip
import pickle
import os

from pandas import read_csv
from numpy import random, array, zeros, nonzero, sort, argsort, where, arange
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

def loadCountData(name, splitting_method = "random", splitting_fraction = 0.8,
    feature_selection = None, feature_size = None,
    filtering_method = None, cluster_data = None):
    
    if name == "sample":
        data_set = createSampleData(m = 1000, n = 100, scale = 2, p = 0.8)
        
        index_features = selectFeatureIndices(data_set, feature_selection, feature_size)
        index_train, index_valid, index_test = splitDataSetIndices(data_set,
            splitting_method, splitting_fraction)
        
        training_set = data_set[index_train, index_features]
        validation_set = data_set[index_valid, index_features]
        test_set = data_set[index_test, index_features]
        
        training_headers, validation_headers, test_headers = [None] * 3
    
    else:
        
        (training_set, training_headers), (validation_set, validation_headers), \
            (test_set, test_headers) = loadSplitDataSets(name, splitting_method,
                splitting_fraction, feature_selection, feature_size, filtering_method,
                cluster_data)
    
    return (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers)

def loadClusterData(name):
    
    cluster_path = data_path(name + text_extension)
    
    clusters = {}
    
    print("Loading cluster data from {}.".format(cluster_path))
    
    with open(cluster_path, "r") as cluster_data:
        for line in cluster_data.read().split("\n"):
            
            if line == "":
                continue
                
            cell, cluster_id = line.split("\t")
            
            cluster_id = int(cluster_id)
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append(cell)
    
    print("Cluster data loaded.")
    
    return clusters

def loadSplitDataSets(name, splitting_method = "random", splitting_fraction = 0.8,
    feature_selection = None, feature_size = None,
    filtering_method = None, cluster_data = None):
    
    split_data_sets_name = name
    split_data_sets_name += "_s_" + splitting_method.replace(" ", "_") + "_" + str(splitting_fraction)
    if filtering_method:
        split_data_sets_name += "_f_" + filtering_method[0].replace(" ", "_")
        if filtering_method[0] == "clusters":
            split_data_sets_name += "_" + "_".join(filtering_method[1:])
            filtering_method[1:] = [clusters[int(c)] for c in filtering_method[1:]]
    if feature_selection:
        split_data_sets_name += "_fs_" + feature_selection.replace(" ", "_") + "_" + str(feature_size)
    split_data_sets_path = preprocessed_path(split_data_sets_name +
        zipped_pickle_extension)
    
    if os.path.isfile(split_data_sets_path):
        print("Loading split data sets from {}.".format(split_data_sets_path))
        (training_set, validation_set, test_set), \
            (training_headers, validation_headers, test_headers) = \
            loadSparseData(split_data_sets_path)
        print("Split data sets loaded as " +
              "training ({} examples), ".format(training_set.shape[0]) +
              "validation ({} examples), ".format(validation_set.shape[0]) +
              "and test ({} examples) sets ".format(test_set.shape[0]) +
              "with {} features.".format(training_set.shape[1]))
    else:
        
        data_set, data_headers = loadDataSet(name)
        
        print("Splitting data set and selecting features.")
        
        index_features = selectFeatureIndices(data_set, feature_selection, feature_size)
        index_train, index_valid, index_test = splitDataSetIndices(data_set,
            splitting_method, splitting_fraction, data_headers, filtering_method)
        
        training_set = data_set[index_train][:, index_features]
        validation_set = data_set[index_valid][:, index_features]
        test_set = data_set[index_test][:, index_features]
        
        training_headers = {"cells": data_headers["cells"][index_train],
            "genes": data_headers["genes"][index_features]}
        validation_headers = {"cells": data_headers["cells"][index_valid],
            "genes": data_headers["genes"][index_features]}
        test_headers = {"cells": data_headers["cells"][index_test],
            "genes": data_headers["genes"][index_features]}
        
        print("Data split into training ({} examples), ".format(len(index_train)) +
              "validation ({} examples), ".format(len(index_valid)) +
              "and test ({} examples) sets ".format(len(index_test)) +
              "with {} features.".format(len(index_features)))
        
        print("Saving split data.")
        saveSparseData([training_set, validation_set, test_set],
            [training_headers, validation_headers, test_headers], split_data_sets_path)
        print("Saved split data sets as {}.".format(split_data_sets_path))
    
    return (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers)

def loadDataSet(name):
    
    original_data_path = data_path(name + zipped_text_extension)
    sparse_data_path = preprocessed_path(name + "_sparse" + zipped_pickle_extension)
    
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

def selectFeatureIndices(data, feature_selection = None, feature_size = None):
    
    D = data.shape[1]
    
    if feature_size is None:
        feature_size = D
    
    if feature_selection == "high_variance":
        
        data_variance = data.var(axis = 0)
        index_features_variance_sorted = argsort(data_variance)
        
        index_features = index_features_variance_sorted[-feature_size:]
        
        index_features = sort(index_features)
    
    elif feature_selection is None:
        index_features = arange(D)
    
    return index_features

def splitDataSetIndices(data, splitting_method = "random", splitting_fraction = 0.8,
    headers = None, filtering_method = None):
    
    if filtering_method and filtering_method[0] == splitting_method:
        raise Error("Filtering method and spitting method can't be the same.")
    
    N, D = data.shape
    
    if splitting_method == "random":
        
        V = int(splitting_fraction * N)
        T = int(splitting_fraction * V)
        
        random.shuffle(data)
        
        index_train = arange(T)
        index_valid = arange(T, V)
        index_test = arange(V, N)
        
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
    
    if filtering_method:
        
        if filtering_method[0] == "clusters":
        
            clusters = filtering_method[1:]
            index_examples = set()
            
            for cluster in clusters:
            
                for cell in cluster:
                    index = where(headers["cells"] == cell)[0]
                    if len(index) == 0:
                        continue
                    index_examples.add(int(index))
        
        elif filtering_method[0] == "Macosko":
            
            minimum_genes_expressed = 900
        
            N_non_zero_elements = (data != 0).sum(axis = 1)
        
            index_examples = nonzero(N_non_zero_elements > minimum_genes_expressed)[0]
            index_examples = set(index_examples)
        
        index_train = [i for i in index_train if i in index_examples]
        index_valid = [i for i in index_valid if i in index_examples]
        index_test = [i for i in index_test if i in index_examples]
    
    return index_train, index_valid, index_test

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

def dataSetBaseName(splitting_method, splitting_fraction,
    filtering_method, feature_selection, feature_size):
    
    base_name = "s_" + splitting_method.replace(" ", "_") + "_" \
        +  str(splitting_fraction)
    
    if filtering_method:
        base_name += "_f_" + filtering_method[0].replace(" ", "_")
    
    if feature_selection:
        base_name += "_fs_" + feature_selection.replace(" ", "_") + "_" \
            + str(feature_size)
    
    return base_name

def modelName(base_name, filtering_method, feature_selection,
    feature_size, splitting_method, splitting_fraction,
    reconstruction_distribution, number_of_reconstruction_classes,
    use_count_sum, latent_size, hidden_structure, learning_rate,
    batch_size, number_of_warm_up_epochs, use_batch_norm, use_gpu,
    number_of_epochs):
    
    model_name = base_name + "_" + \
        dataSetBaseName(splitting_method, splitting_fraction,
        filtering_method, feature_selection, feature_size)
    
    model_name += "_r_" + reconstruction_distribution.replace(" ", "_")
    
    if number_of_reconstruction_classes:
        model_name += "_c_" + str(reconstruction_classes)
    
    if use_count_sum:
        model_name += "_sum"
    
    model_name += "_l_" + str(latent_size) + "_h_" + "_".join(map(str,
        hidden_structure))
    
    if use_batch_norm:
        model_name += "_bn"
    
    model_name += "_lr_{:.1g}".format(learning_rate)
    model_name += "_b_" + str(batch_size) + "_wu_" + str(number_of_warm_up_epochs)
    
    if use_gpu:
        model_name += "_gpu"
    
    model_name += "_e_" + str(number_of_epochs)

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
    figure.savefig(figures_path(figure_name + figure_extension), bbox_inches='tight')

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
    clusters = loadClusterData("retina_clusteridentities")
    loadCountData("GSE63472_P14Retina_merged_digital_expression",
        filtering_method = ["Macosko"],
        feature_selection = "high_variance", feature_size = 5000,
        splitting_method = "random", splitting_fraction = 0.8)
