#!/usr/bin/env python

import gzip
import pickle
import os

from pandas import read_csv
from numpy import nonzero, random, zeros, sort, argsort
from scipy.sparse import csc_matrix

from seaborn import despine
figure_extension = ".png"

from aux import (
    script_directory, data_path, preprocessed_path, models_path, figures_path
)

zipped_text_extension = ".txt.gz"
zipped_pickle_extension = ".pkl.gz"

script_directory()

def loadData(name, filtering_method = None, feature_selection = None,
    feature_size = None, splitting_method = "random", splitting_fraction = 0.8):
    
    if filtering_method == splitting_method:
        raise Error("Splitting and filtering method cannot be the same.")
    
    if name == "sample":
        data_set = createSampleData()
        if filtering_method:
            data_set = filterExamples(data_set, filtering_method)
        if feature_selection:
            data_set = selectFeatures(data_set, feature_selection, feature_size)
        training_set, validation_set, test_set = splitDataSet(data_set,
            splitting_method, splitting_fraction)
    
    else:
        
        # data_set_name = os.path.basename(data_set_name).split(".")[0]
        
        training_set, validation_set, test_set = loadSplitDataSets(name,
            filtering_method, feature_selection, feature_size, splitting_method,
            splitting_fraction)
    
    return training_set, validation_set, test_set

def loadSplitDataSets(name, filtering_method, feature_selection, feature_size,
    splitting_method, splitting_fraction):
    
    split_data_sets_name = name
    if filtering_method:
        split_data_sets_name += "_f_" + filtering_method
    if feature_selection:
        split_data_sets_name += "_fs_" + feature_selection + "_" + str(feature_size)
    split_data_sets_name += "_s_" + splitting_method + "_" + str(splitting_fraction)
    split_data_sets_path = preprocessed_path(split_data_sets_name +
        zipped_pickle_extension)
    
    if os.path.isfile(split_data_sets_path):
        print("Loading split data sets from {}.".format(split_data_sets_path))
        training_set, validation_set, test_set = loadSparseData(split_data_sets_path)
        print("Split data sets loaded.")
    else:
        if feature_selection:
            data_set = loadFeatureSelectedDataSet(name, filtering_method,
                feature_selection, feature_size)
        elif filtering_method:
            data_set = loadFilteredDataSet(name, filtering_method)
        else:
            data_set = loadDataSet(name)
        training_set, validation_set, test_set = splitDataSet(data_set, splitting_method,
            splitting_fraction)
        print("Saving split data.")
        saveSparseData([training_set, validation_set, test_set], split_data_sets_path)
        print("Saved split data sets as {}.".format(split_data_sets_path))
    
    return training_set, validation_set, test_set

def loadFeatureSelectedDataSet(name, filtering_method, feature_selection, feature_size):
    
    feature_selected_data_set_name = name
    if filtering_method:
        feature_selected_data_set_name += "_f_" + filtering_method
    feature_selected_data_set_name += "_fs_" + feature_selection + "_" + str(feature_size)
    feature_selected_data_set_path = preprocessed_path(feature_selected_data_set_name +
        zipped_pickle_extension)
    
    if os.path.isfile(feature_selected_data_set_path):
        print("Loading feature selected data sets from {}.".format(feature_selected_data_set_path))
        feature_selected_data_set = loadSparseData(feature_selected_data_set_path)
        print("Feature selected data set loaded.")
    else:
        if filtering_method:
            data_set = loadFilteredDataSet(name, filtering_method)
        else:
            data_set = loadDataSet(name)
        feature_selected_data_set = selectFeatures(data_set, feature_selection,
            feature_size)
        print("Saving feature selected data set.")
        saveSparseData(feature_selected_data_set, feature_selected_data_set_path)
        print("Saved feature selected data set as {}.".format(feature_selected_data_set_path))
    
    return feature_selected_data_set

def loadFilteredDataSet(name, filtering_method):
    
    filtered_data_set_name = name + "_f_" + filtering_method
    filtered_data_set_path = preprocessed_path(filtered_data_set_name +
        zipped_pickle_extension)
    
    if os.path.isfile(filtered_data_set_path):
        print("Loading filtered data set from {}.".format(filtered_data_set_path))
        filtered_data_set = loadSparseData(filtered_data_set_path)
        print("Filtered data set loaded.")
    else:
        data_set = loadDataSet(name)
        type(data_set)
        filtered_data_set = filterExamples(data_set, filtering_method)
        print("Saving filtered data set.")
        saveSparseData(filtered_data_set, filtered_data_set_path)
        print("Saved filtered data set as {}.".format(filtered_data_set_path))
    
    return filtered_data_set

def loadDataSet(name):
    
    original_data_path = data_path(name + zipped_text_extension)
    sparse_data_path = preprocessed_path(name + "_sparse" +
        zipped_pickle_extension)
    
    if os.path.isfile(sparse_data_path):
        print("Loading original data set in sparse representation from {}.".format(sparse_data_path))
        data_set = loadSparseData(sparse_data_path)
        data_set = data_set.T
        print("Original data set loaded.")
    else:
        print("Loading original data set from {}.".format(original_data_path))
        data_set = loadOriginalData(original_data_path)
        print("Original data set loaded.")
        print("Saving original data set in sparse representation as {}.".format(sparse_data_path))
        saveSparseData(data, sparse_data_path)
        print("Original data set in sparse representation saved.")
    
    return data_set

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

def filterExamples(data, filtering_method = None):
    
    print("Filtering examples.")
    
    N = data.shape[0]
    
    if filtering_method == "Macosko":
        
        minimum_genes_expressed = 900
        
        N_non_zero_elements = (data != 0).sum(axis = 1)
        
        index_examples = nonzero(N_non_zero_elements > minimum_genes_expressed)[0]
    
    elif filtering_method is None:
        index_examples = slice(N)
    
    print("Filtered examples with {} remaining.".format(len(index_examples)))
    
    return data[index_examples, :]

def selectFeatures(data, feature_selection = None, feature_size = None):
    
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
    
    return data[:, index_feature]

def splitDataSet(data, splitting_method, splitting_fraction):
    
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
    
    X_train = data[index_train, :]
    X_valid = data[index_valid, :]
    X_test = data[index_test, :]
    
    print("Data split into training ({} examples), validation ({} examples), and test ({} examples) sets.".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
    
    return X_train, X_valid, X_test

def loadOriginalData(file_path):
    
    data = read_csv(file_path, sep='\s+', index_col = 0,
        compression = "gzip", engine = "python"
    )
    
    return data.values

def loadSparseData(file_path):
    
    with gzip.open(file_path, 'rb') as data_file:
        sparse_data = pickle.load(data_file)
    
    if type(sparse_data) == list:
        data = []
        for sparse_data_set in sparse_data:
            data_set = sparse_data_set.todense().A
            data.append(data_set)
    else:
        data = sparse_data.todense().A
    
    return data

def saveSparseData(data, file_path):
    
    if type(data) != list:
        sparse_data = csc_matrix(data)
    else:
        sparse_data = []
        for data_set in data:
            sparse_data_set = csc_matrix(data_set)
            sparse_data.append(sparse_data_set)
    
    with gzip.open(file_path, "wb") as data_file:
        pickle.dump(sparse_data, data_file)

def modelName(base_name, filtering_method, feature_selection, feature_size,
    splitting_method, splitting_fraction, latent_size, hidden_structure,
    batch_size, number_of_epochs):
    
    model_name = base_name
    
    if filtering_method:
        model_name += "_f_" + filtering_method.replace(" ", "_")
    
    if feature_selection:
        model_name += "_fs_" + feature_selection.replace(" ", "_") + "_" \
            + str(feature_size)
    
    model_name += "_s_" + splitting_method.replace(" ", "_") + "_" \
        + str(splitting_fraction)
    
    model_name += "_l_" + str(latent_size) + "_h_" + "_".join(map(str,
        hidden_structure))
    
    model_name += "_b_" + str(batch_size) + "_e_" + str(number_of_epochs)
    
    return model_name

def saveModel(model, model_name):
    
    model_path = models_path(model_name + zipped_pickle_extension)
    
    print("Saving model parameters and metadata.")
    
    with gzip.open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    
    print("Model parameters and metadata saved in {}.".format(model_path))

def loadModel(model_name):
    
    model_path = models_path(model_name + zipped_pickle_extension)
    
    print("Loading model parameters and metadata.")
    
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

if __name__ == '__main__':
    script_directory()
    number_of_epochs = 20
    model_name = modelName("vae", "Macosko", "high_variance",
        5000, "random", 0.8, 50, [500], 100, number_of_epochs)
    previous_model_name, epochs_still_to_train = \
        findPreviouslyTrainedModel(model_name, number_of_epochs)
    if previous_model_name:
        print(previous_model_name, epochs_still_to_train)
