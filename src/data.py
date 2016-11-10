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

script_directory()

# TODO Rewrite this to load and save preprocessed, split data as necessary.
def loadDataSets(file_name, filtering_method, splitting_method, splitting_fraction,
    feature_selection, feature_size):
    
    # TODO Use real data directory not fixed data path.
    file_name = (os.path.basename(file_name)).split(".")[0]
    
    data_set = loadData(file_name)
    training_set, validation_set, test_set = splitData(data_set,
        example_splitting_method = splitting_method,
        example_splitting_parameter = splitting_fraction,
        feature_splitting_method = feature_selection,
        feature_splitting_parameter = feature_size
    )
    
    return training_set, validation_set, test_set

def loadData(file_name):
    
    original_data_path = data_path(file_name + ".txt.gz")
    sparse_data_path = preprocessed_path(file_name + "_sparse.pkl.gz")
    
    if os.path.isfile(sparse_data_path):
        data = loadSparseData(sparse_data_path)
    else:
        data = loadOriginalData(original_data_path)
        saveDataAsSparseData(data, sparse_data_path)
    
    return data

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

def splitData(data, example_splitting_method = "random", example_splitting_parameter = None, feature_splitting_method = None, feature_splitting_parameter = None):
    
    print("Splitting data.")
    
    # TODO Introduce better way to set (multiple) parameters for splitting methods.
    
    N, D = data.shape
    
    if example_splitting_method == "random":
        
        if example_splitting_parameter is None:
            example_splitting_parameter = 0.8
        
        V = int(example_splitting_parameter * N)
        T = int(example_splitting_parameter * V)
        
        random.shuffle(data)
        
        index_train = slice(T)
        index_valid = slice(T, V)
        index_test = slice(V, N)
        
    # Combine training set of cells (rows, i) expressing more than 900 genes.
    elif example_splitting_method == "Macosko":
        
        if example_splitting_parameter is None:
            example_splitting_parameter = 900
        
        N_non_zero_elements = (data != 0).sum(axis = 1)
        
        index_train = nonzero(N_non_zero_elements > example_splitting_parameter)[0]
        index_test_valid = nonzero(N_non_zero_elements <= example_splitting_parameter)[0]
        
        random.shuffle(index_test_valid)
        
        N_index_test_valid = len(index_test_valid)
        V = int(.2 * N_index_test_valid)
        
        index_valid = index_test_valid[:V]
        index_test = index_test_valid[V:]
    
    # Combine training set of cells (rows, i) expressing more than 900 genes.
    elif example_splitting_method == "Macosko (modified)":
        
        if example_splitting_parameter is None:
            example_splitting_parameter = 900
        
        N_non_zero_elements = (data != 0).sum(axis = 1)
        
        index_examples = nonzero(N_non_zero_elements > example_splitting_parameter)[0]
        
        random.shuffle(index_examples)
        
        N_index_examples = len(index_examples)
        V = int(.8 * N_index_examples)
        T = int(.8 * V)
        
        index_train = index_examples[:T]
        index_valid = index_examples[T:V]
        index_test = index_examples[V:]
    
    if feature_splitting_method is None:
        index_feature = slice(D)
    
    elif feature_splitting_method == "high variance":
        
        if feature_splitting_parameter is None:
            feature_splitting_parameter = 900
        
        data_variance = data.var(axis = 0)
        index_feature_variance_sorted = argsort(data_variance)
        
        index_feature = index_feature_variance_sorted[-feature_splitting_parameter:]
        
        index_feature = sort(index_feature)
    
    X_train = data[index_train, :][:, index_feature]
    X_valid = data[index_valid, :][:, index_feature]
    X_test = data[index_test, :][:, index_feature]
    
    print("Data split into training ({} examples), validation ({} examples), and test ({} examples) sets with {} features.".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1]))
    
    return X_train, X_valid, X_test

def loadOriginalData(file_path, sparse_data_path = None):
    
    print("Loading original data.")
    
    data = read_csv(file_path, sep='\s+', index_col = 0,
        compression = "gzip", engine = "python"
    )
    
    print("Original data loaded.")
    
    return data.values

def loadSparseData(file_path):
    
    print("Loading sparse data.")
    
    with gzip.open(file_path, 'rb') as data_file:
        data_sparse = pickle.load(data_file)
    
    print("Sparse data loaded.")
    
    # Transpose gene expression matrix to index 49,300 examples (cells)
    # in rows (i) and 24,658 genes in columns (j).
    data = data_sparse.todense().T.A
    
    print("Sparse data converted to dense data.")
    
    return data

# TODO Implement separate save function using NumPy saving function.
def saveDataAsSparseData(data, file_path):
    
    print("Saving sparse data.")
    
    data_sparse = csc_matrix(data.values)
    
    with gzip.open(file_path, "wb") as data_file:
        pickle.dump(data_sparse, data_file)
    
    print("Sparse data saved.")

def saveModel(model, model_name):
    
    model_path = models_path(model_name + ".pkl.gz")
    
    print("Saving model.")
    
    with gzip.open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    
    print("Model saved in {}.".format(model_path))

def loadModel(model_name):
    
    model_path = models_path(model_name + ".pkl.gz")
    
    print("Loading model.")
    
    with gzip.open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    
    print("Model loaded.")
    
    return model

def saveFigure(figure, figure_name, no_spine = True):
    
    if no_spine:
        despine()
    figure.savefig(figures_path(figure_name + figure_extension))

if __name__ == '__main__':
    script_directory()
    data = createSampleData(10, 5)
    X_train, X_valid, X_test = splitData(data, example_splitting_method = "Macosko (modified)", example_splitting_parameter = 2, feature_splitting_method = "high variance", feature_splitting_parameter = 2)
    print(data)
    print(X_train)
    print(X_valid)
    print(X_test)
