#!/usr/bin/env python

import gzip
import pickle
import os

from pandas import read_csv
from numpy import nonzero, random, zeros
from scipy.sparse import csc_matrix

from seaborn import despine
figure_extension = ".png"

from aux import (
    script_directory, data_path, preprocessed_path, model_path, figure_path
)

script_directory()

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

def splitData(data, splitting_method = "random", splitting_parameter = None):
    
    print("Splitting data.")
    
    N, D = data.shape
    
    if splitting_method == "random":
        
        if splitting_parameter is None:
            splitting_parameter = 0.8
        
        V = int(splitting_parameter * N)
        T = int(splitting_parameter * V)
        
        random.shuffle(data)
        
        index_train = slice(T)
        index_valid = slice(T, V)
        index_test = slice(V, N)
        
        index_feature = slice(D)
        
    # Combine training set of cells (rows, i) expressing more than 900 genes.
    elif splitting_method == "Macosko":
        
        if splitting_parameter is None:
            splitting_parameter = 900
        
        N_non_zero_elements = (data != 0).sum(axis = 1)
        
        index_train = nonzero(N_non_zero_elements > splitting_parameter)[0]
        index_test_valid = nonzero(N_non_zero_elements <= splitting_parameter)[0]
        
        random.shuffle(index_test_valid)
        
        N_index_test_valid = len(index_test_valid)
        V = int(.2 * N_index_test_valid)
        
        index_valid = index_test_valid[:V]
        index_test = index_test_valid[V:]
        
        index_feature = slice(D)
    
    X_train = data[index_train, index_feature]
    X_valid = data[index_valid, index_feature]
    X_test = data[index_test, index_feature]
    
    print("Data split into training, validation, and test sets.")
    
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
    data = data_sparse.todense().T
    
    print("Sparse data converted to dense data.")
    
    return data

def saveDataAsSparseData(data, file_path):
    
    print("Saving sparse data.")
    
    data_sparse = csc_matrix(data.values)
    
    with gzip.open(file_path, "wb") as data_file:
        pickle.dump(data_sparse, data_file)
    
    print("Sparse data saved.")

def saveModelParameters(parameter_value_sets, model_name):
    
    model_parameters_path = model_path(model_name + ".pkl.gz")
    
    print("Saving model parameters.")
    
    with gzip.open(model_parameters_path, "wb") as model_parameters_file:
        pickle.dump(parameter_value_sets, model_parameters_file)
    
    print("Model parameters saved in {}.".format(model_parameters_path))

def loadModelParameters(model_name):
    
    model_parameters_path = model_path(model_name + ".pkl.gz")
    
    print("Loading model parameters.")
    
    with gzip.open(model_parameters_path, "rb") as model_parameters_file:
        parameter_value_sets = pickle.load(model_parameters_file)
    
    print("Model parameters loaded.")
    
    return parameter_value_sets

def saveFigure(figure, figure_name, no_spine = True):
    
    if no_spine:
        despine()
    figure.savefig(figure_path(figure_name + figure_extension))

if __name__ == '__main__':
    script_directory()
    data = createSampleData(10, 5)
    X_train, X_valid, X_test = splitData(data, splitting_method = "Macosko", splitting_parameter = 3)
    print(data.shape)
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
