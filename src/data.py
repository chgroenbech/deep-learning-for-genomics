#!/usr/bin/env python

import gzip
import pickle
import os

from pandas import read_csv
from numpy import nonzero, random
from scipy.sparse import csc_matrix

from aux import script_directory, data_path, preprocessed_path, model_path

def loadData(file_name):
    
    original_data_path = data_path(file_name + ".txt.gz")
    sparse_data_path = preprocessed_path(file_name + "_sparse.pkl.gz")
    
    if os.path.isfile(sparse_data_path):
        data = loadSparseData(sparse_data_path)
    else:
        data = loadOriginalData(original_data_path, sparse_data_path)
    
    return data

def loadSampleData(m = 100, n = 20, mean = 1):
    print("Creating sample data.")
    return random.poisson(mean, (m, n))

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
        
    # Combine training set of cells (rows, i) expressing more than 900 genes.
    elif splitting_method == "Macosko":
        
        if splitting_parameter is None:
            splitting_parameter = 900
        
        N_non_zero_elements = (data != 0).sum(1)
        
        index_train = nonzero(N_non_zero_elements > splitting_parameter)[1]
        index_train = None
        index_test = nonzero(N_non_zero_elements <= splitting_parameter)[1]
    
    X_train = data[index_train, :]
    X_valid = data[index_valid, :]
    X_test = data[index_test, :]
    
    print("Data split into training, validation, and test sets.")
    
    return X_train, X_valid, X_test

def loadOriginalData(file_path, sparse_data_path = None):
    
    print("Loading original data.")
    
    data = read_csv(file_path, sep='\s+', index_col = 0,
        compression = "gzip", engine = "python"
    )
    
    print("Original data loaded.")
    
    if sparse_data_path:
        
        print("Saving sparse data.")
        
        data_sparse = csc_matrix(data.values)
        
        with gzip.open(sparse_data_path, "wb") as sparse_data_file:
            pickle.dump(data_sparse, sparse_data_file)
        
        print("Sparse data saved.")
    
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

if __name__ == '__main__':
    script_directory()
    data = loadSampleData(10, 5)
    print(data)
