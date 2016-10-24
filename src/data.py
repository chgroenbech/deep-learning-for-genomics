#!/usr/bin/env python

import gzip
import pickle
import os

from pandas import read_csv
from numpy import nonzero
from scipy.sparse import csc_matrix

from aux import script_directory, data_path, preprocessed_path

def load(file_name, splitting_method = "random", splitting_parameter = None):
    
    original_data_path = data_path(file_name + ".txt.gz")
    sparse_data_path = preprocessed_path(file_name + "_sparse.pkl.gz")
    
    if os.path.isfile(sparse_data_path):
        data = loadSparseData(sparse_data_path)
    else:
        data = loadOriginalData(original_data_path, sparse_data_path)
    
    N, D = data.shape
    
    if splitting_method == "random":
        
        if splitting_parameter is None:
            splitting_parameter = 0.8
        
        T = int(splitting_parameter * N)
        
        # numpy.random.shuffle(data)
        
        index_train = range(T)
        index_test = range(T, N)
        
    # Combine training set of cells (rows, i) expressing more than 900 genes.
    elif splitting_method == "Macosko":
        
        if splitting_parameter is None:
            splitting_parameter = 900
        
        N_non_zero_elements = (data != 0).sum(1)
        
        index_train = nonzero(N_non_zero_elements > splitting_parameter)[1]
        index_test = nonzero(N_non_zero_elements <= splitting_parameter)[1]
    
    X_train = data[index_train, :] 
    X_valid = None
    X_test = data[index_test, :]
    
    print("Data split into training and test sets.")
    
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

if __name__ == '__main__':
    script_directory()
    # load("GSE63472_P14Retina_merged_digital_expression")
    load("GSE63472_P14Retina_logDGE")
