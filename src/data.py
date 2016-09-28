from __future__ import division
import gzip
import pickle
import numpy as np

from aux import script_directory, data_path

def load(file_path, shape=None):
    
    with gzip.open(file_path, 'rb') as data_file:
        data = pickle.load(data_file)
    
    # Transpose gene expression matrix to index 49,300 examples (cells)
    # in rows (i) and 24,658 genes in columns (j).
    data = data.todense().T
    
    print("Nonzero-Fraction:")
    print(np.count_nonzero(data)/data.size)

    # Combine training set of cells (rows, i) expressing more than 900 genes.
    N_non_zero_elements  = (data != 0).sum(1)
    index_train = np.nonzero(N_non_zero_elements > 900)
    index_test = np.nonzero(N_non_zero_elements <= 900)
    print(len(index_train))
    X_train = data[index_train[1], :] 
    X_test = data[index_test[1], :]

    print "Shape of the training set with #genes > 900 (Should be 13,155 cells)"
    print X_train.shape
    print "Shape of the test set with #genes <= 900 (Should be 36,145 cells)"
    print X_test.shape

    # X_train = data[:30000, :]
    # X_valid = data[:, 30000:35000]
    X_valid = None
    # X_test = data[35000:, :]
    # X_test = data[30000:, :]

    # X_train = X_train.reshape(-1, *shape)
    # X_valid = X_valid.reshape(-1, *shape)
    # X_test = X_test.reshape(-1, *shape)
    
    return X_train, X_valid, X_test

if __name__ == '__main__':
    script_directory()
    load(data_path("DGE_matrix_counts_sparse.pkl.gz"))