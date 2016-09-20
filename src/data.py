import gzip
import pickle

from aux import data_path, script_directory

def load(file_path, shape):
    
    with gzip.open(data_path(file_path), 'rb') as data_file:
        data = pickle.load(data_file)
    
    data = data.todense().T
    
    X_train = data[:30000, :]
    # X_valid = data[:, 30000:35000]
    X_valid = None
    X_test = data[35000:, :]
    
    # X_train = X_train.reshape(-1, *shape)
    # X_valid = X_valid.reshape(-1, *shape)
    # X_test = X_test.reshape(-1, *shape)
    
    return X_train, X_valid, X_test

if __name__ == '__main__':
    script_directory()
