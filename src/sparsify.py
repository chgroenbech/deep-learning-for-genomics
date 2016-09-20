#!/usr/bin/env python

from pandas import read_csv
from scipy.sparse import csc_matrix
import pickle
from time import time

data_file_path = data_path("GSE63472_P14Retina_merged_digital_expression.txt.gz")

loading_start = time()

data = read_csv(data_file_path, sep='\s+', index_col = 0,
    compression = "gzip", engine = "python"
)

loading_duration = time() - loading_start
print("Loading took {:3g} min.".format(loading_duration / 60))

saving_start = time()

DGE_matrix_sparse = csc_matrix(data.values)

with open(data_path("DGE_matrix_counts_sparse.pkl"), "wb") as DGE_file:
    pickle.dump(DGE_matrix_sparse, DGE_file)

saving_duration = time() - saving_start
print("Saving took {:3g} min.".format(saving_duration / 60))
