#!/usr/bin/env python

import data
import modeling
import analysis

from numpy import random

def main():
    
    latent_size = 50
    hidden_structure = [128]
    
    file_name = "GSE63472_P14Retina_merged_digital_expression"
    
    random.seed(42)
    
    # data_set = data.loadData(file_name)
    data_set = data.createSampleData(1000, 500, p = .95)
    
    analysis.analyseData(data_set, name = "sample")
    
    training_set, validation_set, test_set = data.splitData(data_set)
    
    feature_shape = data_set.shape[1]
    
    model = modeling.VAE(feature_shape, latent_size, hidden_structure)
    
    model.train(training_set, validation_set, N_epochs = 50)
    
    model.save(name = "test")
    
    reconstructed_test_set, sample_set, test_metrics = model.evaluate(test_set)
    print(test_metrics)
    
    model.load(model_name = "test")
    
    reconstructed_test_set, sample_set, test_metrics = model.evaluate(test_set)
    print(test_metrics)
    
    analysis.analyseModel(test_set, reconstructed_test_set, sample_set)

if __name__ == '__main__':
    main()
