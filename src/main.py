#!/usr/bin/env python

import data

import modeling

from aux import script_directory

def main():
    
    latent_size = 500
    hidden_structure = [128]
    
    file_name = "GSE63472_P14Retina_logDGE"
    
    # data_set = data.loadData(file_name)
    data_set = data.loadSampleData(500, 100)
    
    training_set, validation_set, test_set = data.splitData(data_set)
    
    feature_shape = training_set.shape[1]
    
    model = modeling.VAE(feature_shape, latent_size, hidden_structure)
    
    model.train(training_set, validation_set)
    
    # model.save()
    
    # results = model.evaluate(test_set)
    #
    # analyse(results)

if __name__ == '__main__':
    script_directory()
    main()
