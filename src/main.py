#!/usr/bin/env python

import data
import modeling
import analysis

import argparse

from numpy import random
from itertools import product

def main(data_name, cluster_name, splitting_method = "random", splitting_fraction = 0.8,
    filtering_method = None, feature_selection = None, feature_size = None,
    latent_sizes = None, hidden_structure = None, reconstruction_distributions = None, 
    numbers_of_reconstruction_classes = [0], use_count_sum = False, numbers_of_epochs = 10, batch_size = 100,
    learning_rate = 1e-3, force_training = False):
    
    random.seed(42)
    
    # Data
    
    clusters = data.loadClusterData(cluster_name)
    
    (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers) = data.loadCountData(data_name,
        splitting_method, splitting_fraction, feature_selection, feature_size,
        filtering_method, clusters)
    
    print("")
    
    analysis_name = "s_" + splitting_method.replace(" ", "_") + "_" \
        + str(splitting_fraction)
    
    if filtering_method:
        analysis_name += "_f_" + filtering_method[0].replace(" ", "_")
    
    if feature_selection:
        analysis_name += "_fs_" + feature_selection.replace(" ", "_") + "_" \
            + str(feature_size)
    
    data_sets = {"training": training_set,
                 "validation": validation_set,
                 "test": test_set}
    
    for name, data_set in data_sets.items():
        analysis.analyseData(data_set, analysis_name + "_" + name)
    analysis.analyseData(data_sets.values(), name = analysis_name + "_" + "united")
    
    metadata = {
        "filtering method": filtering_method,
        "splitting method": splitting_method,
        "splitting fraction": splitting_fraction,
        "feature selection": feature_selection,
        "feature size": training_set.shape[1],
        "training size": training_set.shape[0],
        "validation size": validation_set.shape[0],
        "test size": test_set.shape[0]
    }
    
    print("")
    
    # Loop
    
    feature_size = training_set.shape[1]
    
    if not hidden_structure:
        hidden_structure = [feature_size / 10]
    
    if not latent_sizes:
        latent_sizes = [feature_size / 100]
    
    for latent_size, reconstruction_distribution, number_of_reconstruction_classes, \
        number_of_epochs in product(latent_sizes, reconstruction_distributions, \
        numbers_of_reconstruction_classes, numbers_of_epochs):
        
        if reconstruction_distribution == "bernoulli":
            if use_count_sum:
                print("Can't use count sum with Bernoulli distribution.\n")
                continue
            if number_of_reconstruction_classes > 0:
                print("Can't use reconstruction classification with Bernoulli distribution.\n")
                continue
        
        if "zero_inflated" in reconstruction_distribution:
            if number_of_reconstruction_classes > 0:
                print("Can't use reconstruction classification with zero-inflated distributions.\n")
                continue
        
        # Model
        
        model_name = data.modelName("VAE", filtering_method, feature_selection,
            feature_size, splitting_method, splitting_fraction,
            reconstruction_distribution, number_of_reconstruction_classes, use_count_sum,
            latent_size, hidden_structure, learning_rate, batch_size, number_of_epochs)
        
        model = modeling.VariationalAutoEncoderForCounts(feature_size, latent_size,
            hidden_structure, reconstruction_distribution, number_of_reconstruction_classes, use_count_sum)
        
        previous_model_name, epochs_still_to_train = \
            data.findPreviouslyTrainedModel(model_name)
        
        print("")
        
        if previous_model_name and not force_training:
            model.load(previous_model_name)
            if epochs_still_to_train > 0:
                print("")
                model.train(training_set, validation_set,
                    N_epochs = epochs_still_to_train, batch_size = batch_size,
                    learning_rate = learning_rate)
                model.save(name = model_name, metadata = metadata)
        else:
            model.train(training_set, validation_set,
                N_epochs = number_of_epochs, batch_size = batch_size,
                learning_rate = learning_rate)
            model.save(name = model_name, metadata = metadata)
        
        print("")
        
        # Analysis
        
        analysis.analyseModel(model, name = model_name)
        
        test_set, reconstructed_test_set, latent_set, sample_set, test_metrics = \
            model.evaluate(test_set)
        
        analysis.analyseResults(test_set, reconstructed_test_set, test_headers,
            clusters, latent_set, sample_set, name = model_name,
            intensive_calculations = True)
        
        print("")



parser = argparse.ArgumentParser(
    description='Model gene counts in single cells.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--data-name", metavar = "name", type = str, default = "sample",
    help = "data set name")
parser.add_argument("--cluster-name", metavar = "name", type = str,
    help = "cluster name")
parser.add_argument("--latent-sizes", metavar = "size", nargs = '+', type = int,
    help = "sizes of latent space for different models")
parser.add_argument("--hidden-structure", metavar = "sizes", nargs = '+',
    type = int, help = "structure of hidden layers")
parser.add_argument("--filtering-method", metavar = "method", type = str,
    nargs = '+', help = "method for filtering examples")
parser.add_argument("--splitting-method", metavar = "method", type = str,
    default = "random", 
    help = "method for splitting data into training,   validation, and test sets")
parser.add_argument("--splitting-fraction", metavar = "fraction", type = float,
    default = 0.8,
    help = "fraction to use when splitting data into training, validation, and test sets")
parser.add_argument("--feature-selection", metavar = "selection", type = str,
    help = "selection of features to use")
parser.add_argument("--feature-size", metavar = "size", type = int,
    help = "size of feature space")
parser.add_argument("--reconstruction-distributions", metavar = "distribution", type = str,
    nargs = '+', help = "distributions for the reconstructions for different models")
parser.add_argument("--numbers-of-reconstruction-classes", metavar = "k", type = int,
    nargs = '+', default = [0],
    help = "the maximum counts for which to use classification for different models")
parser.add_argument("--use-count-sum", action = "store_true",
    help = "use the count sum of each example for the reconstructions")
parser.add_argument("--numbers-of-epochs", metavar = "N", type = int, default = 10,
    nargs = '+', help = "numbers of epochs for which to train and to save model parameters for")
parser.add_argument("--batch-size", metavar = "B", type = int, default = 100,
    help = "batch size used when training")
parser.add_argument("--learning-rate", metavar = "epsilon", type = float,
    default = 1e-3, help = "learning rate when training")
parser.add_argument("--force-training", action = "store_true",
    help = "train model whether or not it was previously trained")

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
