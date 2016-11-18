#!/usr/bin/env python

import data
import modeling
import analysis

import argparse

from numpy import random

random.seed(42)

def main(name, latent_size, hidden_structure, filtering_method = None,
    splitting_method = "random", splitting_fraction = 0.8,
    feature_selection = None, feature_size = None,
    number_of_epochs = 10, batch_size = 100, learning_rate = 1e-3,
    force_training = False):
    
    # Data
    
    training_set, validation_set, test_set = data.loadData(name,
        filtering_method, feature_selection, feature_size,
        splitting_method, splitting_fraction
    )
    
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
    
    # Model
    
    feature_size = training_set.shape[1]
    
    model_name = data.modelName("vae", filtering_method, feature_selection,
        feature_size, splitting_method, splitting_fraction, latent_size,
        hidden_structure, batch_size, number_of_epochs)
    
    model = modeling.VAE(feature_size, latent_size, hidden_structure)
    print(model.number_of_epochs_trained)
    
    previous_model_name, epochs_still_to_train = \
        data.findPreviouslyTrainedModel(model_name)
    
    if previous_model_name and not force_training:
        model.load(previous_model_name)
        if epochs_still_to_train > 0:
            model.train(training_set, validation_set,
                N_epochs = epochs_still_to_train, batch_size = batch_size)
            model.save(name = model_name, metadata = metadata)
    else:
        model.train(training_set, validation_set,
            N_epochs = number_of_epochs, batch_size = batch_size)
        model.save(name = model_name, metadata = metadata)
    
    # Analysis
    
    analysis.analyseModel(model, name = model_name)
    
    reconstructed_test_set, sample_set, test_metrics = model.evaluate(test_set)

    analysis.analyseResults(test_set, reconstructed_test_set, sample_set,
        name = model_name)

parser = argparse.ArgumentParser(
    description='Model gene counts in single cells.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--name", metavar = "name", type = str, default = "sample", help = "data set name")
parser.add_argument("--latent-size", metavar = "size", type = int,
    help = "size of latent space")
parser.add_argument("--hidden-structure", metavar = "sizes", nargs = '+',
    type = int, help = "structure of hidden layers")
parser.add_argument("--filtering-method", metavar = "method", type = str,
    help = "method for filtering examples")
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
parser.add_argument("--number-of-epochs", metavar = "N", type = int, default = 10,
    help = "number of epochs for which to train")
parser.add_argument("--batch-size", metavar = "B", type = int, default = 100,
    help = "batch size used when training")
parser.add_argument("--learning-rate", metavar = "epsilon", type = float,
    default = 1e-3, help = "learning rate when training")
parser.add_argument("--force-training", action = "store_true",
    help = "train model whether or not it was previously trained")

if __name__ == '__main__':
    
    # file_name = "GSE63472_P14Retina_merged_digital_expression"
    # latent_size = 50
    # hidden_structure = [500]
    
    arguments = parser.parse_args()
    main(**vars(arguments))
