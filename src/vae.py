#!/usr/bin/env python

import data

import numpy

import theano
import theano.tensor as T

from lasagne import init, updates

from lasagne.nonlinearities import identity, sigmoid, rectify, softmax, softplus, tanh

from lasagne.layers import (
    InputLayer, DenseLayer,
    Pool2DLayer, ReshapeLayer, DimshuffleLayer,
    NonlinearityLayer,
    get_output,
    get_all_params
)
from parmesan.layers.sample import SimpleSampleLayer, SampleLayer

from parmesan.distributions import (
    log_stdnormal, log_normal2, log_bernoulli,
    kl_normal2_stdnormal
)

from scipy.stats import norm as gaussian

import time

import pickle

from itertools import product

from aux import data_path, figure_path, script_directory, enumerate_reversed

def main():
    
    # Main setup
    
    latent_sizes = [500]
    N_epochs = 50
    
    # Setup
    
    F = 24658 # number of features
    
    hidden_sizes = [200, 200]
    
    batch_size = 100
    
    analytic_kl_term = True
    learning_rate = 0.001 #0.0003
    
    shape = [F]
    
    # Symbolic variables
    symbolic_x = T.matrix()
    symbolic_z = T.matrix()
    symbolic_learning_rate = T.scalar('learning_rate')
    
    # Fix random seed for reproducibility
    numpy.random.seed(1234)
    
    # Data
    
    file_name = "DGE_matrix_counts_sparse.pkl.gz"
    file_path = data_path(file_name)
    
    X_train, X_valid, X_test = data.load(file_path, shape)
    
    # X_train = numpy.concatenate([X_train, X_valid])
    
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)

    N_train_batches = X_train.shape[0] / batch_size
    N_test_batches = X_test.shape[0] / batch_size
    
    # Setup shared variables
    X_train_shared = theano.shared(preprocess(X_train), borrow = True)
    X_test_shared = theano.shared(preprocess(X_test), borrow = True)
    X_test_shared_fixed = theano.shared(preprocess(X_test), borrow = True)
    X_test_shared_normal = theano.shared(X_test, borrow = True)
    
    all_runs_duration = 0
    
    for latent_size in latent_sizes:
        
        run_start = time.time()
        
        print("Training model with a latent size of {}:\n".format(latent_size))
        
        # Models
    
        ## Recognition model q(z|x)
    
        l_enc_in = InputLayer((None, F), name = "ENC_INPUT")
        
        l_enc = l_enc_in
    
        for i, hidden_size in enumerate(hidden_sizes, start = 1):
            l_enc = DenseLayer(l_enc, num_units = hidden_size, nonlinearity = rectify, name = 'ENC_DENSE{:d}'.format(i))
    
        l_z_mu = DenseLayer(l_enc, num_units = latent_size, nonlinearity = tanh, name = 'ENC_Z_MU')
        l_z_log_var = DenseLayer(l_enc, num_units = latent_size, nonlinearity = tanh, name = 'ENC_Z_LOG_VAR')
    
        # Sample the latent variables using mu(x) and log(sigma^2(x))
        l_z = SimpleSampleLayer(mean = l_z_mu, log_var = l_z_log_var) # as Kingma
        # l_z = SampleLayer(mean = l_z_mu, log_var = l_z_log_var)

        ## Generative model p(x|z)
    
        l_dec_in = InputLayer((None, latent_size), name = "DEC_INPUT")
    
        l_dec = l_dec_in
    
        for i, hidden_size in enumerate_reversed(hidden_sizes, start = 0):
            l_dec = DenseLayer(l_dec, num_units = hidden_size, nonlinearity = softplus, name = 'DEC_DENSE{:d}'.format(i))
    
        l_dec_x_p = DenseLayer(l_dec, num_units = F, nonlinearity = sigmoid, name = 'DEC_X_P')
        l_dec_x_r = DenseLayer(l_dec, num_units = F, nonlinearity = lambda x: affine_rectify(x, b = 1), name = 'DEC_X_R')
        
        ## Get outputs from models
    
        # With noise
        z_train, z_mu_train, z_log_var_train = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = False
        )
        x_p_train, x_r_train = get_output([l_dec_x_p, l_dec_x_r], {l_dec_in: z_train}, deterministic = False)
    
        # Without noise
        z_eval, z_mu_eval, z_log_var_eval = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = True
        )
        x_p_eval, x_r_eval = get_output([l_dec_x_p, l_dec_x_r], {l_dec_in: z_eval}, deterministic = True)
    
        # Sampling
        x_p_sample = get_output(l_dec_x_p, {l_dec_in: symbolic_z},
            deterministic = True)
        
        # Likelihood
        
        # Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
        def log_likelihood(z, z_mu, z_log_var, x_p, x_r, x, analytic_kl_term):
            
            log_px_given_z = log_negative_binomial(x, x_r, x_p, eps = 1e-6).sum(axis = 1)
            # log_px_given_z = log_poisson_gamma(x, x_r, x_p, eps = 1e-6).sum(axis = 1)
            
            if analytic_kl_term:
                kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis = 1)
                LL = T.mean(-kl_term + log_px_given_z)
            else:
                log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis = 1)
                log_pz = log_stdnormal(z).sum(axis = 1)
                LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
            
            return LL

        # log-likelihood for training
        ll_train = log_likelihood(
            z_train, z_mu_train, z_log_var_train, x_p_train, x_r_train, symbolic_x, analytic_kl_term)

        # log-likelihood for evaluating
        ll_eval = log_likelihood(
            z_eval, z_mu_eval, z_log_var_eval, x_p_eval, x_r_eval, symbolic_x, analytic_kl_term)
    
        # Parameters to train
        parameters = get_all_params([l_z, l_dec_x_p, l_dec_x_r], trainable = True)
        print("Parameters that will be trained:")
        for parameter in parameters:
            print("{}: {}".format(parameter, parameter.get_value().shape))

        ### Take gradient of negative log-likelihood
        gradients = T.grad(-ll_train, parameters)

        # Adding gradient clipping to reduce the effects of exploding gradients,
        # and hence speed up convergence
        gradient_clipping = 1
        gradient_norm_max = 5
        gradient_constrained = updates.total_norm_constraint(gradients,
            max_norm = gradient_norm_max)
        gradients_clipped = [T.clip(g,-gradient_clipping, gradient_clipping) for g in gradient_constrained]
    
        # Setting up functions for training
    
        symbolic_batch_index = T.iscalar('index')
        batch_slice = slice(symbolic_batch_index * batch_size, (symbolic_batch_index + 1) * batch_size)

        update_expressions = updates.adam(gradients_clipped, parameters,
            learning_rate = symbolic_learning_rate)

        train_model = theano.function(
            [symbolic_batch_index, symbolic_learning_rate], ll_train,
            updates = update_expressions, givens = {symbolic_x: X_train_shared[batch_slice]}
        )

        test_model = theano.function(
            [symbolic_batch_index], ll_eval,
            givens = {symbolic_x: X_test_shared[batch_slice]}
        )
    
        test_model_fixed = theano.function(
            [symbolic_batch_index], ll_eval,
            givens = {symbolic_x: X_test_shared_fixed[batch_slice]}
        )
    
        def train_epoch(learning_rate):
            costs = []
            for i in range(N_train_batches):
                cost_batch = train_model(i, learning_rate)
                costs += [cost_batch]
            return numpy.mean(costs)
    
        def test_epoch():
            costs = []
            for i in range(N_test_batches):
                cost_batch = test_model(i)
                costs += [cost_batch]
            return numpy.mean(costs)
    
        def test_epoch_fixed():
            costs = []
            for i in range(N_test_batches):
                cost_batch = test_model_fixed(i)
                costs += [cost_batch]
            return numpy.mean(costs)
    
        # Training
    
        epochs = []
        cost_train = []
        cost_test = []
    
        print

        for epoch in range(N_epochs):
        
            epoch_start = time.time()
        
            # Shuffle train data
            numpy.random.shuffle(X_train)
            X_train_shared.set_value(preprocess(X_train))
        
            # TODO: Using dynamically changed learning rate
            train_cost = train_epoch(learning_rate)
            test_cost = test_epoch()
            test_cost_fixed = test_epoch_fixed()
        
            epoch_duration = time.time() - epoch_start
        
            epochs.append(epoch + 1)
            cost_train.append(train_cost)
            cost_test.append(test_cost)
        
            print("Epoch {:d} (duration: {:.2f} s, learning rate: {:.1e}):".format(epoch + 1, epoch_duration, learning_rate))
            print("    log-likelihood: {:.3f} (training set), {:.3f} (test set)".format(train_cost, test_cost))
        
        print
        
        run_duration = time.time() - run_start
        
        all_runs_duration += run_duration
        
        print("Run took {:.2f} minutes.".format(run_duration / 60))
        
        print("\n")
    
    print("All runs took {:.2f} minutes in total.".format(all_runs_duration / 60))

def log_poisson_gamma(x, r, p, eps = 0.0, approximation = "simple"):
    """
    Compute log pdf of a negative binomial distribution with success probability p and number of failures, r, until the experiment is stopped, at values x.
    
    A simple variation of Stirling's approximation is used: log x! = x log x - x.
    """
    
    x = T.clip(x, eps, x)
    
    p = T.clip(p, eps, 1.0 - eps)
    r = T.clip(r, eps, r)
    
    if approximation == "simple":
        def stirling(x):
            if x == 0:
                return 0
            else:
                return x * T.log(x) - x
    
    # y = T.gamma(r + x) \
    #     + x * T.log(p) + r * T.log(1-p)
    
    y = stirling(r + x) - stirling(x) - stirling(r) + x * T.log(p) + r * T.log(1-p)
    
    # y = 0.*(x + r + p) - stirling(r)
    
    return y

def log_negative_binomial(x, r, p, eps = 0.0, approximation = "simple"):
    """
    Compute log pdf of a negative binomial distribution with success probability p and number of failures, r, until the experiment is stopped, at values x.
    
    A simple variation of Stirling's approximation is used: log x! = x log x - x.
    """
    
    x = T.clip(x, eps, x)
    
    p = T.clip(p, eps, 1.0 - eps)
    r = T.clip(r, 1 + eps, r)
    # TODO Change the value in the network
    
    if approximation == "simple":
        stirling = lambda x: x * T.log(x) - x
    
    y = stirling(x + r - 1) - stirling(x) - stirling(r - 1) \
        + x * T.log(p) + r * T.log(1 - p)
    
    return y

def affine_rectify(x, a = 1, b = 0):
    return a * rectify(x) + b

def preprocess(x):
    # x[numpy.where(x != 0)] = 1
    return x

if __name__ == '__main__':
    script_directory()
    main()
