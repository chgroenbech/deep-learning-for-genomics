#!/usr/bin/env python

import data

import numpy

import theano
import theano.tensor as T

from lasagne import init, updates

from lasagne.nonlinearities import identity, sigmoid, rectify, softmax, softplus

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
    
    latent_sizes = [100]
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
    
    def preprocess(x):
        x[numpy.where(x != 0)] = 1
        return x

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
            l_enc = DenseLayer(l_enc, num_units = hidden_size, nonlinearity = softplus, name = 'ENC_DENSE{:d}'.format(i))
    
        l_z_mu = DenseLayer(l_enc, num_units = latent_size, nonlinearity = identity, name = 'ENC_Z_MU')
        l_z_log_var = DenseLayer(l_enc, num_units = latent_size, nonlinearity = identity, name = 'ENC_Z_LOG_VAR')
    
        # Sample the latent variables using mu(x) and log(sigma^2(x))
        l_z = SimpleSampleLayer(mean = l_z_mu, log_var = l_z_log_var) # as Kingma
        # l_z = SampleLayer(mean = l_z_mu, log_var = l_z_log_var)

        ## Generative model p(x|z)
    
        l_dec_in = InputLayer((None, latent_size), name = "DEC_INPUT")
    
        l_dec = l_dec_in
    
        for i, hidden_size in enumerate_reversed(hidden_sizes, start = 0):
            l_dec = DenseLayer(l_dec, num_units = hidden_size, nonlinearity = softplus, name = 'DEC_DENSE{:d}'.format(i))
    
        l_dec_x_mu = DenseLayer(l_dec, num_units = F, nonlinearity = sigmoid, name = 'DEC_X_MU')
        l_dec_x_log_var = DenseLayer(l_dec, num_units = F, nonlinearity = sigmoid, name = 'DEC_X_MU')
    
        # TODO: Try ReLU instead of softplus (maybe with more hidden units)
        # TODO: Try softmax instead of sigmoid
        # PROBLEM with this is that we have several pixels activated.

        ## Get outputs from models
    
        # With noise
        z_train, z_mu_train, z_log_var_train = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = False
        )
        x_mu_train, x_log_var_train = get_output([l_dec_x_mu, l_dec_x_log_var], {l_dec_in: z_train}, deterministic = False)
    
        # Without noise
        z_eval, z_mu_eval, z_log_var_eval = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = True
        )
        x_mu_eval, x_log_var_eval = get_output([l_dec_x_mu, l_dec_x_log_var], {l_dec_in: z_eval}, deterministic = True)
    
        # Sampling
        x_mu_sample = get_output(l_dec_x_mu, {l_dec_in: symbolic_z},
            deterministic = True)
        
        # Likelihood
        
        # Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
        def log_likelihood(z, z_mu, z_log_var, x_mu, x_log_var, x, analytic_kl_term):
            if analytic_kl_term:
                kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis = 1)
                log_px_given_z = log_bernoulli(x, x_mu,  eps = 1e-6).sum(axis = 1)
                # log_px_given_z = log_normal2(x, x_mu, x_log_var).sum(axis = 1)
                LL = T.mean(-kl_term + log_px_given_z)
            else:
                log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis = 1)
                log_pz = log_stdnormal(z).sum(axis = 1)
                log_px_given_z = log_bernoulli(x, x_mu,  eps = 1e-6).sum(axis = 1)
                # log_px_given_z = log_normal2(x, x_mu, x_log_var).sum(axis = 1)
                LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
            return LL

        # log-likelihood for training
        ll_train = log_likelihood(
            z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train, symbolic_x, analytic_kl_term)

        # log-likelihood for evaluating
        ll_eval = log_likelihood(
            z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval, symbolic_x, analytic_kl_term)
    
        # Parameters to train
        parameters = get_all_params([l_z, l_dec_x_mu], trainable = True)
        # parameters = get_all_params([l_z, l_dec_x_mu, l_dec_x_log_var], trainable = True)
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
        
            # line = "Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, learning_rate, train_cost, test_cost)
            print("Epoch {:d} (duration: {:.2f} s, learning rate: {:.1e}):".format(epoch + 1, epoch_duration, learning_rate))
            print("    log-likelihood: {:.3f} (training set), {:.3f} (test set)".format(train_cost, test_cost))
        
        print
        
        # Results
    
        ## Reconstruction
    
        N_reconstructions = 50
    
        X_test_eval = X_test_shared.eval()
        X_test_eval_fixed = X_test_shared_fixed.eval()
        X_test_eval_normal = X_test_shared_normal.eval()
        
        subset = numpy.random.randint(0, len(X_test_eval), size = N_reconstructions)
    
        x_original = X_test_eval[numpy.array(subset)]
        x_LR = get_output(l_enc_HR_downsample, x_original).eval()
        z = get_output(l_z, x_LR).eval()
        x_reconstructed = x_mu_sample.eval({symbolic_z: z})
    
        x_original_fixed = X_test_eval_fixed[numpy.array(subset)]
        x_LR_fixed = get_output(l_enc_HR_downsample, x_original_fixed).eval()
        z_fixed = get_output(l_z, x_LR_fixed).eval()
        x_reconstructed_fixed = x_mu_sample.eval({symbolic_z: z_fixed})
        
        originals = X_test_eval_normal[numpy.array(subset)]
        
        reconstructions = {
            "originals": x_original,
            "downsampled":  x_LR,
            "reconstructions": x_reconstructed
        }
        
        reconstructions_fixed = {
            "originals": x_original_fixed,
            "downsampled":  x_LR_fixed,
            "reconstructions": x_reconstructed_fixed
        }
        
        ## Manifold
        
        if latent_size == 2:
        
            x = numpy.linspace(0.1, 0.9, 20)
            # TODO: Ideally sample from the real p(z)
            v = gaussian.ppf(x)
            z = numpy.zeros((20**2, 2))
        
            i = 0
            for a in v:
                for b in v:
                    z[i,0] = a
                    z[i,1] = b
                    i += 1
            z = z.astype('float32')
        
            samples = x_mu_sample.eval({symbolic_z: z})
    
        else:
            samples = None
    
        ## Reconstructions of homemade numbers
    
        if downsampling_factor == 2:
        
            file_names = [
                "hm_7_Avenir.png",
                "hm_7_Noteworthy.png",
                "hm_7_Chalkboard.png",
                "hm_7_drawn.png",
                "hm_A_Noteworthy.png",
                "hm_A_drawn.png",
                "hm_7_0.txt",
                "hm_7_1.txt",
                "hm_7_2.txt",
                "hm_A.txt"
            ]
        
            x_LR_HM = data.loadHomemade(map(data_path, file_names), [h * w])
        
            z = get_output(l_z, x_LR_HM).eval()
            x_HM_reconstructed = x_mu_sample.eval({symbolic_z: z})
    
            reconstructions_homemade = {
                "originals": x_LR_HM,
                "reconstructions": x_HM_reconstructed
            }
    
        else:
            reconstructions_homemade = None
    
        # Saving
    
        setup_and_results = {
            "setup": {
                "image size": (C, H, W),
                "downsampling factor": downsampling_factor,
                "learning rate": learning_rate,
                "analytic K-L term": analytic_kl_term,
                "batch size": batch_size,
                "hidden layer sizes": hidden_sizes,
                "latent size": latent_size,
                "number of epochs": N_epochs
            },
            "results": {
                "learning curve": {
                    "epochs": epochs,
                    "training cost function": cost_train,
                    "test cost function": cost_test
                },
                "originals": originals,
                "reconstructions": reconstructions,
                "reconstructions (fixed)": reconstructions_fixed,
                "manifold": {
                    "samples": samples
                },
                "reconstructed homemade numbers": reconstructions_homemade
            }
        }
        
        file_name = "results{}_ds{}{}_l{}_e{}.pkl".format("_bs" if bernoulli_sampling else "", downsampling_factor, "b" if binarise_downsampling else "", latent_size, N_epochs)
    
        with open(data_path(file_name), "w") as f:
            pickle.dump(setup_and_results, f)
        
        run_duration = time.time() - run_start
        
        all_runs_duration += run_duration
        
        print("Run took {:.2f} minutes.".format(run_duration / 60))
        
        print("\n")
    
    print("All runs took {:.2f} minutes in total.".format(all_runs_duration / 60))

def log_negative_binomial(x, r, p, approximation = "simple"):
    """
    Compute log pdf of a negative binomial distribution with success probability p and number of failures, r, until the experiment is stopped, at values x.
    
    A simple variation of Stirling's approximation is used: log x! = x log x.
    """
    if approximation == "simple":
        stirling = lambda x: x * T.log(x) - x
    
    y = stirling(x + r - 1) - stirling(x - 1) - stirling(r) \
        + x * T.log(p) + r * T.log(1 - p)
    
    return y

if __name__ == '__main__':
    script_directory()
    main()
