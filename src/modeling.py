#!/usr/bin/env python

from __future__ import print_function

import theano
import theano.tensor as T

import numpy

from lasagne.layers import (
    InputLayer, DenseLayer,
    get_output,
    get_all_params, get_all_param_values, set_all_param_values
)
from parmesan.layers.sample import SimpleSampleLayer, SampleLayer

from lasagne.nonlinearities import identity, sigmoid, rectify, softmax, softplus, tanh

from parmesan.distributions import (
    log_stdnormal, log_normal2, log_bernoulli,
    kl_normal2_stdnormal
)

from lasagne import updates

import data

from time import time
from aux import convertTimeToString

class VAE(object):
    def __init__(self, feature_shape, latent_size, hidden_structure):
        
        # Setup
        
        super(VAE, self).__init__()
        
        print("Setting up model.")
        
        self.feature_shape = feature_shape
        self.latent_size = latent_size
        self.hidden_structure = hidden_structure
        
        symbolic_x = T.matrix('x')
        symbolic_z = T.matrix('z')
        
        self.number_of_epochs_trained = 0
        self.learning_curves = {
            "training": {
                "lower bound": [],
                "log p(x|z)": [],
                "KL divergence": []
            },
            "validation": {
                "lower bound": [],
                "log p(x|z)": [],
                "KL divergence": []
            }
        }
        
        # Models
    
        ## Recognition model q(z|x)
    
        l_enc_in = InputLayer(shape = (None, feature_shape), name = "ENC_INPUT")
        l_enc = l_enc_in
        
        for i, hidden_size in enumerate(hidden_structure):
            l_enc = DenseLayer(l_enc, num_units = hidden_size, nonlinearity = rectify, name = 'ENC_DENSE{:d}'.format(i + 1))
        
        l_z_mu = DenseLayer(l_enc, num_units = latent_size, nonlinearity = None, name = 'ENC_Z_MU')
        l_z_log_var = DenseLayer(l_enc, num_units = latent_size, nonlinearity = lambda x: T.clip(x, -10, 10), name = 'ENC_Z_LOG_VAR')
        
        # Sample a latent representation z \sim q(z|x) = N(mu(x), logvar(x))
        l_z = SimpleSampleLayer(mean = l_z_mu, log_var = l_z_log_var, name = "ENC_SAMPLE")
        
        l_enc_in, l_z = self.buildEncoder(feature_shape, hidden_structure)
        self.encoder = l_z
        
        ## Generative model p(x|z)
        
        l_dec_in = InputLayer(shape = (None, latent_size), name = "DEC_INPUT")
        l_dec = l_dec_in
        
        for i, hidden_size in enumerate(reversed(hidden_structure)):
            l_dec = DenseLayer(l_dec, num_units = hidden_size, nonlinearity = rectify, name = 'DEC_DENSE{:d}'.format(len(hidden_structure) - i))
        
        l_x_p = DenseLayer(l_dec, num_units = feature_shape, nonlinearity = sigmoid, name = 'DEC_X_P')
        l_x_log_r = DenseLayer(l_dec, num_units = feature_shape, nonlinearity = identity, name = 'DEC_X_LOG_R')
        
        self.decoder = {"p": l_x_p, "log_r": l_x_log_r}
        
        ## Get outputs from models
        
        ## Training outputs
        z_train, z_mu_train, z_log_var_train = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = False)
        x_p_train, x_log_r_train = get_output(
            [l_x_p, l_x_log_r], {l_dec_in: z_train}, deterministic = False)
        
        ## Evaluation outputs
        z_eval, z_mu_eval, z_log_var_eval = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = True)
        x_p_eval, x_log_r_eval = get_output(
            [l_x_p, l_x_log_r], {l_dec_in: z_eval}, deterministic = True)
        
        ## Sample outputs
        x_p_sample = get_output(l_x_p, {l_dec_in: symbolic_z},
            deterministic = True)
        x_log_r_sample = get_output(l_x_log_r, {l_dec_in: symbolic_z},
            deterministic = True)
        
        # Likelihood
        
        LL_train, logpx_train, KL_train = self.logLikelihood(x_p_train, x_log_r_train, symbolic_x, z_mu_train, z_log_var_train)
        LL_eval, logpx_eval, KL_eval = self.logLikelihood(x_p_eval, x_log_r_eval, symbolic_x, z_mu_eval, z_log_var_eval)

        all_params = get_all_params([l_z, l_x_p, l_x_log_r], trainable = True)
        
        print("Parameters to train:")
        for parameter in all_params:
            print("    {}: {}".format(parameter, parameter.get_value().shape))
        
        # Let Theano do its magic and get all the gradients we need for training
        all_grads = T.grad(-LL_train, all_params)

        # Set the update function for parameters. The Adam optimizer works really well with VAEs.
        update_expressions = updates.adam(all_grads, all_params, learning_rate = 1e-3)

        self.f_train = theano.function(inputs = [symbolic_x],
                                  outputs = [LL_train, logpx_train, KL_train],
                                  updates = update_expressions)

        self.f_eval = theano.function(inputs = [symbolic_x],
                                 outputs = [LL_eval, logpx_eval, KL_eval])

        self.f_z = theano.function(inputs = [symbolic_x],
                                 outputs = [z_eval])

        self.f_sample = theano.function(inputs = [symbolic_z],
                                 outputs = [x_p_sample, x_log_r_sample])

        self.f_recon = theano.function(inputs = [symbolic_x],
                                 outputs = [x_p_eval, x_log_r_eval])
    
    def train(self, x_train, x_valid = None, N_epochs = 50, batch_size = 100):
        
        training_string = "Training model for {}".format(N_epochs)
        if self.number_of_epochs_trained > 0:
            training_string += " additional"
        training_string += " epochs."
        print(training_string)
        
        LL_train, logpx_train, KL_train = [], [], []
        LL_valid, logpx_valid, KL_valid = [], [], []
        
        N = x_train.shape[0]
        
        training_start = time()
        
        for epoch in range(self.number_of_epochs_trained,
            self.number_of_epochs_trained + N_epochs):
            
            epoch_start = time()
            
            shuffled_indices = numpy.random.permutation(N)
            
            for i in range(0, N, batch_size):
                subset = shuffled_indices[i:(i + batch_size)]
                x_batch = x_train[subset]
                out = self.f_train(x_batch)
            
            out = self.f_eval(x_train)
            LL_train += [out[0]] 
            logpx_train += [out[1]]
            KL_train += [out[2]]
            
            evaluation_string = "    Training set:   lower bound: {:.5g}, log p(x|z): {:.5g}, KL divergence: {:.5g}.".format(float(out[0]), float(out[1]), float(out[2]))
            
            if x_valid is not None:
                out = self.f_eval(x_valid)
                LL_valid += [out[0]]
                logpx_valid += [out[1]]
                KL_valid += [out[2]]
                
                evaluation_string += "\n    Validation set: lower bound: {:.5g}, log p(x|z): {:.5g}, KL divergence: {:.5g}.".format(float(out[0]), float(out[1]), float(out[2]))
            
            epoch_duration = time() - epoch_start
                
            print("Epoch {:2d} ({}):".format(epoch + 1, convertTimeToString(epoch_duration)))
            print(evaluation_string)
        
        training_duration = time() - training_start
        
        self.number_of_epochs_trained += N_epochs
        
        self.learning_curves["training"]["lower bound"] += LL_train
        self.learning_curves["training"]["log p(x|z)"] += logpx_train
        self.learning_curves["training"]["KL divergence"] += KL_train
        
        self.learning_curves["validation"]["lower bound"] += LL_valid
        self.learning_curves["validation"]["log p(x|z)"] += logpx_valid
        self.learning_curves["validation"]["KL divergence"] += KL_valid
        
        print("Training finished with a total of {} epochs after {}.".format(self.number_of_epochs_trained, convertTimeToString(training_duration)))
    
    def save(self, name, metadata = None):
        
        model = {
            "feature shape": self.feature_shape,
            "latent size": self.latent_size,
            "hidden structure": self.hidden_structure,
            "encoder": get_all_param_values(self.encoder),
            "decoder": {
                "p": get_all_param_values(self.decoder["p"]),
                "log_r": get_all_param_values(self.decoder["log_r"])
            },
            "number of epochs trained": self.number_of_epochs_trained,
            "learning curves": self.learning_curves,
        }
        
        if metadata:
            model["metadata"] = metadata
        
        model_name = name
        
        data.saveModel(model, model_name)
    
    def load(self, model_name):
        
        model = data.loadModel(model_name)
        
        set_all_param_values(self.encoder, model["encoder"])
        set_all_param_values(self.decoder["p"], model["decoder"]["p"])
        set_all_param_values(self.decoder["log_r"], model["decoder"]["log_r"])
        
        self.number_of_epochs_trained = model["number of epochs trained"]
        self.learning_curves = model["learning curves"]
    
    def evaluate(self, x_test):
        
        LL_test, _, _ = self.f_eval(x_test)
        
        print("log-likelihood for test set: {:.4g}.".format(float(LL_test)))
        
        z_eval = self.f_z(x_test)[0]
        
        x_p_sample, x_log_r_sample = self.f_sample(numpy.random.normal(size = (100, self.latent_size)).astype('float32'))
        
        x_p_recon, x_log_r_recon = self.f_recon(x_test)
        
        metrics = {
            "LL_test": LL_test
        }
        
        x_sample = meanOfNegativeBinomialDistribution(x_p_sample, x_log_r_sample)
        
        x_test_recon = {
            "p": x_p_recon,
            "log_r": x_log_r_recon,
            "mean": meanOfNegativeBinomialDistribution(x_p_recon, x_log_r_recon)
        }
        
        return x_test_recon, z_eval, x_sample, metrics
    
    def logLikelihood(self, x_p, x_log_r, x, z_mu, z_log_var):
        #note that we sum the latent dimension and mean over the samples
        log_px_given_z = log_negative_binomial(x, x_p, x_log_r, eps = 1e-6).sum(axis = 1).mean()
        # TODO Add normalisation in model.
        KL_qp = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis = 1).mean()
        LL = - KL_qp + log_px_given_z
        return LL, log_px_given_z, KL_qp

def log_negative_binomial(x, log_r, p, eps = 0.0, approximation = "simple"):
    """
    Compute log pdf of a negative binomial distribution with success probability p and number of failures, r, until the experiment is stopped, at values x.
    
    A simple variation of Stirling's approximation is used: log x! = x log x - x.
    """
    
    x = T.clip(x, eps, x)
    
    p = T.clip(p, eps, 1.0 - eps)
    
    r = T.exp(log_r)
    r = T.clip(r, eps, r)
    
    y = T.gammaln(x + r) - T.gammaln(x + 1) - T.gammaln(r) \
        + x * T.log(p) + r * T.log(1 - p)
    
    return y

def meanOfNegativeBinomialDistribution(p, log_r):
    return p * numpy.exp(log_r) / (1 - p)
