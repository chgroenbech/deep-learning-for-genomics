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

class VAE(object):
    def __init__(self, feature_shape, latent_size, hidden_structure):
        
        super(VAE, self).__init__()
        
        self.feature_shape = feature_shape
        self.latent_size = latent_size
        self.hidden_structure = hidden_structure
        
        print("Setting up model.")
        
        symbolic_x = T.matrix('x')
        symbolic_z = T.matrix('z')
        
        # Models
    
        ## Recognition model q(z|x)
    
        l_enc_in = InputLayer(shape = (None, feature_shape), name = "ENC_INPUT")
        l_enc = l_enc_in
        
        # TODO Add option for embedding layer.
        
        for i, hidden_size in enumerate(hidden_structure):
            l_enc = DenseLayer(l_enc, num_units = hidden_size, nonlinearity = rectify, name = 'ENC_DENSE{:d}'.format(i + 1))
        
        l_z_mu = DenseLayer(l_enc, num_units = latent_size, nonlinearity = None, name = 'ENC_Z_MU')
        l_z_log_var = DenseLayer(l_enc, num_units = latent_size, nonlinearity = lambda x: T.clip(x, -10, 10), name = 'ENC_Z_LOG_VAR')
        
        # Sample a latent representation z \sim q(z|x) = N(mu(x), logvar(x))
        l_z = SimpleSampleLayer(mean = l_z_mu, log_var = l_z_log_var, name = "ENC_SAMPLE")
        
        self.encoder = l_z
        
        ## Generative model p(x|z)
        
        l_dec_in = InputLayer(shape = (None, latent_size), name = "DEC_INPUT")
        l_dec = l_dec_in
        
        for i, hidden_size in enumerate(reversed(hidden_structure)):
            l_dec = DenseLayer(l_dec, num_units = hidden_size, nonlinearity = rectify, name = 'DEC_DENSE{:d}'.format(len(hidden_structure) - i))
        
        l_x_p = DenseLayer(l_dec, num_units = feature_shape, nonlinearity = sigmoid, name = 'DEC_X_P')
        l_x_log_r = DenseLayer(l_dec, num_units = feature_shape, nonlinearity = rectify, name = 'DEC_X_LOG_R')
        
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
        
        print("Training model.")
        
        LL_train, KL_train, logpx_train = [], [], []
        LL_valid, KL_valid, logpx_valid = [], [], []
        
        N = x_train.shape[0]
        
        for epoch in range(N_epochs):
            
            epoch_string = "Epoch {:2d}: ".format(epoch + 1)
            
            shuffled_indices = numpy.random.permutation(N)
            
            for i in range(0, N, batch_size):
                subset = shuffled_indices[i:(i + batch_size)]
                x_batch = x_train[subset]
                out = self.f_train(x_batch)
            
            out = self.f_eval(x_train)
            LL_train += [out[0]] 
            logpx_train += [out[1]]
            KL_train += [out[2]]
            
            epoch_string += "log-likelihood: {:.4g} (training set)".format(float(out[0]))
            
            if x_valid is not None:
                out = self.f_eval(x_valid)
                LL_valid += [out[0]]
                logpx_valid += [out[1]]
                KL_valid += [out[2]]
                
                epoch_string += ", {:.4g} (validation set)".format(float(out[0]))
                
            epoch_string += "."
            
            print(epoch_string)
        
        self.training_learning_curve = LL_train
        self.validation_learning_curve = LL_valid
        
        print("Training finished.")
    
    def save(self, name):
        
        model = {
            "encoder": get_all_param_values(self.encoder),
            "decoder": {
                "p": get_all_param_values(self.decoder["p"]),
                "log_r": get_all_param_values(self.decoder["log_r"])
            },
            "training learning curve": self.training_learning_curve,
            "validation learning curve": self.validation_learning_curve,
            # self.feature_shape
            # self.latent_size
            # self.hidden_structure
        }
        
        model_name = name
        
        data.saveModel(model, model_name)
    
    def load(self, model_name):
        
        model = data.loadModel(model_name)
        
        set_all_param_values(self.encoder, model["encoder"])
        set_all_param_values(self.decoder["p"], model["decoder"]["p"])
        set_all_param_values(self.decoder["log_r"], model["decoder"]["log_r"])
        self.training_learning_curve = model["training learning curve"]
        self.validation_learning_curve = model["validation learning curve"]
    
    def evaluate(self, x_test):
        LL_test, _, _ = self.f_eval(x_test)
        
        z_eval = self.f_z(x_test)
        
        x_p_sample, x_log_r_sample = self.f_sample(numpy.random.normal(size = (100, self.latent_size)).astype('float32'))#[0]
        
        x_p_recon, x_log_r_recon = self.f_recon(x_test)#[0]
        
        metrics = {
            "LL_test": LL_test
        }
        
        x_sample = meanOfNegativeBinomialDistribution(x_p_sample, x_log_r_sample)
        x_test_recon = meanOfNegativeBinomialDistribution(x_p_recon, x_log_r_recon)
        
        return x_test_recon, x_sample, metrics
    
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
    log_r = T.clip(log_r, eps, log_r)
    
    r = T.exp(log_r)
    
    y = T.gammaln(x + r) - T.gammaln(x + 1) - T.gammaln(r) \
        + x * T.log(p) + r * T.log(1 - p)
    
    return y

def meanOfNegativeBinomialDistribution(p, log_r):
    return p * numpy.exp(log_r) / (1 - p)
