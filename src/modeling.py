#!/usr/bin/env python

from __future__ import print_function

import theano
import theano.tensor as T

import numpy

from lasagne.layers import (
    InputLayer, DenseLayer, ReshapeLayer, ConcatLayer,
    get_output,
    get_all_params, get_all_param_values, set_all_param_values
)
from parmesan.layers.sample import SimpleSampleLayer, SampleLayer

from lasagne.nonlinearities import identity, sigmoid, rectify, softmax, softplus, tanh

from parmesan.distributions import (
    log_normal, log_bernoulli,
    kl_normal2_stdnormal
)

from lasagne import updates, objectives

import data

from time import time
from aux import convertTimeToString

class VariationalAutoEncoderForCounts(object):
    def __init__(self, feature_shape, latent_size, hidden_structure,
        reconstruction_distribution = None, number_of_reconstruction_classes = None,
        use_count_sum = False):
        
        self.use_count_sum = use_count_sum and \
            (reconstruction_distribution != "bernoulli")
        
        print("Setting up model.")
        print("    feature size: {}".format(feature_shape))
        print("    latent size: {}".format(latent_size))
        print("    hidden structure: {}".format(", ".join(map(str, hidden_structure))))
        if type(reconstruction_distribution) == str:
            print("    reconstruction distribution: " + reconstruction_distribution)
        else:
            print("    reconstruction distribution: custom")
        if number_of_reconstruction_classes > 0:
            print("    reconstruction classes: {}".format(number_of_reconstruction_classes),
                  " (including 0s)")
        if self.use_count_sum:
            print("    using count sums")
        print("")
        
        # Setup
        
        super(VariationalAutoEncoderForCounts, self).__init__()
        
        self.feature_shape = feature_shape
        self.latent_size = latent_size
        self.hidden_structure = hidden_structure
        
        symbolic_x = T.matrix('x') # counts
        symbolic_z = T.matrix('z') # latent variable
        
        self.number_of_epochs_trained = 0
        symbolic_learning_rate = T.scalar("epsilon")
        self.learning_curves = {
            "training": {
                "LB": [],
                "ENRE": [],
                "KL": []
            },
            "validation": {
                "LB": [],
                "ENRE": [],
                "KL": []
            }
        }
        
        if reconstruction_distribution:
            
            if type(reconstruction_distribution) == str:
                reconstruction_distribution = \
                    reconstruction_distributions[reconstruction_distribution]
            
            self.x_parameters = reconstruction_distribution["parameters"]
            self.reconstruction_activation_functions = \
                reconstruction_distribution["activation functions"]
            self.expectedNegativeReconstructionError = reconstruction_distribution["function"]
            self.meanOfReconstructionDistribution = reconstruction_distribution["mean"]
            self.preprocess = reconstruction_distribution["preprocess"]
        else:
            reconstruction_distribution = "Gaussian (default)"
            
            # Use a Gaussian distribution as standard
            self.x_parameters = ["mu", "sigma"]
            self.reconstruction_activation_functions = {
                "mu": identity,
                "sigma": identity
            }
            self.expectedNegativeReconstructionError = lambda x, x_theta, eps = 0.0: \
                log_normal(x, x_theta["mu"], x_theta["sigma"], eps)
            self.meanOfReconstructionDistribution = lambda x_theta: x_theta["mu"]
            self.preprocess = lambda x: x
        
        if number_of_reconstruction_classes > 0:
            
            self.x_parameters += ["p_k"]
            self.reconstruction_activation_functions["p_k"] = softmax
            log_distribution = self.expectedNegativeReconstructionError
            self.expectedNegativeReconstructionError = lambda x, x_theta, eps = 0.0: \
                log_cross_entropy_extended(x, x_theta,
                    log_distribution, k_max = number_of_reconstruction_classes - 1,
                    eps = 0.0)
            mean_of_distribution = self.meanOfReconstructionDistribution
            self.meanOfReconstructionDistribution = lambda x_theta: \
                meanOfCrossEntropyExtendedDistibution(x_theta,
                    mean_of_distribution, k_max = number_of_reconstruction_classes - 1)
            self.k_max = number_of_reconstruction_classes - 1
        
        if self.use_count_sum:
            symbolic_n = T.matrix('n') # sum of counts
        
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
        
        self.encoder = l_z
        
        ## Generative model p(x|z)
        
        l_dec_z_in = InputLayer(shape = (None, latent_size), name = "DEC_INPUT")
        
        if self.use_count_sum:
            l_dec_n_in = InputLayer(shape = (None, 1), name = "DEC_N_INPUT")
            l_dec = ConcatLayer([l_dec_z_in, l_dec_n_in], axis = 1,
                name = "DEC_MERGE_INPUT")
        else:
            l_dec = l_dec_z_in
        
        for i, hidden_size in enumerate(reversed(hidden_structure)):
            l_dec = DenseLayer(l_dec, num_units = hidden_size, nonlinearity = rectify, name = 'DEC_DENSE{:d}'.format(len(hidden_structure) - i))
        
        l_x_theta = {}
        
        for p in self.x_parameters:
            p_name = 'DEC_X_' + p.upper()
            if self.reconstruction_activation_functions[p] == softmax:
                l_dense = DenseLayer(l_dec,
                    num_units = feature_shape * (self.k_max + 1),
                    nonlinearity = identity, name = p_name + "_DENSE")
                l_reshape = ReshapeLayer(l_dense, (-1, (self.k_max + 1)))
                l_softmax = DenseLayer(l_reshape, num_units = (self.k_max + 1),
                    nonlinearity = softmax, name = p_name + "_SOFTMAX")
                l_x_theta[p] = ReshapeLayer(l_softmax,
                    (-1, feature_shape, (self.k_max + 1)))
            else:
                l_x_theta[p] = DenseLayer(l_dec, num_units = feature_shape,
                    nonlinearity = self.reconstruction_activation_functions[p],
                    name = p_name)
        
        self.decoder = {p: l_x_theta[p] for p in self.x_parameters}
        
        ## Get outputs from models
        
        ## Training outputs
        z_train, z_mu_train, z_log_var_train = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = False)
        inputs = {l_dec_z_in: z_train}
        if self.use_count_sum:
            inputs[l_dec_n_in] = symbolic_n
        x_theta_train = get_output([l_x_theta[p] for p in self.x_parameters], inputs,
            deterministic = False)
        x_theta_train = {p: o for p, o in zip(self.x_parameters, x_theta_train)}
        
        ## Evaluation outputs
        z_eval, z_mu_eval, z_log_var_eval = get_output(
            [l_z, l_z_mu, l_z_log_var], {l_enc_in: symbolic_x}, deterministic = True)
        inputs = {l_dec_z_in: z_eval}
        if self.use_count_sum:
            inputs[l_dec_n_in] = symbolic_n
        x_theta_eval = get_output([l_x_theta[p] for p in self.x_parameters], inputs,
            deterministic = True)
        x_theta_eval = {p: o for p, o in zip(self.x_parameters, x_theta_eval)}
        
        ## Sample outputs
        
        inputs = {l_dec_z_in: symbolic_z}
        if self.use_count_sum:
            inputs[l_dec_n_in] = symbolic_n
        x_theta_sample = get_output([l_x_theta[p] for p in self.x_parameters], inputs,
            deterministic = True)
        x_theta_sample = {p: o for p, o in zip(self.x_parameters, x_theta_sample)}
        
        # Likelihood
        
        lower_bound_train, log_p_x_train, KL__train = \
            self.lowerBound(symbolic_x, x_theta_train, z_mu_train, z_log_var_train)
        lower_bound_eval, log_p_x_eval, KL__eval = \
            self.lowerBound(symbolic_x, x_theta_eval, z_mu_eval, z_log_var_eval)
        
        all_parameters = get_all_params([l_z] + [l_x_theta[p] for p in self.x_parameters], trainable = True)
        
        print("Parameters to train:")
        for parameter in all_parameters:
            print("    {}: {}".format(parameter, parameter.get_value().shape))
        
        # Let Theano do its magic and get all the gradients we need for training
        all_gradients = T.grad(-lower_bound_train, all_parameters)

        # Set the update function for parameters. The Adam optimizer works really well with VAEs.
        update_expressions = updates.adam(all_gradients, all_parameters,
            learning_rate = symbolic_learning_rate)
        
        inputs = [symbolic_x]
        if self.use_count_sum:
            inputs.append(symbolic_n)
        inputs.append(symbolic_learning_rate)
        
        self.f_train = theano.function(
            inputs = inputs,
            outputs = [lower_bound_train, log_p_x_train, KL__train],
            updates = update_expressions)
        
        inputs = [symbolic_x]
        if self.use_count_sum:
            inputs.append(symbolic_n)
        
        self.f_eval = theano.function(
            inputs = inputs,
            outputs = [lower_bound_eval, log_p_x_eval, KL__eval])
       
        self.f_z = theano.function(inputs = [symbolic_x], outputs = [z_eval])
        
        inputs = [symbolic_z]
        if self.use_count_sum:
            inputs.append(symbolic_n)
        
        self.f_sample = theano.function(
            inputs = inputs,
            outputs = [x_theta_sample[p] for p in self.x_parameters])
        
        inputs = [symbolic_x]
        if self.use_count_sum:
            inputs.append(symbolic_n)
        
        self.f_recon = theano.function(
            inputs = inputs,
            outputs = [x_theta_eval[p] for p in self.x_parameters])
    
    def lowerBound(self, x, x_theta, z_mu, z_log_var):
        #note that we sum the latent dimension and mean over the samples
        log_px_given_z = self.expectedNegativeReconstructionError(x, x_theta, eps = 1e-6).sum(axis = 1).mean()
        KL_qp = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis = 1).mean()
        LL = - KL_qp + log_px_given_z
        return LL, log_px_given_z, KL_qp
    
    def train(self, training_set, validation_set = None, N_epochs = 50, batch_size = 100,
        learning_rate = 1e-3):
        
        M = training_set.shape[0]
        
        x_train = training_set
        x_valid = validation_set
        
        if self.use_count_sum:
            n_train = x_train.sum(axis = 1).reshape(-1, 1)
            n_valid = x_valid.sum(axis = 1).reshape(-1, 1)
        
        x_train = self.preprocess(x_train)
        x_valid = self.preprocess(x_valid)
        
        training_string = "Training model for {}".format(N_epochs)
        if self.number_of_epochs_trained > 0:
            training_string += " additional"
        training_string += " epoch{}".format("s" if N_epochs > 1 else "")
        training_string += " with a learning rate of {:.3g}".format(learning_rate)
        training_string += "."
        print(training_string)
        
        LL_train, logpx_train, KL_train = [], [], []
        LL_valid, logpx_valid, KL_valid = [], [], []
        
        training_start = time()
        
        for epoch in range(self.number_of_epochs_trained,
            self.number_of_epochs_trained + N_epochs):
            
            epoch_start = time()
            
            shuffled_indices = numpy.random.permutation(M)
            
            for i in range(0, M, batch_size):
                subset = shuffled_indices[i:(i + batch_size)]
                inputs = [x_train[subset]]
                if self.use_count_sum:
                    inputs.append(n_train[subset])
                inputs.append(learning_rate)
                out = self.f_train(*inputs)
            
            inputs = [x_train]
            if self.use_count_sum:
                inputs.append(n_train)
            out = self.f_eval(*inputs)
            
            LL_train += [out[0]] 
            logpx_train += [out[1]]
            KL_train += [out[2]]
            
            evaluation_string = "    Training set:   LB: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(float(out[0]), float(out[1]), float(out[2]))
            
            if x_valid is not None:
                inputs = [x_valid]
                if self.use_count_sum:
                    inputs.append(n_valid)
                out = self.f_eval(*inputs)
                
                LL_valid += [out[0]]
                logpx_valid += [out[1]]
                KL_valid += [out[2]]
                
                evaluation_string += "\n    Validation set: LB: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(float(out[0]), float(out[1]), float(out[2]))
            
            epoch_duration = time() - epoch_start
                
            print("Epoch {:2d} ({}):".format(epoch + 1, convertTimeToString(epoch_duration)))
            print(evaluation_string)
        
        training_duration = time() - training_start
        
        self.number_of_epochs_trained += N_epochs
        
        self.learning_curves["training"]["LB"] += LL_train
        self.learning_curves["training"]["ENRE"] += logpx_train
        self.learning_curves["training"]["KL"] += KL_train
        
        self.learning_curves["validation"]["LB"] += LL_valid
        self.learning_curves["validation"]["ENRE"] += logpx_valid
        self.learning_curves["validation"]["KL"] += KL_valid
        
        print("Training finished with a total of {} epoch{} after {}.".format(self.number_of_epochs_trained, "s" if N_epochs > 1 else "", convertTimeToString(training_duration)))
    
    def evaluate(self, test_set):
        
        x_test = test_set
        
        if self.use_count_sum:
            n_test = x_test.sum(axis = 1).reshape(-1, 1)
        
        x_test = self.preprocess(x_test)
        
        inputs = [x_test]
        if self.use_count_sum:
            inputs.append(n_test)
        lower_bound_test, _, _ = self.f_eval(*inputs)
        
        print("Lower bound for test set: {:.4g}.".format(float(lower_bound_test)))
        
        z_eval = self.f_z(x_test)[0]
        
        inputs = [numpy.random.normal(size = (100, self.latent_size)).astype('float32')]
        if self.use_count_sum:
            inputs.append(numpy.random.poisson(0.5, size = (100, 1)).astype('float32'))
        x_theta_sample = self.f_sample(*inputs)
        x_theta_sample = {p: o for p, o in zip(self.x_parameters, x_theta_sample)}
        x_sample = self.meanOfReconstructionDistribution(x_theta_sample)
        
        inputs = [x_test]
        if self.use_count_sum:
            inputs.append(n_test)
        x_test_recon = self.f_recon(*inputs)
        x_test_recon = {p: o for p, o in zip(self.x_parameters, x_test_recon)}
        x_test_recon["mean"] = self.meanOfReconstructionDistribution(x_test_recon)
        
        metrics = {
            "LL_test": lower_bound_test
        }
        
        return x_test, x_test_recon, z_eval, x_sample, metrics
    
    def save(self, name, metadata = None):
        
        model = {
            "feature shape": self.feature_shape,
            "latent size": self.latent_size,
            "hidden structure": self.hidden_structure,
            "encoder": get_all_param_values(self.encoder),
            "decoder": {
                p: get_all_param_values(self.decoder[p]) for p in self.x_parameters
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
        
        for p in self.x_parameters:
            set_all_param_values(self.decoder[p], model["decoder"][p])
        
        self.number_of_epochs_trained = model["number of epochs trained"]
        self.learning_curves = model["learning curves"]

def log_poisson(x, log_lambda, eps = 0.0):
    
    x = T.clip(x, eps, x)
    
    lambda_ = T.exp(log_lambda)
    lambda_ = T.clip(lambda_, eps, lambda_)
    
    y = x * log_lambda - lambda_ - T.gammaln(x + 1)
    
    return y

def log_negative_binomial(x, p, log_r, eps = 0.0):
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

def log_zero_inflated_poisson(x, pi, log_lambda, eps = 0.0):
    """
    Compute log pdf of a zero-inflated Poisson distribution with success probability pi and number of failures, r, until the experiment is stopped, at values x.
    
    A simple variation of Stirling's approximation is used: log x! = x log x - x.
    """
    
    pi = T.clip(pi, eps, 1.0 - eps)
    
    lambda_ = T.exp(log_lambda)
    lambda_ = T.clip(lambda_, eps, lambda_)
    
    y_0 = T.log(pi + (1 - pi) * T.exp(-lambda_))
    y_1 = T.log(1 - pi) + log_poisson(x, log_lambda, eps)
    
    y = T.eq(x, 0) * y_0 + T.gt(x, 0) * y_1
    
    return y

# TODO Gives NaNs?
def log_zero_inflated_negative_binomial(x, pi, p, log_r, eps = 0.0):
    
    pi = T.clip(pi, eps, 1.0 - eps)
    
    p = T.clip(p, eps, 1.0 - eps)
    
    r = T.exp(log_r)
    r = T.clip(r, eps, r)
    
    y_0 = T.log(pi + (1 - pi) * T.pow(1 - p, r))
    y_1 = T.log(1 - pi) + log_negative_binomial(x, p, log_r, eps)
    
    y = T.eq(x, 0) * y_0 + T.gt(x, 0) * y_1
    
    return y

def log_cross_entropy_extended(x, x_theta, log_distribution, k_max, eps = 0.0):
    
    p_k = x_theta["p_k"]
    
    F = x.shape[1]
    
    p_k = T.clip(p_k, eps, 1.0)
    x_k = T.clip(x, 0, k_max)
    
    p_k = T.reshape(p_k, (-1, k_max + 1))
    x_k = T.reshape(x_k, (-1, 1))
    
    y_cross_entropy = objectives.categorical_crossentropy(p_k, x_k)
    y_cross_entropy = T.reshape(y_cross_entropy, (-1, F))
    
    y_log_distribution = T.ge(x, k_max) * log_distribution(x - k_max, x_theta, eps)

    # y = - T.lt(x, 0) * y_cross_entropy + y_log_distribution
    y = - y_cross_entropy + T.lt(x, 0) * y_log_distribution
    # y = - y_cross_entropy + y_log_distribution
    
    return y

def meanOfCrossEntropyExtendedDistibution(x_theta, mean_of_distribution, k_max):
    
    p_k = x_theta["p_k"]
    
    mean = numpy.argmax(p_k, axis = -1).astype("float64")
    
    k_max_indices = mean == k_max
    
    mean[k_max_indices] += numpy.exp(mean_of_distribution(x_theta)[k_max_indices])
    
    return mean

def log_softmax_poisson(x, p_k, log_lambda, k_max = 10, eps = 0.0):
    
    F = x.shape[1]
    
    p_k = T.clip(p_k, eps, 1.0 - eps)
    x_k = T.clip(x, 0, k_max)
    
    p_k = T.reshape(p_k, (-1, k_max + 1))
    x_k = T.reshape(x_k, (-1, 1))
    
    y_cross_entropy = objectives.categorical_crossentropy(p_k, x_k)
    y_cross_entropy = T.reshape(y_cross_entropy, (-1, F))
    
    y_log_poisson = T.ge(x, k_max) * log_poisson(x - k_max, log_lambda, eps)

    y = - y_cross_entropy + y_log_poisson
    
    return y

def meanOfSoftmaxPoissonDistribution(p_k, log_lambda, k_max):
    
    mean = numpy.argmax(p_k, axis = -1).astype("float64")
    
    k_max_indices = mean == k_max
    
    mean[k_max_indices] += numpy.exp(log_lambda[k_max_indices])
    
    return mean

reconstruction_distributions = {
    "bernoulli": {
        "parameters": ["p"],
        "activation functions": {
            "p": sigmoid,
        },
        "function": lambda x, x_theta, eps = 0.0: \
            log_bernoulli(x, x_theta["p"], eps),
        "mean": lambda x_theta: x_theta["p"],
        # TODO Consider switching to Bernouilli sampling
        "preprocess": lambda x: (x != 0).astype('float32')
    },
    
    "negative_binomial": {
        "parameters": ["p", "log_r"],
        "activation functions": {
            "p": sigmoid,
            "log_r": lambda x: T.clip(x, -10, 10)
        },
        "function": lambda x, x_theta, eps = 0.0: \
            log_negative_binomial(x, x_theta["p"], x_theta["log_r"], eps),
        "mean": lambda x_theta: \
            meanOfNegativeBinomialDistribution(x_theta["p"], x_theta["log_r"]),
        "preprocess": lambda x: x
    },
    
    "poisson": {
        "parameters": ["log_lambda"],
        "activation functions": {"log_lambda": lambda x: T.clip(x, -10, 10)},
        "function": lambda x, x_theta, eps = 0.0: \
            log_poisson(x, x_theta["log_lambda"], eps),
        "mean": lambda x_theta: numpy.exp(x_theta["log_lambda"]),
        "preprocess": lambda x: x
    },
    
    "zero_inflated_poisson": {
        "parameters": ["pi", "log_lambda"],
        "activation functions": {
            "pi": sigmoid,
            "log_lambda": lambda x: T.clip(x, -10, 10)
        },
        "function": lambda x, x_theta, eps = 0.0: \
            log_zero_inflated_poisson(x, x_theta["pi"], x_theta["log_lambda"], eps),
        "mean": lambda x_theta: (1 - x_theta["pi"]) * numpy.exp(x_theta["log_lambda"]),
        "preprocess": lambda x: x
    },
    
    "zero_inflated_negative_binomial": {
        "parameters": ["pi", "p", "log_r"],
        "activation functions": {
            "pi": sigmoid,
            "p": sigmoid,
            "log_r": lambda x: T.clip(x, -10, 10)
        },
        "function": lambda x, x_theta, eps = 0.0: \
            log_zero_inflated_negative_binomial(x, x_theta["pi"], x_theta["p"],
                x_theta["log_r"], eps),
        "mean": lambda x_theta: (1 - x_theta["pi"]) \
            * meanOfNegativeBinomialDistribution(x_theta["p"], x_theta["log_r"]),
        "preprocess": lambda x: x
    },
    
    "softmax_poisson": {
        "parameters": ["p_k", "log_lambda"],
        "activation functions": {
            "p_k": softmax,
            "log_lambda": lambda x: T.clip(x, -10, 10)
        },
        "function": lambda x, x_theta, eps = 0.0: \
            log_softmax_poisson(x, x_theta["p_k"], x_theta["log_lambda"], k_max = 10,
                eps = eps),
        "mean": lambda x_theta: \
            meanOfSoftmaxPoissonDistribution(x_theta["p_k"], x_theta["log_lambda"], k_max = 10),
        "preprocess": lambda x: x
    }
}



if __name__ == '__main__':
    (training_set, training_headers), (validation_set, validation_headers), \
        (test_set, test_headers) = data.loadCountData("sample")
    feature_size = training_set.shape[1]
    model = VariationalAutoEncoder(feature_size, latent_size = 2, hidden_structure = [5], reconstruction_distribution = "negative_binomial")
    model.train(training_set, validation_set, N_epochs = 1, batch_size = 1)
    model.save("test")
    model.load("test")
    model.evaluate(test_set)
