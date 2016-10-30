#!/usr/bin/env python

import theano
import theano.tensor as T

import numpy

from lasagne.layers import (
    InputLayer, DenseLayer,
    get_output,
    get_all_params
)
from parmesan.layers.sample import SimpleSampleLayer, SampleLayer

from lasagne.nonlinearities import identity, sigmoid, rectify, softmax, softplus, tanh

from parmesan.distributions import (
    log_stdnormal, log_normal2, log_bernoulli,
    kl_normal2_stdnormal
)

from lasagne import updates

class VAE(object):
    def __init__(self, feature_shape, latent_size, hidden_structure):
        
        super(VAE, self).__init__()
        
        self.feature_shape = feature_shape
        self.latent_size = latent_size
        self.hidden_structure = hidden_structure
        
        print("Setting up model.")
        
        sym_x = T.matrix('x')
        sym_z = T.matrix('z')
        
        #ENCODER
        l_in_x = InputLayer(shape=(None, feature_shape), name = "ENC_INPUT")
        l_enc = l_in_x
        for i, hidden_size in enumerate(hidden_structure):
            l_enc = DenseLayer(l_enc, num_units = hidden_size, nonlinearity = rectify, name = 'ENC_DENSE{:d}'.format(i + 1))
        l_muq = DenseLayer(l_enc, num_units=latent_size, nonlinearity=None, name = 'ENC_Z_MU')
        l_logvarq = DenseLayer(l_enc, num_units=latent_size, nonlinearity=lambda x: T.clip(x,-10,10), name = 'ENC_Z_LOG_VAR')
        # Sample a latent representation z \sim q(z|x) = N(mu(x),logvar(x))
        l_z = SimpleSampleLayer(mean=l_muq, log_var=l_logvarq, name = "ENC_SAMPLE")
        
        #we split the model into two parts to allow sampling from the decoder model separately
        #DECODER
        l_in_z = InputLayer(shape=(None, latent_size), name = "DEC_INPUT")
        l_dec = l_in_z
        for i, hidden_size in enumerate(reversed(hidden_structure)):
            l_dec = DenseLayer(l_in_z, num_units = hidden_size, nonlinearity = rectify, name = 'DEC_DENSE{:d}'.format(len(hidden_structure) - i)) 
        l_mux = DenseLayer(l_dec, num_units=feature_shape, nonlinearity=sigmoid, name = 'DEC_X_MU')  #reconstruction of input using a sigmoid output since mux \in [0,1]
        
        z_train, muq_train, logvarq_train = get_output([l_z,l_muq,l_logvarq],{l_in_x:sym_x},deterministic=False)
        mux_train = get_output(l_mux,{l_in_z:z_train},deterministic=False)
        
        z_eval, muq_eval, logvarq_eval = get_output([l_z,l_muq,l_logvarq],{l_in_x:sym_x},deterministic=True)
        mux_eval = get_output(l_mux,{l_in_z:z_eval},deterministic=True)
        
        mux_sample = get_output(l_mux,{l_in_z:sym_z},deterministic=True)
        
        LL_train, logpx_train, KL_train = self.logLikelihood(mux_train, sym_x, muq_train, logvarq_train)
        LL_eval, logpx_eval, KL_eval = self.logLikelihood(mux_eval, sym_x, muq_eval, logvarq_eval)

        all_params = get_all_params([l_z, l_mux], trainable = True)
        
        print("Parameters to train:")
        for parameter in all_params:
            print("{}: {}".format(parameter, parameter.get_value().shape))
        
        # Let Theano do its magic and get all the gradients we need for training
        all_grads = T.grad(-LL_train, all_params)

        # Set the update function for parameters. The Adam optimizer works really well with VAEs.
        update_expressions = updates.adam(all_grads, all_params, learning_rate=1e-2)

        self.f_train = theano.function(inputs=[sym_x],
                                  outputs=[LL_train, logpx_train, KL_train],
                                  updates = update_expressions)

        self.f_eval = theano.function(inputs=[sym_x],
                                 outputs=[LL_eval, logpx_eval, KL_eval])

        self.f_z = theano.function(inputs=[sym_x],
                                 outputs=[z_eval])

        self.f_sample = theano.function(inputs=[sym_z],
                                 outputs=[mux_sample])

        self.f_recon = theano.function(inputs=[sym_x],
                                 outputs=[mux_eval])
    
    def train(self, x_train, x_valid = None, N_epochs = 50, batch_size = 100):
        
        print("Training model.")
        
        LL_train, KL_train, logpx_train = [], [], []
        LL_valid, KL_valid, logpx_valid = [], [], []
        
        N = x_train.shape[0]
        
        for epoch in range(N_epochs):
            
            print("Epoch {:d}".format(epoch + 1))
            
            shuffled_indices = numpy.random.permutation(N)
            
            for i in range(0, N, batch_size):
                subset = shuffled_indices[i:(i + batch_size)]
                x_batch = x_train[subset]
                out = self.f_train(x_batch)
            
            out = self.f_eval(x_train)
            LL_train += [out[0]] 
            logpx_train += [out[1]]
            KL_train += [out[2]]
            
            if x_valid is not None:
                out = self.f_eval(x_valid)
                LL_valid += [out[0]]
                logpx_valid += [out[1]]
                KL_valid += [out[2]]
                
                z_eval = self.f_z(x_valid)[0]
                x_sample = self.f_sample(numpy.random.normal(size=(100, self.latent_size)).astype('float32'))[0]
                x_recon = self.f_recon(x_valid)[0]
    
    def evaluate(self):
        pass
    
    
    def logLikelihood(self, mux, x, muq, logvarq):
        log_px_given_z = log_bernoulli(x, mux, eps=1e-6).sum(axis=1).mean() #note that we sum the latent dimension and mean over the samples
        KL_qp = kl_normal2_stdnormal(muq, logvarq).sum(axis=1).mean()
        LL = log_px_given_z - KL_qp
        return LL, log_px_given_z, KL_qp
