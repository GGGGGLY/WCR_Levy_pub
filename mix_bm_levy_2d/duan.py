# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:53:49 2023

@author: gly
"""

import tensorflow as tf

from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

import keras
import keras.backend as K
#import tensorflow_probability as tfp

import sys
import numpy as np
from scipy.stats import levy_stable
import datetime

#tfd = tfp.distributions

# notebook parameters
random_seed = 1
step_size = 1e-2  # 5e-2 # step size
n_pts = 20000      # number of points

n_x=20 #number of different x

n_layers = 2
n_dim_per_layer = 25

n_dimensions = 1

ACTIVATIONS = tf.nn.elu
VALIDATION_SPLIT = .1
BATCH_SIZE = 100
#N_EPOCHS = 25

tf.random.set_seed(random_seed)

# data generation 

class SDEIntegrators:
    """
    Implements the common Euler-Maruyama 
    scheme used in integration of SDE.
    """

    def __init__(self):
        pass

    
    @staticmethod
    def euler_maruyama(xn, h, _f_sigma, rng):

        dW = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        #np.random.seed(random_seed)
        dL = levy_stable.rvs(alpha=1, beta=0, size=xn.shape ) #scale=h)    # added for levy 
        
        xk = xn.reshape(1, -1)  # we only allow a single point as input

        fk, sk = _f_sigma(xk)
        if np.prod(sk.shape) == xk.shape[-1]:
            skW = sk * dW
            skL = sk * dL *h 
        else:
            sk = sk.reshape(xk.shape[-1], xk.shape[-1])
            skW = (sk @ dW.T).T
            skL = ((sk @ dL.T).T)*h
        # return xk + h * fk + skW 
        return xk + h * fk + skL   # added for levy 

    
def sample_data(drift_diffusivity, step_size, n_dimensions, low, high, n_pts, rng, n_subsample=1, n_x=20):
    x_data=np.linspace(low,high,n_x+1)
    x_data=np.repeat(x_data[:-1], n_pts/n_x).reshape(-1, n_dimensions)
    y_data = x_data.copy()
    for k in range(n_subsample):
        y_data = np.row_stack([
            SDEIntegrators.euler_maruyama(y_data[k, :],
                                          step_size / n_subsample,
                                          drift_diffusivity,
                                          rng)
            for k in range(x_data.shape[0])
        ])

    return x_data, y_data


# EXAMPLE 1
def true_drift(x):
    return -x+1


def true_diffusivity(x):
    return (x+np.sqrt(25))*0.1 


def true_drift_diffusivity(x):
    return true_drift(x), true_diffusivity(x)


rng = np.random.default_rng(random_seed)

x_data, y_data = sample_data(true_drift_diffusivity,
                             step_size=step_size, n_dimensions=n_dimensions,
                             low=-3, high=3, n_pts=n_pts,
                             rng=rng)
print('data x shape', x_data.shape)

print('data y shape', y_data.shape)

step_sizes = np.zeros((x_data.shape[0],)) + step_size


    