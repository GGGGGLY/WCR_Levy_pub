# -*- coding: utf-8 -*-
"""
plot distribution of Black-Scholes driven by Levy noise

@author: gly
"""



import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
import math
from visdom import Visdom
import seaborn as sns
from scipy import stats
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from generate_data_BS_levy import DataSet
from weak_BlackScholes_Levy_pltDist import Gaussian
from weak_BlackScholes_Levy_pltDist import Model



if __name__ == '__main__':
    np.random.seed(100)
    torch.manual_seed(100)

    drift = torch.tensor([0, 1, 0, -1])
    xi = torch.tensor([0, 1.0])
    samples = 10000 
    t_raw = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    plt.figure()
    label1 = "test"
    dataset_raw = DataSet(t_raw, dt=0.0001, samples_num=samples, dim=1,
                      drift_term=drift, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label1, explosion_prevention=False) #sample_num = 2000: out of range
    data_raw = dataset_raw.get_data(plot_hist=True)
    print("data_raw.size: ", data_raw.size())
    #print("data.max: ", data.max(), "data.min: ", data.min())
    
    '''
    To estimate the unknown coefficient, the data used is [0,1], while the actual data is [0,1.2]. 
    In other words, we divide the actual data into test data and train data.
    '''
    
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    data = data_raw[:10, :, :]
    
    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=3,basis_xi_order=1, gauss_variance=0.3, type='LMM_2_nonequal', drift_term=drift, xi_term=xi,\
                  gauss_samp_way='lhs', lhs_ratio=1.0)
    drift_est, xi_est = model.train(gauss_samp_number=70, lam=0.0, STRidge_threshold=0.1)
    
    '''
    Predictive distribution of extrapolation time for estimated values
    '''
    
    label = "train"
    dataset_plt = DataSet(t_raw, dt=0.0001, samples_num=10000, dim=1, drift_term=drift_est, \
                      xi_term= xi_est, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label, explosion_prevention=False)
    data_plt = dataset_plt.get_data(plot_hist=True)
    plt.legend()
    plt.show()
    

