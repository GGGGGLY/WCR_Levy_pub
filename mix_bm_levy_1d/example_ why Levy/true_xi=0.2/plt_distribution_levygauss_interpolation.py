# -*- coding: utf-8 -*-
"""
Plot distribution: weak levy

interpolation 

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

from generate_data_1d import DataSet
from generate_data_1d_interpolate import DataSet_plot_inter
from WeakGaussianLevy_1d_dist import Gaussian
from WeakGaussianLevy_1d_dist import Model



if __name__ == '__main__':
    np.random.seed(7)
    torch.manual_seed(7)

    drift = torch.tensor([0, 1, 0, -1])
    diffusion = torch.tensor([1])
    xi = torch.tensor([0.2])     
    t_raw = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    plt.figure()
    label1 = "Groundtruth"
    dataset_raw = DataSet(t_raw, dt=0.0001, samples_num=10000, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label1, explosion_prevention=False) #sample_num = 2000: out of range
    data_raw = dataset_raw.get_data(plot_hist=False)
    print("data_raw.size: ", data_raw.size())
    #print("data.max: ", data.max(), "data.min: ", data.min())
    
    
    t_inter = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])
    groundtruth = DataSet_plot_inter(t_inter, dt=0.0001, samples_num=10000, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label1, explosion_prevention=False) #sample_num = 2000: out of range
    
    inter_plot_true = groundtruth.get_data(plot_hist=True)
    
    
    '''
    To estimate the unknown coefficient, the data used is [0,1], while the actual data is [0,1.2]. 
    In other words, we divide the actual data into test data and train data.
    '''
    
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    data = data_raw[:10, :, :]
    
    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=3,basis_xi_order=1, gauss_variance=0.55, type='LMM_2_nonequal', drift_term=drift, diffusion_term=diffusion, xi_term=xi,
                  gauss_samp_way='lhs', lhs_ratio=0.85)
    drift_est, diffusion_est, xi_est = model.train(gauss_samp_number=70, lam=0.0, STRidge_threshold=0.05)
    
    '''
    Predictive distribution of extrapolation time for estimated values
    '''
    
    label = "Estimated SDE"
    dataset_plt = DataSet_plot_inter(t_inter, dt=0.0001, samples_num=10000, dim=1,
                      drift_term=drift_est, diffusion_term=diffusion_est, xi_term= xi_est, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label, explosion_prevention=False)
    data_plt = dataset_plt.get_data(plot_hist=True)
    plt.legend()
    plt.title("Distribution of extrapolation: t=0.75")
    plt.show()
    

