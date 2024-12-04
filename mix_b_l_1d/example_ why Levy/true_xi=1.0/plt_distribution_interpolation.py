# -*- coding: utf-8 -*-
"""
Plot distribution： interpolation t = 0.6

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

from scipy.stats import gaussian_kde
from generate_data_1d import DataSet
from generate_data_1d_interpolate import DataSet_plot_inter
from weak_gaussian_1d_dist import Gaussian
from weak_gaussian_1d_dist import Model
import scipy.stats


if __name__ == '__main__':
    np.random.seed(7)
    torch.manual_seed(7)

    drift = torch.tensor([0, 1, 0, -1])
    diffusion = torch.tensor([1])
    xi = torch.tensor([1.0])      #0, 1, 2, 5, 8, 9, 10
    t_raw = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    plt.figure()
    label1 = "Interpolation of Real Data"
    dataset_raw = DataSet(t_raw, dt=0.0001, samples_num=10000, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label1, explosion_prevention=False) #sample_num = 2000: out of range
    
    data_raw, data_distT1 = dataset_raw.get_data(plot_hist=False)
    print("data_raw.size: ", data_raw.size())
    #print("data.max: ", data.max(), "data.min: ", data.min())
    px1 = data_distT1
    
    
    '''
    To estimate the unknown coefficient, the data used is [0,1], while the actual data is [0,1.2]. 
    In other words, we divide the actual data into test data and train data.
    '''
    
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #dataset = DataSet(t, dt=0.001, samples_num=10000, dim=1,
     #                 drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
     #                 explosion_prevention=False)
    #data = dataset.get_data(plot_hist=False)
    data = data_raw[:10, :, :]
    
    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=3, gauss_variance=1.0, type='LMM_2_nonequal', drift_term=drift, diffusion_term=diffusion,
                  gauss_samp_way='lhs', lhs_ratio=1.0)
    drift_est, diffusion_est = model.train(gauss_samp_number=100, lam=0.0, STRidge_threshold=0.05)
    
    '''
    Distribution of interpolation time for initial SDE
    '''
    t_inter = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])
    groundtruth = DataSet_plot_inter(t_inter, dt=0.0001, samples_num=10000, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label1, explosion_prevention=False) #sample_num = 2000: out of range
    
    inter_plot_true = groundtruth.get_data(plot_hist=True)
    
    '''
    Predictive distribution of interpolation time for estimated values
    '''
    
    label = "Estimated SDE"
    dataset_plt = DataSet_plot_inter(t_inter, dt=0.0001, samples_num=10000, dim=1,
                      drift_term=drift_est, diffusion_term=diffusion_est, xi_term=torch.tensor([0.0]) , alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      label = label, explosion_prevention=False)
    data_plt, data_distT2 = dataset_plt.get_data(plot_hist=True)
    px2 = data_distT2
    
    kde_x1 = gaussian_kde(px1)
    kde_x2 = gaussian_kde(px2)

    x_vals = np.linspace(min(min(px1), min(px2)), max(max(px1), max(px2)), 1000)

    l1_error = np.mean(np.abs(kde_x1(x_vals) - kde_x2(x_vals)) * np.diff(x_vals[::len(x_vals)-1]))
    #L1_err = torch.mean(torch.abs(px1-px2))
    #scipy.stats.wasserstein_distance(px1, px2)
    plt.legend()
    plt.title("Distribution of interpolation: t=0.75, L1 error = %f" %(l1_error))
    #plt.title("Distribution of interpolation: t=0.75, WD = %f" %(WD))
    plt.show()
    

