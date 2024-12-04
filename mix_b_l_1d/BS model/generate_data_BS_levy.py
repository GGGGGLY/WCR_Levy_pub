# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:36:38 2023

Generate data for BS model with levy noise

@author: gly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
import math
import seaborn as sns
from scipy import stats
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, xi_term,\
                 alpha_levy, initialization, label, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.xi_term = xi_term
        self.alpha_levy = alpha_levy
        self.initialization = initialization
        self.explosion_prevention = explosion_prevention
        self.label = label
        self.explosion_prevention_N = 0

    def drift(self, x):
        y = 0
        for i in range(self.drift_term.shape[0]):
            y = y + self.drift_term[i] * x ** i
        return y

    
    def xi(self, x):
        y = 0
        for i in range(self.xi_term.shape[0]):
            y = y + self.xi_term[i] * x ** i
        return y

    
    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.99999
        U = torch.rand(self.samples_num,self.dim)*0.99999
        W = -torch.log(U+1e-6)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    

    def subSDE(self, t0, t1, x):  # t0：时间间隔的开始；t1：时间间隔的结束；x：上次得到的位置
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                y = y + self.drift(y) * self.dt + self.xi(y) * torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable()
                #+ self.diffusion(y) * torch.sqrt(torch.tensor(self.dt)) * torch.randn(self.samples_num, self.dim)
                if self.explosion_prevention:
                    if any(y < 0):
                        y[y < 0] = 0
                        self.explosion_prevention_N = self.explosion_prevention_N + 1
            return y
        
                

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data[0, :, :] = self.subSDE(0, self.time_instants[0], self.initialization)  # self.initialization
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :, :] = self.subSDE(self.time_instants[i], self.time_instants[i + 1], data[i, :, :])
        if self.explosion_prevention:
            print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            for i in range(self.dim):
                #plt.figure()
                #plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[data.min().numpy(), data.max().numpy()], density=True)
                sns.distplot(x=data[-1, :, i].numpy(),bins=100, hist=False, kde_kws={"label":"%s" % (self.label)},axlabel='x') 
                #sns.distplot(x=data[-1, :, i].numpy(),bins=1000,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" }) 
            plt.show()
        
        return data




if __name__ == '__main__':
    drift = torch.tensor([0, 1, 0, -1])
    xi = torch.tensor([0, 1])      #0, 1, 2, 5, 8, 9, 10
    label = "Levy"
    dataset = DataSet(torch.tensor([0.1, 0.3, 0.4, 0.7, 0.8, 1, 1.2]), dt=0.001, samples_num=10000, dim=1,
                      drift_term=drift, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]), label = label, explosion_prevention=False) #sample_num = 2000: out of range
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
    






