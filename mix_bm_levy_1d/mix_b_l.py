# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:37:00 2022

@author: gly

Generate Data of    dX_t = drift(X_t) dt + diffusion(X_t) dB_t  + xi(X_t) dL_t,  0<=t<=1
time_instants: E.g. torch.tensor([0, 0.2, 0.5, 1])
samples_num: E.g. 10000
dim: E.g. 1
drift_term of SDE: E.g. torch.tensor([0, 1, 0, -1]) -- that means drift = x - x^3
diffusion_term of SDE: E.g. torch.tensor([1]) -- that means diffusion = 1
xi only test 1, x, ..., x^p
levy increments satisfies alpha in (0,1)cup(1,2]: general: paper generative regession
alpha_levy: the stability of levy increments
increment ~ S^{alpha}(1,0,0); beta = 0 symmetric
return data: [time, samples, dim]
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

class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, diffusion_term, xi_term,
                 alpha_levy, initialization, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.xi_term = xi_term
        self.alpha_levy = alpha_levy
        self.initialization = initialization
        self.explosion_prevention = explosion_prevention

        self.explosion_prevention_N = 0

    def drift(self, x):
        y = 0
        for i in range(self.drift_term.shape[0]):
            y = y + self.drift_term[i] * x ** i
        return y

    def diffusion(self, x):
        y = 0
        for i in range(self.diffusion_term.shape[0]):
            y = y + self.diffusion_term[i] * x ** i
        return y
    
    def xi(self, x):
        y = 0
        for i in range(self.xi_term.shape[0]):
            y = y + self.xi_term[i] * x ** i
        return y

    
    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.999993
        U = torch.rand(self.samples_num,self.dim)*0.999993
        W = -torch.log(U+1e-7)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*\
            (torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    

    def subSDE(self, t0, t1, x):  # t0：时间间隔的开始；t1：时间间隔的结束；x：上次得到的位置
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                y = y + self.drift(y) * self.dt + self.xi(y) * torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() + self.diffusion(y) * torch.sqrt(torch.tensor(self.dt)) * torch.randn(self.samples_num, self.dim)
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
                plt.figure()
                #plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[data.min().numpy(), data.max().numpy()], density=True)
                sns.distplot(x=data[-1, :, i].numpy(),bins=1000,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" }) 
            plt.show()
        return data




if __name__ == '__main__':
    drift = torch.tensor([0, 0.94, 0, -1])
    diffusion = torch.tensor([1])
    xi = torch.tensor([1])      #0, 1, 2, 5, 8, 9, 10
    dataset = DataSet(torch.tensor([0, 1, 5]), dt=0.001, samples_num=10000, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[10000, 1]),
                      explosion_prevention=False) #sample_num = 2000: out of range
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())

