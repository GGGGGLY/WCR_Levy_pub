# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:42:38 2023

trajectory fig

@author: gly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
import os
import seaborn as sns 
import matplotlib as mpl 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, diffusion_term, xi_term, drift_term_simu, \
                 diffusion_term_simu, xi_term_simu, alpha_levy, initialization, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.xi_term = xi_term
        self.drift_term_simu = drift_term_simu
        self.diffusion_term_simu = diffusion_term_simu
        self.xi_term_simu = xi_term_simu
        self.alpha_levy = alpha_levy
        self.initialization = initialization
        self.explosion_prevention = explosion_prevention
        self.explosion_prevention_N = 0

    #def drift(self, x):
    #    y = torch.zeros_like(x)
    #    for i in range(self.drift_term.shape[0]):
    #        for j in range(self.drift_term.shape[1]):
    #            y[:, i] = y[:, i] + self.drift_term[i, j] * x[:, i] ** j
    #    return y
    
    def drift(self, x):
        y = 0
        for i in range(self.drift_term.shape[0]):
            y = y + self.drift_term[i] * x ** i
        return y

    def drift_simu(self, x):
        y = 0
        for i in range(self.drift_term_simu.shape[0]):
            y = y + self.drift_term_simu[i] * x ** i
        return y

    #def drift_simu(self, x):
    #    y = torch.zeros_like(x)
    #    for i in range(self.drift_term_simu.shape[0]):
    #        for j in range(self.drift_term_simu.shape[1]):
    #            y[:, i] = y[:, i] + self.drift_term_simu[i, j] * x[:, i] ** j
    #    return y
    
    
    def diffusion(self, x):
        y = 0
        for i in range(self.diffusion_term.shape[0]):
            y = y + self.diffusion_term[i] * x ** i
        return y
    
    def diffusion_simu(self, x):
        y = 0
        for i in range(self.diffusion_term_simu.shape[0]):
            y = y + self.diffusion_term_simu[i] * x ** i
        return y
    
    def xi(self, x):
        y = 0
        for i in range(self.xi_term.shape[0]):
            y = y + self.xi_term[i] * x ** i
        return y
    
    def xi_simu(self, x):
        y = 0
        for i in range(self.xi_term_simu.shape[0]):
            y = y + self.xi_term_simu[i] * x ** i
        return y
    

    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.99999
        U = torch.rand(self.samples_num,self.dim)*0.99999
        W = -torch.log(U+1e-6)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    
    #def levy_variable(self):
    #    V = (torch.rand(self.samples_num,1)*np.pi - np.pi/2)*0.99999
    #    U = torch.rand(self.samples_num,1)*0.99999
    #    W = -torch.log(U+1e-6)
    #    X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
     #   LEVY = X
     #   for k in range(1, self.dim):
     #       V = (torch.rand(self.samples_num,1)*np.pi - np.pi/2)*0.99999
     #       U = torch.rand(self.samples_num,1)*0.99999
     #       W = -torch.log(U+1e-6)
     #       X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
     #       LEVY = torch.cat((LEVY, X), dim=1)
     #   return LEVY
    
    
    
    def subSDE_simu(self, t0, t1, x1, x2):
        if t0 == t1:
            return x1
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x1
            y_simu = x2
            for i in range(t.shape[0] - 1):
                rand_incre = torch.randn(self.samples_num, self.dim)
                levy_incre = self.levy_variable()
                y = y + self.drift(y) * self.dt + self.diffusion(y) * torch.sqrt(torch.tensor(self.dt)) * rand_incre \
                                + self.xi(y) * torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * levy_incre
                                 
                y_simu = y_simu + self.drift_simu(y_simu) * self.dt + self.diffusion_simu(y_simu) * torch.sqrt(torch.tensor(self.dt)) * rand_incre  \
                    + self.xi_simu(y_simu) *  torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * levy_incre
                                  
                if self.explosion_prevention:
                    if any(y < 0):
                        y[y < 0] = 0
                        self.explosion_prevention_N = self.explosion_prevention_N + 1
            return y, y_simu

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data_simu = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data[0, :, :] = self.subSDE_simu(0, self.time_instants[0], self.initialization, self.initialization)  
        data_simu[0, :, :] = data[0, :, :]
        
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :, :], data_simu[i + 1, :, :] = self.subSDE_simu(self.time_instants[i], self.time_instants[i + 1], data[i, :, :], data_simu[i, :, :])
            
        if self.explosion_prevention:
            print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            
            N = self.samples_num
            fig, ax = plt.subplots()
            plot1 = []
            plot2 = []
            for j in range(N):
                tt = torch.linspace(0, 1, steps=100)
                xx=data[:, j, 0].numpy()
                yy=data_simu[:, j, 0].numpy()
                p1 = ax.plot(tt, xx, color='blue', linewidth =3.0, linestyle='--',label='Exact trajectories')
                p2 = ax.plot(tt, yy, color='red', linewidth =1.5, label = 'Simulated trajectories')
                plot1.append(p1)
                plot2.append(p2)
            plt.title("Trajectories",fontsize=12)
            plt.xlabel('t',fontsize=12)
            plt.ylabel('X_t',fontsize=11)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.setp(plot1[1:], label="_") 
            plt.setp(plot2[1:], label="_")
            ax.legend(fontsize=14)
            plt.grid(True)
            plt.show()    
        return data


if __name__ == '__main__':
    
    dim = 1
    
    ###gauss
    #drift = torch.tensor([0, 1.0, 0, -1.0])
    #diffusion = torch.tensor([0.0, 1.0])
    #xi = torch.tensor([0.0])
      #xi = torch.tensor([1.0]).repeat(dim)
    #drift_simu = torch.tensor([0.0, 0.9674, 0.0, -0.9115])
      # np.sqrt(0.9158) = 0.9570
    #diffusion_simu = torch.tensor([0, 0.9570])
    #xi_simu = torch.tensor([0.0])
    
    ###levy
    drift = torch.tensor([0, 1.0, 0, -1.0])
    diffusion = torch.tensor([0.0])
    xi = torch.tensor([0.0, 1.0]) 
    drift_simu = torch.tensor([0.0, 0.90805, 0.0, -1.039701])
    diffusion_simu = torch.tensor([0.0])
    xi_simu = torch.tensor([0.0, 0.9089])
    
    samples_num = 6
    dataset = DataSet(torch.linspace(0, 1, steps=100), dt=0.001, samples_num=samples_num, dim=dim, \
                      drift_term=drift, diffusion_term=diffusion, xi_term = xi, \
                      drift_term_simu =drift_simu, diffusion_term_simu =diffusion_simu, xi_term_simu = xi_simu, \
                      alpha_levy = 3/2, initialization=torch.normal(0, 0.2, [samples_num, dim]), explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())