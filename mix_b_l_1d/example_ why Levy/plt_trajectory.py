
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:33:13 2023

trajectory fig: 1d

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
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, diffusion_term, xi_term, drift_gauss_est, \
                 diffusion_gauss_est, drift_mixed_est, diffusion_mixed_est, xi_mixed_est, alpha_levy, initialization, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.xi_term = xi_term
        self.drift_gauss_est = drift_gauss_est
        self.diffusion_gauss_est = diffusion_gauss_est
        self.drift_mixed_est = drift_mixed_est
        self.diffusion_mixed_est = diffusion_mixed_est
        self.xi_mixed_est = xi_mixed_est
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
    
    def diffusion(self, x):
        y = 0
        for i in range(self.diffusion_term.shape[0]):
            y = y + self.diffusion_term[i] * x ** i
        return y
    
    def drift_gauss(self, x):
        y = 0
        for i in range(self.drift_gauss_est.shape[0]):
            y = y + self.drift_gauss_est[i] * x ** i
        return y
    
    def diffusion_gauss(self, x):
        y = 0
        for i in range(self.diffusion_gauss_est.shape[0]):
            y = y + self.diffusion_gauss_est[i] * x ** i
        return y
    
    def drift_mixed(self, x):
        y = 0
        for i in range(self.drift_mixed_est.shape[0]):
            y = y + self.drift_mixed_est[i] * x ** i
        return y
    
    def diffusion_mixed(self, x):
        y = 0
        for i in range(self.diffusion_mixed_est.shape[0]):
            y = y + self.diffusion_mixed_est[i] * x ** i
        return y
    
    

    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.99999
        U = torch.rand(self.samples_num,self.dim)*0.99999
        W = -torch.log(U+1e-6)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    
 
    
    def subSDE_simu(self, t0, t1, x, x1, x2):
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            y_gauss = x1
            y_mixed = x2
            for i in range(t.shape[0] - 1):
                random_incre = torch.randn(self.samples_num, self.dim)
                levy_incre = self.levy_variable()
                y = y + self.drift(y) * self.dt + self.xi_term * torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * levy_incre \
                + self.diffusion(y) * torch.sqrt(torch.tensor(self.dt)) * random_incre
                                 
                
                y_mixed = y_mixed + self.drift_mixed(y_mixed) * self.dt + torch.sqrt(torch.tensor(self.dt)) * \
                    random_incre* self.diffusion_mixed(y_mixed) + \
                        self.xi_mixed_est * torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * levy_incre
                        
                y_gauss = y_gauss + self.drift_gauss(y_gauss) * self.dt  + self.diffusion_gauss(y_gauss) * torch.sqrt(torch.tensor(self.dt)) * random_incre
                 
                    
                
                if self.explosion_prevention:
                    if any(y < 0):
                        y[y < 0] = 0
                        self.explosion_prevention_N = self.explosion_prevention_N + 1
            return y, y_gauss, y_mixed

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data_gauss = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data_mixed = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        x0 = self.initialization
        data[0, :, :] = self.subSDE_simu(0, self.time_instants[0], x0, x0, x0)  
        #y0, y1, y2 = self.subSDE_simu(0, self.time_instants[0], x0, x0, x0)  
        #data[0, :, :],  data_gauss[0, :, :],  data_mixed[0, :, :] = y0, y1, y2
        data_gauss[0, :, :] = data[0, :, :]
        data_mixed[0, :, :] = data[0, :, :]      
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :, :], data_gauss[i + 1, :, :], data_mixed[i + 1, :, :] = self.subSDE_simu(self.time_instants[i], self.time_instants[i + 1], data[i, :, :], data_gauss[i, :, :], data_mixed[i, :, :])
            
        if self.explosion_prevention:
            print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            
            N = self.samples_num
            fig, ax = plt.subplots()
            plot0 = []
            plot1 = []
            plot2 = []
            for j in range(N):
                tt = torch.linspace(0, 1, steps=100)
                xx=data[:, j, 0].numpy()
                yy1=data_gauss[:, j, 0].numpy()
                yy2=data_mixed[:, j, 0].numpy()
                p0 = ax.plot(tt, xx, color='royalblue', linewidth =6.0, linestyle=':',label='Exact trajectories')
                p1 = ax.plot(tt, yy1, color='purple', linewidth =1.5, label = 'Simulated: Only Gaussian')
                p2 = ax.plot(tt, yy2, color='red', linewidth =1.5, label = 'Simulated: Both Gaussian and Levy')
                plot0.append(p0)
                plot1.append(p1)
                plot2.append(p2)
            plt.title("Trajectories",fontsize=20)
            plt.xlabel('t',fontsize=18)
            plt.ylabel('X_t',fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.setp(plot0[1:], label="_") 
            plt.setp(plot1[1:], label="_") 
            plt.setp(plot2[1:], label="_")
            ax.legend(fontsize=16)
            plt.grid(True)
            plt.show()    
        return data


if __name__ == '__main__':
    
    dim = 1
    drift = torch.tensor([0, 1.0, 0, -1.0])
    diffusion = torch.tensor([1.0])
    xi = torch.tensor([0.2])
    
    drift_gauss_est = torch.tensor([0, 0.9606894, 0, -0.9507338])
    diffusion_gauss_est = torch.tensor([1.0426])
    
    
    drift_mixed_est = torch.tensor([0, 0.99895173, 0, -1.0620492])
    diffusion_mixed_est = torch.tensor([1.0134])
    xi_mixed_est = torch.tensor([0.2132])
    
    
    samples_num = 6
    dataset = DataSet(torch.linspace(0, 1, steps=100), dt=0.001, samples_num=samples_num, dim=dim, \
                      drift_term=drift, diffusion_term=diffusion, xi_term = xi, \
                      drift_gauss_est =drift_gauss_est, diffusion_gauss_est =drift_gauss_est, \
                      drift_mixed_est = drift_mixed_est, diffusion_mixed_est = diffusion_mixed_est, xi_mixed_est = xi_mixed_est,     
                      alpha_levy = 3/2, initialization=torch.normal(0, 0.2, [samples_num, dim]), explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())