# -*- coding: utf-8 -*-

''''
Generate Data of    dX_t = drift(X_t) dt + diffusion(X_t) dB_t,  0<=t<=1
time_instants: E.g. torch.tensor([0, 0.2, 0.5, 1])
samples_num: E.g. 10000
dim: E.g. 2
drift_term: E.g. torch.tensor([[0, 1, 0, -1], [0, 1, 0, 1]]) -- that means drift = x - x^3
diffusion_term: E.g. torch.tensor([1, 1]) -- that means diffusion = I2
return data: [time, samples, dim]
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
import os
import seaborn as sns 
import matplotlib as mpl 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, xi_term,
                 alpha_levy, initialization, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.xi_term = xi_term
        self.alpha_levy = alpha_levy
        self.initialization = initialization
        self.explosion_prevention = explosion_prevention
        self.explosion_prevention_N = 0

    def drift(self, x):
        y = torch.zeros_like(x)
        for i in range(self.drift_term.shape[0]):
            for j in range(self.drift_term.shape[1]):
                y[:, i] = y[:, i] + self.drift_term[i, j] * x[:, i] ** j
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
    
    def subSDE(self, t0, t1, x):
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                y = y + self.drift(y) * self.dt + \
                        torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() * \
                            torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term))
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
                sns.set_palette("hls") 
                mpl.rc("figure", figsize=(5,4))
                sns.distplot(x=data[-1, :, 0].numpy(),bins=1000,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
            plt.show()
        return data


if __name__ == '__main__':
    # drift = torch.tensor([[0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1]])
    # diffusion = torch.tensor([1, 1, 1])
    dim = 2
    drift = torch.tensor([[0,1.0,0,-1.0]]).repeat(dim,1)
    xi = torch.tensor([1.0]).repeat(dim)
    samples_num = 6000 
    dataset = DataSet(torch.tensor([0,1.0]), dt=0.001, samples_num=samples_num, dim=dim, \
                      drift_term=drift, xi_term = xi, alpha_levy = 3/2, \
                      initialization=torch.normal(0, 0.2,[samples_num, dim]), explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())