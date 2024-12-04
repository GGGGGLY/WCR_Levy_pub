# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:45:56 2023

@author: gly
"""

import numpy as np
import torch
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns 
import matplotlib as mpl 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.stats import levy_stable


class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, xi_term, alpha_levy,
                 initialization, drift_independence=True, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.xi_term = xi_term
        self.alpha_levy = alpha_levy
        self.initialization = initialization
        self.drift_independence = drift_independence
        self.explosion_prevention = explosion_prevention
        self.shape_t = self.time_instants.shape[0]
        

    def drift(self, x): 
        y = 0
        for i in range(self.drift_term.shape[1]):
            y = y + self.drift_term[:, i] * x ** i
        return y
    
          

    def hat3d_ex1(self, x):
        ####  V = -5(x2+y2+z2) + (x2+y2+z2)2,   drift = - grad V
        norm = torch.sum(x ** 2, dim=1).unsqueeze(1)
        norm2 = norm.repeat(1, x.shape[1])
        return 10*x - 4 * x * norm2
       
    
    def hat3d_ex2(self, x):
        # V =    drift = - grad V
        Cross = x[:,0].reshape(self.samples_num,1)*x[:,1].reshape(1,self.samples_num)
        #print(Cross.shape) 
        grad = -4*torch.mm(Cross, x ) - 2.5*x - 2*torch.mm(Cross, torch.ones(self.samples_num, self.dim))\
            - (7/4)*torch.ones(self.samples_num, self.dim)-\
            torch.cat( (x[:,1].reshape(self.samples_num,1)**2 + 5* x[:,1].reshape(self.samples_num,1), x[:,0].reshape(self.samples_num,1)**2 + 5* x[:,0].reshape(self.samples_num,1)), dim=1)
        return grad
   
   
    def SubDiag(self, x): #x: N*d 
        """
        one-order: [X1, X1+X2, X2+X3]
        """
        x_prime = x[:, [0,0,1,2] ]
        x_prime[:,0] = 0
        row_index_to_remove = self.dim
        xx_prime = x_prime[:, : row_index_to_remove]
        y = xx_prime + x
        return y
    
  
    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.9999
        U = torch.rand(self.samples_num,self.dim)*0.9999
        W = -torch.log(U+1e-5)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    
   
    def subSDE(self, t0, t1, x):
        if self.drift_independence: 
            """
            each dim is independent and the same
            """
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
            
        elif self.drift_term.shape == torch.Size([self.dim, self.dim + 1]):  
            """
            one-order polynomial
            """
            if t0 == t1:
                return x
            else:
                t = torch.arange(t0, t1 + self.dt, self.dt)
                y = x
                for i in range(t.shape[0] - 1):
                    y = y + self.SubDiag(y) * self.dt + torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() * \
                            torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term))
                    if self.explosion_prevention:
                        if any(y < 0):
                            y[y < 0] = 0
                            self.explosion_prevention_N = self.explosion_prevention_N + 1
                return y
            
        
        
        elif self.drift_term.shape == torch.Size([3, 20]) or self.drift_term.shape == torch.Size([2, 10]):   #2d,3d含交叉项 （2d最高次为3次）3-order
            """
            two or three dims, three-order polynomial, has cross-terms
            """
            if t0 == t1:
                return x
            else:
                t = torch.arange(t0, t1 + self.dt, self.dt)
                y = x
                for i in range(t.shape[0] - 1):
                   # dL = self.levy_rv(y, torch.tensor(self.dt))
                    #print("dL shape", dL.shape)
                    #y = y + self.hat3d_ex4(y)* self.dt + torch.mm(dL, torch.diag(self.xi_term))
                    y = y + self.hat3d_ex1(y)* self.dt + torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) *\
                        torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term)) * self.levy_variable()
                    if self.explosion_prevention:
                        if any(y < 0):
                            y[y < 0] = 0
                            self.explosion_prevention_N = self.explosion_prevention_N + 1
                return y
            
                

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data[0, :, :] = self.subSDE(0, self.time_instants[0], self.initialization)
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :, :] = self.subSDE(self.time_instants[i], self.time_instants[i + 1], data[i, :, :])
          
        #data = torch.where(torch.isinf(data), torch.full_like(data, 0), data)
        #data = torch.where(torch.isnan(data), torch.full_like(data, 0), data)
        if self.explosion_prevention:
            print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            for i in range(self.dim):
                plt.figure()
                plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[-5,5], density=True)
                #plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[data.min().numpy(), data.max().numpy()], density=True)
                #sns.set_palette("hls") 
                #mpl.rc("figure", figsize=(5,4))
                #sns.distplot(x=data[-1, :, i].numpy(),bins=1000,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
            plt.show()
        return data




if __name__ == '__main__':
    torch.manual_seed(6)
    dim=3
    # drift = torch.tensor([[0, -0.5, 0, 0], [0, 0, -0.7, 0], [0, 0, 0, -1]])
    # drift = torch.tensor([[0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1]])
    drift = torch.tensor([[0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, -4, 0, -4, 0, 0, 0, 0],
                           [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4, 0],
                           [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4]])
    #drift = torch.tensor([[-7/4, -5/2, -5, 0, -2, -1, 0, -4,0, 0], [-7/4, -5/2, -5, 0, -2, -1, 0, -4,0, 0]])
    #or
    #drift = torch.tensor([[-7/4, -5/2, -5, 0, -2, -1, 0, 0, -4, 0], [-7/4, -5/2, -5, 0, -2, -1, 0, 0, -4, 0]])
    #drift = torch.tensor([[-1.75, -2.5, -5, 0, -2, -1, 0, 0, -4, 0], [-1.75, -5, -2.5, -1, -2, 0, 0, -4, 0, 0]]) 
    #drift = torch.tensor([[-1.75, -2.5, -5, 0, -2, -1, 0, -4, 0, 0], [-1.75, -5, -2.5, -1, -2, 0, 0, 0, -4, 0]]) 
    #drift = torch.tensor([[0, 10, 0, 0, 0, 0, -4, 0, -4, 0], [0, 10, 0, 0, 0, 0, -4, 0, -4, 0]])
   
    #drift = torch.tensor([[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    #drift = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, -4, 0, -4, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4, 0],
    #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4]])
    
   
    xi = torch.tensor([1.0]).repeat(dim)
    samples = 10000
    #t = np.array([0.1, 0.3, 0.5, 0.7, 1.0]).astype(np.float32)
    #t = torch.tensor(t)
    t =torch.linspace(0, 1.0, 11)
    dataset = DataSet(t, dt=0.001, samples_num=samples, dim=dim,\
                      drift_term=drift, xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, dim]),\
                      drift_independence=False, explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
   
    plt.figure(figsize=(10, 10))
    plt.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], density=True)
    plt.figure(figsize=(10, 10))
    plt.hist2d(x=data[-1, :, 1].numpy(), y=data[-1, :, 2].numpy(), bins=50, range=[[-5, 5], [-5, 5]], density=True)
    
   
    
   