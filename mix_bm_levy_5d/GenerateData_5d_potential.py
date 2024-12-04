# -*- coding: utf-8 -*-
"""
Generate data for 5d problem with potential drift terms.

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
    
          

    def hat5d_ex1(self, x):
        ####  V = -5(x2+y2+z2) + (x2+y2+z2)2,   drift = - grad V
        norm = torch.sum(x ** 2, dim=1).unsqueeze(1)
        norm2 = norm.repeat(1, x.shape[1])
        return 10 * x - 4 * x * norm2
       
    
    def hat3d_ex2(self, x):
        # V =    drift = - grad V
        Cross = x[:,0].reshape(self.samples_num,1)*x[:,1].reshape(1,self.samples_num)
        #print(Cross.shape) 
        grad = -4*torch.mm(Cross, x ) - 2.5*x - 2*torch.mm(Cross, torch.ones(self.samples_num, self.dim))\
            - (7/4)*torch.ones(self.samples_num, self.dim)-\
            torch.cat( (x[:,1].reshape(self.samples_num,1)**2 + 5* x[:,1].reshape(self.samples_num,1), x[:,0].reshape(self.samples_num,1)**2 + 5* x[:,0].reshape(self.samples_num,1)), dim=1)
        return grad
    
    def hat3d_ex3(self, x):
        if x.shape[1] == 2:
            y1 = 5*x[:,0] - x[:,1]**2
            y2 = 5*x[:,1] + x[:,0]**2
            y1 = y1.reshape(self.samples_num,1)
            y2 = y2.reshape(self.samples_num,1)            
   
            #Cross = x[:,0].reshape(self.samples_num,1)*x[:,1].reshape(1,self.samples_num)
            #y1 = 5*x[:,0].reshape(self.samples_num,1) - torch.mm(Cross, torch.ones(self.samples_num, 1))
            #y2 = 5*x[:,1].reshape(self.samples_num,1) - torch.mm(Cross, torch.ones(self.samples_num, 1))
            y = torch.cat((y1, y2), dim=1)
        else:
            print("The dim should be 2.")
        return y
    
    def hat3d_ex4(self, x):
        x0_reshape = x[:,0].reshape(1,self.samples_num)
        x1_reshape = x[:,1].reshape(self.samples_num,1)
        Cross = torch.mm(x0_reshape, x1_reshape) #1*1
        #print("Cross.shape", Cross.shape)
        if x.shape[1] == 2:
            y1 = -4 * torch.mm(Cross, x[:,1].reshape(1,self.samples_num)) - (x[:,1]**2).reshape(1,self.samples_num) - 2*Cross*torch.ones(1,x.shape[0]) - 5*x[:,1].reshape(1,self.samples_num) - 2.5*x[:,0].reshape(1,self.samples_num) - 1.75 *torch.ones(1,x.shape[0])
            y2 = -4 * torch.mm(Cross, x[:,0].reshape(1,self.samples_num)) - (x[:,0]**2).reshape(1,self.samples_num) - 2*Cross*torch.ones(1,x.shape[0]) - 5*x[:,0].reshape(1,self.samples_num) - 2.5*x[:,1].reshape(1,self.samples_num) - 1.75 *torch.ones(1,x.shape[0])
            #print("y1.shape", y1.shape) #1*10000
            y1 = y1.t()
            y2 = y2.t()
            y = torch.cat((y1, y2), dim=1)
            
        else:
            print("The dim should be 2.")
        return y
    
    def SubDiag(self, x): #x: N*d 
        """
        one-order: [X1, X1+X2, X2+X3, X3+X4, X4+X5]
        """
        x_prime = x[:, [0,0,1,2,3,4] ]
        x_prime[:,0] = 0
        row_index_to_remove = self.dim
        xx_prime = x_prime[:, : row_index_to_remove]
        y = xx_prime + x
        return y
    
    def SubDiag_2(self, x): #x: N*d 
        """
        one-order: [X2, X3, X4, X5, X1]
        """
        y = x[:, [1, 2, 3, 4, 1] ]
        #y = x[:, [1, 2, 4 ,4 ,1] ]
        #y = x[:, [4,3,2,1,0] ]
        return y
    
    
    #def SubDiag_2Order(self, x): #x: N*d 
    #    """
    #    2-order: [-X1^2, X1-X2^2, X2-X3^2, X3-X4^2, X4-X5^2]
    #    """
    #    x_prime = x[:, [0,0,1,2,3,4] ]
    #    x_prime[:,0] = 0
    #    row_index_to_remove = self.dim
    #    xx_prime = x_prime[:, : row_index_to_remove]
    #    y = xx_prime - torch.pow(x, 2)
    #    return y
    
    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.9999
        U = torch.rand(self.samples_num,self.dim)*0.9999
        W = -torch.log(U+1e-5)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    
    def levy_rv(self, x, dt):
        dL = levy_stable.rvs(alpha=self.alpha_levy, beta=0, size=x.shape, scale=dt**(1/self.alpha_levy)) 
        dL = torch.tensor(dL,dtype=torch.float)
        return dL
    
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
                    y = y + self.SubDiag_2(y) * self.dt + torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() * \
                            torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term))
                    if self.explosion_prevention:
                        if any(y < 0):
                            y[y < 0] = 0
                            self.explosion_prevention_N = self.explosion_prevention_N + 1
                return y
            
        elif self.drift_term.shape == torch.Size([self.dim, 2*self.dim + 1]): 
            """
            2-order polynomial, no cross terms, e.g. {1, x1, x2, x3, x1^2, x2^2, x3^2}
            """
            if t0 == t1:
                return x
            else:
                t = torch.arange(t0, t1 + self.dt, self.dt)
                y = x
                for i in range(t.shape[0] - 1):
                    y = y + self.SubDiag_2Order(y) * self.dt + torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() * \
                            torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term))
                    if self.explosion_prevention:
                        if any(y < 0):
                            y[y < 0] = 0
                            self.explosion_prevention_N = self.explosion_prevention_N + 1
                return y
        
        
        
        elif self.drift_term.shape == torch.Size([2,6]):  
            """
            two dim, two-order polynomial, has cross-terms
            """
            if t0 == t1:
                return x
            else:
                t = torch.arange(t0, t1 + self.dt, self.dt)
                y=x
                for i in range(t.shape[0] - 1):
                    #y = y +  self.hat3d_ex3(y)* self.dt + torch.sqrt(torch.tensor(self.dt)) * \
                    #    torch.mm(torch.randn(self.samples_num, self.dim), torch.diag(self.diffusion_term)) + \
                    #    torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() * \
                    #        torch.mm(torch.randn(self.samples_num, self.dim), torch.diag(self.xi_term))
                            
                    y = y +  self.hat5d_ex1(y)* self.dt + \
                        torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * self.levy_variable() * \
                            torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term))
                    if self.explosion_prevention:
                        if any(y < 0):
                            y[y < 0] = 0
                            self.explosion_prevention_N = self.explosion_prevention_N + 1
                return y
        
        
        elif self.drift_term.shape == torch.Size([3, 20]) or self.drift_term.shape == torch.Size([5, 31]):   #3d含交叉项 3-order; 5d 只考虑对potential的系数进行估计，3-order
            """
            two or three dims, three-order polynomial, has cross-terms
            """
            if t0 == t1:
                return x
            else:
                t = torch.arange(t0, t1 + self.dt, self.dt)
                y = x
                for i in range(t.shape[0] - 1):
                    dL = self.levy_rv(y, torch.tensor(self.dt))
                    #print("dL shape", dL.shape)
                    #y = y + self.hat3d_ex4(y)* self.dt + torch.mm(dL, torch.diag(self.xi_term))
                    y = y + self.hat5d_ex1(y)* self.dt +\
                        torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term)) * self.levy_variable()
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
                plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[-10,10], density=True)
                #plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[data.min().numpy(), data.max().numpy()], density=True)
                #sns.set_palette("hls") 
                #mpl.rc("figure", figsize=(5,4))
                #sns.distplot(x=data[-1, :, i].numpy(),bins=1000,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "b" })
            plt.show()
        return data




if __name__ == '__main__':
    torch.manual_seed(6)
    dim=5
    
    drift = torch.tensor([[0, 10, 0, 0, 0, 0,  -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 10, 0, 0, 0,  0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 10, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 10, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 10,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4]])
    
    xi = torch.tensor([1.0]).repeat(dim)
    samples = 50000
    #t = np.array([0.1, 0.3, 0.5, 0.7, 1.0]).astype(np.float32)
    #t = torch.tensor(t)
    t =torch.linspace(0.1, 1.0, 10)
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=dim, drift_term=drift, \
                      xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, dim]),\
                      drift_independence=False, explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
    torch.save(data, "./5D_potential_data_50000.pt")
   
    plt.figure(figsize=(16,16))
    plt.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=50, range=[[-10, 10], [-10, 10]], density=True)
    plt.figure(figsize=(16,16))
    plt.hist2d(x=data[-1, :, 1].numpy(), y=data[-1, :, 2].numpy(), bins=50, range=[[-10, 10], [-10, 10]], density=True)
    
   
    
   