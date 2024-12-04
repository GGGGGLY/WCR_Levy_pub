# -*- coding: utf-8 -*-
"""
3D couple plot

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
from matplotlib import colors
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.stats import levy_stable
import scipy.stats

class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_term, xi_term, drift_term_est, xi_term_est, alpha_levy,
                 initialization, drift_independence=True, explosion_prevention=False):
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.xi_term = xi_term
        self.drift_term_est = drift_term_est
        self.xi_term_est = xi_term_est
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
    
    def hat3d_estimate(self, X, drift):
        """ 
        X: N*3
        drift.shape torch.size([3, 20])
        Theta: N*20
        
        output: y torch.size([N, 5])
        """
        Theta = torch.zeros(X.size(0), 20)
        Theta[:, 0] = 1
        count = 1
        for ii in range(0,self.dim):
            Theta[:, count] = X[:,ii]
            count += 1

        
        for ii in range(0,self.dim):
            for jj in range(ii,self.dim):
                Theta[:, count] = torch.mul(X[:,ii],X[:,jj])
                count += 1

        for ii in range(0,self.dim):
            for jj in range(ii,self.dim):
                for kk in range(jj,self.dim):
                    Theta[:,count] = torch.mul(torch.mul(X[:,ii],X[:,jj]),X[:,kk])
                    count += 1
        drift = drift.unsqueeze(0)
        Theta = Theta.unsqueeze(1)
        y = torch.mul(drift, Theta)

        """y torch.size([N, 3, 20])"""
        y = torch.sum(y, dim=2)
        
        return y
       
   
   
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
    
    def subSDE_estimate(self, t0, t1, x):
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                y = y + self.hat3d_estimate(y, self.drift_term_est)* self.dt + torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) *\
                    torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term_est)) * self.levy_variable()
                if self.explosion_prevention:
                    if any(y < 0):
                        y[y < 0] = 0
                        self.explosion_prevention_N = self.explosion_prevention_N + 1
            return y
    
   
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
          
        data_est = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data_est[0, :, :] = self.subSDE_estimate(0, self.time_instants[0], self.initialization)
        for i in range(self.time_instants.shape[0] - 1):
            data_est[i + 1, :, :] = self.subSDE_estimate(self.time_instants[i], self.time_instants[i + 1], data[i, :, :])
          
        if self.explosion_prevention:
            print("explosion_prevention * %s" % self.explosion_prevention_N)
        if plot_hist:
            for i in range(self.dim):
                plt.figure()
                #plt.hist(x=data[-1, :, i].numpy(), bins=100, range=[-5,5], density=True, color = 'bisque', label="True")
                #plt.hist(x=data_est[-1, :, i].numpy(), bins=100, range=[-5,5], density=True, color = 'thistle', label="Estimated")
                #plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[data.min().numpy(), data.max().numpy()], density=True)
                sns.set_palette("hls") 
                true_bd = data[-1, :, i].numpy()
                est_bd = data_est[-1, :, i].numpy()
                WD = scipy.stats.wasserstein_distance(true_bd, est_bd)
                sns.distplot(x=true_bd, bins=80, kde_kws={"color":"seagreen", "lw":3 }, hist= False, label="True")
                sns.distplot(x=est_bd,bins=80,kde_kws={"color":"mediumpurple", "lw":3 }, hist= False, label="Estimated")
                plt.legend(labels=["True distribution", "Estimated distribution"])
                plt.title("Data distribution at t=1 of dimension %d, WD= %f" %(i+1, WD))
                plt.legend(labels=["True distribution", "Estimated distribution"])
            
            plt.show()
        return data




if __name__ == '__main__':
    torch.manual_seed(6)
    dim=3
    drift = torch.tensor([[0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, -4, 0, -4, 0, 0, 0, 0],
                           [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4, 0],
                           [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4]])
    xi = torch.tensor([1.0]).repeat(dim)
    
    drift_est = torch.tensor([[0, 11.0432, 0, 0, 0, 0, 0, 0, 0, 0, -4.4727, 0, 0, -4.4839, 0, -4.2990, 0, 0, 0, 0],
                          [0, 0, 11.8475, 0, 0, 0, 0, 0, 0, 0, 0, -4.5651, 0, 0, 0, 0, -4.7658, 0, -4.9104, 0],
                         [0, 0, 0, 10.8256, 0, 0, 0, 0, 0, 0, 0, 0, -4.5086, 0, 0, 0, 0, -4.0030, 0, -4.4259]])
    
    xi_est = torch.tensor([1.12534, 0.98300, 1.1436028])
    
    samples = 30000
    t =torch.linspace(0, 1.0, 11)
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=dim, drift_term=drift, xi_term = xi, \
                      drift_term_est=drift_est, xi_term_est = xi_est, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, dim]),\
                      drift_independence=False, explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
   
    plt.figure(figsize=(10, 8))
    #fig, ax = plt.subplots()
    #hh = ax.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=50)
    plt.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=100, cmap = "RdYlGn_r", norm = colors.LogNorm(), range=[[-3, 3], [-3, 3]], density=True)
    #fig.colorbar(hh[3], ax=ax)
    cbar = plt.colorbar()
    density = cbar.get_ticks()
    plt.clim(density.min(), density.max())
    plt.xlabel("Dim1",fontsize=14)
    plt.ylabel("Dim2",fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('True particle distribution density at the final moment',fontsize=18)
    
   