# -*- coding: utf-8 -*-
"""
5D couple plot
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
    
    def hat5d_estimate(self, X, drift):
        """ 
        X: N*5
        drift.shape torch.size([5, 31])
        Theta: N*31
        
        output: y torch.size([N, 5])
        """
        Theta = torch.zeros(X.size(0), 31)
        Theta[:, 0] = torch.ones(X.size(0))
        count = 1
        for ii in range(0, self.dim):
            for jj in range(0, self.dim):
                Theta[:, count] = torch.mul(X[:, ii], X[:, jj]**2)
                count = count+1
        drift = drift.unsqueeze(0)
        Theta = Theta.unsqueeze(1)
        y = torch.mul(drift, Theta)
        """y torch.size([N, 5, 31])"""
        y = torch.sum(y, dim=2)
        
        return y
       
    
    def hat3d_ex2(self, x):
        # V =    drift = - grad V
        Cross = x[:,0].reshape(self.samples_num,1)*x[:,1].reshape(1,self.samples_num)
        #print(Cross.shape) 
        grad = -4*torch.mm(Cross, x ) - 2.5*x - 2*torch.mm(Cross, torch.ones(self.samples_num, self.dim))\
            - (7/4)*torch.ones(self.samples_num, self.dim)-\
            torch.cat( (x[:,1].reshape(self.samples_num,1)**2 + 5* x[:,1].reshape(self.samples_num,1), x[:,0].reshape(self.samples_num,1)**2 + 5* x[:,0].reshape(self.samples_num,1)), dim=1)
        return grad
    
   
    
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
                    y = y + self.hat5d_ex1(y)* self.dt +\
                        torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term)) * self.levy_variable()
                    if self.explosion_prevention:
                        if any(y < 0):
                            y[y < 0] = 0
                            self.explosion_prevention_N = self.explosion_prevention_N + 1
                return y
            
            
    def subSDE_estimate(self, t0, t1, x):
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                #dL = self.levy_rv(y, torch.tensor(self.dt))
                y = y + self.hat5d_estimate(y, self.drift_term)* self.dt +\
                    torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term)) * self.levy_variable()
                if self.explosion_prevention:
                    if any(y < 0):
                        y[y < 0] = 0
                        self.explosion_prevention_N = self.explosion_prevention_N + 1
            return y
                

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim)
        data[0, :, :] = self.subSDE_estimate(0, self.time_instants[0], self.initialization)
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :, :] = self.subSDE_estimate(self.time_instants[i], self.time_instants[i + 1], data[i, :, :])
          
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
    #drift = torch.tensor([[0, 10, 0, 0, 0, 0,  -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 10, 0, 0, 0,  0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 10, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 10, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 10,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4]])
    
    #xi = torch.tensor([1.0]).repeat(dim)
    
    #drift = torch.tensor([[0, 12.0645, -0.9119, 0, 0, 0,  -4.8857, -5.2054, -4.8540, -5.0956, -3.9317, 0, 0, 0.8976, 0, 0.9562, 0, 0, 0, 0.7486, 0, 0, 0, 0, 0, 0, 0, 0, -2.2888, 1.511, 0],
    #                          [0, 0.9204, 11.8742, 0.6924, -0.5990, 0, 0, 0, -1.0205, 0, -0.9048, -4.2805, -4.7726, -4.0807, -4.7871, -5.7, 0.6959, 0, 0, -1.01, 0, 0, 0, 1.6990, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, -0.5920, 11.2834, 0, 0.4211,  0, 0, 0, -0.6829, 0, -0.7887, 0, 0, 0.8869, 0, -4.3589, -5.1078, -4.6232, -4.7825, -3.5390, 0, 0, 0, 0, 0, -0.6658, 0, 0, 0, 0],
    #                          [0, 0, 0.579, 0, 11.661, 0,  0, 0, 0, 0, 0, 0, 0, -1.6089, 0, 0, 0, 0, 0, 0, 0, -4.3233, -4.6188, -4.4046, -4.7548, -5.1419, 0, 0, 0.5522, 0, 0],
    #                          [0, 0, 0, -0.4604, 0, 12.3708,  0, 0, 2.2778, -1.5617, 0, 0, 0, 0, 0, 0, 0.6924, 0, 0, 0, 0, 0, 0, -0.4672, 0, 0, -5.5825, -3.7152, -5.8929, -4.3259, -5.0139]])
    
    #xi = torch.tensor([1.0341, 1.03810, 1.0177953, 1.06686, 1.003])
    
    #drift = torch.tensor([[0, 11.9372, 0, 0, 0, 0,  -4.7916, -4.6736, -4.9238, -5.5738, -3.7302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.1274, 0, 0],
    #                          [0, 0, 11.5378, 0, 0, 0,  0, 0, 0, 0, 0, -4.6173, -4.7683, -4.1349, -4.6328, -4.8811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 12.0892, 0, 4.9304,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.4586, -5.2778, -4.8045, -4.3700, -5.0443, 0, 0, 0, 0, 0, -1.9247, -0.9820, -1.3989, -1.6270, -1.5252],
    #                          [0, 0, 0, 0, 11.6683, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.8463, -4.7761, -5.0826, -4.7934, -4.8863, 0, -1.0248, 0, 0, 0],
    #                          [0, 0, 0, -4.7030, 0, 12.2094, 0, 0, 1.1316, 0, 0, 0, 0, 0, 0, 0, 1.9532, 1.0489, 1.2850, 1.6240, 1.3944, 0, 0, 0, 0, 0, -5.7040, -4.5842, -4.4766, -4.5950, -4.8293]])
    
    #xi = torch.tensor([1.010027, 1.05781, 0.995245, 1.0895337, 0.9733109])
    drift = torch.tensor([[0, 12.0952, -0.8888, 0, 0, 0,  -4.8970, -5.2087, -4.7776, -5.1907, -3.9431, 0, 0, 0.7991, 0, 0.8675, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.1186, 1.4110, 0],
                              [0, 0.8904, 11.8919, 0, -0.6115, 0, 0, 0, -0.9106, 0, -0.8033, -4.2927, -4.7752, -3.8731, -4.9062, -5.8023, 1.5217, 0, 0, 0, 0, 0, 0, 1.7316, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 11.3288, 0, 0, 0, 0, 0, 0, 0, -1.5174, 0, 0, 0, 0, -4.4473, -5.3355, -4.6312, -4.6515, -3.4184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0.5811, 0, 11.6164, 0, 0, 0, 0, 0, 0, 0, 0, -1.6329, 0, 0, 0, 0, 0, 0, 0, -4.2266, -4.4768, -4.5517, -4.7485, -5.1724, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 12.3517, 0, 0, 2.0903, -1.4510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.5718, -3.6215, -6.0135, -4.2868, -5.0090]])
    
    xi = torch.tensor([1.0302591, 1.0353966, 1.0140418, 1.0729787, 1.0048746])
    
    samples = 30000
    t =torch.linspace(0, 1.0, 11)
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=dim,\
                      drift_term=drift, xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, dim]),\
                      drift_independence=False, explosion_prevention=False)
    data = dataset.get_data(plot_hist=False)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
   
    plt.figure(figsize=(10, 8))
    #fig, ax = plt.subplots()
    #hh = ax.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=50)
    dim1_ = data[-1, :, 0].numpy()[ (data[-1, :, 0].numpy()>-5) & (data[-1, :, 0].numpy()<5)]
    dim2_ = data[-1, :, 1].numpy()[ (data[-1, :, 1].numpy()>-5) & (data[-1, :, 1].numpy()<5)]
    #plot_n = min(dim1_.size(), dim2_.size())
    plot_n = min(len(dim1_), len(dim2_))
    plt.hist2d(x=dim1_[:plot_n], y=dim2_[:plot_n], bins=100, cmap = "RdYlGn_r", norm = colors.LogNorm(), range=[[-3, 3], [-3, 3]], density=True)
    #fig.colorbar(hh[3], ax=ax)
    cbar = plt.colorbar()
    density = cbar.get_ticks()
    plt.clim(density.min(), density.max())
    plt.xlabel("Dim1",fontsize=14)
    plt.ylabel("Dim2",fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Estimated particle distribution density at the final moment',fontsize=18)
    #plt.figure(figsize=(10, 10))
    #plt.hist2d(x=data[-1, :, 1].numpy(), y=data[-1, :, 2].numpy(), bins=50, range=[[-5, 5], [-5, 5]], density=True)
    
   
    
   