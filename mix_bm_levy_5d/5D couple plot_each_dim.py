# -*- coding: utf-8 -*-
"""
5D couple plot: plot dist. each dimension

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
    
          

    def hat5d_ex1(self, x):
        ####  V = -5(x2+y2+z2) + (x2+y2+z2)2,   drift = - grad V
        norm = torch.sum(x ** 2, dim=1).unsqueeze(1)
        norm2 = norm.repeat(1, x.shape[1])
        return 10 * x - 4 * x * norm2
    
    def hat5d_estimate(self, X, drift):
        """ 
        X: N*5
        drift.shape torch.size([5, 31])
        Theta: [N, 31]
        
        output: y torch.size([N, 5])
        """
        Theta = torch.zeros(X.size(0), 31)
        Theta[:, 0] = torch.ones(X.size(0))
        count = 1
        for ii in range(0,self.dim):
            Theta[:, count] = X[:,ii]
            count += 1
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
        return y
    
   
    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.99999
        U = torch.rand(self.samples_num,self.dim)*0.99999
        W = -torch.log(U+1e-6)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    
    #def levy_rv(self, x, dt):
     #   dL = levy_stable.rvs(alpha=self.alpha_levy, beta=0, size=x.shape, scale=dt**(1/self.alpha_levy)) 
     #   dL = torch.tensor(dL,dtype=torch.float)
     #   return dL
    
    
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
                    #dL = self.levy_rv(y, torch.tensor(self.dt))
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
                y = y + self.hat5d_estimate(y, self.drift_term_est)* self.dt +\
                    torch.pow(torch.tensor(self.dt), 1/self.alpha_levy) * torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term_est)) * self.levy_variable()
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
                #plt.hist(x=data[-1, :, i].numpy(), bins=80, range=[-10,10], density=True, label="True")
                #plt.hist(x=data_est[-1, :, i].numpy(), bins=80, range=[-10,10], density=True, label="Estimated")
                #plt.legend(['True','Estimated'])
                sns.set_palette("hls") 
                true_bd = data[-1, :, i].numpy()[ (data[-1, :, i].numpy()>-4) & (data[-1, :, i].numpy()<4)]
                est_bd = data_est[-1, :, i].numpy()[ (data_est[-1, :, i].numpy()>-4) & (data_est[-1, :, i].numpy()<4)]
                
                WD = scipy.stats.wasserstein_distance(true_bd, est_bd)
                sns.distplot(x=true_bd, bins=100, kde_kws={"color":"seagreen", "lw":3 }, hist= False, label="True")
                sns.distplot(x=est_bd,bins=100,kde_kws={"color":"mediumpurple", "lw":3 }, hist= False, label="Estimated")
                plt.legend(labels=["True distribution", "Estimated distribution"])
                #plt.title("Data distribution at t=1 of dimension %d" %(i+1))
                plt.title("Data distribution at t=1 of dimension %d, WD= %f" %(i+1, WD))
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
    
    #drift1 = torch.tensor([[0, 12.0645, -0.9119, 0, 0, 0,  -4.8857, -5.2054, -4.8540, -5.0956, -3.9317, 0, 0, 0.8976, 0, 0.9562, 0, 0, 0, 0.7486, 0, 0, 0, 0, 0, 0, 0, 0, -2.2888, 1.511, 0],
    #                          [0, 0.9204, 11.8742, 0.6924, -0.5990, 0, 0, 0, -1.0205, 0, -0.9048, -4.2805, -4.7726, -4.0807, -4.7871, -5.7, 0.6959, 0, 0, -1.01, 0, 0, 0, 1.6990, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, -0.5920, 11.2834, 0, 0.4211,  0, 0, 0, -0.6829, 0, -0.7887, 0, 0, 0.8869, 0, -4.3589, -5.1078, -4.6232, -4.7825, -3.5390, 0, 0, 0, 0, 0, -0.6658, 0, 0, 0, 0],
    #                          [0, 0, 0.579, 0, 11.661, 0,  0, 0, 0, 0, 0, 0, 0, -1.6089, 0, 0, 0, 0, 0, 0, 0, -4.3233, -4.6188, -4.4046, -4.7548, -5.1419, 0, 0, 0.5522, 0, 0],
    #                          [0, 0, 0, -0.4604, 0, 12.3708,  0, 0, 2.2778, -1.5617, 0, 0, 0, 0, 0, 0, 0.6924, 0, 0, 0, 0, 0, 0, -0.4672, 0, 0, -5.5825, -3.7152, -5.8929, -4.3259, -5.0139]])
    
    #xi1 = torch.tensor([1.0341, 1.03810, 1.0177953, 1.06686, 1.003])
    
    #drift_est = torch.tensor([[0, 11.9372, 0, 0, 0, 0,  -4.7916, -4.6736, -4.9238, -5.5738, -3.7302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.1274, 0, 0],
    #                          [0, 0, 11.5378, 0, 0, 0,  0, 0, 0, 0, 0, -4.6173, -4.7683, -4.1349, -4.6328, -4.8811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 12.0892, 0, 4.9304,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.4586, -5.2778, -4.8045, -4.3700, -5.0443, 0, 0, 0, 0, 0, -1.9247, -0.9820, -1.3989, -1.6270, -1.5252],
    #                          [0, 0, 0, 0, 11.6683, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.8463, -4.7761, -5.0826, -4.7934, -4.8863, 0, -1.0248, 0, 0, 0],
    #                          [0, 0, 0, -4.7030, 0, 12.2094, 0, 0, 1.1316, 0, 0, 0, 0, 0, 0, 0, 1.9532, 1.0489, 1.2850, 1.6240, 1.3944, 0, 0, 0, 0, 0, -5.7040, -4.5842, -4.4766, -4.5950, -4.8293]])
    
    #xi_est = torch.tensor([1.010027, 1.05781, 0.995245, 1.0895337, 0.9733109])
    
    #drift_est = torch.tensor([[0, 12.0952, -0.8888, 0, 0, 0,  -4.8970, -5.2087, -4.7776, -5.1907, -3.9431, 0, 0, 0.7991, 0, 0.8675, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.1186, 1.4110, 0],
    #                          [0, 0.8904, 11.8919, 0, -0.6115, 0, 0, 0, -0.9106, 0, -0.8033, -4.2927, -4.7752, -3.8731, -4.9062, -5.8023, 1.5217, 0, 0, 0, 0, 0, 0, 1.7316, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 11.3288, 0, 0, 0, 0, 0, 0, 0, -1.5174, 0, 0, 0, 0, -4.4473, -5.3355, -4.6312, -4.6515, -3.4184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0.5811, 0, 11.6164, 0, 0, 0, 0, 0, 0, 0, 0, -1.6329, 0, 0, 0, 0, 0, 0, 0, -4.2266, -4.4768, -4.5517, -4.7485, -5.1724, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 12.3517, 0, 0, 2.0903, -1.4510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.5718, -3.6215, -6.0135, -4.2868, -5.0090]])
    
    #xi_est = torch.tensor([1.0302591, 1.0353966, 1.0140418, 1.0729787, 1.0048746])
    
    #drift_est = torch.tensor([[0, 12.1343, 0, 0, 0, 0,  -4.8860, -5.5086, -4.3934, -5.2097, -4.0037, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8277, 1.2317, 0],
    #                          [0, 0, 12.0820, 0, 0, -0.5877, 0, 0, 0, 0, 0, -3.9742, -4.8898, -5.5104, -4.5258, -5.1252, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.3595, 0, 0, 0, 0],
    #                          [0, 0, 0, 11.3528, -0.6186, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.8915, -3.8222, -4.7240, -4.5875, -4.5698, 1.2405, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0.6348, 11.6513, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.2296, 0, 0, 0, 0, -4.2056, -4.9806, -4.6601, -4.7406, -4.6922, 0, 0, 0, 0, 0],
    #                          [0, 0, 0.5568, 0, 0, 12.4118, 0, 0, 0.8515, -1.2960, 0, -1.3174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.5065, -4.5083, -4.7773, -4.8510, -4.8909]])
    
    #xi_est = torch.tensor( [0.9924439, 1.0430527, 1.0464699, 1.0799823, 0.96604615])
    
    
    
    #drift_est = torch.tensor([[0, 11.9679, 0, 0, 0, 0,  -4.8567, -5.4584, -4.3835, -5.2354, -3.8455, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7943, 1.0730, 0],
    #                          [0, 0, 12.1663, 0, 0, 0, 0, 0, 0, 0, 0, -4.0337, -4.9128, -5.5266, -4.6023, -5.1343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 11.4217, -0.6292, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.8986, -3.8466, -4.7348, -4.5932, -4.6660, 1.2447, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 11.6478, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.2593, 0, 0, 0, 0, -4.1635, -4.9419, -4.6707, -4.7392, -4.7902, 0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 12.3708, 0, 0, 0.8279, -1.1314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.6571, -4.5232, -4.6903, -4.7477, -4.8782]])
    
    #xi_est = torch.tensor( [1.0497667, 1.0759219, 1.0719243, 1.1171424, 1.0001796])
    
    
    #drift_est = torch.tensor([[0, 12.0277, 0, 0, 0, 0,  -4.8678, -5.5059, -4.3570, -5.2534, -3.8668,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
    #                          [0, 0, 12.1030, 0, 0, 0,  0, 0, 0, 0, 0,  -3.9660, -4.8950, -5.4470, -4.6594, -5.1001,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
    #                          [0, 0, 0, 11.4616, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  -4.9526, -3.9227, -4.7460, -4.4497, -4.7283,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 11.6261, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  -4.1419, -4.8413, -4.8210, -4.7334, -4.7018,  0, 0, 0, 0, 0],
    #                          [0, 0, 0, 0, 0, 12.4077,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  -5.6421, -4.5394, -4.6437, -4.8234, -4.8880]])
    
    #xi_est = torch.tensor( [1.0174625, 1.0531102, 1.0492412, 1.0917829, 0.97529674])
    
    
    drift_est = torch.tensor([[0, 12.0793, 0, 0, 0, 0,  -4.9077, -5.3753, -4.3347, -5.1961, -4.1174,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
                              [0, 0, 11.9558, 0, 0, 0,  0, 0, 0, 0, 0,  -4.0659, -4.8606, -5.2558, -4.5814, -5.0243,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
                              [0, 0, 0, 11.5183, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  -4.9810, -4.0673, -4.7440, -4.4935, -4.6254,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 11.6109, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  -4.1980, -4.8642, -4.7723, -4.7349, -4.6403,  0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 12.3710,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  -5.3796, -4.5660, -4.7634, -4.9006, -4.9102]])
    
    xi_est = torch.tensor( [1.0317707, 1.0801454, 1.065147, 1.1174623, 1.013833 ])
    
    
    
    
    samples = 30000
    t =torch.linspace(0, 1.0, 11)
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=dim, drift_term=drift, xi_term = xi,\
                      drift_term_est=drift_est, xi_term_est = xi_est, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, dim]),\
                      drift_independence=False, explosion_prevention=False)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
   
    plt.figure(figsize=(10, 8))
    plt.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=100, cmap = "RdYlGn_r", norm = colors.LogNorm(), range=[[-3, 3], [-3, 3]], density=True)
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
   
   
     