# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 11:40:03 2023

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from generate_data_NonPoly_OnlyLevy import DataSet
import time
import utils
import scipy.io
import scipy.special as sp

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import seaborn as sns
from scipy import stats

class Gaussian(torch.nn.Module): 
    def __init__(self, mu, sigma, lap_alpha):
        super(Gaussian, self).__init__() 
        self.mu = mu
        self.sigma = sigma
        self.lap_alpha = lap_alpha
        self.dim = 1

    def gaussB(self, x):
        func = 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi))) * torch.exp(-0.5*(x-self.mu)**2/self.sigma**2)
        return func

    def gaussZero(self, x):
        func = 1
        for d in range(x.shape[2]):
            func = func * self.gaussB(x[:, :, d])
        return func

    def gaussFirst(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]])
        for k in range(x.shape[2]):
            func[:, :, k] = -(x[:, :, k] - self.mu)/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]])
        for k in range(x.shape[2]):
            for j in range(x.shape[2]):
                
                if k == j:
                    func[:, :, k, j] =  (
                                    -1/self.sigma**2 + (-(x[:, :, k]-self.mu)/self.sigma**2)
                                    * (-(x[:, :, j]-self.mu)/self.sigma**2)
                                    ) * g0
                else:
                    func[:, :, k, j] =  (-(x[:, :, k]-self.mu)/self.sigma**2)*(
                        -(x[:, :, j]-self.mu)/self.sigma**2
                        ) * g0
        return func
    
    def LapGauss(self, x):
        
        # (x.shape[2] + self.lap_alpha)/2, x.shape[2]/2, -torch.sum((x-self.mu)**2, dim=2) / (2*self.sigma**2)) 
        x = (x - self.mu)/self.sigma/np.sqrt(2)
        func = (1/self.sigma/np.sqrt(2))**self.lap_alpha * 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi))) \
            *sp.gamma((self.dim+self.lap_alpha)/2)*2**self.lap_alpha/sp.gamma(self.dim/2)*sp.hyp1f1((self.dim+self.lap_alpha)/2, self.dim/2, -torch.sum(x**2,dim = 2))
        return func  
    
    #def LapGauss(self, x):
        
            # (x.shape[2] + self.lap_alpha)/2, x.shape[2]/2, -torch.sum((x-self.mu)**2, dim=2) / (2*self.sigma**2))  
    #    func = (1/self.sigma/np.sqrt(2))**self.lap_alpha * 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi))) \
    #        *sp.gamma((self.dim+self.lap_alpha)/2)*2**self.lap_alpha/sp.gamma(self.dim/2)*sp.hyp1f1((self.dim+self.lap_alpha)/2, self.dim/2, -(x[:, :, 0]-self.mu)**2 / (2*self.sigma**2))
    #    return func  ##1/(2*pi)
    
    
   # def LapGauss_1d(self,x, g0):
        #func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
    #    for k in range(x.shape[2]):
    #        func = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (1 + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(1/2) * \
   #                 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*\
    #                    sp.hyp1f1((1 + self.lap_alpha)/2, 1/2, -(x[:, :, k]-self.mu[k])**2 / (2*self.sigma**2))  
    #    return func 
   
    
    
    def forward(self, x, diff_order=0): #diff_order=0不写，默认为0   #forward是内置函数
        g0 = self.gaussZero(x)
        if diff_order == 0:
            return g0
        elif diff_order == 1:
            return self.gaussFirst(x, g0)
        elif diff_order == 2:
            return self.gaussSecond(x, g0)
        elif diff_order == 'frac':
            return self.LapGauss(x)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not bee implemented!")

class Model(object):
    """A ``Model`` solve the true coefficients of the basis on the data by the outloop for linear regression and 
    and the inner loop of increasing the parameters in the test function TestNet.
    Args:
        t : `` t'' vector read from the file
        data: ``data`` matrix read from the file.
        testFunc: ``DNN`` instance.
    """
    def __init__(self, t, data, testFunc):
        self.t = t
        self.itmax = len(t)
        self.data = data
        self.net = testFunc
        self.basis = None # given by build_basis
        self.A = None # given by build_A
        self.b = None # given by build_b
        self.dimension = None
        self.basis_number = None
        self.basis1_number = None
        self.basis2_number = None
        self.basis_order = None
        self.bash_size = data.shape[1]
        self.basis_xi_order = None
        
        self.zeta = None # coefficients of the unknown function
        self.error_tolerance = None
        self.max_iter = None
        self.loss = None

    def _get_data_t(self, it):
        X = self.data[it,:,:] 
        return X
    
    @utils.timing # decorator
    @torch.no_grad()
    def build_basis(self): # \Lambda matrix
        """build the basis list for the different time snapshot 
        """
        self.t_number = len(self.t)
        self.basis1_number = int(np.math.factorial(self.dimension+self.basis_order)
                /(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order))) #int取整， np.math.factorial阶乘
        
        self.basis2_number = int( self.dimension ) 
        

        # Construct Theta
        basis1 = []

        for it in range(self.t_number):
            X = self._get_data_t(it)
            basis_count1 = 0
            Theta = torch.zeros(X.size(0),self.basis1_number)
            Theta[:,0] = 1
            basis_count1 += 1
            for i in range(self.basis1_number):
                Theta[:, i] = X[:, 0]**i
            basis1.append(Theta)
            # print("X", X)
            # print("theta", Theta.shape)

        basis_theta = torch.stack(basis1)
        
            #basis_theta = torch.stack(basis1)
        #print("basis_theta", basis_theta.shape)
            
            
        # Construct Xi
        basis2 = []     
        for it in range(self.t_number):
            #basis_count2 = 0
            X = self._get_data_t(it)
            Xi = torch.ones(X.size(0),1)
            
            #assert basis_count2 == self.basis2_number
            basis2.append(Xi)
            basis_xi = torch.stack(basis2)
        #print("basis_xi", basis_xi.shape)
        self.basis_theta = basis_theta
        self.basis = torch.cat([basis_theta, basis_xi],dim=2)         
        #self.basis = torch.stack(basis)
        print("self.basis.shape", self.basis.shape)
        self.basis_number = self.basis1_number + self.basis2_number
        print("self.basis1_number ", self.basis1_number)
        print("self.basis_number ", self.basis_number)
        
        
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2)     

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number  #db
        C_number = self.dimension * self.dimension #* self.basis2_number  #d^2c
        
        A = torch.zeros([self.t_number, H_number+C_number]) #A is a L* (db+d^2c)
        rb = torch.zeros(self.t_number)
        b = torch.zeros(self.t_number)


        # ##########################################################
        #  Tensor form of computing A and b for parallel computing
        # ##########################################################

        TX = self.data
        TX.requires_grad = True
        # Phi = self.net(TX)
        gauss0 = gauss(TX, diff_order=0)
        gauss1 = gauss(TX, diff_order=1)
        #gauss2 = gauss(TX, diff_order=2)
        gauss_lap = gauss(TX, diff_order='frac')
        

        # print("self.basis_number", self.basis_number)
        # print("self.dimension", self.dimension)
        for kd in range(self.dimension):
            for jb in range(self.basis1_number):
                # print("gauss1[:, :, %s]" % kd, gauss1[:, :, kd].size())
                H = 1/self.bash_size * torch.sum(
                    gauss1[:, :, kd]
                     *
                    self.basis_theta[:, :, jb], dim=1
                    )
                A[:, kd*self.basis1_number+jb] = H

                
                               
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                E = -torch.mean(gauss_lap, dim=1)
                A[:, H_number] = E 
                
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
       
       

        # print("b", b)
        # print("A.shape", A.shape)
        if self.type == 'PDEFind':
            b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
            return A, b
        if self.type == 'LMM_2':
            AA = torch.ones(A.size(0) - 1, A.size(1))
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + A[i + 1, :]) / 2
            bb = torch.from_numpy(np.diff(rb.numpy())) / dt
            return AA, bb
        if self.type == 'LMM_3':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + 4*A[i + 1, :] + A[i + 2, :]) * dt / 3
                bb[i] = rb[i + 2] - rb[i]
            return AA, bb
        
        if self.type == 'LMM_6':
            AA = torch.ones(A.size(0) - 5, A.size(1))
            bb = torch.ones(A.size(0) - 5)
            for i in range(AA.size(0)):
                AA[i, :] = (
                        A[i+1, :] +
                        1/2 * (A[i+2, :] + A[i+1, :]) +
                        5/12 * A[i+3, :] + 8/12 * A[i+2, :] - 1/12 * A[i+1, :] +
                        9/24 * A[i+4, :] + 19/24 * A[i+3, :] - 5/24 * A[i+2, :] + 1/24 * A[i+1, :] +
                        251/720 * A[i + 5, :] + 646/720 * A[i + 4, :] - 264/720 * A[i + 3, :] + 106/720 * A[i + 2, :] - 19/720 * A[i + 1, :]
                            ) * dt
                bb[i] = rb[i + 5] - rb[i]
            return AA, bb  

        if self.type == 'bdf2':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + 4*A[i + 1, :] + A[i + 2, :]) * dt / 3
                bb[i] = rb[i + 2] - rb[i]
            return AA, bb  
        if self.type == 'LMM_2_nonequal':
            AA = torch.ones(A.size(0) - 1, A.size(1))
            bb = torch.ones(A.size(0) - 1)
            ht = torch.from_numpy(np.diff(self.t.numpy()))
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + A[i + 1, :]) / 2 * ht[i]
                bb[i] = rb[i + 1] - rb[i]
            return AA, bb
        if self.type == 'non-equal3':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            ht = torch.from_numpy(np.diff(self.t.numpy()))
            # print("ht: ", ht)
            wt = torch.tensor([ht[i + 1] / ht[i] for i in range(ht.size(0) - 1)])
            # print("wt: ", wt)
            for i in range(AA.size(0)):
                print("ht[i + 1]", ht[i + 1], "wt[i]", wt[i])
                AA[i, :] = ht[i + 1] * (1 + wt[i]) / (1 + 2 * wt[i]) * A[i + 2, :]
                bb[i] = rb[i + 2] - (1 + wt[i]) ** 2 / (1 + 2 * wt[i]) * rb[i + 1] + wt[i] ** 2 / (1 + 2 * wt[i]) * rb[i]
            return AA, bb
        if self.type == 'non-equal-adams':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            ht = torch.from_numpy(np.diff(self.t.numpy()))
            # print("ht: ", ht)
            wt = torch.tensor([ht[i + 1] / ht[i] for i in range(ht.size(0) - 1)])
            # print("wt: ", wt)
            for i in range(AA.size(0)):
                AA[i, :] = ht[i + 1] / (6 * (1 + wt[i])) * (
                        (3 + 2 * wt[i]) * A[i + 2, :] 
                        + (3 + wt[i]) * (1 + wt[i]) * A[i + 1, :]
                        - wt[i] ** 2 * A[i, :])
                bb[i] = rb[i + 2] - rb[i + 1]
            return AA, bb
    def sampleTestFunc(self, samp_number):
        # for i in range(self.sampling_number):
        if self.gauss_samp_way == 'lhs':
            #factor = 1/2
            mu_list = self.lhs_ratio * torch.rand(samp_number)*(1.5-(-1.5)) -1.5
            # 1/2比3/8和2/3要好？
            #mu_list = self.lhs_ratio * torch.rand(samp_number)*(self.data.max()-self.data.min())*factor + self.data.min()*factor
            
        if self.gauss_samp_way == 'SDE':
            if samp_number <= self.bash_size:
                index = np.arange(self.bash_size)
                np.random.shuffle(index)
                mu_list = data[-1, index[0: samp_number], :]
        # print("mu_list", mu_list)
        sigma_list = torch.ones(samp_number)*self.variance
        return mu_list, sigma_list

    def buildLinearSystem(self, samp_number):
        mu_list, sigma_list = self.sampleTestFunc(samp_number)
        A_list = []
        b_list = []
        for i in range(mu_list.shape[0]):
            mu = mu_list[i]
            sigma = sigma_list[i]
            #gauss = self.net(mu, sigma)
            gauss = self.net(mu, sigma, 3/2) 
            A, b = self.computeAb(gauss)
            A_list.append(A)
            b_list.append(b)
        # print("A_list", A_list)
        # print("b_list", b_list)
        self.A = torch.cat(A_list, dim=0) # 2-dimension
        self.b = torch.cat(b_list, dim=0).unsqueeze(-1) # 1-dimension

    @utils.timing
    def solveLinearRegress(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression

    @utils.timing
    def STRidge(self, X0, y, lam, maxit, tol, normalize = 0, print_results = False):
        """
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """
        n,d = X0.shape
        X = np.zeros((n,d), dtype=np.complex64)
        # First normalize data
        if normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
                X[:,i] = Mreg[i]*X0[:,i]
        else: X = X0

        # Get the standard ridge esitmate
        if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y))[0]
        else:
            #w = np.linalg.lstsq(X,y)[0] #########################
            X_inv = np.linalg.pinv(X)
            w = np.dot(X_inv,y)
        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):
            # Figure out which items to cut out
            smallinds = np.where(abs(w) < tol)[0]
            print("STRidge_j: ", j)
            print("smallinds", smallinds)
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                print("here1")
                break
            else: num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    print("here2")
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else:
                    print("here3")
                    break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0
            if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        if normalize != 0: return np.multiply(Mreg,w)
        else: return w
    
    @utils.timing
    def compile(self, basis_order, gauss_variance, type, xi_term, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.basis_order = basis_order
        self.build_basis()
        self.variance = gauss_variance
        self.type = type
        self.xi = xi_term
        self.gauss_samp_way = gauss_samp_way
        self.lhs_ratio = lhs_ratio if self.gauss_samp_way == 'lhs' else 1
    
    
    def drift_est(self, x):
        #y = 0
        #est_drifts = self.zeta[0: -1]
        #for i in range(est_drifts.shape[0]):
        #    y = y + est_drifts[i] * x ** i
            
        y = torch.zeros_like(x)
        for i in range(self.basis_order+1):
            y = y + self.zeta[i, 0] * x**i    
        return y
    
    

    @utils.timing
    @torch.no_grad()
    def train(self, sample_num, gauss_samp_number, lam, STRidge_threshold):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        print("A: ", self.A.size(), "b: ", self.b.size())
        print("A",np.linalg.cond(self.A, p=None))
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        print("zeta: ", self.zeta)

        drift = [self.zeta[0].numpy()]
        for i in range(1, self.basis_number-1):
            drift.extend([" + ", self.zeta[i].numpy(), 'x^', i])
        print("Drift term: ", "".join([str(_) for _ in drift]))
        self.zeta[-1] = (self.zeta[-1])**(2/3) 
        print("Diffusion term of Levy Noise: ", self.zeta[-1])
        #true = torch.cat((self.diffusion, self.xi))
        #index = torch.nonzero(true).squeeze()
        #relative_error = torch.abs((self.zeta.squeeze()[index] - true[index]) / true[index])
        #x = self.data
        #x = torch.randn((10000,1))
        x = torch.linspace(-1.5, 1.5, 3000)
        
        #L1_error = torch.sum(torch.abs(x-2*x* torch.exp(-x**2) - self.drift_est(x))).numpy() / torch.sum(torch.abs(-2*x * torch.exp(-x**2))).numpy()
        #L2_error = torch.sum((-2*x* torch.exp(-x**2) - self.drift_est(x))**2).numpy() / torch.sum((-2*x * torch.exp(-x**2))**2).numpy()
        L2_error =(1/ x.numel() )* torch.norm(x-2*x * torch.exp(-x**2) - self.drift_est(x))
        print("L2 error: ", L2_error)

if __name__ == '__main__':
    np.random.seed(6) 
    torch.manual_seed(6)

    dt = 0.0001
    # t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    #t = torch.linspace(0.1,0.4,4)
    #t = torch.tensor([0.02, 0.05, 0.15, 0.25, 0.3])
    t = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    #t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #t = torch.tensor([0.2, 0.5, 1])
   
    xi = torch.tensor([1.0])
    samples = 20000
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=1, xi_term=xi, alpha_levy = 3/2, \
                      initialization=torch.normal(0, 0.2,[samples, 1]),explosion_prevention=False) 
                      #initialization=torch.randint(-1000,1000,[10000, 1])
    data = dataset.get_data(plot_hist=False)
    print("data.max: ", data.max(), "data.min: ", data.min())

    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=10, gauss_variance=0.5, type='LMM_2_nonequal', xi_term=xi,\
                  gauss_samp_way='lhs', lhs_ratio=1.0)
    model.train(sample_num = samples, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0)
    
     
  
   ################################        
      
      #####  t = torch.linspace(0,1,11)
      
  
  ################ basis_order=5     
    ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      #
      
 ################ basis_order=6     
   ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
     #Drift term:  [-0.01472363] + [-0.91135603]x^1 + [0.08226614]x^2 + [0.2745569]x^3 + [-0.00886573]x^4 + [-0.02043224]x^5 + [0.00036193]x^6
     #Diffusion term of Levy Noise:  tensor([0.8643])
     #L2 error:  0.22371158
     #'train' took 4.161567 s
    
    
    
   ################ basis_order=7     
     ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
       #  Drift term:  [-0.0180993] + [-1.1519837]x^1 + [-0.04395188]x^2 + [0.44346815]x^3 + [0.01883244]x^4 + [-0.04915128]x^5 + [-0.00206866]x^6 + [0.0010749]x^7
       #Diffusion term of Levy Noise:  tensor([0.8958])
       #L2 error:  0.124843724
       #'train' took 4.283743 s
    
    
    
   ################ basis_order=8     
     ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
       #  Drift term:  [-0.01834646] + [-1.1515284]x^1 + [-0.03952326]x^2 + [0.444074]x^3 + [0.01694677]x^4 + [-0.04927403]x^5 + [-0.00185248]x^6 + [0.00108577]x^7 + [-5.152078e-06]x^8
       #Diffusion term of Levy Noise:  tensor([0.8957])
       #L2 error:  0.12331158
       #'train' took 5.209281 s
    
    
    
  ################ basis_order=9     
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.6, lhs = 1.0
      #   Drift term:  [-0.01461075] + [-1.2149938]x^1 + [-0.07445611]x^2 + [0.5150415]x^3 + [0.03996538]x^4 + [-0.07201964]x^5 + [-0.00464053]x^6 + [0.00306173]x^7 + [0.00011203]x^8 + [-4.2602354e-05]x^9
      #Diffusion term of Levy Noise:  tensor([0.9016])
      #L2 error:  0.11888959
      #'train' took 3.663644 s
    
    #------------------------
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      # Drift term:  [-0.01318259] + [-1.2411803]x^1 + [-0.08882214]x^2 + [0.5512496]x^3 + [0.04585158]x^4 + [-0.07859498]x^5 + [-0.0052851]x^6 + [0.00344506]x^7 + [0.00012925]x^8 + [-4.8753765e-05]x^9
      #Diffusion term of Levy Noise:  tensor([0.9029])
      #L2 error:  0.09240185
      #'train' took 4.200607 s
     #------------------------- 
    
    ###sample = 10000, gauss_samp_number=60, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      #Drift term:  [-0.01553657] + [-1.2361933]x^1 + [-0.0656732]x^2 + [0.5378784]x^3 + [0.03762379]x^4 + [-0.07493166]x^5 + [-0.0045812]x^6 + [0.00317522]x^7 + [0.00010826]x^8 + [-4.335146e-05]x^9
      #Diffusion term of Levy Noise:  tensor([0.9020])
      #L2 error:  0.09781369
      #'train' took 5.156705 s
    
  ################ basis_order= 10  
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      #Drift term:  [-0.01984424] + [-1.2884624]x^1 + [-0.0182099]x^2 + [0.6157201]x^3 + [-0.00226566]x^4 + [-0.09791375]x^5 + [0.00302374]x^6 + [0.00512418]x^7 + [-0.00033172]x^8 + [-8.804127e-05]x^9 + [8.17962e-06]x^10
      #Diffusion term of Levy Noise:  tensor([0.9076])
      #L2 error:  0.074956864
      #'train' took 4.306605 s
      
      # gauss_var = 0.52
      #Drift term:  [-0.01912476] + [-1.3008037]x^1 + [-0.02403562]x^2 + [0.63799775]x^3 + [-0.00168368]x^4 + [-0.10222293]x^5 + [0.00312856]x^6 + [0.0054085]x^7 + [-0.0003479]x^8 + [-9.310986e-05]x^9 + [8.595213e-06]x^10
      #Diffusion term of Levy Noise:  tensor([0.9078])
      #L2 error:  0.06298099
      #'train' took 4.640334 s
    
    
    
  ################ basis_order= 11  
  
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      # Drift term:  [-0.01579971] + [-1.3213495]x^1 + [-0.06404009]x^2 + [0.6691167]x^3 + [0.03346895]x^4 + [-0.11914919]x^5 + [-0.00430508]x^6 + [0.00786592]x^7 + [0.00018245]x^8 + [-0.00021456]x^9 + [-2.111941e-06]x^10 + [1.9757292e-06]x^11
      #Diffusion term of Levy Noise:  tensor([0.9091])
      #L2 error:  0.069929086
      #'train' took 4.547649 s  
      
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
      #  Drift term:  [-0.01508181] + [-1.3371543]x^1 + [-0.07138138]x^2 + [0.6960255]x^3 + [0.03635037]x^4 + [-0.12574202]x^5 + [-0.00477249]x^6 + [0.00846437]x^7 + [0.00021501]x^8 + [-0.00023401]x^9 + [-2.7374867e-06]x^10 + [2.1936246e-06]x^11
      #Diffusion term of Levy Noise:  tensor([0.9097])
      #L2 error:  0.058145747
      #'train' took 4.656719 s
   
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.5, lhs = 1.0
      # 不收敛
    
    
  ################ basis_order= 12  
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      #   Drift term:  [-0.02512893] + [-1.4207127]x^1 + [-0.01113998]x^2 + [0.8000165]x^3 + [-0.03089279]x^4 + [-0.17018779]x^5 + [0.01146896]x^6 + [0.01443475]x^7 + [-0.00132237]x^8 + [-0.00053922]x^9 + [5.7019322e-05]x^10 + [7.1716477e-06]x^11 + [-8.2936486e-07]x^12
      #Diffusion term of Levy Noise:  tensor([0.9186])
      #L2 error:  0.08294476
      #'train' took 4.292853 s
    
    #gauss_var = 0.52
    #Drift term:  [-0.02479601] + [-1.4353062]x^1 + [-0.01550247]x^2 + [0.8267061]x^3 + [-0.03115091]x^4 + [-0.17747338]x^5 + [0.01173895]x^6 + [0.01517562]x^7 + [-0.00135629]x^8 + [-0.00056799]x^9 + [5.8808047e-05]x^10 + [7.559923e-06]x^11 + [-8.61354e-07]x^12
    #Diffusion term of Levy Noise:  tensor([0.9195])
    #L2 error:  0.07324849
    #'train' took 5.067543 s
    
    
    
    
    
    