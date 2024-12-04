# -*- coding: utf-8 -*-
"""
Non-poly problem: drift = -x-sinx; diffusion = 1, xi = 1
With more basis
"""

import torch
import torch.nn as nn
import numpy as np
from generate_data_NonPoly_sindrift import DataSet
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
    
    
    def forward(self, x, diff_order=0): 
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
        #self.basis = None # given by build_basis
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
        self.basis1_number = int(1+ self.basis_order) + 4
        # +4 means adding {sinx, cosx, sin^2 x, cos^2 x}      
        # int(1+ self.basis_order): 1, x, x^2, x^3,...
        self.basis2_number = int( self.dimension ) 
        

        # Construct Theta
        basis1 = []

        for it in range(self.t_number):
            X = self._get_data_t(it)
            basis_count1 = 0
            Theta = torch.zeros(X.size(0),self.basis1_number)
            Theta[:,0] = 1
            basis_count1 += 1
            for i in range(self.basis1_number -4):
                Theta[:, i] = X[:, 0]**i
            Theta[:, -4] = torch.sin(X[:, 0])
            Theta[:, -3] = torch.cos(X[:, 0])
            Theta[:, -2] = torch.sin(X[:, 0])**3
            Theta[:, -1] = torch.cos(X[:, 0])**3
            basis1.append(Theta)
            # print("X", X)
            # print("theta", Theta.shape)

        basis_theta = torch.stack(basis1)
            #basis_theta = torch.stack(basis1)
        print("basis_theta", basis_theta.shape)
        """
        basis_theta = {1, x, x^2, x^3, sinx, cosx, sin^3 x, cos^3 x}
        basis_theta torch.Size([10, 20000, 6])
        """
            
            
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
        self.basis_xi = basis_xi
        #self.basis = torch.cat([basis_theta, basis_xi],dim=2)         
        #self.basis = torch.stack(basis)
        #print("self.basis.shape", self.basis.shape)
        print("self.basis1_number ", self.basis1_number)
       
        
        
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2)     

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number  #db
        F_number = 1 if self.dimension ==1 else self.dimension * self.dimension * self.basis1_number
        C_number = self.dimension
        
        A = torch.zeros([self.t_number, H_number+F_number+C_number]) #A is a L* (db+d^2c)
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
        gauss2 = gauss(TX, diff_order=2)
        gauss_lap = gauss(TX, diff_order='frac')
        
        for kd in range(self.dimension):
            for jb in range(self.basis1_number):
                # print("gauss1[:, :, %s]" % kd, gauss1[:, :, kd].size())
                H = 1/self.bash_size * torch.sum(
                    gauss1[:, :, kd] * self.basis_theta[:, :, jb], dim=1)
                A[:, kd*self.basis1_number+jb] = H
        
        # for ld in range(self.dimension):
        #     for kd in range(self.dimension):
        #         for jb in range(self.basis1_number):
        #             F = 1/self.bash_size * torch.sum(
        #                 gauss2[:, :, ld, kd] *
        #                self.basis_theta[:, :, jb], dim=1)
        #             A[:, H_number + ld *self.basis1_number + kd *self.basis1_number + jb] = F

                 
                
        if self.dimension == 1:
            F = 1/self.bash_size * torch.sum(
                gauss2[:, :, 0, 0], dim=1)
            A[:, H_number] = F
                
            for ld in range(self.dimension):
                for kd in range(self.dimension):
                    E = -torch.mean(gauss_lap, dim=1)
                    A[:, H_number+F_number] = E                    
        else:
            print("The dimension should be 1.")                   
               
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
       
       
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
            factor = 1/2
            #mu_list = self.lhs_ratio * torch.rand(samp_number)*(1.5-(-1.5)) -1.5
            mu_list = self.lhs_ratio * torch.rand(samp_number)*(self.data.max()-self.data.min())*factor + self.data.min()*factor
            
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
        #if biginds != []:
        if len(biginds) != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

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
            
        y = y + self.zeta[self.basis_order+1, 0] *torch.sin(x) + self.zeta[self.basis_order+2, 0] *torch.cos(x)
        return y
 
    @utils.timing
    @torch.no_grad()
    def train(self, sample_num, gauss_samp_number, lam, STRidge_threshold):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        print("A: ", self.A.size(), "b: ", self.b.size())
        """ A:  torch.Size([450, 6]), b:  torch.Size([450, 1]) """
        print("A",np.linalg.cond(self.A, p=None))
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        print("zeta: ", self.zeta)

        drift = [self.zeta[0].numpy()]
        for i in range(1, self.basis1_number-4):
            drift.extend([" + ", self.zeta[i].numpy(), 'x^', i])
        drift.extend([" + ", self.zeta[self.basis1_number-4].numpy(), 'sinx'])
        drift.extend([" + ", self.zeta[self.basis1_number-3].numpy(), 'cosx'])
        drift.extend([" + ", self.zeta[self.basis1_number-2].numpy(), 'sin^3 x'])
        drift.extend([" + ", self.zeta[self.basis1_number-1].numpy(), 'cos^3 x'])
        print("Drift term: ", "".join([str(_) for _ in drift]))
        
        diffusion = np.sqrt(self.zeta[-2].numpy()*2)
        print("Diffusion term: ", diffusion)
        
        self.zeta[-1] = (self.zeta[-1])**(2/3) 
        print("Noise Intensity of Levy Noise: ", self.zeta[-1])
        #true = torch.cat((self.drift, self.xi))
        #index = torch.nonzero(true).squeeze()
        #relative_error = torch.abs((self.zeta.squeeze()[index] - true[index]) / true[index])
       

if __name__ == '__main__':
    np.random.seed(6) 
    torch.manual_seed(6)

    dt = 0.0001
    # t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    #t = torch.linspace(0.1,0.4,4)
    #t = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #t = torch.tensor([0.2, 0.5, 1])
   
    #drift = torch.tensor([-3, 0, 0, -0.5])
    diffusion = torch.tensor([1])
    xi = torch.tensor([1])
    samples = 20000
    dataset = DataSet(t, dt=dt, samples_num=samples, dim=1, diffusion_term = diffusion, xi_term=xi, alpha_levy = 3/2, \
                      initialization=torch.normal(0, 0.2,[samples, 1]),explosion_prevention=False) 
                      #initialization=torch.randint(-1000,1000,[10000, 1])
    data = dataset.get_data(plot_hist=False)
    print("data.max: ", data.max(), "data.min: ", data.min())

    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=3, gauss_variance=0.45, type='LMM_2_nonequal', xi_term=xi,\
                  gauss_samp_way='lhs', lhs_ratio=0.8)
    model.train(sample_num = samples, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.1)
    
     
  
   ################################        
      
      #####  t = torch.linspace(0,1,11), dt = 0.0001, 1e-6, 0.99999
      
      ####sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.5, lhs = 1.0
      #Drift term:  [-0.00523797] + [-0.8167456]x^1 + [-1.1124642]sinx + [0.01882189]cosx
      #Diffusion term:  [1.1855001]
      #Noise Intensity of Levy Noise:  tensor([0.7273])
      
  
    ####sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.4, lhs = 1.0
    # Drift term:  [-0.02972674] + [-0.86956376]x^1 + [-1.0475278]sinx + [0.04472015]cosx
    #Diffusion term:  [1.1181195]
    #Noise Intensity of Levy Noise:  tensor([0.7980])
    #'train' took 4.445840 s
    
    ####sample = 20000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.4, lhs = 1.0
    # Drift term:  [-0.01379205] + [-0.9585566]x^1 + [-0.9725849]sinx + [0.03530917]cosx
    #Diffusion term:  [1.0938927]
    #Noise Intensity of Levy Noise:  tensor([0.8290])
    #'train' took 9.128231 s
    
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.4, lhs = 1.0
    # Drift term:  [0.01073073] + [-0.9696316]x^1 + [-0.97837114]sinx + [0.0146821]cosx
    #Diffusion term:  [1.092775]
    #Noise Intensity of Levy Noise:  tensor([0.8379])
    #'train' took 14.878052 s
    
    #-----------------------
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_var = 0.4, lhs = 1.0
    #Drift term:  [0.] + [-1.0040958]x^1 + [-0.93468744]sinx + [0.]cosx
    #Diffusion term:  [1.08956]
    #Noise Intensity of Levy Noise:  tensor([0.8422])
    #------------------
    
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_var = 0.35, lhs = 1.0
    #Drift term:  [0.] + [-1.0931318]x^1 + [-0.8404603]sinx + [0.]cosx
    #Diffusion term:  [1.0414554]
    #Noise Intensity of Levy Noise:  tensor([0.8906])
    #'train' took 17.657118 s
    
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_var = 0.3, lhs = 1.0
    #Drift term:  [0.] + [-1.2234541]x^1 + [-0.69810987]sinx + [0.]cosx
    #Diffusion term:  [0.9786858]
    #Noise Intensity of Levy Noise:  tensor([0.9480])
    
    ####sample = 20000, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, gauss_var = 0.3, lhs = 1.0
    #Drift term:  [0.] + [-1.2486002]x^1 + [-0.6651421]sinx + [0.]cosx
    #Diffusion term:  [0.9728026]
    #Noise Intensity of Levy Noise:  tensor([0.9522])
    #'train' took 22.467597 s
    
    ####sample = 20000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.05, gauss_var = 0.3, lhs = 1.2
    #误差变大了
    
    
###############################################################    
    #####  t = torch.linspace(0,1,11), dt = 0.0001, 1e-7, 0.999999
    
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.4, lhs = 0.8
    # Drift term:  [0.] + [-1.0171366]x^1 + [-0.95780075]sinx + [0.]cosx
    #Diffusion term:  [0.94479793]
    #Noise Intensity of Levy Noise:  tensor([1.0262])
    #'train' took 13.099318 s
    
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.42, lhs = 0.8
    # Drift term:  [0.] + [-0.9910881]x^1 + [-0.98944914]sinx + [0.]cosx
    #Diffusion term:  [0.957115]
    #Noise Intensity of Levy Noise:  tensor([1.0174])
    #'train' took 14.321518 s
    
    ####sample = 20000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.45, lhs = 0.8
    #
    
    
    
    
 