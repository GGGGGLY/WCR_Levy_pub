# -*- coding: utf-8 -*-
"""
5D couple: only estimate the parameters of potential a, b

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from GenerateData_5d_potential import DataSet
from pyDOE import lhs
import time
import utils
import scipy.io
import scipy.special as sp
import math
from sympy import symbols, integrate, cos

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Gaussian(torch.nn.Module):
    def __init__(self, mu, sigma, lap_alpha, device):
        super(Gaussian, self).__init__()
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.dim = mu.shape[0]# 5
        self.lap_alpha = lap_alpha
        self.device = device

    def gaussZero(self, x):
        func = 1
        for d in range(self.dim):
            func = func * 1 / (self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(\
                -0.5 * (x[:, :, d] - self.mu[d]) ** 2 / self.sigma ** 2)
        return func

    def gaussFirst(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        for k in range(self.dim):
            func[:, :, k] = -(x[:, :, k] - self.mu[k]) / self.sigma ** 2 * g0
        return func

    def gaussSecond(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]]).to(self.device)
        for k in range(x.shape[2]):
            for j in range(x.shape[2]):
                if k == j:
                    func[:, :, k, j] = (-1 / self.sigma ** 2 + (-(x[:, :, k] - self.mu[k]) / self.sigma ** 2)\
                    * (-(x[:, :, j] - self.mu[j]) / self.sigma ** 2)  ) * g0                                 
                else:
                    func[:, :, k, j] = (-(x[:, :, k] - self.mu[k]) / self.sigma ** 2) * (
                            -(x[:, :, j] - self.mu[j]) / self.sigma ** 2
                    ) * g0
        return func
    
    def LapGauss(self, x):
        Gamma_ = sp.hyp1f1( (x.cpu().shape[2] + self.lap_alpha)/2, x.cpu().shape[2]/2, -torch.sum((x.cpu()-self.mu.cpu())**2, dim=2) / (2*self.sigma.cpu()**2))
        Gamma_ = Gamma_.to(device)
        func = (1/(torch.sqrt(torch.tensor(2))*self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*Gamma_
        #func = (1/(torch.sqrt(torch.tensor(2**2 *self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*Gamma_

        return func
    
    def LapGauss_VaryDim(self,x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        for k in range(x.shape[2]):
            # fractional derivate to k-th variable
            Gamma_1 = sp.hyp1f1((1 + self.lap_alpha)/2, 1/2, -(x[:, :, k].cpu()-self.mu[k].cpu())**2 / (2*self.sigma.cpu()**2))
            Gamma_1 = Gamma_1.to(device)
            func_k = (torch.sqrt(1/(torch.sqrt(torch.tensor(2))*self.sigma)) )** self.lap_alpha * sp.gamma( (1 + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(1/2) * \
                    1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))* Gamma_1

       
            func[:,:,k] = g0 * (self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))* torch.exp(\
                0.5 * (x[:, :, k] - self.mu[k]) ** 2 / self.sigma ** 2) *func_k
            #print("func", func)
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
        elif diff_order == 'frac_diag':
            return self.LapGauss_VaryDim(x, g0) 
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
    def __init__(self, t, data, alpha, Xi_type, testFunc, device):
        self.device = device
        self.t = t.to(self.device)
        self.itmax = len(t)
        self.data = data.to(self.device)
        self.net = testFunc
        self.basis_theta = None # given by build_basis
        self.A = None # given by build_A
        self.b = None # given by build_b
        self.dimension = None
        self.basis_number = None
        self.basis1_number = None
        self.basis2_number = None
        self.basis_order = None
        self.basis_xi_order = None
        self.bash_size = data.shape[1]
        self.alpha_levy = alpha
        self.Xi_type = Xi_type
        
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
        #self.basis1_number = int(np.math.factorial(self.dimension+self.basis_order)/(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order)))                            
        self.basis1_number = int(self.dimension * (1+5) + 1)
        self.basis2_number = int(self.dimension) 
             
        basis1 = []

        for it in range(self.t_number):
            X = self._get_data_t(it)
            basis_count1 = 0
            Theta = torch.zeros(X.size(0),self.basis1_number)
            Theta[:,0] = 1
            basis_count1 += 1
            for ii in range(0,self.dimension):
                Theta[:,basis_count1] = X[:,ii]
                basis_count1 += 1

            if self.basis_order ==3:
                for ii in range(0,self.dimension):
                    for jj in range(0,self.dimension):
                        Theta[:,basis_count1] = torch.mul(X[:,ii],X[:,jj]**2)
                        basis_count1 += 1

           
            #print("basis_count1", basis_count1) #31
            #print("basis1_number", self.basis1_number)
            assert basis_count1 == self.basis1_number
            basis1.append(Theta)
            # print("theta", Theta)
            basis_theta = torch.stack(basis1)
        print("basis_theta", basis_theta.shape)
            
            
        # Construct Xi  
        basis2 = []     
        for it in range(self.t_number):
            basis_count2 = 0
            X = self._get_data_t(it)
            Xi = torch.ones(X.size(0),1)
           
            basis2.append(Xi)
            basis_xi = torch.stack(basis2)
        print("basis_xi", basis_xi.shape)
            
        self.basis_theta = basis_theta
        self.basis = torch.cat([basis_theta, basis_xi],dim=2)         
        #self.basis = torch.stack(basis)
        print("self.basis.shape", self.basis.shape)
        self.basis_number = self.basis1_number + self.basis2_number
        print("self.basis1_number ", self.basis1_number)
        #print("self.basis_number ", self.basis_number)
        
        #self.basis = torch.stack(basis1).to(self.device)
        # print("self.basis.shape", self.basis.shape)

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number # mu: db
        C_number = self.dimension if self.xi_independence else 1
        A = torch.zeros([self.t_number, H_number+C_number]).to(self.device)

        # ##########################################################
        #  Tensor form of computing A and b for parallel computing
        # ##########################################################

        TX = self.data
        TX.requires_grad = True
        gauss0 = gauss(TX, diff_order=0)
        gauss1 = gauss(TX, diff_order=1)
        gauss2 = gauss(TX, diff_order=2)
        gauss_lap = gauss(TX, diff_order='frac')
        gauss_LapDiag = gauss(TX, diff_order='frac_diag')
        
        for kd in range(self.dimension):
            for jb in range(self.basis1_number):
                if self.drift_independence:
                    H = torch.mean(gauss1[:, :, kd] * self.data[:, :, kd].to(device) ** jb, dim=1)  ##这里有问题
                else:                    
                    H = torch.mean(gauss1[:, :, kd] * self.basis_theta[:, :, jb].to(device), dim=1)
                A[:, kd*self.basis1_number+jb] = H

        # second constru_xi
        if self.xi_independence:
            if self.Xi_type == "cI":
                for ld in range(self.dimension):
                    E = -torch.mean(gauss_lap, dim=1) 
                    #print("E",E)  # (1) 10^{-3}阶
                    
                    A[:, H_number+ld] = E
            elif self.Xi_type == "Diag":
                for kd in range(self.dimension):
                    E = -torch.mean(gauss_LapDiag[:, :, kd], dim=1)
                    E = torch.nan_to_num(E) 
                    #transfer "nan" to 0.0, transfer "inf" to 3.4028e+38
                    A[:, H_number+kd] = E
                    #print("E",E)      ##怎么几乎都是nan                   
        else:                                     
            E = np.sum([-torch.mean(gauss_lap, dim=1) for i in range(self.dimension)])
            A[:, H_number] = E
         
       
            
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number)
        #print("b", rb)

        if self.type == 'PDEFind':
            b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
            return A, b
        if self.type == 'LMM_2':
            AA = torch.ones(A.size(0) - 1, A.size(1)).to(self.device)
            bb = torch.ones(A.size(0) - 1).to(self.device)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + A[i + 1, :]) * dt / 2
                bb[i] = rb[i + 1] - rb[i]
            return AA, bb
        if self.type == 'LMM_3':
            AA = torch.ones(A.size(0) - 2, A.size(1)).to(self.device)
            bb = torch.ones(A.size(0) - 2).to(self.device)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + 4*A[i + 1, :] + A[i + 2, :]) * dt / 3
                bb[i] = rb[i + 2] - rb[i]
            return AA, bb
        if self.type == 'LMM_6':
            AA = torch.ones(A.size(0) - 5, A.size(1)).to(self.device)
            bb = torch.ones(A.size(0) - 5).to(self.device)
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

    def sampleTestFunc(self, samp_number):
        if self.gauss_samp_way == 'lhs':
            lb = torch.tensor([self.data[:, :, i].min()*0.6 for i in range(self.dimension)]).to(self.device)
            ub = torch.tensor([self.data[:, :, i].max()*0.6 for i in range(self.dimension)]).to(self.device)
            #lb = torch.tensor([-5 for i in range(self.dimension)]).to(self.device)
            #ub = torch.tensor([5 for i in range(self.dimension)]).to(self.device)
            #lb = torch.tensor([-4,-4]).to(self.device)
            #ub = torch.tensor([4,4]).to(self.device)
            mu_list = lb + self.lhs_ratio * (ub - lb) * torch.tensor(lhs(self.dimension, samp_number), dtype=torch.float32).to(self.device)
        if self.gauss_samp_way == 'SDE':
            if samp_number <= self.bash_size:
                index = np.arange(self.bash_size)
                np.random.shuffle(index)
                mu_list = data[-1, index[0: samp_number], :]
            else:
                print("The number of samples shall not be less than the number of tracks!")
        print("mu_list", mu_list)
        sigma_list = torch.ones(samp_number).to(self.device)*self.variance
        print("sigma_list", sigma_list.shape)
        return mu_list, sigma_list

    def buildLinearSystem(self, samp_number):
        mu_list, sigma_list = self.sampleTestFunc(samp_number)
        # print("mu_list: ", mu_list.device)
        # print("sigma_list: ", sigma_list.device)
        A_list = []
        b_list = []
        for i in range(mu_list.shape[0]):
            if i % 20 == 0:
                print('buildLinearSystem:', i)
            mu = mu_list[i]
            sigma = sigma_list[i]
            gauss = self.net(mu, sigma, 3/2,self.device)
            A, b = self.computeAb(gauss)
            A_list.append(A)
            b_list.append(b)
        self.A = torch.cat(A_list, dim=0) 
        #self.A = torch.where(torch.isnan(self.A), torch.full_like(self.A, 0), self.A)
        self.b = torch.cat(b_list, dim=0).unsqueeze(-1) # 1-dimension
        #self.b = torch.where(torch.isnan(self.b), torch.full_like(self.b, 0), self.b)


    @torch.no_grad()
    @utils.timing
    def solveLinearRegress(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression

    @utils.timing
    def STRidge(self, X0, y, lam, maxit, tol, normalize=0, print_results = False):
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
        else: #w = np.linalg.lstsq(X,y)[0]
            X_inv = np.linalg.pinv(X)  #用了伪逆
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
    
    # @utils.timing
    def compile(self, basis_order, gauss_variance, type, drift_term, xi_term,
                drift_independence, xi_independence, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.t_number = len(self.t)
        self.basis_order = basis_order
        self.variance = gauss_variance
        self.type = type
        self.drift = drift_term
        self.xi = xi_term
        self.drift_independence = drift_independence
        self.xi_independence = xi_independence
        if self.drift_independence:
            self.basis1_number = self.basis_order + 1
        else:
            self.build_basis()
        self.gauss_samp_way = gauss_samp_way
        self.lhs_ratio = lhs_ratio if self.gauss_samp_way == 'lhs' else 1

    @utils.timing
    @torch.no_grad()
    def train(self, gauss_samp_number, Xi_type, lam, STRidge_threshold, only_13=None):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        if only_13 == "hat":
            I = torch.tensor([1, 2, 6, 7, 8, 9, 11, 12, 16, 17, 18, 19, 20, 21])
            self.A = self.A[:, I]
        if only_13 == "5D":
            I = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 24])
            self.A = self.A[:, I]
        self.A = self.A.to("cpu")
        self.b = self.b.to("cpu")
        AA = torch.mm(torch.t(self.A), self.A)
        Ab = torch.mm(torch.t(self.A), self.b)
        print("A.max: ", self.A.max(), "b.max: ", self.b.max())
        print("ATA.max: ", AA.max(), "ATb.max: ", Ab.max())
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        print("zeta: ", self.zeta.size(), self.zeta)

      
        if self.xi_independence:
            self.zeta.squeeze()[self.dimension * self.basis1_number: ] \
                = (self.zeta.squeeze()[self.dimension * self.basis1_number:])**(2/3)
            print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number: ].numpy())
        else:
            print("Xi term: ", (self.zeta.squeeze()[self.dimension * self.basis1_number: ])**(2/3).numpy())

        
        true = torch.cat((self.drift.view(-1), self.xi))
        if only_13 == "hat" or only_13 == "5D":
            true = true[I]
        index = torch.nonzero(true).squeeze()
        relative_error = torch.abs((self.zeta.squeeze()[index] - true[index]) / true[index])
        print("Maximum relative error: ", relative_error.max().numpy())
        print("Maximum index: ", torch.argmax(relative_error))

        

if __name__ == '__main__':
    np.random.seed(100)
    torch.manual_seed(100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # device = torch.device('mps')
    print("device: ", device)

    T, dt = 1, 0.0001
    #t = np.array([0.1, 0.3, 0.5, 0.7, 1.0]).astype(np.float32)
    t = torch.linspace(0, 1, 11)
    t = torch.tensor(t)
    dim = 5 
    
    drift = torch.tensor([[0, 10, 0, 0, 0, 0,  -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 10, 0, 0, 0,  0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 10, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 10, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 10,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4, -4, -4]])
    
    xi = torch.tensor([1.0]).repeat(dim)
    #xi = torch.tensor([1.0, 1.0, 1.5, 1.2, 1.5], dtype=torch.float)
    samples = 30000
    alpha =3/2
    Xi_type = "Diag" 
    #dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=dim, drift_term=drift, \
    #                  xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, dim]), \
     #                 drift_independence=False, explosion_prevention=False)
    #data = dataset.get_data(plot_hist=False)
    
    data = torch.load("./5D_potential_data_99999.pt")
    #data = torch.load("./5D_potential_data_50000.pt")
    data = data * (1 + 0.*torch.rand(data.shape))
    print("data: ", data.shape, data.max(), data.min())
    

    testFunc = Gaussian
    model = Model(t, data, alpha, Xi_type, testFunc, device)
    model.compile(basis_order=3, gauss_variance=0.62, type='LMM_3', drift_term=drift, xi_term = xi, \
                  drift_independence=False, xi_independence=True, gauss_samp_way='lhs', lhs_ratio=1.0)
    
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=4000,  Xi_type = Xi_type, lam=0.0, STRidge_threshold=0.8, only_13="None")













# t = torch.linspace(0, 1, 11)
# 将E中的nan化成0了
# smaple = 50000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.65, lhs_ratio=0.8
#zeta:  torch.Size([35, 1]) tensor([[-1.4746],
   #     [ 1.0330],
   #     [-0.6939],
   #     [-0.4182],
   #     [-0.4224],
   #     [ 0.0000],
   #     [-2.6023],
   #     [ 1.0152],
   #     [-0.6737],
   #     [ 1.2099],
   #     [-0.5385],
   #     [-0.9858],
   #     [-0.8596],
   #     [ 0.0000],
   #     [ 0.0000],
   #     [ 0.9391],
   #     [-1.0677],
   #     [ 0.0000],
   #     [ 0.0000],
   #     [ 0.0000],
   #     [ 0.0000],
   #     [ 0.9206],
   #     [ 0.0000],
   #     [ 1.1760],
   #     [-1.8342],
   #     [ 0.0000],
   #     [-0.6110],
   #     [ 0.9868],
   #     [ 0.0000],
   #     [ 1.2871],
   #     [-3.5870],
   #     [ 3.2610],
    #    [-3.9284],
    #    [ 0.0000],
     #   [ 6.7873]])
#Xi term:  diag  [      nan 2.199058        nan 0.        3.5847852]
#Maximum relative error:  nan
#Maximum index:  tensor(9)
#'train' took 1260.671707 s  

# smaple = 50000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.3, gauss_variance=0.65, lhs_ratio=0.6
#tensor([[-1.6832],
 #       [ 2.4632],
 #       [-0.8692],
#        [ 0.5825],
#        [ 0.5247],
#        [-0.9526],
#        [-1.3020],
#        [ 1.2377],
#        [ 0.0000],
#        [ 0.0000],
#        [ 1.0695],
#        [ 0.4227],
#        [-3.9603],
#        [ 0.0000],
#        [ 0.3212],
#        [ 0.8338],
#        [-1.5293],
#        [ 0.0000],
#        [ 4.4863],
#        [-1.3263],
#        [ 1.1632],
#        [ 0.0000],
#        [ 2.0488],
#        [ 0.5180],
 #       [-1.5142],
#        [ 0.0000],
#        [ 0.0000],
 #       [-0.6235],
 #       [ 0.0000],
#        [ 0.7293],
#        [ 5.7640],
#        [11.3841],
#        [-1.5089],
#        [-1.1533],
#        [-0.9756]])
#Xi term:  diag  [3.2147605 5.060558        nan       nan       nan]
#Maximum relative error:  nan


# smaple = 100000, gauss_samp_number=500, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.65, lhs_ratio=0.8
# zeta:  torch.Size([35, 1]) tensor([[ 0.0000],
#        [ 1.2723],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.7296],
#        [ 0.9863],
#        [ 0.7530],
#        [-1.1493],
#        [ 0.7776],
#        [-1.1418],
#        [ 0.0000],
#        [ 0.0000],
#        [ 1.5132],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.0000],
#        [-0.6956],
#        [ 1.1325],
#        [ 0.0000],
#        [ 1.5910],
#        [ 0.0000],
#        [ 0.5411],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.0000],
#        [ 0.4198],
#        [ 1.9806],
#        [-2.4236],
#        [10.3080],
#        [-4.4534],
#        [-5.6198],
#        [ 6.1095]])
#Xi term:  diag  [      nan 4.7364054       nan       nan 3.3419802]
#Maximum relative error:  nan
#Maximum index:  tensor(9)
#'train' took 5701.526322 s


# smaple = 100000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.65, lhs_ratio=0.6
#tensor([[-3.4812],      [ 0.5601],       [ 0.0000],       [ 0.0000],       [ 0.0000],        [-1.0858], 
#     [-7.7431],       [ 0.0000],       [ 0.0000],      [-0.7881],       [ 0.0000],       [-0.7844],
#        [-2.4490],       [ 0.0000],       [ 0.7078],       [ 0.0000],       [-0.5096],       [ 0.0000],
#        [ 6.7559],       [ 0.0000],      [ 1.2054],       [ 0.0000],       [ 1.0659],      [ 1.1443],
#        [-2.1957],       [ 0.0000],       [ 0.0000],       [ 0.0000],      [ 0.0000],      [ 0.6770],
#        [ 0.0000],       [-5.8261],       [-3.1055],       [-0.4826],       [ 2.0851]])
#Xi term:  diag  [0.              nan       nan       nan 1.6320951]
#Maximum relative error:  nan
#Maximum index:  tensor(10)



######  lb = min*0.5, ub = max*0.5

# smaple = 50000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.65, lhs_ratio=0.6
# 全是0.....



# smaple = 50000, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.65, lhs_ratio=1.0
#不收敛

# smaple = 50000, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.3, gauss_variance=0.65, lhs_ratio=0.65
#不收敛

# smaple = 50000, gauss_samp_number=900, lam=0.0, STRidge_threshold=0.3, gauss_variance=0.65, lhs_ratio=0.65
#不收敛


#数据生成的时候*0.999了，之前是*0.9999，给的噪声方差大，会让数据更集中，跑得慢一点。
# smaple = 50000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.6, lhs_ratio=0.8
# 不收敛

# smaple = 50000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.4, gauss_variance=0.65, lhs_ratio=0.8
# 不收敛



