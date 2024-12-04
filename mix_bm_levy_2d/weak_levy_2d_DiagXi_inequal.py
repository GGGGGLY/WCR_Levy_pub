# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:01:08 2023

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from GenerateData_n_inequal_ import DataSet
from pyDOE import lhs
import time
import utils
import scipy.io
import scipy.special as sp
import math
from sympy import symbols, integrate, cos
from mpmath import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Gaussian(torch.nn.Module):
    def __init__(self, mu, sigma, lap_alpha, device):
        super(Gaussian, self).__init__()
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.dim = mu.shape[0]
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
        
        #func = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * \
        #    (1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))**x.shape[2])* hyp1f1( (x.shape[2] + self.lap_alpha, 2), (x.shape[2], 2), -torch.sum((x-self.mu)**2, dim=2) / (2*self.sigma**2)) 
        #print("LapGauss.shape", func.shape) #11,10000
        func = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * \
            (1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))**x.shape[2])* sp.hyp1f1((x.shape[2] + self.lap_alpha)/2, x.shape[2]/2, -torch.sum((x-self.mu)**2, dim=2) / (2*self.sigma**2)) 
        
        return func    
    
    def LapGauss_VaryDim(self,x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        for k in range(x.shape[2]):
            # 对第k个变量求分数阶导数
            func_k = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (1 + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(1/2) * \
                    1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*\
                        sp.hyp1f1((1 + self.lap_alpha)/2, 1/2, -(x[:, :, k]-self.mu[k])**2 / (2*self.sigma**2))  
                        
                        #hyp1f1(2, (-1,3), -1000)
            #func_k = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (1 + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(1/2) * \
            #         1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*\
            #             hyp1f1( (1 + self.lap_alpha, 2), (1,2), -(x[:, :, k]-self.mu[k])**2 / (2*self.sigma**2))  
              
        
        #其余变量没有求导，但是密度函数仍然包含
            func[:,:,k] = g0 * (self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))* torch.exp(\
                0.5 * (x[:, :, k] - self.mu[k]) ** 2 / self.sigma ** 2) *func_k
           
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
            return self.LapGauss_VaryDim(x, g0) ##################################
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

        # self.batch_size = None # tbd
        # self.train_state = TrainState() # tbd
        # self.losshistory = LossHistory() # tbd

    def _get_data_t(self, it):
        X = self.data[it,:,:]
        return X
    
    @utils.timing # decorator
    @torch.no_grad()
    def build_basis(self): # \Lambda matrix
        """build the basis list for the different time snapshot 
        """
        self.basis1_number = int(np.math.factorial(self.dimension+self.basis_order)/(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order)))                            
             
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

            if self.basis_order >= 2:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        Theta[:,basis_count1] = torch.mul(X[:,ii],X[:,jj])
                        basis_count1 += 1

            if self.basis_order >= 3:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            Theta[:,basis_count1] = torch.mul(torch.mul(X[:,ii],\
                                X[:,jj]),X[:,kk])
                            basis_count1 += 1

            if self.basis_order >= 4:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                Theta[:,basis_count1] = torch.mul(torch.mul(torch.mul(X[:,ii],\
                                    X[:,jj]),X[:,kk]),X[:,ll])
                                basis_count1 += 1

            if self.basis_order >= 5:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                for mm in range(ll,self.dimension):
                                    Theta[:,basis_count1] = torch.mul(torch.mul(torch.mul(torch.mul(\
                                        X[:,ii],X[:,jj]),X[:,kk]), X[:,ll]),X[:,mm])
                                    basis_count1 += 1
            assert basis_count1 == self.basis1_number
            basis1.append(Theta)
            # print("theta", Theta)
            basis_theta = torch.stack(basis1)
        print("basis_theta", basis_theta.shape)
            
            
        # Construct Xi   #d(d+1)/2+1 = 3+1 = 4
        basis2 = []     
        for it in range(self.t_number):
            X = self._get_data_t(it)
            Xi = torch.ones(X.size(0),1)
           
            basis2.append(Xi)
            basis_xi = torch.stack(basis2)
        print("basis_xi", basis_xi.shape)
            
        self.basis_theta = basis_theta
        self.basis = torch.cat([basis_theta, basis_xi],dim=2)         
        #self.basis = torch.stack(basis)
        print("self.basis.shape", self.basis.shape)
        print("self.basis1_number ", self.basis1_number)
        
        #self.basis = torch.stack(basis1).to(self.device)
        # print("self.basis.shape", self.basis.shape)

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number # mu: db
        F_number = self.dimension if self.diffusion_independence else 1  #sigma  d^2 b 这里只有对角元
        C_number = self.dimension if self.xi_independence else 1#* self.basis2_number #* self.basis2_number  #d^2 c

        A = torch.zeros([self.t_number, H_number+F_number+C_number]).to(self.device)

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
        gauss_LapDiag = gauss(TX, diff_order='frac_diag')
        # print("gauss0: ", gauss0.device)
        # print("gauss1: ", gauss1.device)
        # print("gauss2: ", gauss2.device)
        # print("self.data: ", self.data.device)

        # print("self.basis_number", self.basis_number)
        # print("self.dimension", self.dimension)
        for kd in range(self.dimension):
            for jb in range(4):#range(self.basis1_number): order+1
                if self.drift_independence:
                    H = torch.mean(gauss1[:, :, kd] * self.data[:, :, kd] ** jb, dim=1)
                else:                    
                    H = torch.mean(gauss1[:, :, kd] * self.basis_theta[:, :, jb], dim=1)
                A[:, kd*self.basis1_number+jb] = H

        # compute A by F_lkj
        if self.diffusion_independence:
            for ld in range(self.dimension):
                F = torch.mean(gauss2[:, :, ld, ld], dim=1)  #涉及维度
                #print("F",F)
                A[:, H_number+ld] = F
        else:
            F = np.sum([torch.mean(gauss2[:, :, i, i], dim=1) for i in range(self.dimension)])
            A[:, H_number] = F
        
     
        # second constru_xi
        if self.xi_independence:
            if self.Xi_type == "cI":
                for ld in range(self.dimension):
                    E = -torch.mean(gauss_lap, dim=1) 
                    print("E",E)  # (1) 10^{-3}阶
                    
                    A[:, H_number+F_number+ld] = E
            elif self.Xi_type == "Diag":
                for kd in range(self.dimension):
                    E = -torch.mean(gauss_LapDiag[:, :, kd], dim=1)
                    A[:, H_number+F_number+kd] = E
                        
        else:                                     
            E = np.sum([-torch.mean(gauss_lap, dim=1) for i in range(self.dimension)])
            A[:, H_number+F_number] = E
            
            
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
            lb = torch.tensor([self.data[:, :, i].min()*(2/3) for i in range(self.dimension)]).to(self.device)
            ub = torch.tensor([self.data[:, :, i].max()*(2/3)  for i in range(self.dimension)]).to(self.device)
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
        #print("A_list", A_list)
        #print("b_list", b_list)  #A, b都是10^{-3}阶
       
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
    def compile(self, basis_order, gauss_variance, type, drift_term,diffusion_term, xi_term,
                drift_independence, diffusion_independence, xi_independence, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.t_number = len(self.t)
        self.basis_order = basis_order
        self.variance = gauss_variance
        self.type = type
        self.drift = drift_term
        self.xi = xi_term
        self.diffusion = diffusion_term
        self.drift_independence = drift_independence
        self.diffusion_independence = diffusion_independence
        self.xi_independence = xi_independence
        if self.drift_independence:
            self.basis1_number = self.basis_order + 1
        else:
            self.build_basis()
        self.gauss_samp_way = gauss_samp_way
        self.lhs_ratio = lhs_ratio if self.gauss_samp_way == 'lhs' else 1

    @utils.timing
    @torch.no_grad()
    def train(self, gauss_samp_number, Xi_type, lam, STRidge_threshold, only_hat_13=False):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        if only_hat_13:
            I = torch.tensor([1, 2, 6, 7, 8, 9, 11, 12, 16, 17, 18, 19, 20, 21])
            self.A = self.A[:, I]
        #self.A = torch.where(self.A == 0, 0.0001, self.A) #####
        self.A = self.A.to("cpu")
        self.b = self.b.to("cpu")
        AA = torch.mm(torch.t(self.A), self.A)
        Ab = torch.mm(torch.t(self.A), self.b)
        print("A.max: ", self.A.max(), "b.max: ", self.b.max())
        print("ATA.max: ", AA.max(), "ATb.max: ", Ab.max())
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        print("zeta: ", self.zeta.size(), self.zeta)
        

        if self.drift_independence:
            for i in range(self.dimension):
                drift = [self.zeta[i*self.basis1_number].numpy()]
                for j in range(self.basis1_number - 1):
                    drift.extend([" + ", self.zeta[i*self.basis1_number + j + 1].numpy(), 'x_', i + 1, '^', j + 1])
                print("Drift term: ", "".join([str(_) for _ in drift]))
        else:
            if only_hat_13 :
                self.basis1_number = 6
            for i in range(self.dimension):
                drift = [self.zeta[i*self.basis1_number].numpy()]
                for j in range(self.basis1_number - 1):
                    drift.extend([" + ", self.zeta[i*self.basis1_number + j + 1].numpy(), 'x_', j + 1])
                print("Drift term : ", i+1, "".join([str(_) for _ in drift]))
                
        if self.diffusion_independence:
            self.zeta.squeeze()[self.dimension*self.basis1_number: self.dimension*self.basis1_number + self.dimension] \
                = torch.sqrt(2*self.zeta.squeeze()[self.dimension * self.basis1_number:self.dimension*self.basis1_number + self.dimension])
            print("Diffusion term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number: self.dimension*self.basis1_number + self.dimension].numpy())
        else:
            print("Diffusion term: ", np.sqrt(2*self.zeta.squeeze()[self.dimension * self.basis1_number : self.dimension*self.basis1_number + self.dimension].numpy()))
            
            
        if self.xi_independence:
            if Xi_type == "cI":
                self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: self.dimension*self.basis1_number+2*self.dimension] \
                    = ((self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension:]))**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number+ self.dimension: ].numpy())
            elif Xi_type == "Diag":
                self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: self.dimension*self.basis1_number+2*self.dimension] \
                    = (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension:])**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number+ self.dimension: ].numpy())
            
        else:
            print("Xi term: ", (1.7973 * (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: ]))**(2/3).numpy())

        true = torch.cat((self.drift.view(-1), self.diffusion, self.xi)) 
        if only_hat_13:
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
    #t = torch.linspace(0,1,11)
    #t = np.array([0.1, 0.3, 0.5, 0.7, 1.0]).astype(np.float32)
    t = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).astype(np.float32) 
    t = torch.tensor(t)
    dim = 2
    #Xi_type = "cI" #对角元相同的对角阵  case1
    Xi_type = "Diag" # 对角元分别估计 case2
    
    drift = torch.tensor([0, 1, 0, -1]).repeat(dim, 1)
    diffusion = torch.ones(dim)
    xi = torch.tensor([1.0, 1.0], dtype=torch.float) #torch.ones(dim)
    sample, dim = 20000, dim 
    alpha =3/2
    dataset = DataSet(t, dt=dt, samples_num=sample, dim=dim, drift_term=drift, diffusion_term=diffusion,\
                      xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(mean=0., std=0.2, size=(sample, dim)),   # torch.normal(mean=0., std=0.1, size=(sample, dim)),
                      explosion_prevention=False)
    data = dataset.get_data(plot_hist=False)
    data = data * (1 + 0.*torch.rand(data.shape))
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(t, data, alpha, Xi_type, testFunc, device)
    model.compile(basis_order=3, gauss_variance=0.65, type='LMM_3', drift_term=drift, \
                  diffusion_term=diffusion, xi_term = xi, drift_independence=True, \
                  diffusion_independence=True, xi_independence=True, gauss_samp_way='lhs', lhs_ratio=1.0) 
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=200, Xi_type = Xi_type, lam=0.0, STRidge_threshold=0.05, only_hat_13=False)
    
    
      
    
   
    ###之前数据是  #t = np.array([0.1, 0.3, 0.5, 0.7, 1.0]).astype(np.float32),  initial N(0, 0.4)
     
    ####################################
    
    #from GenerateData_n_inequal_ import DataSet
    
    # mu的范围 [data.min*(2/3), data.max*(2/3)]
    
    ###################################
    
    #-------------------------------------------------
    #optimal
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.57, lhs_ratio=0.8
    # Drift term:  [-0.07353797] + [0.99198866]x_1^1 + [-0.1344004]x_1^2 + [-1.0893017]x_1^3
    #Drift term:  [0.] + [1.0828545]x_2^1 + [0.]x_2^2 + [-1.1107061]x_2^3
    #Diffusion term:  diag  [1.0147268 1.1188667]
    #Xi term:  diag  [0.9754883  0.97845006]
    #Maximum relative error:  0.11886668
    #Maximum index:  tensor(5)
    #'train' took 9.243611 s
    #-------------------------------------------------------
    
    #DIAG
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.65, lhs_ratio=0.69
    # Drift term:  [-0.12053064] + [0.8639358]x_1^1 + [-0.14629099]x_1^2 + [-1.0473537]x_1^3
    #Drift term:  [-0.06257191] + [1.0348771]x_2^1 + [0.]x_2^2 + [-1.1029449]x_2^3
    #Diffusion term:  diag  [0.86476505 0.9117134 ]
    #Xi term:  diag  [1.094403  1.1515654]
    #Maximum relative error:  0.15156543
    #Maximum index:  tensor(7)
    #'train' took 8.840933 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.65, lhs_ratio=0.75
    # Drift term:  [-0.1058533] + [0.9302817]x_1^1 + [-0.1657696]x_1^2 + [-1.0885401]x_1^3
    #Drift term:  [0.] + [1.0783752]x_2^1 + [0.]x_2^2 + [-1.0978059]x_2^3
    #Diffusion term:  diag  [0.9439144 1.0673467]
    #Xi term:  diag  [1.0349684 1.0187892]
   # Maximum relative error:  0.09780586    # [-0.1657696]x_1^2
    #Maximum index:  tensor(3)
    #'train' took 8.490396 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.65, lhs_ratio=0.8
    # Drift term:  [0.] + [1.0509495]x_1^1 + [-0.10637312]x_1^2 + [-1.0959104]x_1^3
    #Drift term:  [0.] + [1.1068624]x_2^1 + [0.]x_2^2 + [-1.1363474]x_2^3
    #Diffusion term:  diag  [1.212629  1.2515866]
    #Xi term:  diag  [0.8015558  0.86185384]
    #Maximum relative error:  0.25158656
    #Maximum index:  tensor(5)
    #'train' took 7.279283 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.6, lhs_ratio=0.8
    # Drift term:  [-0.06595272] + [1.0047542]x_1^1 + [-0.13510326]x_1^2 + [-1.1004355]x_1^3
    #Drift term:  [0.] + [1.0890139]x_2^1 + [0.]x_2^2 + [-1.1146631]x_2^3
    #Diffusion term:  diag  [1.0432283 1.1498988]
    #Xi term:  diag  [0.95897865 0.95166844]
    #Maximum relative error:  0.14989877
    #Maximum index:  tensor(5)
    #'train' took 11.426560 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.57, lhs_ratio=0.8
    # Drift term:  [-0.07353797] + [0.99198866]x_1^1 + [-0.1344004]x_1^2 + [-1.0893017]x_1^3
    #Drift term:  [0.] + [1.0828545]x_2^1 + [0.]x_2^2 + [-1.1107061]x_2^3
    #Diffusion term:  diag  [1.0147268 1.1188667]
    #Xi term:  diag  [0.9754883  0.97845006]
    #Maximum relative error:  0.11886668
    #Maximum index:  tensor(5)
    #'train' took 9.243611 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.55, lhs_ratio=0.85
    # Drift term:  [0.] + [1.0692555]x_1^1 + [-0.07892312]x_1^2 + [-1.0795624]x_1^3
    #Drift term:  [0.] + [1.1038513]x_2^1 + [0.]x_2^2 + [-1.1386262]x_2^3
    #Diffusion term:  diag  [1.1387353 1.2160732]
    #Xi term:  diag  [0.85746676 0.8904347 ]
    #Maximum relative error:  0.21607316
    #Maximum index:  tensor(5)
    #'train' took 7.750430 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.52, lhs_ratio=0.85
    # 不收敛
    
    
    #从最优的那个出发
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.1, gauss_variance=0.65, lhs_ratio=0.75
    #
    #Drift term:  [0.] + [1.012684]x_1^1 + [-0.10691252]x_1^2 + [-1.0605516]x_1^3
    #Drift term:  [0.] + [1.0942826]x_2^1 + [0.]x_2^2 + [-1.1198795]x_2^3
    #Diffusion term:  diag  [1.2215496 1.2135655]
    #Xi term:  diag  [0.76586145 0.8880911 ]
    #Maximum relative error:  0.23413855
    #Maximum index:  tensor(6)
    #'train' took 7.887193 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.1, gauss_variance=0.6, lhs_ratio=0.75
    #
    #Drift term:  [-0.12062377] + [0.9024822]x_1^1 + [-0.16161804]x_1^2 + [-1.0649972]x_1^3
    #Drift term:  [0.] + [1.0697185]x_2^1 + [0.]x_2^2 + [-1.0920025]x_2^3
    #Diffusion term:  diag  [0.90618527 1.0151756 ]
    #Xi term:  diag  [1.050311  1.0596731]
    #Maximum relative error:  0.09751779
    #Maximum index:  tensor(0)
    #'train' took 11.491998 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.15, gauss_variance=0.52, lhs_ratio=0.75
    #不收敛
    
    
    
    
    
    ####################################
    
    #from GenerateData_n_inequal_ import DataSet 没有L2
    
    ###################################
    
    ##########
    #optimal
    ##########
    #--------------------------------------------------------------------
    # sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.66, lhs_ratio=0.69
    # Drift term:  [-0.08049033] + [0.9393768]x_1^1 + [-0.12519318]x_1^2 + [-1.0691279]x_1^3
    #Drift term:  [0.] + [1.1080954]x_2^1 + [0.]x_2^2 + [-1.1143446]x_2^3
    #Diffusion term:  diag  [1.1002458 1.0979556]
    #Xi term:  diag  [0.9009149  0.98562676]
    #Maximum relative error:  0.1143446
    #Maximum index:  tensor(3)
    #'train' took 12.113438 s
    #---------------------------------------------------------------------------
    
    #DIAG
    ## sample =5000, gauss_samp_number=120, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.58, lhs_ratio=0.55
    # Drift term:  [-0.37254772] + [1.0496702]x_1^1 + [-0.05687075]x_1^2 + [-1.1403295]x_1^3
    #Drift term:  [0.17882846] + [1.1850079]x_2^1 + [-0.16981317]x_2^2 + [-1.1428268]x_2^3
    #Diffusion term:  diag  [1.103809 0.829588]
    #Xi term:  diag  [0.8188951 1.1024588]
    #Maximum relative error:  0.18500793
    #Maximum index:  tensor(2)  #-0.37254772吧
    #'train' took 4.988494 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.7, lhs_ratio=0.65
    # Drift term:  [-0.05650951] + [0.9275985]x_1^1 + [-0.10679854]x_1^2 + [-1.0515444]x_1^3
    #Drift term:  [0.] + [1.1112449]x_2^1 + [0.]x_2^2 + [-1.1358012]x_2^3
    #Diffusion term:  diag  [1.1919078 1.1620922]
    #Xi term:  diag  [0.8282573  0.96274096]
    #Maximum relative error:  0.19190776
    #Maximum index:  tensor(4)
    #'train' took 14.102639 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.68, lhs_ratio=0.66
    # Drift term:  [-0.07036675] + [0.9186223]x_1^1 + [-0.11305157]x_1^2 + [-1.0509124]x_1^3
    #Drift term:  [0.] + [1.1114173]x_2^1 + [0.]x_2^2 + [-1.1350636]x_2^3
    #Diffusion term:  diag  [1.1510923 1.1246367]
    #Xi term:  diag  [0.86300665 0.9931723 ]
    #Maximum relative error:  0.15109229
    #Maximum index:  tensor(4)
    #'train' took 12.353562 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.67, lhs_ratio=0.66
    # Drift term:  [-0.07309051] + [0.9131677]x_1^1 + [-0.11271831]x_1^2 + [-1.0469681]x_1^3
    #Drift term:  [0.] + [1.110042]x_2^1 + [0.]x_2^2 + [-1.1357461]x_2^3
    #Diffusion term:  diag  [1.1417366 1.1119242]
    #Xi term:  diag  [0.86970896 1.0052642 ]
    #Maximum relative error:  0.14173663
    #Maximum index:  tensor(4)
    #'train' took 10.075519 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.66, lhs_ratio=0.66
    # Drift term:  [-0.07589438] + [0.90758646]x_1^1 + [-0.11240481]x_1^2 + [-1.0429409]x_1^3
    #Drift term:  [0.] + [1.1085134]x_2^1 + [0.]x_2^2 + [-1.1363955]x_2^3
    #Diffusion term:  diag  [1.1320186 1.0989399]
    #Xi term:  diag  [0.87665784 1.0174628 ]
    #Maximum relative error:  0.13639545
    #Maximum index:  tensor(3)
    #'train' took 10.695056 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.64, lhs_ratio=0.66
    # 不收敛
    
    # sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.65, lhs_ratio=0.67
    # 不收敛
    
    # sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.66, lhs_ratio=0.67
    # Drift term:  [-0.08210044] + [0.9125389]x_1^1 + [-0.11845487]x_1^2 + [-1.0502769]x_1^3
    #Drift term:  [0.] + [1.1094059]x_2^1 + [0.]x_2^2 + [-1.1302055]x_2^3
    #Diffusion term:  diag  [1.112649  1.0914397]
    #Xi term:  diag  [0.89262795 1.014515  ]
    #Maximum relative error:  0.13020551
    #Maximum index:  tensor(3)
    #'train' took 10.507943 s
    
    # sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.66, lhs_ratio=0.7
    # Drift term:  [-0.07412212] + [0.959492]x_1^1 + [-0.1245936]x_1^2 + [-1.0793287]x_1^3
    #Drift term:  [0.] + [1.1075042]x_2^1 + [0.]x_2^2 + [-1.1079117]x_2^3
    #Diffusion term:  diag  [1.1078929 1.111567 ]
    #Xi term:  diag  [0.89329886 0.9629712 ]
    #Maximum relative error:  0.11156702  #-0.1245936?
    #Maximum index:  tensor(5)
    #'train' took 9.577089 s
    
    ## sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.15, gauss_variance=0.66, lhs_ratio=0.7
    # 误差贼大
    
    # sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, gauss_variance=0.66, lhs_ratio=0.69
    # Drift term:  [-0.08049033] + [0.9393768]x_1^1 + [-0.12519318]x_1^2 + [-1.0691279]x_1^3
    #Drift term:  [0.] + [1.1080954]x_2^1 + [0.]x_2^2 + [-1.1143446]x_2^3
    #Diffusion term:  diag  [1.1002458 1.0979556]
    #Xi term:  diag  [0.9009149  0.98562676]
    #Maximum relative error:  0.1143446
    #Maximum index:  tensor(3)
    #'train' took 12.113438 s
    
    
    
    
    
    
    
    
    #为什么时间增加误差增大？？？
    
    #t = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).astype(np.float32) , initial N(0, 0.4)
    ####################################
    
    #from GenerateData_n_inequal_ import DataSet
    
    # mu的范围 [data.min*(2/3), data.max*(2/3)]
    
    ###################################
    
    ### sample =10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.1, gauss_variance=0.55, lhs_ratio=0.65
    # Drift term:  [0.] + [0.9676437]x_1^1 + [0.]x_1^2 + [-1.0207026]x_1^3
    #Drift term:  [-0.14671272] + [0.98668736]x_2^1 + [0.]x_2^2 + [-1.1318477]x_2^3
    #Diffusion term:  diag  [1.1977617  0.86429876]
    #Xi term:  diag  [0.8635545 1.1726943]
    #Maximum relative error:  0.19776165
    #Maximum index:  tensor(4)
    #'train' took 11.082053 s
    
    
    ### sample =10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.15, gauss_variance=0.55, lhs_ratio=0.6
    #Drift term:  [0.] + [0.96110505]x_1^1 + [0.]x_1^2 + [-1.0185347]x_1^3
    #Drift term:  [-0.19692892] + [0.95976007]x_2^1 + [0.]x_2^2 + [-1.1453832]x_2^3
    #Diffusion term:  diag  [1.1366186 0.7770305]
    #Xi term:  diag  [0.9384847 1.253293 ]
    #Maximum relative error:  0.25329304
    #Maximum index:  tensor(7)
    #'train' took 15.086055 s
    #改lhs还是会增加误差
    
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.15, gauss_variance=0.57, lhs_ratio=0.6
    # Drift term:  [0.] + [0.9722197]x_1^1 + [0.]x_1^2 + [-1.0235944]x_1^3
    #Drift term:  [-0.18345831] + [0.9892171]x_2^1 + [0.]x_2^2 + [-1.1529031]x_2^3
    #Diffusion term:  diag  [1.1792015  0.87299937]
    #Xi term:  diag  [0.89589113 1.1947986 ]
    #Maximum relative error:  0.19479859
    #Maximum index:  tensor(7)
    #'train' took 16.555500 s
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.15, gauss_variance=0.55, lhs_ratio=0.7
    # Drift term:  [0.] + [0.99690104]x_1^1 + [0.]x_1^2 + [-1.0176724]x_1^3
    #Drift term:  [0.] + [1.1242888]x_2^1 + [0.]x_2^2 + [-1.1606857]x_2^3
    #Diffusion term:  diag  [1.3317364 1.129832 ]
    #Xi term:  diag  [0.64710695 0.9481224 ]
    #Maximum relative error:  0.35289305
    #Maximum index:  tensor(6)
    #'train' took 17.915074 s
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.15, gauss_variance=0.55, lhs_ratio=0.75
    # Drift term:  [0.] + [1.0236412]x_1^1 + [0.]x_1^2 + [-1.0270369]x_1^3
    #Drift term:  [0.] + [1.0974127]x_2^1 + [0.]x_2^2 + [-1.1411872]x_2^3
    #Diffusion term:  diag  [1.2690526 1.2253997]
    #Xi term:  diag  [0.7259889 0.8437529]
    #Maximum relative error:  0.27401108
    #Maximum index:  tensor(6)
    #'train' took 16.247693 s
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.55, lhs_ratio=0.8
    # Drift term:  [0.] + [1.0477905]x_1^1 + [0.]x_1^2 + [-1.0386426]x_1^3
    #Drift term:  [0.] + [1.1015192]x_2^1 + [0.]x_2^2 + [-1.1537049]x_2^3
    #Diffusion term:  diag  [1.2551907 1.2770077]
    #Xi term:  diag  [0.74450743 0.80262274]
    #Maximum relative error:  0.2770077
    #Maximum index:  tensor(5)
    #'train' took 15.305257 s
    
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.55, lhs_ratio=0.8
    # Drift term:  [-0.05086284] + [1.0204582]x_1^1 + [-0.09074201]x_1^2 + [-1.1002847]x_1^3
    #Drift term:  [-0.0033519] + [1.093075]x_2^1 + [0.02231453]x_2^2 + [-1.1204814]x_2^3
    #Diffusion term:  diag  [1.2134708 1.16476  ]
    #Xi term:  diag  [0.8246634 0.912099 ]
    #Maximum relative error:  0.21347082
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.53, lhs_ratio=0.8
    #不收敛
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.55, lhs_ratio=0.7
    # Drift term:  [-0.07918018] + [0.93386734]x_1^1 + [-0.09998056]x_1^2 + [-1.0810698]x_1^3
    #Drift term:  [-0.0132711] + [1.0889314]x_2^1 + [0.02659745]x_2^2 + [-1.1173548]x_2^3
    #Diffusion term:  diag  [1.1928343 1.0229841]
    #Xi term:  diag  [0.86079496 1.0402884 ]
    #Maximum relative error:  0.19283426
    #Maximum index:  tensor(4)
    #'train' took 19.911226 s
    
    ### sample =10000, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.55, lhs_ratio=0.9
    # Drift term:  [-0.01820757] + [1.0737541]x_1^1 + [-0.04444551]x_1^2 + [-1.086421]x_1^3
    #Drift term:  [-0.00931217] + [1.1228732]x_2^1 + [0.02680913]x_2^2 + [-1.1364326]x_2^3
    #Diffusion term:  diag  [1.2707313 1.2393861]
    #Xi term:  diag  [0.73713326 0.83001184]
    #Maximum relative error:  0.27073133
    
    ### sample =10000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.55, lhs_ratio=0.8
    # Drift term:  [-0.05518915] + [1.0193186]x_1^1 + [-0.07663606]x_1^2 + [-1.0987645]x_1^3
    #Drift term:  [0.] + [1.0968524]x_2^1 + [0.]x_2^2 + [-1.1286937]x_2^3
    #Diffusion term:  diag  [1.2079074 1.1461842]
    #Xi term:  diag  [0.83187383 0.9179107 ]
    #Maximum relative error:  0.20790744
    #Maximum index:  tensor(4)
    #'train' took 34.294730 s
    
    ### sample =10000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.55, lhs_ratio=0.9
    # Drift term:  [0.] + [1.0668776]x_1^1 + [0.]x_1^2 + [-1.0706314]x_1^3
    #Drift term:  [0.] + [1.1432275]x_2^1 + [0.]x_2^2 + [-1.1592765]x_2^3
    #Diffusion term:  diag  [1.2752827 1.2820778]
    #Xi term:  diag  [0.73555535 0.76241326]
    #Maximum relative error:  0.2820778
    
    
    ### sample =10000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.5, lhs_ratio=0.9
    # 不收敛
   
   ### sample =10000, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0, gauss_variance=0.5, lhs_ratio=0.9
   # Drift term:  [-0.0370905] + [1.0586356]x_1^1 + [0.]x_1^2 + [-1.0760782]x_1^3
   #Drift term:  [0.] + [1.1147876]x_2^1 + [0.]x_2^2 + [-1.1485877]x_2^3
  # Diffusion term:  diag  [1.253689  1.2483158]
   #Xi term:  diag  [0.7680705 0.8031119]
   #Maximum relative error:  0.25368905
   #Maximum index:  tensor(4)
   #'train' took 36.002276 s
    
    
    
    
    
    
    
    