# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:30:28 2023

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('/home/v-liyaguo/Levy_wcr/gen_data'))
print(sys.path) 

from collections import OrderedDict
from gen_data.generateData_n_inequal import DataSet
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
        
        x_cpu = x.cpu()  # Move tensor x to CPU
        mu_cpu = self.mu.cpu()  # Move self.mu to CPU if it's a tensor
        sigma_cpu = self.sigma.cpu()  # Move self.sigma to CPU if it's a tensor
        # Now compute the arguments for the hypergeometric function
        a = (x_cpu.shape[2] + self.lap_alpha) / 2
        b = x_cpu.shape[2] / 2
        z = -torch.sum((x_cpu - mu_cpu)**2, dim=2) / (2 * sigma_cpu**2)
        z_np = z.cpu().numpy()
        frac_result = sp.hyp1f1(a, b, z_np)
        frac_result_tensor = torch.tensor(frac_result, device=x.device)
                
        # func = (1/(torch.sqrt(torch.tensor(2))*self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * \
        #         (1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))**x.shape[2])* \
                # sp.hyp1f1((x.shape[2] + self.lap_alpha)/2, x.shape[2]/2, - torch.sum((x-self.mu)**2, dim=2) / (2 * self.sigma**2)) 
        func = (1/(torch.sqrt(torch.tensor(2))*self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * \
                (1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))**x.shape[2])* frac_result_tensor
        return func   
    
    
    

    def LapGauss_VaryDim(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        
        mu_cpu = self.mu.cpu()  # Move self.mu to CPU
        sigma_cpu = self.sigma.cpu()  # Move self.sigma to CPU

        for k in range(x.shape[2]):
            x_k_cpu = x[:, :, k].cpu()  # Get x[:, :, k] and move it to CPU
            mu_k_cpu = mu_cpu[k]  # Get the k-th component of mu

            # Compute the arguments for the hypergeometric function
            a = (1 + self.lap_alpha) / 2
            b = 1 / 2
            z = -(x_k_cpu - mu_k_cpu) ** 2 / (2 * sigma_cpu ** 2)
            z_np = z.numpy()  # Convert to NumPy array
            frac_result = sp.hyp1f1(a, b, z_np)  # Compute the hypergeometric function
            frac_result_tensor = torch.tensor(frac_result, device=self.device)

            func_k = (1 / (np.sqrt(2) * sigma_cpu)) ** self.lap_alpha * sp.gamma((1 + self.lap_alpha) / 2) * 2 ** self.lap_alpha / sp.gamma(1 / 2) * \
                    1 / (sigma_cpu * torch.sqrt(2 * torch.tensor(np.pi))) * frac_result_tensor

            # Store the result in the func tensor
            func[:, :, k] = g0 * (sigma_cpu * torch.sqrt(2 * torch.tensor(np.pi))) * torch.exp(0.5 * (x[:, :, k] - mu_k_cpu) ** 2 / sigma_cpu ** 2) * func_k

        return func
    
    
    # def LapGauss_VaryDim(self,x, g0):
    #     func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
    #     for k in range(x.shape[2]):
            
    #         # the fractional derivative of the kth variable
    #         func_k = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (1 + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(1/2) * \
    #                 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*\
    #                     sp.hyp1f1((1 + self.lap_alpha)/2, 1/2, -(x[:, :, k]-self.mu[k])**2 / (2*self.sigma**2))  
        
    #         func[:,:,k] = g0 * (self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))* torch.exp(\
    #             0.5 * (x[:, :, k] - self.mu[k]) ** 2 / self.sigma ** 2) *func_k
           
    #     return func
    

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
    def __init__(self, t, data, alpha, xi_q, Xi_type, testFunc, device):
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
        self.xi_q = xi_q
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
    
    def cauchy_int(self): #alpha=3/2
        x = symbols('x')
        f = integrate((cos(x)-1)/(x**(5/2)), x)
        result = (f.subs(x, 10) - f.subs(x, math.pow(10,-6))).evalf()
        result = result * (3*(2**0.5))/(4*(math.pi**0.5))
        #print("cauchy_int", result)
        return result
    
    @utils.timing # decorator
    @torch.no_grad()
    def build_basis(self): # \Lambda matrix
        """build the basis list for the different time snapshot 
        """
        self.basis1_number = int(np.math.factorial(self.dimension+self.basis_order)/(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order)))                            
        self.basis2_number = int( self.dimension*(self.dimension+1)/2 ) + 1
             
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
        print("self.basis_number ", self.basis_number)
        
        #self.basis = torch.stack(basis1).to(self.device)
        # print("self.basis.shape", self.basis.shape)

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number # mu: db
        F_number = self.dimension if self.diffusion_independence else 1  #sigma  d^2 b There are only diagonal elements
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
                F = torch.mean(gauss2[:, :, ld, ld], dim=1)  
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
                    # print("E",E)  
                    
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
        #L = self.data.shape[0]
        #L2 = round(L/2) #round(2L/3)
        if self.gauss_samp_way == 'lhs':
            lb = torch.tensor([self.data[:, :, i].min()*(2/3) for i in range(self.dimension)]).to(self.device)
            ub = torch.tensor([self.data[:, :, i].max()*(2/3) for i in range(self.dimension)]).to(self.device)
            mu_list = lb + self.lhs_ratio * (ub - lb) * torch.tensor(lhs(self.dimension, samp_number), dtype=torch.float32).to(self.device)
        if self.gauss_samp_way == 'SDE':
            if samp_number <= self.bash_size:
                index = np.arange(self.bash_size)
                np.random.shuffle(index)
                mu_list = data[-1, index[0: samp_number], :]
            else:
                print("The number of samples shall not be less than the number of tracks!")
        # print("mu_list", mu_list)
        sigma_list = torch.ones(samp_number).to(self.device)*self.variance
        # print("sigma_list", sigma_list.shape)
        return mu_list, sigma_list

    def buildLinearSystem(self, samp_number):
        mu_list, sigma_list = self.sampleTestFunc(samp_number)
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
        else: 
            X = X0

        # Get the standard ridge esitmate
        if lam != 0: 
            w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y))[0]
        else: #w = np.linalg.lstsq(X,y)[0]
            X_inv = np.linalg.pinv(X)  
            w = np.dot(X_inv,y)
        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):
            # Figure out which items to cut out
            smallinds = np.where(abs(w) < tol)[0]
            # print("STRidge_j: ", j)
            # print("smallinds", smallinds)
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                break
            else: num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else:
                    break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0
            if lam != 0: 
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            else: 
                w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        # if biginds != []: 
        #     w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        if biginds.size > 0:  # 确保 biginds 不是空的
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]
        else:  # 如果 biginds 为空，则直接返回
            print("Warning: biginds is empty. Returning w as-is.")
            return w

        if normalize != 0: 
            return np.multiply(Mreg,w)
        else: 
            return w
    
    # @utils.timing
    def compile(self, basis_order, basis_xi_order, gauss_variance, xi_q, type, drift_term,diffusion_term, xi_term,
                drift_independence, diffusion_independence, xi_independence, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.t_number = len(self.t)
        self.basis_order = basis_order
        self.basis_xi_order = basis_xi_order
        self.variance = gauss_variance
        self.xi_q = xi_q
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
        #self.A = torch.where(self.A == 0, 0.0001, self.A) 
        self.A = self.A.to("cpu")
        self.b = self.b.to("cpu")
        AA = torch.mm(torch.t(self.A), self.A)
        Ab = torch.mm(torch.t(self.A), self.b)
        # print("A.max: ", self.A.max(), "b.max: ", self.b.max())
        # print("ATA.max: ", AA.max(), "ATb.max: ", Ab.max())

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
            if Xi_type == "cI": #2.9112
                self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: self.dimension*self.basis1_number+2*self.dimension] \
                    = (2.5 * (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension:]))**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number+ self.dimension: ].numpy())
            elif Xi_type == "Diag":
                self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: self.dimension*self.basis1_number+2*self.dimension] \
                    = (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension:])**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number+ self.dimension: ].numpy())
            
        else:
            print("Xi term: ", (2.5 * (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: ]))**(2/3).numpy())

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
    #t = np.array([0.1,0.3,0.5,0.7,0.9]).astype(np.float32)
    t = np.array([0.1, 0.3, 0.5, 0.7, 1.0]).astype(np.float32)
    t = torch.tensor(t)
    dim = 3
    xi_q = 0
    #Xi_type = "cI" 
    Xi_type = "Diag" 
    
    drift = torch.tensor([0, 1, 0, -1]).repeat(dim, 1)
    diffusion = torch.ones(dim)
    xi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float) #torch.ones(dim)
    sample, dim = 20000, dim 
    alpha =3/2
    dataset = DataSet(t, dt=dt, samples_num=sample, dim=dim, drift_term=drift, diffusion_term=diffusion,\
                      xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(mean=0., std=0.4, size=(sample, dim)),   # torch.normal(mean=0., std=0.1, size=(sample, dim)),
                      explosion_prevention=False)
    data = dataset.get_data(plot_hist=False)
    data = data * (1 + 0.*torch.rand(data.shape))
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(t, data, alpha, xi_q, Xi_type, testFunc, device)
    model.compile(basis_order=3, basis_xi_order=1, gauss_variance=0.48, xi_q = xi_q, type='LMM_3', \
                  drift_term=drift, diffusion_term=diffusion, xi_term = xi,
                  drift_independence=True, diffusion_independence=True, xi_independence=True, gauss_samp_way='lhs', lhs_ratio=0.6) 
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=150, Xi_type = Xi_type, lam=0.01, STRidge_threshold=0.05, only_hat_13=False)
    
    