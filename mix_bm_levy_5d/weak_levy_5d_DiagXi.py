# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:57:51 2023

Weak Levy: 5d; uniform snapshots

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from GenerateData_n import DataSet
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
        
        func = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (x.shape[2] + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(x.shape[2]/2) * \
            (1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))**x.shape[2])* sp.hyp1f1((x.shape[2] + self.lap_alpha)/2, x.shape[2]/2, -torch.sum((x-self.mu)**2, dim=2) / (2*self.sigma**2)) 
        #print("LapGauss.shape", func.shape) #11,10000
        return func    
    
    def LapGauss_VaryDim(self,x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]]).to(self.device)
        for k in range(x.shape[2]):
            # 对第k个变量求分数阶导数
            func_k = (1/(np.sqrt(2)*self.sigma)) ** self.lap_alpha * sp.gamma( (1 + self.lap_alpha)/2 )* 2**self.lap_alpha / sp.gamma(1/2) * \
                    1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))*\
                        sp.hyp1f1((1 + self.lap_alpha)/2, 1/2, -(x[:, :, k]-self.mu[k])**2 / (2*self.sigma**2))  
        
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
        self.basis2_number = int( self.dimension)
             
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
            X = self._get_data_t(it) # N*d
            Xi = torch.ones(X.size(0),1) # N*1
           
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
                    #print("E",E)  # (1) 10^{-3}阶
                    
                    A[:, H_number+F_number+ld] = E
            elif self.Xi_type == "Diag":
                for kd in range(self.dimension):
                    E = -torch.mean(gauss_LapDiag[:, :, kd], dim=1)
                    A[:, H_number+F_number+kd] = E
                    #print("E",E) #10^{-5}
                        
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
        #L2 = round(L/2) #round(2L/3)  #之前没有2想刨除0时刻的随机分布，从运动一小段时间之后算起
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
        #print("b_list", b_list)  #A, b都是10^{-3}阶啊
       
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
    def compile(self, basis_order, basis_xi_order, gauss_variance, type, drift_term,diffusion_term, xi_term,
                drift_independence, diffusion_independence, xi_independence, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.t_number = len(self.t)
        self.basis_order = basis_order
        self.basis_xi_order = basis_xi_order
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
        #self.A = torch.where(self.A == 0, 0.0001, self.A) ####################
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
            if Xi_type == "cI": #18.2662
                self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: self.dimension*self.basis1_number+2*self.dimension] \
                    = (2.24405 * (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension:]))**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number+ self.dimension: ].numpy())
            elif Xi_type == "Diag":
                self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: self.dimension*self.basis1_number+2*self.dimension] \
                    = (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension:])**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number+ self.dimension: ].numpy())
            
        else:
            print("Xi term: ", (self.zeta.squeeze()[self.dimension * self.basis1_number + self.dimension: ])**(2/3).numpy())

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

    T, dt, true_dt = 1, 0.0001, 0.1
    t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    t = torch.tensor(t)
    dim = 5
    #Xi_type = "cI" #对角元相同的对角阵
    Xi_type = "Diag" # 对角元分别估计
    
    drift = torch.tensor([0, 1, 0, -1]).repeat(dim, 1)
    diffusion = torch.ones(dim)
    #xi = torch.tensor([1.5, 1.0, 1.2, 1.0, 1.0], dtype=torch.float) #torch.ones(dim)
    xi = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float) 
    sample = 30000
    alpha =3/2
    dataset = DataSet(t, true_dt=true_dt, samples_num=sample, dim=dim, drift_term=drift, diffusion_term=diffusion,\
                      xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(mean=0., std=0.2, size=(sample, 1)),   # torch.normal(mean=0., std=0.1, size=(sample, dim)),
                      drift_independence=True, explosion_prevention=False, trajectory_information=True)
    data = dataset.get_data(plot_hist=False)
    data = data * (1 + 0.*torch.rand(data.shape))
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    #model = Model(torch.linspace(0, T, int(T/true_dt) + 1), data, alpha, xi_q, testFunc, device)
    model = Model(torch.linspace(0,1,11), data, alpha, Xi_type, testFunc, device)
    model.compile(basis_order=3, basis_xi_order=1, gauss_variance=0.4, type='LMM_3', \
                  drift_term=drift, diffusion_term=diffusion, xi_term = xi, \
                  drift_independence=True, diffusion_independence=True, xi_independence=True, gauss_samp_way='lhs', lhs_ratio=1) 
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=1500, Xi_type = Xi_type, lam=0.0, STRidge_threshold=0.2, only_hat_13=False)
    
    
 ####这个setting下学出来的xi的系数都是0，sigma系数比较大   

###############################################

# Xi_type = "Diag" ##没有L2

# mu [ self.data[:, :, i].min()*(2/3), self.data[:, :, i].max()*(2/3) ]

############################################3

# sample = 50000, gauss_variance=0.7, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005, lhs_ratio=0.58
# Drift term:  [-0.10522743] + [0.9391944]x_1^1 + [-0.0868182]x_1^2 + [-1.109688]x_1^3
#Drift term:  [-0.05224833] + [1.1072574]x_2^1 + [-0.06089751]x_2^2 + [-1.1276115]x_2^3
#Drift term:  [-0.05338878] + [0.99114674]x_3^1 + [0.06122211]x_3^2 + [-0.9952341]x_3^3
#Drift term:  [-0.08506441] + [1.0510772]x_4^1 + [-0.08816133]x_4^2 + [-1.1343815]x_4^3
#Drift term:  [-0.04183708] + [1.148861]x_5^1 + [-0.06352343]x_5^2 + [-1.163426]x_5^3
#Diffusion term:  diag  [1.1303669 1.1879646 1.202348  1.169863  1.1604825]
#Xi term:  diag  [0.94936657 0.8550758  0.85938656 0.8926147  0.878902  ]
#Maximum relative error:  0.202348
#Maximum index:  tensor(12)
#'train' took 1267.623997 s

#-------------------------------
# sample = 50000, gauss_variance=0.65, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005, lhs_ratio=0.58
# Drift term:  [-0.12000132] + [0.93020815]x_1^1 + [-0.10496778]x_1^2 + [-1.1212871]x_1^3
#Drift term:  [-0.07088655] + [1.0873672]x_2^1 + [-0.06265919]x_2^2 + [-1.1261857]x_2^3
#Drift term:  [-0.05240659] + [0.999507]x_3^1 + [0.06522602]x_3^2 + [-0.9960823]x_3^3
#Drift term:  [-0.09751202] + [1.041346]x_4^1 + [-0.10804728]x_4^2 + [-1.1485909]x_4^3
#Drift term:  [-0.04412302] + [1.1565177]x_5^1 + [-0.08664627]x_5^2 + [-1.1861836]x_5^3
#Diffusion term:  diag  [1.0527982 1.1448588 1.1572613 1.1243781 1.1126705]
#Xi term:  diag  [1.0197511  0.9033003  0.90616125 0.94195074 0.9244608 ]
#Maximum relative error:  0.18618357
#Maximum index:  tensor(9)
#'train' took 1207.528970 s
#------------------------------

# sample = 50000, gauss_variance=0.6, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005, lhs_ratio=0.58
# 不收敛

# sample = 50000, gauss_variance=0.62, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005, lhs_ratio=0.58
# Drift term:  [-0.13060176] + [0.9232894]x_1^1 + [-0.11940845]x_1^2 + [-1.129778]x_1^3
#Drift term:  [-0.08639574] + [1.0711994]x_2^1 + [-0.06556904]x_2^2 + [-1.1263069]x_2^3
#Drift term:  [-0.05117377] + [1.0080707]x_3^1 + [0.0675813]x_3^2 + [-0.9975999]x_3^3
#Drift term:  [-0.10966805] + [1.029111]x_4^1 + [-0.1245553]x_4^2 + [-1.1591828]x_4^3
#Drift term:  [-0.04615127] + [1.1604944]x_5^1 + [-0.10270628]x_5^2 + [-1.2024096]x_5^3
#Diffusion term:  diag  [0.99183035 1.1115353  1.1236103  1.0842406  1.079418  ]
#Xi term:  diag  [1.0704695  0.939658   0.93896693 0.98365885 0.9558892 ]
#Maximum relative error:  0.20240963
#Maximum index:  tensor(9)
#'train' took 918.554184 s

# sample = 50000, gauss_variance=0.62, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005, lhs_ratio=0.65
# Drift term:  [-0.05409105] + [0.98727655]x_1^1 + [-0.09002106]x_1^2 + [-1.1328347]x_1^3
#Drift term:  [-0.04461824] + [1.099483]x_2^1 + [-0.07645468]x_2^2 + [-1.1295179]x_2^3
#Drift term:  [-0.02148827] + [1.0696963]x_3^1 + [0.01057533]x_3^2 + [-1.0693011]x_3^3
#Drift term:  [-0.03493555] + [1.0819607]x_4^1 + [-0.03099309]x_4^2 + [-1.0925632]x_4^3
#Drift term:  [-0.03749421] + [1.1124529]x_5^1 + [-0.02562609]x_5^2 + [-1.1292237]x_5^3
#Diffusion term:  diag  [1.050625  1.2450291 1.1808105 1.1810634 1.201731 ]
#Xi term:  diag  [1.0406514  0.76547176 0.88186824 0.8650167  0.81106603]
#Maximum relative error:  0.24502909
#Maximum index:  tensor(11)
#'train' took 1112.962478 s

# sample = 50000, gauss_variance=0.6, gauss_samp_number=600, lam=0.01, STRidge_threshold=0.005, lhs_ratio=0.65
# 不收敛

# sample = 50000, gauss_variance=0.6, gauss_samp_number=700, lam=0.01, STRidge_threshold=0.005, lhs_ratio=0.65
# 不收敛














###############################################

# Xi_type = "Diag" #####没有L2

############################################3

# sample = 30000, gauss_variance=0.85, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.8  ##gauss_var<= 0.75 不收敛
#Drift term:  [0.01857012] + [1.1495193]x_1^1 + [-0.0069842]x_1^2 + [-1.1203772]x_1^3
#Drift term:  [0.00486915] + [1.1356069]x_2^1 + [-0.00390138]x_2^2 + [-1.135156]x_2^3
#Drift term:  [-0.05630079] + [1.1460364]x_3^1 + [0.06281389]x_3^2 + [-1.1229614]x_3^3
#Drift term:  [0.03695966] + [1.1712569]x_4^1 + [0.00033836]x_4^2 + [-1.0952399]x_4^3
#Drift term:  [0.01715054] + [1.0196391]x_5^1 + [-0.01040696]x_5^2 + [-1.0498663]x_5^3
#Diffusion term:  diag  [1.1113195 1.4897544 1.2835311 1.4517463 1.6301196]
#Xi term:  diag  [0.908154   0.5465875  0.750116   0.54666585 0.03774279]
#Maximum index:  tensor(19)
#'train' took 725.748288 s



# sample = 50000, gauss_variance=0.78, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.7
#         [ 0.4656]])
#Drift term:  [0.00546262] + [1.0646152]x_1^1 + [-0.04934685]x_1^2 + [-1.1211108]x_1^3
#Drift term:  [-0.00982816] + [1.0864588]x_2^1 + [0.]x_2^2 + [-1.0696344]x_2^3
#Drift term:  [-0.04084948] + [1.0577393]x_3^1 + [-0.01727113]x_3^2 + [-1.0829139]x_3^3
#Drift term:  [-0.02932361] + [1.123395]x_4^1 + [-0.10450761]x_4^2 + [-1.1662672]x_4^3
#Drift term:  [-0.02809282] + [1.1573993]x_5^1 + [0.]x_5^2 + [-1.1278033]x_5^3
#Diffusion term:  diag  [1.3021731 1.4557285 1.275527  1.33376   1.3924932]
#Xi term:  diag  [0.7729423  0.525295   0.7904277  0.6896216  0.60069174]
#Maximum relative error:  0.47470498
#Maximum index:  tensor(16)
#'train' took 1078.218696 s

# sample = 30000, gauss_variance=0.78, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.05,
# lhs_ratio=0.62
# Drift term:  [-0.04985496] + [1.0040197]x_1^1 + [-0.10845833]x_1^2 + [-1.1516942]x_1^3
#Drift term:  [-0.03913917] + [1.1216147]x_2^1 + [-0.07318242]x_2^2 + [-1.1318047]x_2^3
#Drift term:  [-0.01987777] + [1.0068867]x_3^1 + [0.0443616]x_3^2 + [-1.009319]x_3^3
#Drift term:  [-0.05446376] + [0.9877712]x_4^1 + [0.01679846]x_4^2 + [-1.0275956]x_4^3
#Drift term:  [-0.01506427] + [1.1605462]x_5^1 + [0.]x_5^2 + [-1.1112621]x_5^3
#Diffusion term:  diag  [1.2977976 1.2750328 1.3688618 1.38753   1.3375255]
#Xi term:  diag  [0.7829874  0.74822944 0.67939365 0.6353241  0.67570466]
#Maximum relative error:  0.38752997
#Maximum index:  tensor(13)
#'train' took 1219.600111 s



# sample = 30000, gauss_variance=0.75, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.05,
# lhs_ratio=0.58
#Drift term:  [-0.08805128] + [0.94190866]x_1^1 + [-0.09705963]x_1^2 + [-1.1229966]x_1^3
#Drift term:  [-0.05125373] + [1.0855328]x_2^1 + [-0.04418882]x_2^2 + [-1.118837]x_2^3
#Drift term:  [-0.06663483] + [0.93080324]x_3^1 + [0.14447844]x_3^2 + [-0.92162466]x_3^3
#Drift term:  [-0.07098735] + [1.0345199]x_4^1 + [0.]x_4^2 + [-1.0894195]x_4^3
#Drift term:  [-0.01246378] + [1.1771368]x_5^1 + [-0.02124437]x_5^2 + [-1.1342826]x_5^3
#Diffusion term:  diag  [1.2645581 1.279685  1.3307589 1.310898  1.2750235]
#Xi term:  diag  [0.8111163  0.77044666 0.71057636 0.7547512  0.75818616]
#Maximum relative error:  0.33075893
#Maximum index:  tensor(12)
#'train' took 1125.610290 s


#GAUSS = 0.7 不收敛



































#####有L2, L2=L/3
# sample = 30000, gauss_variance=0.85, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.7
#Drift term:  [0.00099251] + [1.1400251]x_1^1 + [-0.07831785]x_1^2 + [-1.1282876]x_1^3
#Drift term:  [-0.0117675] + [1.172067]x_2^1 + [0.01213324]x_2^2 + [-1.1588596]x_2^3
#Drift term:  [-0.0595053] + [1.1456655]x_3^1 + [0.0494774]x_3^2 + [-1.1112776]x_3^3
#Drift term:  [0.02679144] + [1.1062448]x_4^1 + [0.01295401]x_4^2 + [-1.0734105]x_4^3
#Drift term:  [0.01700902] + [1.0781854]x_5^1 + [-0.02455888]x_5^2 + [-1.0841749]x_5^3
#Diffusion term:  diag  [1.156717  1.3958372 1.2986585 1.4314698 1.4045273]
#Xi term:  diag  [0.8288008  0.6787965  0.7338234  0.55842817 0.571013  ]
#Maximum relative error:  0.44157183
#Maximum index:  tensor(18)
#'train' took 654.920283 s

# sample = 30000, gauss_variance=0.82, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.65
#Drift term:  [0.01875896] + [1.1535642]x_1^1 + [-0.08635899]x_1^2 + [-1.1449159]x_1^3
#Drift term:  [0.00010587] + [1.1821451]x_2^1 + [-0.05970506]x_2^2 + [-1.2071327]x_2^3
#Drift term:  [-0.0595767] + [1.1333762]x_3^1 + [-0.04622347]x_3^2 + [-1.1626909]x_3^3
#Drift term:  [0.01629847] + [1.1047906]x_4^1 + [0.09812666]x_4^2 + [-1.0268546]x_4^3
#Drift term:  [0.02745276] + [1.0704622]x_5^1 + [-0.01004535]x_5^2 + [-1.0565063]x_5^3
#Diffusion term:  diag  [1.1985788 1.3197912 1.2954688 1.398072  1.4099655]
#Xi term:  diag  [0.804874  0.7765655 0.7286739 0.5967949 0.5711273]
#Maximum relative error:  0.4288727
#Maximum index:  tensor(19)
#'train' took 567.017826 s


# sample = 30000, gauss_variance=0.78, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.65
# Drift term:  [0.01816864] + [1.1545432]x_1^1 + [-0.09056512]x_1^2 + [-1.1500374]x_1^3
#Drift term:  [-0.00099305] + [1.1853695]x_2^1 + [-0.05695412]x_2^2 + [-1.2095387]x_2^3
#Drift term:  [-0.0615264] + [1.1321282]x_3^1 + [-0.04764912]x_3^2 + [-1.1704012]x_3^3
#Drift term:  [0.01985807] + [1.1178036]x_4^1 + [0.10811721]x_4^2 + [-1.0335269]x_4^3
#Drift term:  [0.03193806] + [1.0758471]x_5^1 + [-0.00612636]x_5^2 + [-1.0586519]x_5^3
#Diffusion term:  diag  [1.1890258 1.3169751 1.2826093 1.3808686 1.3712453]
#Xi term:  diag  [0.81301606 0.78091496 0.7449313  0.62049013 0.6202478 ]
#Maximum relative error:  0.38086855
#Maximum index:  tensor(13)
#'train' took 787.350846 s



################################

### L2 = L/2

################################

# sample = 30000, gauss_variance=0.78, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.6
# Drift term:  [0.01199022] + [1.1425645]x_1^1 + [-0.05135273]x_1^2 + [-1.111898]x_1^3
#Drift term:  [-0.02183562] + [1.0675989]x_2^1 + [-0.09559099]x_2^2 + [-1.1597282]x_2^3
#Drift term:  [-0.12884346] + [0.99206686]x_3^1 + [-0.06137985]x_3^2 + [-1.1146784]x_3^3
#Drift term:  [0.02744084] + [1.1609843]x_4^1 + [0.04387367]x_4^2 + [-1.0707432]x_4^3
#Drift term:  [0.0055328] + [1.1233121]x_5^1 + [-0.02833347]x_5^2 + [-1.0954908]x_5^3
#Diffusion term:  diag  [1.1479911 1.2632809 1.2258217 1.2480074 1.3115087]
#Xi term:  diag  [0.8512425  0.837362   0.8099782  0.76054907 0.6905474 ]
#Maximum relative error:  0.31150866
#Maximum index:  tensor(14)
#'train' took 745.426677 s

# sample = 30000, gauss_variance=0.78, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.63
#Drift term:  [0.02861719] + [1.1627783]x_1^1 + [-0.07306976]x_1^2 + [-1.1416607]x_1^3
#Drift term:  [0.] + [1.1490773]x_2^1 + [-0.08996817]x_2^2 + [-1.2033623]x_2^3
#Drift term:  [-0.08292124] + [1.0724577]x_3^1 + [-0.08587252]x_3^2 + [-1.1642444]x_3^3
#Drift term:  [0.03165898] + [1.1596427]x_4^1 + [0.1046755]x_4^2 + [-1.0473994]x_4^3
#Drift term:  [0.02934205] + [1.101941]x_5^1 + [0.]x_5^2 + [-1.0685825]x_5^3
#Diffusion term:  diag  [1.1861613 1.2921802 1.2674279 1.3463556 1.3640919]
#Xi term:  diag  [0.8192265  0.8048523  0.75761765 0.65750253 0.62592244]
#Maximum relative error:  0.37407756
#Maximum index:  tensor(19)
#'train' took 706.018368 s

#————————————————————————————————————————————————————————————————————————————————————————————————————

# sample = 30000, gauss_variance=0.78, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.57
# Drift term:  [-0.04762833] + [1.07066]x_1^1 + [-0.01598833]x_1^2 + [-1.0617436]x_1^3
#Drift term:  [-0.07352924] + [1.0128148]x_2^1 + [-0.06148456]x_2^2 + [-1.1215664]x_2^3
#Drift term:  [-0.13441047] + [1.018663]x_3^1 + [-0.01825509]x_3^2 + [-1.1091115]x_3^3
#Drift term:  [-0.02791567] + [1.0843539]x_4^1 + [0.]x_4^2 + [-1.0678625]x_4^3
#Drift term:  [-0.01579063] + [1.1265659]x_5^1 + [-0.0552995]x_5^2 + [-1.1167085]x_5^3
#Diffusion term:  diag  [1.1253893 1.2263231 1.1901327 1.1773779 1.2466853]
#Xi term:  diag  [0.87559634 0.8842366  0.87548965 0.83573323 0.7779583 ]
#Maximum relative error:  0.24668527
#Maximum index:  tensor(14)
#'train' took 748.286187 s
#——————————————————————————————————————————————————————————————————————————————————————————————






# sample = 30000, gauss_variance=0.78, gauss_samp_number=500, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.6
# Drift term:  [-0.01241457] + [1.0186304]x_1^1 + [-0.1031161]x_1^2 + [-1.0584757]x_1^3
#Drift term:  [-0.05310422] + [1.0908058]x_2^1 + [-0.10612909]x_2^2 + [-1.1912246]x_2^3
#Drift term:  [-0.09867477] + [0.99682224]x_3^1 + [0.01631058]x_3^2 + [-1.0578892]x_3^3
#Drift term:  [-0.09583614] + [0.98085445]x_4^1 + [0.09553336]x_4^2 + [-0.9526217]x_4^3
#Drift term:  [-0.00852945] + [1.0493684]x_5^1 + [0.05697521]x_5^2 + [-1.0436536]x_5^3
#Diffusion term:  diag  [1.1912454 1.1593925 1.3064692 1.2682303 1.3548304]
#Xi term:  diag  [0.8144107  0.9587131  0.7262947  0.73533326 0.69679505]
#Maximum relative error:  0.35483038
#Maximum index:  tensor(14)
#'train' took 609.391548 s

#减小gauss_num误差变大，所以增加gauss_num

# sample = 30000, gauss_variance=0.78, gauss_samp_number=700, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.6
#Drift term:  [-0.07660235] + [0.99649745]x_1^1 + [-0.28659332]x_1^2 + [-1.1887631]x_1^3
#Drift term:  [-0.03969583] + [1.1577135]x_2^1 + [-0.11508608]x_2^2 + [-1.2256732]x_2^3
#Drift term:  [-0.10120013] + [1.0018437]x_3^1 + [0.01799484]x_3^2 + [-1.0337269]x_3^3
#Drift term:  [0.00700624] + [1.0893533]x_4^1 + [0.21010564]x_4^2 + [-0.95674586]x_4^3
#Drift term:  [0.] + [1.175464]x_5^1 + [-0.03769615]x_5^2 + [-1.1066157]x_5^3
#Diffusion term:  diag  [1.1710706 1.218364  1.3536528 1.2673925 1.2033131]
#Xi term:  diag  [0.827089   0.8881029  0.6758733  0.77828103 0.8024105 ]
#Maximum relative error:  0.35365283
#Maximum index:  tensor(12)
#'train' took 826.816296 s

#为了增大xi,要减小gauss_var, 为了避免不收敛，尝试更换位置，即lhs_ratio

# sample = 30000, gauss_variance=0.77, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.55
# Drift term:  [-0.32905215] + [0.63468283]x_1^1 + [-0.47167313]x_1^2 + [-1.1934196]x_1^3
#Drift term:  [-0.10180234] + [0.93762136]x_2^1 + [0.01088819]x_2^2 + [-1.0528033]x_2^3
#Drift term:  [-0.05832498] + [1.0469297]x_3^1 + [0.13248584]x_3^2 + [-1.01908]x_3^3
#Drift term:  [0.13177222] + [1.2974532]x_4^1 + [0.2628546]x_4^2 + [-0.9600463]x_4^3
#Drift term:  [0.11604698] + [1.35002]x_5^1 + [-0.08494141]x_5^2 + [-1.1711361]x_5^3
#Diffusion term:  diag  [1.207007  1.3528498 1.3769186 1.2331333 1.1352651]
#Xi term:  diag  [0.7732387  0.7503901  0.7235788  0.77356046 0.8506987 ]
#Maximum relative error:  0.37691855
#Maximum index:  tensor(12)
#'train' took 889.096902 s

#那还是增大lhs_ratio吧
# sample = 30000, gauss_variance=0.77, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.005,
# lhs_ratio=0.7
# Drift term:  [0.] + [1.152638]x_1^1 + [-0.122918]x_1^2 + [-1.1810641]x_1^3
#Drift term:  [-0.01383696] + [1.1667458]x_2^1 + [-0.02397324]x_2^2 + [-1.1635222]x_2^3
#Drift term:  [-0.06135058] + [1.0459423]x_3^1 + [0.01052635]x_3^2 + [-1.0715897]x_3^3
#Drift term:  [-0.0090132] + [1.0782841]x_4^1 + [0.0108895]x_4^2 + [-1.074568]x_4^3
#Drift term:  [-0.02070069] + [1.0793037]x_5^1 + [-0.00667235]x_5^2 + [-1.0895547]x_5^3
#Diffusion term:  diag  [1.1993656 1.2762328 1.4405808 1.293619  1.4317164]
#Xi term:  diag  [0.83692175 0.8075922  0.55602986 0.73330015 0.5864697 ]
#Maximum relative error:  0.44397014
#Maximum index:  tensor(17)
#'train' took 755.173283 s

#还是变大了...







##################################(1) Xi_type = "cI", L2 = round(L/2)


# sample = 30000, gauss_variance=0.55, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.05,
# lhs_ratio=0.7, L2 = L/2 (L/4也不收敛)
#Drift term:  [-0.06063394] + [0.98676854]x_1^1 + [-0.05567907]x_1^2 + [-1.0456811]x_1^3
#Drift term:  [0.] + [1.302529]x_2^1 + [0.]x_2^2 + [-1.2783366]x_2^3
#Drift term:  [0.] + [1.1485043]x_3^1 + [0.05698742]x_3^2 + [-1.1369673]x_3^3
#Drift term:  [-0.06594657] + [1.1558071]x_4^1 + [0.05722119]x_4^2 + [-1.1049668]x_4^3
#Drift term:  [0.] + [0.99039865]x_5^1 + [0.]x_5^2 + [-1.0533185]x_5^3
#Diffusion term:  diag  [1.1524591 1.3315146 1.2491883 1.188635  1.2110304]
#Xi term:  diag  [0.9849569 0.9849569 0.9849569 0.9849569 0.9849569]
#Maximum relative error:  0.3315146
#Maximum index:  tensor(11)
#'train' took 445.389432 s


# sample = 30000, gauss_variance=0.54, gauss_samp_number=600, lam=0.0, STRidge_threshold=0.05,
# lhs_ratio=0.7, L2 = L/2 (L/4也不收敛)
#Drift term:  [-0.06379609] + [0.97692317]x_1^1 + [-0.05159175]x_1^2 + [-1.0445251]x_1^3
#Drift term:  [0.] + [1.3106256]x_2^1 + [0.]x_2^2 + [-1.2852563]x_2^3
#Drift term:  [0.] + [1.1454114]x_3^1 + [0.05692328]x_3^2 + [-1.1404593]x_3^3
#Drift term:  [-0.07644022] + [1.1552083]x_4^1 + [0.06351483]x_4^2 + [-1.1065539]x_4^3
#Drift term:  [0.] + [0.9840061]x_5^1 + [0.]x_5^2 + [-1.0520673]x_5^3
#Diffusion term:  diag  [1.1485993 1.3251182 1.242389  1.1763186 1.2018912]
#Xi term:  diag  [1.0476298 1.0476298 1.0476298 1.0476298 1.0476298]
#Maximum relative error:  0.32511818
#Maximum index:  tensor(11)
#'train' took 450.331679 s








