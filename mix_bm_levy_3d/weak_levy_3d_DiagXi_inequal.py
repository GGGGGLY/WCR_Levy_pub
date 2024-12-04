# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:30:28 2023

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
        #self.A = torch.where(self.A == 0, 0.0001, self.A) ####################3
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
    #Xi_type = "cI" #对角元相同的对角阵  case1
    Xi_type = "Diag" # 对角元分别估计 case2
    
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
    model.compile(basis_order=3, basis_xi_order=1, gauss_variance=0.52, xi_q = xi_q, type='LMM_3', \
                  drift_term=drift, diffusion_term=diffusion, xi_term = xi,
                  drift_independence=True, diffusion_independence=True, xi_independence=True, gauss_samp_way='lhs', lhs_ratio=0.6) 
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=150, Xi_type = Xi_type, lam=0.01, STRidge_threshold=0.05, only_hat_13=False)
    
    #############################################################
    #
    #         lb = min(data)*（2/3）;  ub = max(data)*(2/3)
    #
    ##############################################################
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0, lhs_ratio=0.63
    #Drift term:  [-0.12290064] + [1.0490242]x_1^1 + [-0.09396362]x_1^2 + [-1.1572473]x_1^3
    #Drift term:  [-0.0135221] + [1.0048532]x_2^1 + [-0.0479507]x_2^2 + [-1.0663042]x_2^3
    #Drift term:  [-0.0948139] + [1.0660797]x_3^1 + [0.00623799]x_3^2 + [-1.0971451]x_3^3
    #Diffusion term:  diag  [1.3012275  0.98420143 1.0455792 ]
    #Xi term:  diag  [0.75241303 1.0714219  1.0392944 ]
    #Maximum relative error:  0.30122745
    #Maximum index:  tensor(6)
    #'train' took 17.417178 s
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.65
    # Drift term:  [-0.11067076] + [1.0650665]x_1^1 + [-0.08603198]x_1^2 + [-1.1586328]x_1^3
    # Drift term:  [0.] + [1.0173696]x_2^1 + [0.]x_2^2 + [-1.0302768]x_2^3
    #Drift term:  [-0.11069273] + [1.0624058]x_3^1 + [0.]x_3^2 + [-1.1046572]x_3^3
    #Diffusion term:  diag  [1.3118597 1.113848  1.076723 ]
    #Xi term:  diag  [0.7415709 0.9426534 1.0136127]
    #Maximum relative error:  0.31185973
    #Maximum index:  tensor(6)
    #'train' took 16.041503 s
    
    #sample = 20000, gauss_variance=0.6, gauss_samp_number=150, lam=0.01, STRidge_threshold=0.05, lhs_ratio=0.6
    #Drift term:  [0.] + [1.1622864]x_1^1 + [0.]x_1^2 + [-1.1319834]x_1^3
    #Drift term:  [0.] + [1.0548855]x_2^1 + [0.]x_2^2 + [-1.0506729]x_2^3
    #Drift term:  [0.] + [1.020789]x_3^1 + [0.]x_3^2 + [-0.9841913]x_3^3
   # Diffusion term:  diag  [1.3405434 1.1228681 1.3100096]
    #Xi term:  diag  [0.6408081  0.88951373 0.6374542 ]
    #Maximum relative error:  0.3625458
    #Maximum index:  tensor(11)
    #'train' took 29.206952 s
    
    
    
    
    
    
    
    
    
    
    ############没有L2
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.01,
    # lhs_ratio=0.7
    #Drift term:  [-0.04667174] + [1.1651403]x_1^1 + [-0.04856925]x_1^2 + [-1.2028114]x_1^3
    #Drift term:  [-0.04274896] + [1.0113282]x_2^1 + [-0.03974266]x_2^2 + [-1.06103]x_2^3
    #Drift term:  [-0.08241579] + [1.1324509]x_3^1 + [-0.04862981]x_3^2 + [-1.1359037]x_3^3
    #Diffusion term:  diag  [1.1983372 1.2521968 1.1024728]
    #Xi term:  diag  [0.91711676 0.7756138  0.9283289 ]
    #Maximum relative error:  0.2521968
    #Maximum index:  tensor(7)
    #'train' took 18.107153 s
    
    #sample = 15000, gauss_variance=0.62, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.01,
    # lhs_ratio=0.66
    # 不收敛
    
    #sample = 15000, gauss_variance=0.64, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.01,
    # lhs_ratio=0.66
    # 不收敛
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.01,
    # lhs_ratio=0.66
    # Drift term:  [-0.07415561] + [1.1022241]x_1^1 + [-0.0852901]x_1^2 + [-1.1903477]x_1^3
    #Drift term:  [0.] + [1.0589346]x_2^1 + [0.]x_2^2 + [-1.092308]x_2^3
    #Drift term:  [-0.12130165] + [1.1045544]x_3^1 + [-0.06041889]x_3^2 + [-1.1339586]x_3^3
    #Diffusion term:  diag  [1.2053503 1.1785918 1.0737344]
    #Xi term:  diag  [0.9133294  0.9224684  0.94972324]
    #Maximum relative error:  0.20535028
    #Maximum index:  tensor(6)
    #'train' took 17.868062 s
    
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.005,
    # lhs_ratio=0.66
    #Drift term:  [-0.07296313] + [1.1025078]x_1^1 + [-0.08354153]x_1^2 + [-1.1885928]x_1^3
    #Drift term:  [-0.01427944] + [1.0490959]x_2^1 + [-0.00754692]x_2^2 + [-1.0949533]x_2^3
    #Drift term:  [-0.11674138] + [1.107467]x_3^1 + [-0.05720404]x_3^2 + [-1.133536]x_3^3
    #Diffusion term:  diag  [1.1999384 1.158283  1.0729331]
    #Xi term:  diag  [0.9191495  0.94045603 0.95415103]
    #Maximum relative error:  0.19993842
    #Maximum index:  tensor(6)
    #'train' took 18.218004 s
    
    #---------------------------------------------------------------------------------
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.005,
    # lhs_ratio=0.63
    #Drift term:  [-0.09888585] + [1.0438708]x_1^1 + [-0.10302562]x_1^2 + [-1.1634395]x_1^3
    #Drift term:  [0.02784298] + [1.1062193]x_2^1 + [-0.02549192]x_2^2 + [-1.1371629]x_2^3
    #Drift term:  [-0.09554765] + [1.1281836]x_3^1 + [-0.03595298]x_3^2 + [-1.1211078]x_3^3
    #Diffusion term:  diag  [1.1990484 1.133478  1.151991 ]
    #Xi term:  diag  [0.9023957  0.9814625  0.88445973]
    #Maximum relative error:  0.1990484
    #Maximum index:  tensor(6)
    #'train' took 17.104169 s
    
    #________________________________________________________________________________
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.005,
    # lhs_ratio=0.6
    # 误差0.24变大
    
    #sample = 15000, gauss_variance=0.63, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.005,
    # lhs_ratio=0.6
    # 不收敛
    
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    # 同上
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=130, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    # Drift term:  [-0.00825405] + [1.1063527]x_1^1 + [-0.06342747]x_1^2 + [-1.1338086]x_1^3
    #Drift term:  [0.02240479] + [1.1089762]x_2^1 + [0.01828244]x_2^2 + [-1.0760231]x_2^3
    #Drift term:  [-0.18986602] + [0.9864787]x_3^1 + [-0.06836836]x_3^2 + [-1.1163422]x_3^3
    #Diffusion term:  diag  [1.2952299 1.2079029 1.1418253]
    #Xi term:  diag  [0.7773142  0.85779303 0.9134093 ]
    #Maximum relative error:  0.2952299
    #Maximum index:  tensor(6)
    #'train' took 18.675811 s
    
    #sample = 15000, gauss_variance=0.65, gauss_samp_number=170, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    #Drift term:  [0.00245688] + [1.0919468]x_1^1 + [-0.07743729]x_1^2 + [-1.131832]x_1^3
    #Drift term:  [0.02908665] + [1.1334572]x_2^1 + [0.00641087]x_2^2 + [-1.0932932]x_2^3
    #Drift term:  [-0.14663249] + [1.0159096]x_3^1 + [-0.07807838]x_3^2 + [-1.0293212]x_3^3
    #Diffusion term:  diag  [1.2135986 1.186981  1.1804622]
    #Xi term:  diag  [0.84554416 0.86385334 0.741364  ]
    #Maximum relative error:  0.258636
    #Maximum index:  tensor(11)
    #'train' took 29.947539 s
    
    #sample = 15000, gauss_variance=0.62, gauss_samp_number=170, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    # 不收敛
    
    #sample = 15000, gauss_variance=0.635, gauss_samp_number=170, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    #不收敛
    
    #sample = 15000, gauss_variance=0.64, gauss_samp_number=170, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    #不收敛
    
    #sample = 15000, gauss_variance=0.64, gauss_samp_number=140, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    #不收敛
    
    #sample = 16000, gauss_variance=0.64, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    #不收敛
    
    #sample = 16000, gauss_variance=0.65, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.0,
    # lhs_ratio=0.66
    # 不收敛...
    
    
    
    

    
    
# sample = 20000, gauss_variance=0.7, gauss_samp_number=500, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.65
# Drift term:  [-0.02738558] + [1.1730068]x_1^1 + [-0.01909689]x_1^2 + [-1.147875]x_1^3
#Drift term:  [0.00411133] + [1.109257]x_2^1 + [-0.00531719]x_2^2 + [-1.0919734]x_2^3
#Drift term:  [0.00345768] + [1.0796399]x_3^1 + [-0.08689588]x_3^2 + [-1.07089]x_3^3
#Diffusion term:  diag  [1.2771256 1.1987263 1.2171433]
#Xi term:  diag  [1.5917132 1.7177963 1.5653797]
#Maximum relative error:  0.7177963
#Maximum index:  tensor(10)
#'train' took 91.812426 s

# sample = 20000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.65
# Drift term:  [-0.01435946] + [1.1653777]x_1^1 + [-0.00835906]x_1^2 + [-1.1496866]x_1^3
#Drift term:  [-0.02236399] + [1.0959526]x_2^1 + [-0.03710198]x_2^2 + [-1.0968041]x_2^3
#Drift term:  [0.0392829] + [1.1122428]x_3^1 + [-0.05415969]x_3^2 + [-1.0603265]x_3^3
#Diffusion term:  diag  [1.310606  1.2305566 1.2872903]
#Xi term:  diag  [1.5445777 1.6458929 1.4009038]
#Maximum relative error:  0.64589286
#Maximum index:  tensor(10)
#'train' took 64.747387 s

# sample = 20000, gauss_variance=0.85, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.65
# 误差很大1.0

# sample = 20000, gauss_variance=0.75, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.6
# 误差0.8

#加了gauss，误差变大
## sample = 20000, gauss_variance=0.75, gauss_samp_number=400, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.6
# error = 0.94

# 还是300个gauss吧，调一下lhs
## sample = 20000, gauss_variance=0.75, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
#lhs_ratio=0.7
# 误差0.78

## sample = 20000, gauss_variance=0.75, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
#lhs_ratio=0.85
# Drift term:  [-0.00750764] + [1.171856]x_1^1 + [0.02101536]x_1^2 + [-1.1257731]x_1^3
#Drift term:  [0.03563853] + [1.0966969]x_2^1 + [-8.7630186e-05]x_2^2 + [-1.0516529]x_2^3
#Drift term:  [0.02723403] + [1.1089652]x_3^1 + [-0.02530076]x_3^2 + [-1.0730951]x_3^3
#Diffusion term:  diag  [1.4015026 1.3888807 1.4291594]
#Xi term:  diag  [1.4095837 1.2854443 1.1921312]
#Maximum relative error:  0.4291594
#Maximum index:  tensor(8)
#'train' took 50.759411 s


## sample = 20000, gauss_variance=0.75, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
#lhs_ratio=0.95
#Drift term:  [-0.00612523] + [1.1411691]x_1^1 + [0.0064662]x_1^2 + [-1.1020498]x_1^3
#Drift term:  [0.03123475] + [1.1243774]x_2^1 + [0.00958313]x_2^2 + [-1.0649078]x_2^3
#Drift term:  [0.00972319] + [1.133302]x_3^1 + [-0.00658965]x_3^2 + [-1.0939608]x_3^3
#Diffusion term:  diag  [1.4106821 1.3905663 1.3956727]
#Xi term:  diag  [1.3462144 1.2778554 1.3482935]
#Maximum relative error:  0.41068208
#Maximum index:  tensor(6)
#'train' took 45.100278 s

## sample = 20000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.005,
#lhs_ratio=0.95
#Drift term:  [-0.00664729] + [1.1478103]x_1^1 + [0.00512436]x_1^2 + [-1.1053642]x_1^3
#Drift term:  [0.03431674] + [1.1267499]x_2^1 + [0.00877541]x_2^2 + [-1.0624124]x_2^3
#Drift term:  [0.00618274] + [1.1466409]x_3^1 + [-0.00516451]x_3^2 + [-1.1000612]x_3^3
#Diffusion term:  diag  [1.3819691 1.3599696 1.3799435]
#Xi term:  diag  [0.61291075 0.58112794 0.5957689 ]
#Maximum relative error:  0.41887206
#Maximum index:  tensor(10)
#'train' took 56.391734 s

## sample = 20000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.95
# Drift term:  [0.] + [1.1483006]x_1^1 + [0.]x_1^2 + [-1.1062567]x_1^3
#Drift term:  [0.0349783] + [1.1256341]x_2^1 + [0.]x_2^2 + [-1.0598338]x_2^3
#Drift term:  [0.] + [1.1523286]x_3^1 + [0.]x_3^2 + [-1.103087]x_3^3
#Diffusion term:  diag  [1.3827564 1.3625777 1.380043 ]
#Xi term:  diag  [0.6129077 0.5768588 0.5958516]
#Maximum relative error:  0.42314118
#Maximum index:  tensor(10)
#'train' took 103.041761 s

## sample = 20000, gauss_variance=0.66, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.9
# 不收敛

## sample = 15000, gauss_variance=0.66, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.9
# Drift term:  [0.04002222] + [1.194994]x_1^1 + [-0.04492139]x_1^2 + [-1.1906079]x_1^3
#Drift term:  [0.02610987] + [1.1080941]x_2^1 + [0.]x_2^2 + [-1.0729823]x_2^3
#Drift term:  [-0.04332176] + [1.148737]x_3^1 + [0.]x_3^2 + [-1.071854]x_3^3
#Diffusion term:  diag  [1.4492993 1.4049407 1.3825033]
#Xi term:  diag  [0.616602   0.5375197  0.58043736]
#Maximum relative error:  0.4624803
#Maximum index:  tensor(10)
#'train' took 24.675154 s

## sample = 15000, gauss_variance=0.66, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.75
#Drift term:  [0.] + [1.1912994]x_1^1 + [-0.01867066]x_1^2 + [-1.158471]x_1^3
#Drift term:  [0.05181782] + [1.1098309]x_2^1 + [0.]x_2^2 + [-1.0413187]x_2^3
#Drift term:  [-0.0108116] + [1.1664716]x_3^1 + [-0.06043624]x_3^2 + [-1.1160158]x_3^3
#Diffusion term:  diag  [1.3573061 1.3797841 1.3317211]
#Xi term:  diag  [0.6765563  0.56243867 0.6212924 ]
#Maximum relative error:  0.43756133
#Maximum index:  tensor(10)
#'train' took 21.331657 s

## sample = 15000, gauss_variance=0.6, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.75
#bushoulian


## sample = 20000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.85
# Drift term:  [0.] + [1.1759064]x_1^1 + [0.01973064]x_1^2 + [-1.1275102]x_1^3
#Drift term:  [0.03835744] + [1.0955154]x_2^1 + [0.]x_2^2 + [-1.0488313]x_2^3
#Drift term:  [0.02683825] + [1.1137437]x_3^1 + [-0.02648697]x_3^2 + [-1.076491]x_3^3
#Diffusion term:  diag  [1.3706872 1.368245  1.428773 ]
#Xi term:  diag  [0.6450853  0.57352436 0.51181656]
#Maximum relative error:  0.48818344
#Maximum index:  tensor(11)
#'train' took 63.261130 s


## sample = 30000, gauss_variance=0.7, gauss_samp_number=500, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.85
#Drift term:  [0.] + [1.1291924]x_1^1 + [0.]x_1^2 + [-1.1092639]x_1^3
#Drift term:  [0.] + [1.0724841]x_2^1 + [-0.01270107]x_2^2 + [-1.0706688]x_2^3
#Drift term:  [0.] + [1.1230974]x_3^1 + [-0.03136531]x_3^2 + [-1.0896473]x_3^3
#Diffusion term:  diag  [1.4013411 1.4045388 1.3631223]
#Xi term:  diag  [0.5909488  0.60063285 0.60606813]
#Maximum relative error:  0.40905118
#Maximum index:  tensor(9)
#'train' took 246.714709 s

## sample = 30000, gauss_variance=0.7, gauss_samp_number=400, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.85
#Drift term:  [0.] + [1.1225196]x_1^1 + [0.01279195]x_1^2 + [-1.1217335]x_1^3
#Drift term:  [0.0157262] + [1.0813578]x_2^1 + [0.]x_2^2 + [-1.0483204]x_2^3
#Drift term:  [0.] + [1.1149884]x_3^1 + [-0.03667769]x_3^2 + [-1.0720714]x_3^3
#Diffusion term:  diag  [1.4403576 1.3404663 1.3518212]
#Xi term:  diag  [0.5653369  0.64803416 0.6086431 ]
#Maximum relative error:  0.44035757
#Maximum index:  tensor(6)
#'train' took 176.906948 s


## sample = 30000, gauss_variance=0.68, gauss_samp_number=500, lam=0.0, STRidge_threshold=0.01,
#lhs_ratio=0.85
#Drift term:  [0.] + [1.129305]x_1^1 + [0.]x_1^2 + [-1.1089095]x_1^3
#Drift term:  [0.] + [1.0734537]x_2^1 + [-0.01218013]x_2^2 + [-1.0720363]x_2^3
#Drift term:  [0.] + [1.1228194]x_3^1 + [-0.03173284]x_3^2 + [-1.0892713]x_3^3
#Diffusion term:  diag  [1.3929126 1.4007598 1.3568339]
#Xi term:  diag  [0.60055614 0.6056294  0.61168176]
#Maximum relative error:  0.40075982
#Maximum index:  tensor(7)
#'train' took 139.342790 s





## sample = 30000, gauss_variance=0.75, gauss_samp_number=400, lam=0.0, STRidge_threshold=0.005,
#lhs_ratio=0.95
# Drift term:  [0.00374095] + [1.0711255]x_1^1 + [0.0126834]x_1^2 + [-1.0952034]x_1^3
#Drift term:  [-0.00257051] + [1.0673621]x_2^1 + [-0.02017114]x_2^2 + [-1.0445986]x_2^3
#Drift term:  [-0.0163595] + [1.1497493]x_3^1 + [-0.01530516]x_3^2 + [-1.075079]x_3^3
#Diffusion term:  diag  [1.4347689 1.3526837 1.3692037]
#Xi term:  diag  [1.365511  1.4727091 1.3008865]
#Maximum relative error:  0.47270906
#Maximum index:  tensor(10)
#'train' took 138.371342 s

## sample = 30000, gauss_variance=0.75, gauss_samp_number=400, lam=0.0, STRidge_threshold=0.05,
#lhs_ratio=1.0
# Drift term:  [0.0079364] + [1.0710933]x_1^1 + [0.0071697]x_1^2 + [-1.095881]x_1^3
#Drift term:  [0.] + [1.0636781]x_2^1 + [-0.01956402]x_2^2 + [-1.0296887]x_2^3
#Drift term:  [-0.01517125] + [1.1215907]x_3^1 + [-0.01073506]x_3^2 + [-1.07115]x_3^3
#Diffusion term:  diag  [1.4905522 1.3235744 1.4042314]
#Xi term:  diag  [1.1957473 1.5098274 1.2457111]
#Maximum relative error:  0.5098274
#Maximum index:  tensor(10)
#'train' took 123.706061 s


## sample = 30000, gauss_variance=0.75, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.05,
#lhs_ratio=1.0
#Drift term:  [0.] + [1.091803]x_1^1 + [0.02197594]x_1^2 + [-1.0920459]x_1^3
#Drift term:  [0.00620903] + [1.0578042]x_2^1 + [-0.02243189]x_2^2 + [-1.0551909]x_2^3
#Drift term:  [0.02839795] + [1.0774043]x_3^1 + [-0.03920317]x_3^2 + [-1.0918856]x_3^3
#Diffusion term:  diag  [1.423458  1.4543138 1.3940984]
#Xi term:  diag  [0.59087986 0.52461    0.64450413]
#Maximum relative error:  0.47539002
#Maximum index:  tensor(10)
#'train' took 70.137107 s

## sample = 30000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.05,
#lhs_ratio=1.0
# rift term:  [0.] + [1.0991313]x_1^1 + [0.02136128]x_1^2 + [-1.0992113]x_1^3
#Drift term:  [0.00933231] + [1.0656627]x_2^1 + [-0.02179191]x_2^2 + [-1.0640126]x_2^3
#Drift term:  [0.03275481] + [1.0793098]x_3^1 + [-0.04269766]x_3^2 + [-1.097557]x_3^3
#Diffusion term:  diag  [1.4035811 1.4406474 1.3633547]
#Xi term:  diag  [0.6202697 0.544515  0.6886799]
#Maximum relative error:  0.455485
#Maximum index:  tensor(10)
#'train' took 87.634535 s

## sample = 30000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.05,
#lhs_ratio=0.85
# Drift term:  [0.] + [1.1138825]x_1^1 + [0.02801118]x_1^2 + [-1.1113338]x_1^3
#Drift term:  [0.01085704] + [1.083227]x_2^1 + [-0.0320203]x_2^2 + [-1.0398291]x_2^3
#Drift term:  [0.00983249] + [1.1219922]x_3^1 + [-0.04261061]x_3^2 + [-1.0737274]x_3^3
#Diffusion term:  diag  [1.4601953 1.3658674 1.379986 ]
#Xi term:  diag  [0.5436424  0.59063935 0.57787883]
#Maximum relative error:  0.4601953
#Maximum index:  tensor(6)
#'train' took 70.256072 s




## sample = 15000, gauss_variance=0.65, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.55
# Drift term:  [-0.15393746] + [0.9293317]x_1^1 + [-0.07403987]x_1^2 + [-1.0636867]x_1^3
#Drift term:  [0.00161039] + [1.0099119]x_2^1 + [0.01770337]x_2^2 + [-1.0066937]x_2^3
#Drift term:  [0.03754193] + [1.1776719]x_3^1 + [0.03282848]x_3^2 + [-1.1196017]x_3^3
#Diffusion term:  diag  [1.1518366 1.2683858 1.308365 ]
#Xi term:  diag  [1.6479354 1.4217664 1.5484989]
#Maximum relative error:  0.6479354
#Maximum index:  tensor(9)
#'train' took 43.399286 s

## sample = 15000, gauss_variance=0.75, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.0,
# lhs_ratio=0.55
# 误差太大 0.9




