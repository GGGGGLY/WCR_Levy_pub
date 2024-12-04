# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:58:45 2023

Only levy noise

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from GenerateData_n_inequal_OnlyLevy import DataSet
from pyDOE import lhs
import time
import utils
import scipy.io
import scipy.special as sp
import math
from sympy import symbols, integrate, cos


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
                    1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))* sp.hyp1f1((1 + self.lap_alpha)/2, 1/2, -(x[:, :, k]-self.mu[k])**2 / (2*self.sigma**2))  
        
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
            #basis_count2 = 0
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
        C_number = self.dimension if self.xi_independence else 1#* self.basis2_number #* self.basis2_number  #d^2 c

        A = torch.zeros([self.t_number, H_number+C_number]).to(self.device)

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
        

        # print("self.basis_number", self.basis_number)
        # print("self.dimension", self.dimension)
        for kd in range(self.dimension):
            for jb in range(4):#range(self.basis1_number): order+1
                if self.drift_independence:
                    H = torch.mean(gauss1[:, :, kd] * self.data[:, :, kd] ** jb, dim=1)
                else:                    
                    H = torch.mean(gauss1[:, :, kd] * self.basis_theta[:, :, jb], dim=1)
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
                    A[:, H_number+kd] = E
                        
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
        #L = self.data.shape[0]
        #L2 = round(L/3) #round(2L/3)
        if self.gauss_samp_way == 'lhs':
            lb = torch.tensor([self.data[:, :, i].min()*0.9 for i in range(self.dimension)]).to(self.device)
            ub = torch.tensor([self.data[:, :, i].max()*0.9 for i in range(self.dimension)]).to(self.device)
            mu_list = lb + self.lhs_ratio * (ub - lb) * torch.tensor(lhs(self.dimension, samp_number), dtype=torch.float32).to(self.device)
        if self.gauss_samp_way == 'SDE':
            if samp_number <= self.bash_size:
                index = np.arange(self.bash_size)
                np.random.shuffle(index)
                mu_list = data[-1, index[0: samp_number], :]
            else:
                print("The number of samples shall not be less than the number of tracks!")
        #print("mu_list", mu_list)
        sigma_list = torch.ones(samp_number).to(self.device)*self.variance
        #print("sigma_list", sigma_list.shape)
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
        print("A_list", A_list)
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
    def compile(self, basis_order, basis_xi_order, gauss_variance, type, drift_term, xi_term,
                drift_independence,xi_independence, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.t_number = len(self.t)
        self.basis_order = basis_order
        self.basis_xi_order = basis_xi_order
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
                
            
        if self.xi_independence:
            if Xi_type == "cI": #2.9112
                self.zeta.squeeze()[self.dimension * self.basis1_number : self.dimension*self.basis1_number+self.dimension] \
                    = (2.5 * (self.zeta.squeeze()[self.dimension * self.basis1_number:]))**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number: ].numpy())
            elif Xi_type == "Diag":
                self.zeta.squeeze()[self.dimension * self.basis1_number : self.dimension*self.basis1_number+self.dimension] \
                    = (self.zeta.squeeze()[self.dimension * self.basis1_number :])**(2/3)
                print("Xi term: ", "diag ", self.zeta.squeeze()[self.dimension*self.basis1_number: ].numpy())
            
        else:
            print("Xi term: ", (2.5 * (self.zeta.squeeze()[self.dimension * self.basis1_number: ]))**(2/3).numpy())

        true = torch.cat((self.drift.view(-1), self.xi)) 
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
    #Xi_type = "cI" #对角元相同的对角阵  case1
    Xi_type = "Diag" # 对角元分别估计 case2
    
    drift = torch.tensor([0, 1.0, 0.0, -1.5]).repeat(dim, 1)
    xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) #torch.ones(dim)
    sample, dim = 20000, dim 
    alpha =3/2
    ## torch.normal(mean=0., std=0.1, size=(sample, dim)),
    dataset = DataSet(t, dt=dt, samples_num=sample, dim=dim, drift_term=drift, \
                      xi_term = xi, alpha_levy = 3/2, initialization=torch.normal(mean=0., std=0.4, size=(sample, dim)), explosion_prevention=False)
    data = dataset.get_data(plot_hist=False)
    data = data * (1 + 0.*torch.rand(data.shape))
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(t, data, alpha, Xi_type, testFunc, device)
    model.compile(basis_order=3, basis_xi_order=1, gauss_variance=0.72, type='LMM_3', \
                  drift_term=drift, xi_term = xi, drift_independence=True, xi_independence=True, gauss_samp_way='lhs', lhs_ratio=0.48) 
    # For 1 dimension, "drift_independence" and "diffusion_independence"  make no difference
    model.train(gauss_samp_number=400, Xi_type = Xi_type, lam=0.0, STRidge_threshold=0.2, only_hat_13=False)
    
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.72, gauss_samp_number=150, lam=0.01, STRidge_threshold=0.1, lhs_ratio=0.6
    # 不收敛
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.72, gauss_samp_number=200, lam=0.01, STRidge_threshold=0.05, lhs_ratio=0.7
    # 不收敛
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.01, STRidge_threshold=0.05, lhs_ratio=0.7
    #Drift term:  [0.] + [1.5489409]x_1^1 + [0.]x_1^2 + [-2.2377303]x_1^3
    #Drift term:  [0.] + [1.1540221]x_2^1 + [0.]x_2^2 + [-1.731031]x_2^3
    #Drift term:  [0.] + [0.64351034]x_3^1 + [0.]x_3^2 + [0.]x_3^3
    #Xi term:  diag  [1.034159  1.5670196       nan]  #第三个xi是负的
    #Maximum relative error:  nan
    #Maximum index:  tensor(8)
    #'train' took 39.573880 s
    
    #还是上面的setting, 只不过把lb, ub都乘以0.8了；
    #Drift term:  [0.] + [1.4780195]x_1^1 + [0.]x_1^2 + [-2.1891706]x_1^3
    #Drift term:  [0.] + [1.1519613]x_2^1 + [0.]x_2^2 + [-1.7662269]x_2^3
    #Drift term:  [0.] + [0.60028696]x_3^1 + [0.]x_3^2 + [0.]x_3^3
    #Xi term:  diag  [1.0507483 1.5903636       nan]
    #Maximum relative error:  nan
    #Maximum index:  tensor(8)
    #'train' took 33.276966 s
    
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.0, lhs_ratio=0.7
    # Drift term:  [-0.02445211] + [1.3140308]x_1^1 + [-0.08210882]x_1^2 + [-1.8819877]x_1^3
    #Drift term:  [0.00154196] + [1.0605398]x_2^1 + [-0.00321328]x_2^2 + [-1.6337235]x_2^3
    #Drift term:  [0.02675578] + [1.9548036]x_3^1 + [-0.43213126]x_3^2 + [-2.680483]x_3^3
    #Xi term:  diag  [0.98630047 1.5815028  0.19029593]
    #Maximum relative error:  0.9548036
    #Maximum index:  tensor(4)
    #'train' took 23.189481 s
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.0, lhs_ratio=0.5
    #         [ 0.2657]])
    #Drift term:  [-0.15262087] + [1.0928934]x_1^1 + [-0.25627932]x_1^2 + [-1.9402273]x_1^3
    #Drift term:  [-0.05241412] + [0.9932214]x_2^1 + [-0.04419562]x_2^2 + [-1.6918318]x_2^3
    #Drift term:  [0.00442473] + [1.039233]x_3^1 + [0.15291508]x_3^2 + [-1.1984918]x_3^3
    #Xi term:  diag  [1.0204054  1.6289517  0.41331258]
    #Maximum relative error:  0.29348484
    #Maximum index:  tensor(1)
    #'train' took 24.633920 s
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.0, lhs_ratio=0.4
    #Drift term:  [-0.03489234] + [1.167134]x_1^1 + [-0.2998851]x_1^2 + [-2.0834603]x_1^3
    #Drift term:  [-0.28158244] + [0.8164271]x_2^1 + [-0.11372298]x_2^2 + [-1.7215849]x_2^3
    #Drift term:  [0.07888909] + [1.2300751]x_3^1 + [0.32070687]x_3^2 + [-1.3140583]x_3^3
    #Xi term:  diag  [1.0894699  1.6249205  0.43815228]
    #Maximum relative error:  0.38897356
    #Maximum index:  tensor(1)
    #'train' took 26.038531 s
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.5
    #Drift term:  [-0.1786263] + [1.0599055]x_1^1 + [-0.2690534]x_1^2 + [-1.9239374]x_1^3
    #Drift term:  [0.] + [1.031212]x_2^1 + [0.]x_2^2 + [-1.6824934]x_2^3
    #Drift term:  [0.] + [1.0526124]x_3^1 + [0.0609502]x_3^2 + [-1.2802414]x_3^3
    #Xi term:  diag  [1.0117072  1.6442373  0.40188354]
    #Maximum relative error:  0.28262496
    #Maximum index:  tensor(1)
    #'train' took 30.028000 s
    
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 2, 1.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.5
    #不收敛
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -2]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 2, 1.5], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.5
    #不收敛
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.0, lhs_ratio=0.6
    #Drift term:  [-0.11255225] + [1.2240549]x_1^1 + [-0.30055106]x_1^2 + [-1.9920113]x_1^3
    #Drift term:  [-0.04449852] + [1.0465646]x_2^1 + [-0.04501295]x_2^2 + [-1.6667421]x_2^3
    #Drift term:  [-0.03234225] + [1.1207352]x_3^1 + [0.03220876]x_3^2 + [-1.5395833]x_3^3
    #Xi term:  diag  [0.9766761 1.5789251 0.9471344]
    #Maximum relative error:  0.32800755
    #Maximum index:  tensor(1)
    #'train' took 23.223940 s
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.4, lhs_ratio=0.6
    #Drift term:  [0.] + [1.4456904]x_1^1 + [0.]x_1^2 + [-2.0154786]x_1^3
    #Drift term:  [0.] + [1.1066314]x_2^1 + [0.]x_2^2 + [-1.6748486]x_2^3
    #Drift term:  [0.] + [1.2387185]x_3^1 + [0.]x_3^2 + [-1.6857988]x_3^3
    #Xi term:  diag  [0.96117055 1.5271941  0.8939928 ]
    #Maximum relative error:  0.4456904
    #Maximum index:  tensor(0)
    #'train' took 27.729759 s
    
    #lb, ub都乘以0.8
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.4, lhs_ratio=0.5
    #Drift term:  [0.] + [1.3576263]x_1^1 + [0.]x_1^2 + [-1.9328728]x_1^3
    #Drift term:  [0.] + [1.1179574]x_2^1 + [0.]x_2^2 + [-1.7209647]x_2^3
    #Drift term:  [0.] + [1.2066848]x_3^1 + [0.]x_3^2 + [-1.6073099]x_3^3
    #Xi term:  diag  [0.97298086 1.5571716  0.8393159 ]
    #Maximum relative error:  0.35762632
    #Maximum index:  tensor(0)
    #'train' took 49.012474 s
    
    #lb, ub都乘以0.7
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.4, lhs_ratio=0.5
    #Drift term:  [0.] + [1.4207077]x_1^1 + [0.]x_1^2 + [-1.9982996]x_1^3
    #Drift term:  [0.] + [1.112249]x_2^1 + [0.]x_2^2 + [-1.7146262]x_2^3
    #Drift term:  [0.] + [1.2344879]x_3^1 + [0.]x_3^2 + [-1.6266906]x_3^3
    #Xi term:  diag  [0.96593463 1.5512481  0.8464049 ]
    #Maximum relative error:  0.4207077
    #Maximum index:  tensor(0)
    #'train' took 23.612160 s
    
    #lb, ub都乘以1.0
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.4, lhs_ratio=0.5
    #不收敛
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.4, lhs_ratio=0.5
    # Drift term:  [0.] + [1.2969662]x_1^1 + [0.]x_1^2 + [-1.8791288]x_1^3
    #Drift term:  [0.] + [1.1063596]x_2^1 + [0.]x_2^2 + [-1.7072107]x_2^3
    #Drift term:  [0.] + [1.2292025]x_3^1 + [0.]x_3^2 + [-1.6379051]x_3^3
    #Xi term:  diag  [0.97816116 1.5610842  0.83365244]
    #Maximum relative error:  0.2969662
    #Maximum index:  tensor(0)
    #'train' took 35.751869 s
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 30000, gauss_variance=0.73, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.35, lhs_ratio=0.5
    #不收敛
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.72, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.35, lhs_ratio=0.5
    # Drift term:  [0.] + [1.2848004]x_1^1 + [0.]x_1^2 + [-1.8690301]x_1^3
    #Drift term:  [0.] + [1.1015021]x_2^1 + [0.]x_2^2 + [-1.7049786]x_2^3
    #Drift term:  [0.] + [1.2239257]x_3^1 + [0.]x_3^2 + [-1.6387614]x_3^3
    #Xi term:  diag  [0.97944486 1.5629221  0.83721524]
    #Maximum relative error:  0.2848004
    #Maximum index:  tensor(0)
    #'train' took 58.000751 s
    
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.72, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.3, lhs_ratio=0.5
    # Drift term:  [0.] + [1.2848004]x_1^1 + [0.]x_1^2 + [-1.8690301]x_1^3
    #Drift term:  [0.] + [1.1015021]x_2^1 + [0.]x_2^2 + [-1.7049786]x_2^3
    #Drift term:  [0.] + [1.2239257]x_3^1 + [0.]x_3^2 + [-1.6387614]x_3^3
    #Xi term:  diag  [0.97944486 1.5629221  0.83721524]
    #Maximum relative error:  0.2848004
    #Maximum index:  tensor(0)
    #'train' took 58.000751 s
   
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.72, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.3, lhs_ratio=0.5
    #Drift term:  [0.] + [1.3165654]x_1^1 + [0.]x_1^2 + [-1.9295572]x_1^3
    #Drift term:  [0.] + [1.0675888]x_2^1 + [0.]x_2^2 + [-1.6511608]x_2^3
    #Drift term:  [0.] + [1.2247883]x_3^1 + [0.]x_3^2 + [-1.6177632]x_3^3
    #Xi term:  diag  [0.97568613 1.5291185  0.8663632 ]
    #Maximum relative error:  0.3165654
    #Maximum index:  tensor(0)
    #'train' took 49.761553 s
    
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.7, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.3, lhs_ratio=0.48
    #不收敛
    
    
    #lb, ub都乘以0.9
   #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
   #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
   #sample = 20000, gauss_variance=0.72, gauss_samp_number=300, lam=0.0, STRidge_threshold=0.25, lhs_ratio=0.48
   # Drift term:  [0.] + [1.297338]x_1^1 + [0.]x_1^2 + [-1.9032178]x_1^3
   #Drift term:  [0.] + [1.0958261]x_2^1 + [0.]x_2^2 + [-1.6612945]x_2^3
   #Drift term:  [0.] + [1.2186593]x_3^1 + [0.]x_3^2 + [-1.5908823]x_3^3
   #Xi term:  diag  [0.9759674  1.5230446  0.85304075]
   #Maximum relative error:  0.297338
  # Maximum index:  tensor(0)
   #'train' took 52.377368 s
   
   #lb, ub都乘以0.9
  #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
  #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
  #sample = 20000, gauss_variance=0.7, gauss_samp_number=350, lam=0.0, STRidge_threshold=0.2, lhs_ratio=0.48
  #不收敛
  
  
  #----------------------------------------------------------------------------
  #lb, ub都乘以0.9
 #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
 #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
 #sample = 20000, gauss_variance=0.72, gauss_samp_number=350, lam=0.0, STRidge_threshold=0.2, lhs_ratio=0.48
 # Drift term:  [0.] + [1.2025664]x_1^1 + [0.]x_1^2 + [-1.8322487]x_1^3
 #Drift term:  [0.] + [1.2003728]x_2^1 + [0.]x_2^2 + [-1.7621499]x_2^3
 #Drift term:  [0.] + [1.1729821]x_3^1 + [0.]x_3^2 + [-1.5038378]x_3^3
 #Xi term:  diag  [1.0076364  1.5254347  0.81978256]
 #Maximum relative error:  0.22149913
 #Maximum index:  tensor(1)
 #'train' took 64.260721 s
    
 #--------------------------------------------------------------------------------
 
    #lb, ub都乘以0.9
   #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
   #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
   #sample = 20000, gauss_variance=0.7, gauss_samp_number=400, lam=0.0, STRidge_threshold=0.2, lhs_ratio=0.48
   # 不收敛
   
   #lb, ub都乘以0.9
  #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
  #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
  #sample = 20000, gauss_variance=0.72, gauss_samp_number=400, lam=0.0, STRidge_threshold=0.2, lhs_ratio=0.48
  # Drift term:  [0.] + [1.3820693]x_1^1 + [-0.7254786]x_1^2 + [-2.42861]x_1^3
  #Drift term:  [0.] + [1.0486931]x_2^1 + [0.]x_2^2 + [-1.6445024]x_2^3
  #Drift term:  [0.] + [1.1973693]x_3^1 + [0.]x_3^2 + [-1.5682102]x_3^3
  #Xi term:  diag  [1.0130888 1.5957894 0.9027308]
  #Maximum relative error:  0.6190734
  #Maximum index:  tensor(1)
  #'train' took 60.863481 s
    
    
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.7, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.3, lhs_ratio=0.5
    # 不收敛
    
    #lb, ub都乘以0.9
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.7, gauss_samp_number=150, lam=0.0, STRidge_threshold=0.3, lhs_ratio=0.5
    #不收敛
    
    
    #############################
    #
    # 有L2, L2 = L/3， ub, lb不乘系数
    #
    #############################
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.5
    # Drift term:  [-0.07488542] + [1.1530454]x_1^1 + [-0.10988828]x_1^2 + [-1.873642]x_1^3
    #Drift term:  [0.] + [1.0541714]x_2^1 + [0.]x_2^2 + [-1.6959413]x_2^3
    #Drift term:  [-0.15353404] + [1.052651]x_3^1 + [-0.13014254]x_3^2 + [-1.6626136]x_3^3
    #Xi term:  diag  [1.0516931 1.6418839 0.9254014]
    #Maximum relative error:  0.24909465
    #Maximum index:  tensor(1)
    #'train' took 26.180262 s
    
    ##gauss_var换成0.74就不收敛了
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.4
    #不行，误差很大
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=200, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.55
    #Drift term:  [-0.08199472] + [1.2098564]x_1^1 + [-0.17796282]x_1^2 + [-1.9160746]x_1^3
    #Drift term:  [0.] + [1.0587955]x_2^1 + [0.]x_2^2 + [-1.6797913]x_2^3
    #Drift term:  [-0.1248812] + [1.0683051]x_3^1 + [-0.11092284]x_3^2 + [-1.6130414]x_3^3
    #Xi term:  diag  [1.0123508 1.6299075 0.9077773]
    #Maximum relative error:  0.2773831
    #Maximum index:  tensor(1)
    #'train' took 25.220184 s
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=180, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.55
    #误差变大 （减少了20个gauss samples）
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=220, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.55
    # Drift term:  [0.] + [1.2226145]x_1^1 + [0.16406079]x_1^2 + [-1.7530477]x_1^3
    #Drift term:  [0.] + [1.0796115]x_2^1 + [-0.07026378]x_2^2 + [-1.7178311]x_2^3
   # Drift term:  [-0.23415405] + [0.91381913]x_3^1 + [-0.27925503]x_3^2 + [-1.6651473]x_3^3
    #Xi term:  diag  [1.0870355 1.5979935 0.910296 ]
    #Maximum relative error:  0.22261453
    #Maximum index:  tensor(0)
    #'train' took 28.973886 s
    # 增加20个gauss, 误差减小
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.75, gauss_samp_number=240, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.55
    # Drift term:  [0.] + [1.2945472]x_1^1 + [0.]x_1^2 + [-1.918002]x_1^3
    #Drift term:  [0.] + [1.019573]x_2^1 + [0.]x_2^2 + [-1.6925603]x_2^3
    #Drift term:  [-0.24137197] + [0.89006364]x_3^1 + [-0.17610298]x_3^2 + [-1.5412933]x_3^3
   # Xi term:  diag  [1.0790051 1.6557271 0.8821426]
    #Maximum relative error:  0.2945472
    #Maximum index:  tensor(0)
    #'train' took 31.595845 s
    
    #继续增加，误差又变大了
    
    #drift = torch.tensor([0, 1, 0, -1.5]).repeat(dim, 1)
    #xi = torch.tensor([1.0, 1.5, 1.0], dtype=torch.float) 
    #sample = 20000, gauss_variance=0.73, gauss_samp_number=220, lam=0.0, STRidge_threshold=0.05, lhs_ratio=0.55
    #不收敛
    
    
    