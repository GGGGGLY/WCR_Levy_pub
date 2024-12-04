# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:41:12 2023

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from generate_data_1d import DataSet
#from levy_data_test import Dataset
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

#save.tensor
class Gaussian(torch.nn.Module): 
    def __init__(self, mu, sigma, lap_alpha):
        super(Gaussian, self).__init__()  #gaussian()里面不是object,就需要super
        #if 下面用到了nn.Module, e.g. nn.Linear(),就要定义类的时候，里面要加上nn.Module
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
        
        #         (x.shape[2] + self.lap_alpha)/2, x.shape[2]/2, -torch.sum((x-self.mu)**2, dim=2) / (2*self.sigma**2)) 
        x = (x - self.mu)/self.sigma/np.sqrt(2)
        func = (1/self.sigma/np.sqrt(2))**self.lap_alpha * 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi))) \
            *sp.gamma((self.dim+self.lap_alpha)/2)*2**self.lap_alpha/sp.gamma(self.dim/2)*sp.hyp1f1((self.dim+self.lap_alpha)/2, self.dim/2, -torch.sum(x**2,dim = 2))
        return func  ##1/(2*pi)
    
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
        self.t_number = len(self.t)
        self.basis1_number = int(np.math.factorial(self.dimension+self.basis_order)
                /(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order))) 
        
        if self.dimension ==1:
            self.basis2_number = 1
        else:
            self.basis2_number = int( self.dimension*(self.dimension+1)/2 ) + 1
        

        # Construct Theta
        basis1 = [] #用1带进去基， 得到一向量，用2带进去，又得到一个向量
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
                            Theta[:,basis_count1] = torch.mul(torch.mul(X[:,ii],
                                X[:,jj]),X[:,kk])
                            basis_count1 += 1

            if self.basis_order >= 4:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                Theta[:,basis_count1] = torch.mul(torch.mul(torch.mul(X[:,ii],
                                    X[:,jj]),X[:,kk]),X[:,ll])
                                basis_count1 += 1

            if self.basis_order >= 5:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                for mm in range(ll,self.dimension):
                                    Theta[:,basis_count1] = torch.mul(torch.mul(torch.mul(torch.mul(
                                        X[:,ii],X[:,jj]),X[:,kk]),
                                            X[:,ll]),X[:,mm])
                                    basis_count1 += 1
            
            assert basis_count1 == self.basis1_number
            basis1.append(Theta)
            basis_theta = torch.stack(basis1)
            # print("X", X)
        #print("basis_theta", basis_theta.shape)
            
            
            # Construct Xi
        basis2 = []     
        for it in range(self.t_number):
            basis_count2 = 0
            X = self._get_data_t(it)
            Xi = torch.zeros(X.size(0),self.basis2_number)
            #q = 0 # q为任意给定的常数，XI的基由x^{2q}次给出(x_1^{2q}, ..., x_d^{2q}, x_1^{q}x_2^{q}, ..., x_1^{q}x_d^{q}, ..., x_{d-1}^{q}x_d^{q})
            Xi[:,0] = 1
            basis_count2 += 1

            if self.basis_xi_order == 1 & self.dimension > 1:  #一个次幂的基底展开，只能是2q次方
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        Xi[:,basis_count2] = torch.mul(X[:,ii]** self.xi_q, X[:,jj]** self.xi_q)
                        basis_count2 += 1
            else:
                print("The basis of levy noise is not suitable or dimension is 1")
                
                    
            print("basis_count2",basis_count2)
            print("basis2_number",self.basis2_number)
            print("Xi",Xi)
                
            assert basis_count2 == self.basis2_number
            basis2.append(Xi)
            basis_xi = torch.stack(basis2)
        #print("basis_xi", basis_xi.shape)
            
        self.basis_theta = basis_theta   
        self.basis = torch.cat([basis_theta, basis_xi],dim=2)         
        #self.basis = torch.stack(basis)
        #print("self.basis.shape", self.basis.shape)
        self.basis_number = self.basis1_number + self.basis2_number
    
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2)     

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number  #db
        F_number = self.dimension * self.dimension #* self.basis1_number  #d^2b
        C_number = self.dimension * self.dimension #* self.basis2_number  #d^2c
        
        A = torch.zeros([self.t_number, H_number+F_number+C_number]) #A is a L* (db+d^2b+d^2c)
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

        # compute A by F_lkj
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                F = 1/self.bash_size * torch.sum(
                    gauss2[:, :, ld, kd], dim=1
                    )
                A[:, H_number] = F
                
                               
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                E = -torch.mean(gauss_lap, dim=1)
                A[:, H_number+1] = E 
                
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
        # print("b", rb)

        # b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
        # print("b.shape", b.shape)

        # plt.clf()
        # plt.plot(rb.detach().numpy(),'-*')
        # plt.plot(b.detach().numpy(),'-o')
        # plt.draw()
        # plt.pause(1)

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
            mu_list = self.lhs_ratio * torch.rand(samp_number)*(self.data.max()-self.data.min()) + self.data.min()
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
            gauss = self.net(mu, sigma,3/2) 
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
    def compile(self, basis_order, basis_xi_order, gauss_variance, type, drift_term, diffusion_term, xi_term, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.basis_order = basis_order
        self.build_basis()
        self.basis_xi_order = basis_xi_order
        self.variance = gauss_variance
        self.type = type
        self.drift = drift_term
        self.diffusion = diffusion_term
        self.xi = xi_term
        self.gauss_samp_way = gauss_samp_way
        self.lhs_ratio = lhs_ratio if self.gauss_samp_way == 'lhs' else 1

    @utils.timing
    @torch.no_grad()
    def train(self, gauss_samp_number, lam, STRidge_threshold):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        print("A: ", self.A.size(), "b: ", self.b.size())
        print("A",np.linalg.cond(self.A, p=None))
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        print("zeta: ", self.zeta)

        drift = [self.zeta[0].numpy()]
        for i in range(self.basis_number-2):
            drift.extend([" + ", self.zeta[i+1].numpy(), 'x^', i+1])
        print("Drift term: ", "".join([str(_) for _ in drift]))
        self.zeta[-2] = torch.sqrt(self.zeta[-2]*2) #G = (1/2) \sigma \sigma^T
        print("Diffusion term of Brown Motion: ", self.zeta[-2])
        self.zeta[-1] = (self.zeta[-1])**(2/3) #\Sigma = xi xi^T
        print("Diffusion term of Levy Noise: ", self.zeta[-1])
        true = torch.cat((self.drift, self.diffusion, self.xi))
        index = torch.nonzero(true).squeeze()
        relative_error = torch.abs((self.zeta.squeeze()[index] - true[index]) / true[index])
        print("Maximum relative error: ", relative_error.max().numpy())

if __name__ == '__main__':
    np.random.seed(7) #seed=200;100估计就很差； seed=1000就挺好;
    #  "SVD did not converge in Linear Least Squares"
    torch.manual_seed(7)

    dt = 0.0001
    # t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    #t = torch.tensor([0.1, 0.3, 0.5])
    #t = torch.tensor([0.2, 0.5, 1.0])
    #t = torch.linspace(0,1,10)
    #t = torch.tensor([0.1, 0.4, 0.7, 1])

    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #t = torch.tensor([0.2, 0.5, 1])
    # data = scipy.io.loadmat('./data/data1d.mat')['bb'].astype(np.float32)
    # data = torch.tensor(data).unsqueeze(-1)
    # drift = torch.tensor([0, -3, -0.5, 4, 0.5, -1])   # -(x+1.5)(x+1)x(x-1)(x-2)
    # drift = torch.tensor([0, -24, 50, -35, 10, -1])     # -x(x-1)(x-2)(x-3)(x-4)
    # drift = torch.tensor([0, -4, 0, 5, 0, -1])  # -x(x-1)(x+1)(x-2)(x+2)
    drift = torch.tensor([0, 1, 0, -1])
    diffusion = torch.tensor([1.0])
    xi = torch.tensor([1.0])
    samples = 10000
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, 1]),
                      explosion_prevention=False) #initialization=torch.randint(-1000,1000,[10000, 1])
    data = dataset.get_data(plot_hist=False)
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=3,basis_xi_order=1, gauss_variance=0.7, type='LMM_2_nonequal', drift_term=drift, diffusion_term=diffusion, xi_term=xi,
                  gauss_samp_way='lhs', lhs_ratio=0.7)
    model.train(gauss_samp_number=100, lam=0.01, STRidge_threshold=0.1)
    
        
        
 ##### (1) t = torch.tensor([0.1, 0.4, 0.7, 1.0])     
  
##########sample = 10000, gauss_samp_number=100, lam=0.02 （或者0.0）, STRidge_threshold=0.1（或者0.2，结果一样）, gauss_var = 0.95, lhs = 0.9
    #Drift term:  [0.] + [0.8734925]x^1 + [0.]x^2 + [-0.93648255]x^3
    #Diffusion term of Brown Motion:  tensor([0.9173])
    #Diffusion term of Levy Noise:  tensor([1.0145])
    #Maximum relative error:  0.12650752
    #'train' took 1.610035 s (没乘1.03)
  
##########sample = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2, gauss_var = 0.95, lhs = 0.95
    #Drift term:  [0.] + [0.86041343]x^1 + [0.]x^2 + [-0.9271305]x^3
    #Diffusion term of Brown Motion:  tensor([0.9658])
    #Diffusion term of Levy Noise:  tensor([0.9865])
    #Maximum relative error:  0.13958657
    #'train' took 1.845096 s


##########sample = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2, gauss_var = 0.95, lhs = 0.92
    #Drift term:  [0.] + [0.87223333]x^1 + [0.]x^2 + [-0.9349003]x^3
    #Diffusion term of Brown Motion:  tensor([0.9343])
    #Diffusion term of Levy Noise:  tensor([1.0046])
    #Maximum relative error:  0.12776667
    #'train' took 1.874035 s
    
    
##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.1, gauss_var = 0.9, lhs = 0.92
    # Drift term:  [0.] + [0.8984596]x^1 + [0.]x^2 + [-0.94664985]x^3
    #Diffusion term of Brown Motion:  tensor([0.8632])
    #Diffusion term of Levy Noise:  tensor([1.0402])
    #Maximum relative error:  0.13682973
    #'train' took 1.274552 s
     
 ##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.1, gauss_var = 1.0, lhs = 0.9
     #Drift term:  [0.] + [0.87764007]x^1 + [0.]x^2 + [-0.93691117]x^3
     #Diffusion term of Brown Motion:  tensor([0.9140])
     #Diffusion term of Levy Noise:  tensor([1.0133])
     #Maximum relative error:  0.12235993
     #'train' took 0.926758 s   
     
##########sample = 10000, gauss_samp_number=50, lam=0.005, STRidge_threshold=0.1, gauss_var = 1.0, lhs = 0.9
    #Drift term:  [0.] + [0.87764007]x^1 + [0.]x^2 + [-0.93691117]x^3
    #Diffusion term of Brown Motion:  tensor([0.9140])
    #Diffusion term of Levy Noise:  tensor([1.0133])
    #Maximum relative error:  0.12235993
    #'train' took 0.692720 s
    
##########sample = 10000, gauss_samp_number=50, lam=0.005, STRidge_threshold=0.1, gauss_var = 1.1, lhs = 0.9
    # error变大
    
 ##########sample = 10000, gauss_samp_number=80, lam=0.005, STRidge_threshold=0.1, gauss_var = 1.0, lhs = 0.9   
    #Drift term:  [0.] + [0.88194793]x^1 + [0.]x^2 + [-0.9428406]x^3
    #Diffusion term of Brown Motion:  tensor([0.9385])
    #Diffusion term of Levy Noise:  tensor([1.0033])
    #Maximum relative error:  0.118052065
    #'train' took 1.104586 s
    
##########sample = 10000, gauss_samp_number=75, lam=0.005, STRidge_threshold=0.1, gauss_var = 1.0, lhs = 0.9   
       
    #Drift term:  [0.] + [0.8914281]x^1 + [0.]x^2 + [-0.948983]x^3
    #Diffusion term of Brown Motion:  tensor([0.9256])
    #Diffusion term of Levy Noise:  tensor([1.0115])
    #Maximum relative error:  0.10857189
    #'train' took 0.943515 s
    
##########sample = 10000, gauss_samp_number=70, lam=0.005, STRidge_threshold=0.1, gauss_var = 1.0, lhs = 0.9   
       
    # Drift term:  [0.] + [0.8943298]x^1 + [0.]x^2 + [-0.950215]x^3
    #Diffusion term of Brown Motion:  tensor([0.9276])
    #Diffusion term of Levy Noise:  tensor([1.0083])
    #Maximum relative error:  0.10567021
    #'train' took 1.021149 s

##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2, gauss_var = 1.0, lhs = 0.75  
#Drift term:  [0.] + [0.9025586]x^1 + [0.]x^2 + [-0.9541948]x^3
#Diffusion term of Brown Motion:  tensor([0.9374])
#Diffusion term of Levy Noise:  tensor([1.0146])
#Maximum relative error:  0.097441375
#'train' took 1.333390 s

##########sample = 10000, gauss_samp_number=80, lam=0.0, STRidge_threshold=0.2, gauss_var = 1.0, lhs = 0.75  
# Drift term:  [0.] + [0.90533805]x^1 + [0.]x^2 + [-0.9547661]x^3
#Diffusion term of Brown Motion:  tensor([0.9315])
#Diffusion term of Levy Noise:  tensor([1.0208])
#Maximum relative error:  0.09466195
#'train' took 1.215698 s

#--------------------------------------

##########sample = 10000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_var = 1.0, lhs = 0.75  
# Drift term:  [0.] + [0.91395336]x^1 + [0.]x^2 + [-0.9585366]x^3
#Diffusion term of Brown Motion:  tensor([0.9221])
#Diffusion term of Levy Noise:  tensor([1.0262])
#Maximum relative error:  0.086046636
#'train' took 0.914090 s

#------------------------------------

##########sample = 10000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.05, gauss_var = 1.0, lhs = 0.75  
#同上

##########sample = 10000, gauss_samp_number=60, lam=0.0, STRidge_threshold=0.2, gauss_var = 1.0, lhs = 0.75  
#Drift term:  [0.] + [0.916197]x^1 + [0.]x^2 + [-0.958148]x^3
#Diffusion term of Brown Motion:  tensor([0.9017])
#Diffusion term of Levy Noise:  tensor([1.0339])
#Maximum relative error:  0.098341525
#'train' took 1.198055 s



   
    
    
    
    
    
      
        
 ##### (2) t = torch.tensor([0.2, 0.5, 1.0])
            
        
##########sample = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2, gauss_var = 0.75, lhs = 1.0
     ##### t = torch.tensor([0.2, 0.5, 1.0])
     # diffusion=1; xi = 1:   
         #Drift term:  [0.] + [0.8462193]x^1 + [0.]x^2 + [-0.89018095]x^3
         #Diffusion term of Brown Motion:  tensor([1.0578])
         #Diffusion term of Levy Noise:  tensor([0.8686])
         #Maximum relative error:  0.1537807
         #'train' took 1.771243 s
         
         
##########sample = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2, gauss_var = 0.75, lhs = 0.75
     ##### t = torch.tensor([0.2, 0.5, 1.0])
     # diffusion=1; xi = 1:  
         
        #Drift term:  [0.] + [0.85324836]x^1 + [0.]x^2 + [-0.9026397]x^3
        #Diffusion term of Brown Motion:  tensor([1.0283])
        #Diffusion term of Levy Noise:  tensor([0.9023])
        #Maximum relative error:  0.14675164
        #'train' took 1.590582 s

##########sample = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1, gauss_var = 0.72, lhs = 0.75
     # diffusion=1; xi = 1:          
    #Drift term:  [0.] + [0.85202056]x^1 + [0.]x^2 + [-0.9014467]x^3
    #Diffusion term of Brown Motion:  tensor([1.0144])
    #Diffusion term of Levy Noise:  tensor([0.9113])
    #Maximum relative error:  0.14797944
    #'train' took 1.716157 s
    
##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2, gauss_var = 0.72, lhs = 0.75
     # diffusion=1; xi = 1:   同上
          
        
 ##########sample = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1, gauss_var = 0.72, lhs = 0.78
      #       Drift term:  [0.] + [0.8137923]x^1 + [0.]x^2 + [-0.8807643]x^3
      #Diffusion term of Brown Motion:  tensor([1.0276])
      #Diffusion term of Levy Noise:  tensor([0.9031])
      #Maximum relative error:  0.18620771
      #'train' took 1.982755 s
        
        
##########sample = 10000, gauss_samp_number=120, lam=0.02, STRidge_threshold=0.1, gauss_var = 0.72, lhs = 0.75
     # diffusion=1; xi = 1:          
     # Drift term:  [0.] + [0.8456846]x^1 + [0.]x^2 + [-0.896343]x^3
     #Diffusion term of Brown Motion:  tensor([1.0409])
     #Diffusion term of Levy Noise:  tensor([0.8909])
     #Maximum relative error:  0.15431541
     #'train' took 2.241779 s   
        
        
        
        
        
 ##### (3) t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])     
  
##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.1, gauss_var = 0.72, lhs = 0.75        
 #Drift term:  [0.] + [0.9704476]x^1 + [0.]x^2 + [-0.9933872]x^3
 #Diffusion term of Brown Motion:  tensor([1.0390])
 #Diffusion term of Levy Noise:  tensor([0.9729])
 #Maximum relative error:  0.03903663
 #'train' took 10.571525 s       
        
##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.1, gauss_var = 0.7, lhs = 0.75        
 # Drift term:  [0.] + [0.9705612]x^1 + [0.]x^2 + [-0.9934679]x^3
 #Diffusion term of Brown Motion:  tensor([1.0328])
 #Diffusion term of Levy Noise:  tensor([0.9772])
 #Maximum relative error:  0.032756686
 #'train' took 8.262494 s       
 
 ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.1, gauss_var = 0.7, lhs = 0.85        
  #Drift term:  [0.] + [0.93727624]x^1 + [0.]x^2 + [-0.9848175]x^3
 # Diffusion term of Brown Motion:  tensor([0.9970])
 # Diffusion term of Levy Noise:  tensor([1.0085])
 # Maximum relative error:  0.062723756
 # 'train' took 9.919689 s
 
 #-----------------------------------------
 ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.1, gauss_var = 0.7, lhs = 0.7       
  #Drift term:  [0.] + [0.97959614]x^1 + [0.]x^2 + [-1.0140877]x^3
  #Diffusion term of Brown Motion:  tensor([1.0167])
  #Diffusion term of Levy Noise:  tensor([0.9948])
  #Maximum relative error:  0.020403862
  #'train' took 10.443834 s
  #------------------------------------------
  
        
  ##########sample = 10000, gauss_samp_number=80, lam=0.0, STRidge_threshold=0.1, gauss_var = 0.7, lhs = 0.75        
   #Drift term:  [0.] + [0.96179646]x^1 + [0.]x^2 + [-0.9858595]x^3
   #Diffusion term of Brown Motion:  tensor([1.0239])
   #Diffusion term of Levy Noise:  tensor([0.9836])
   #Maximum relative error:  0.038203537
   #'train' took 7.860995 s       
        
        
        
        