# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:30:10 2023

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
#from collections import OrderedDict
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
        X = self.data[it,:,:]  #三个维度，时间，轨道数，问题的维度
        return X
    
    @utils.timing # decorator
    @torch.no_grad()
    def build_basis(self): # \Lambda matrix
        """build the basis list for the different time snapshot 
        """
        self.t_number = len(self.t)
        self.basis1_number = int(np.math.factorial(self.dimension+self.basis_order)
                /(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order))) #int取整， np.math.factorial阶乘
        
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
                                    
            
            if self.basis_order >= 6:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                for mm in range(ll,self.dimension):
                                    for nn in range(mm,self.dimension):
                                        Theta[:,basis_count1] = torch.mul(torch.mul(torch.mul(torch.mul(torch.mul(
                                            X[:,ii],X[:,jj]),X[:,kk]), \
                                            X[:,ll]),X[:,mm]), X[:,nn])
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

            #if self.basis_xi_order == 1 & self.dimension > 1:  #一个次幂的基底展开，只能是2q次方
            #    for ii in range(0,self.dimension):
             #       for jj in range(ii,self.dimension):
             #           Xi[:,basis_count2] = torch.mul(X[:,ii]** self.xi_q, X[:,jj]** self.xi_q)
             #           basis_count2 += 1
            #else:
             #   print("The basis of levy noise is not suitable or dimension is 1")
                                 
            #print("basis_count2",basis_count2)
            #print("basis2_number",self.basis2_number)
            #print("Xi",Xi)
                
            assert basis_count2 == self.basis2_number
            basis2.append(Xi)
            basis_xi = torch.stack(basis2)
            
        self.basis_theta = basis_theta   
        self.basis = torch.cat([basis_theta, basis_xi],dim=2)         
        #self.basis = torch.stack(basis)
        #print("self.basis.shape", self.basis.shape)
        self.basis_number = self.basis1_number + self.basis2_number
    
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 
        #unsqueeze()用于增加一个维度

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2)     
        #torch.matmul(b, a) 矩阵b与a相乘

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis1_number  #db
        C_number = self.dimension * self.dimension #* self.basis2_number  #d^2c
        
        A = torch.zeros([self.t_number, H_number+C_number]) #A is a L* (db+d^2b+d^2c)
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
            gauss = self.net(mu, sigma,3/2) #########alpha在哪赋值？？？
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
        y = 0
        #y = self.zeta[0]
        est_drifts = self.zeta[0: -1]
        for i in range(est_drifts.shape[0]):
            y = y + est_drifts[i] * x ** i
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
        x = torch.randn((1000,1))
        #x = torch.rand((10,3))
        #u_values = -x**4 * torch.exp(-x**2)
        #v_values = self.drift_est(x)
        #Werr = scipy.stats.wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None)
        #print("Wasserstein error: ", Werr.numpy())
        L2_error =(1/ x.numel() )* torch.norm(-2*x * torch.exp(-x**2) - self.drift_est(x))
        print("L2 error: ", L2_error.numpy())

if __name__ == '__main__':
    np.random.seed(6) 
    torch.manual_seed(6)

    dt = 0.0001
    # t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    #t = torch.tensor([0.1, 0.3, 0.5])
    #t = torch.tensor([0.2, 0.5, 1.0])
    #t = torch.linspace(0,1,10)
    t = torch.tensor([0.1, 0.2, 0.4, 0.7, 0.8, 1])

    # t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #t = torch.tensor([0.2, 0.5, 1])
    # data = scipy.io.loadmat('./data/data1d.mat')['bb'].astype(np.float32)
    # data = torch.tensor(data).unsqueeze(-1)
    # drift = torch.tensor([0, -3, -0.5, 4, 0.5, -1])   # -(x+1.5)(x+1)x(x-1)(x-2)
    
    xi = torch.tensor([1.0])
    samples = 10000
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=1, xi_term=xi, alpha_levy = 3/2, \
                      initialization=torch.normal(0, 0.2,[samples, 1]),explosion_prevention=False) 
                      #initialization=torch.randint(-1000,1000,[10000, 1])
    data = dataset.get_data(plot_hist=False)
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=2, gauss_variance=0.55, type='LMM_2_nonequal', xi_term=xi,\
                  gauss_samp_way='lhs', lhs_ratio=1.0)
    model.train(sample_num = samples, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0)
    
    
  ###############################
 ### 计算误差的数据x是随机产生的torch.randn((10,3)) 正态
 ################################        
    
    #####  t = torch.tensor([0.1, 0.2, 0.4, 0.7, 0.8, 1])  
    
################ basis_order=2
 ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
   #Drift term:  [-0.04234535] + [-0.514166]x^1 + [-0.09216347]x^2
   #Diffusion term of Levy Noise:  tensor([0.7481])
   #L2 error:  0.015869588
   #'train' took 2.824617 s
   
    
################ basis_order=3
 ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
   #Drift term:  [-0.06770282] + [-0.7849851]x^1 + [-0.16422017]x^2 + [0.12534173]x^3
   #Diffusion term of Levy Noise:  tensor([0.8087])
   #L2 error:  0.01467079
   #'train' took 2.663183 s
   
 ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
  # Drift term:  [-0.07771354] + [-0.77549636]x^1 + [-0.18400773]x^2 + [0.12678364]x^3
  #Diffusion term of Levy Noise:  tensor([0.8009])
  #L2 error:  0.015539539
  #'train' took 3.176082 s
  
  
################ basis_order=4
 ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
   # Drift term:  [-0.05945932] + [-0.7836889]x^1 + [-0.23470138]x^2 + [0.1334491]x^3 + [0.01405139]x^4
   #Diffusion term of Levy Noise:  tensor([0.8115])
   #L2 error:  0.014756083
   #'train' took 2.754304 s
    
    
################ basis_order=5
 ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
   # Drift term:  [-0.06062147] + [-1.0296358]x^1 + [-0.29857883]x^2 + [0.2794859]x^3 + [0.02178926]x^4 + [-0.0217872]x^5
   # Diffusion term of Levy Noise:  tensor([0.8603])
   #L2 error:  0.015352209
   # 'train' took 3.016591 s


################ basis_order=6
 ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
   # Drift term:  [-0.06285699] + [-1.0003705]x^1 + [-0.35796738]x^2 + [0.2696021]x^3 + [0.03348419]x^4 + [-0.02027802]x^5 + [-0.00129061]x^6
   #Diffusion term of Levy Noise:  tensor([0.8547])
   #L2 error:  0.016793806
   #'train' took 3.015200 s
    
    
    
    
    
    
    
    
    
    
    
    ##############################3
    
    # 之前L2_error =(1/ x.numel() )\sum...分母错了，应该是检验的x的元素个数，之前除成了sample的个数(data的个数)
    
    ##############################3
    
    
    
 ###############################
### 计算误差的数据x是随机产生的torch.rand((10,3)) 均匀
################################       
        
 #####  t = torch.tensor([0.1, 0.2, 0.4, 0.7, 0.8, 1])   
 ###basis_order=3
  
##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 1.0, lhs = 1.0
# Drift term:  [-0.02230229] + [-0.58228815]x^1 + [-0.00057352]x^2 + [0.05696728]x^3
#Diffusion term of Levy Noise:  tensor([0.8042])
#L2 error:  1.9099435e-04
#'train' took 1.318512 s

##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.0, gauss_var = 1.0, lhs = 1.0
# Drift term:  [-0.01896776] + [-0.5321587]x^1 + [0.00173822]x^2 + [0.05385792]x^3
#Diffusion term of Levy Noise:  tensor([0.7928])
#L2 error:  2.1838958e-04
#'train' took 2.544805 s


##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.85, lhs = 1.0
# Drift term:  [-0.02504484] + [-0.69420147]x^1 + [-0.02810544]x^2 + [0.08245508]x^3
#Diffusion term of Levy Noise:  tensor([0.8227])
#L2 error:  1.5879127e-04
#'train' took 1.273218 s

##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.75, lhs = 1.0
# Drift term:  [-0.03133397] + [-0.75396645]x^1 + [-0.05961446]x^2 + [0.09988371]x^3
#Diffusion term of Levy Noise:  tensor([0.8283])
#L2 error:  1.3782315e-04
#'train' took 1.901745 s


#--------------------------------

##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
# Drift term:  [-0.06770282] + [-0.7849851]x^1 + [-0.16422017]x^2 + [0.12534173]x^3
# Diffusion term of Levy Noise:  tensor([0.8087])
#L2 error:  1.0302984e-04
# 'train' took 2.209337 s

#-----------------------------------

##########sample =10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.45, lhs = 1.0
# 不收敛

##########sample =10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 0.85
# Drift term:  [0.02529648] + [-0.2284942]x^1 + [0.02420173]x^2 + [0.01936538]x^3 + [0.58043826]x^4
#Diffusion term of Levy Noise:  tensor([0.6958])
#L2 error:  3.08012e-04
#'train' took 2.672321 s

##########sample =20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 0.85
# Drift term:  [-0.05321235] + [-0.6409784]x^1 + [-0.031448]x^2 + [0.08274561]x^3 + [0.6468183]x^4
#Diffusion term of Levy Noise:  tensor([0.7479])
#L2 error:  1.7149837e-04
#'train' took 4.162059 s

  
 ###### basis_order = 4
 
 ##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.05945932] + [-0.7836889]x^1 + [-0.23470138]x^2 + [0.1334491]x^3 + [0.01405139]x^4
 # Diffusion term of Levy Noise:  tensor([0.8115])
 #L2 error:  1.0013332e-04
 #'train' took 2.014093 s
 
 ########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.5, lhs = 1.0
 # 不收敛
 
 
 #######basis_order = 2
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 #Drift term:  [-0.04234535] + [-0.514166]x^1 + [-0.09216347]x^2
 # Diffusion term of Levy Noise:  tensor([0.7481])
 #L2 error:  1.7761477e-04
 #'train' took 1.876019 s
 
 
 #######basis_order = 5
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.06062147] + [-1.0296358]x^1 + [-0.29857883]x^2 + [0.2794859]x^3 + [0.02178926]x^4 + [-0.0217872]x^5
 # Diffusion term of Levy Noise:  tensor([0.8603])
 #L2 error:  6.9840205e-05
 #'train' took 2.021271 s
 
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.5, lhs = 1.0
 # 不收敛
 
 #---------------------------
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
 # Drift term:  [-0.06930447] + [-1.0136533]x^1 + [-0.322403]x^2 + [0.28052768]x^3 + [0.02433956]x^4 + [-0.02172366]x^5
 #Diffusion term of Levy Noise:  tensor([0.8527])
 #L2 error: 6.9655627e-05
 #'train' took 2.488726 s
 #----------------------------
 
 
 ############ basis_order = 6
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
 # Drift term:  [-0.07210229] + [-0.9840788]x^1 + [-0.38280383]x^2 + [0.27087006]x^3 + [0.03630953]x^4 + [-0.02013306]x^5 + [-0.00133744]x^6
 #Diffusion term of Levy Noise:  tensor([0.8471])
 #L2 error:  7.237276e-05
 #'train' took 2.180866 s
 
 #----------------------------
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.06285699] + [-1.0003705]x^1 + [-0.35796738]x^2 + [0.2696021]x^3 + [0.03348419]x^4 + [-0.02027802]x^5 + [-0.00129061]x^6
 #Diffusion term of Levy Noise:  tensor([0.8547])
 #L2 error:  7.23355e-05
 #'train' took 1.999630 s
 #--------------------------------
 
 
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.04431906] + [-1.0838614]x^1 + [0.03180895]x^2 + [0.24861413]x^3 + [-0.00691995]x^4 + [-0.01780893]x^5 + [0.00079371]x^6
 #Diffusion term of Levy Noise:  tensor([0.8484])
 #L2 error:  4.685231e-05
 #'train' took 3.843506 s
 
 
 
 
 
 
 
 ####################################
 ## 计算误差的x = self.data, 不影响未知项的估计，只是影响误差值
 ############################
 ##### t = torch.tensor([0.1, 0.2, 0.4, 0.7, 0.8, 1])   
 ### basis_order=3
 
 
 ############ basis_order = 6
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.04431906] + [-1.0838614]x^1 + [0.03180895]x^2 + [0.24861413]x^3 + [-0.00691995]x^4 + [-0.01780893]x^5 + [0.00079371]x^6
 #Diffusion term of Levy Noise:  tensor([0.8484])
 #L2 error:  0.043288928
 #'train' took 3.727845 s
 
 
 
 ############ basis_order = 5
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.04402484] + [-1.0728511]x^1 + [-0.01885872]x^2 + [0.24888024]x^3 + [0.00503431]x^4 + [-0.01736384]x^5
 #Diffusion term of Levy Noise:  tensor([0.8455])
 #L2 error:  0.039050326
 #'train' took 4.197603 s
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 #Drift term:  [-0.06062147] + [-1.0296358]x^1 + [-0.29857883]x^2 + [0.2794859]x^3 + [0.02178926]x^4 + [-0.0217872]x^5
 #Diffusion term of Levy Noise:  tensor([0.8603])
 #L2 error:  0.0917404
 #'train' took 2.736982 s
 
 
 
 
 ############### basis_order = 4
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.04912507] + [-0.64845544]x^1 + [-0.04141122]x^2 + [0.08696736]x^3 + [0.00522086]x^4
 #Diffusion term of Levy Noise:  tensor([0.7496])
 #L2 error:  0.0102087855
 #'train' took 4.648601 s
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 #Drift term:  [-0.05945932] + [-0.7836889]x^1 + [-0.23470138]x^2 + [0.1334491]x^3 + [0.01405139]x^4
 #Diffusion term of Levy Noise:  tensor([0.8115])
 #L2 error:  0.02709006
 #'train' took 2.828524 s
 
 
 
 
 ################ basis_order = 3
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
 # Drift term:  [-0.06160183] + [-0.6809114]x^1 + [-0.02786956]x^2 + [0.08567537]x^3
 #Diffusion term of Levy Noise:  tensor([0.7530])
 #L2 error:  0.009629369
 #'train' took 4.353461 s
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.06770282] + [-0.7849851]x^1 + [-0.16422017]x^2 + [0.12534173]x^3
 #Diffusion term of Levy Noise:  tensor([0.8087])
 #L2 error:  0.020542737
 #'train' took 3.499747 s
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
 # Drift term:  [-0.07771354] + [-0.77549636]x^1 + [-0.18400773]x^2 + [0.12678364]x^3
 #Diffusion term of Levy Noise:  tensor([0.8009])
 #L2 error:  0.021266308
 #'train' took 2.979031 s
 
 
 
 
 ################### basis_order = 2
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.05683747] + [-0.49413723]x^1 + [-0.02207983]x^2
 #Diffusion term of Levy Noise:  tensor([0.7136])
 #L2 error:  0.008754043
 #'train' took 4.478622 s
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.04234535] + [-0.514166]x^1 + [-0.09216347]x^2
 # Diffusion term of Levy Noise:  tensor([0.7481])
 #L2 error:  0.013608242
 #'train' took 2.768482 s
 
 
 
 ################ basis_order = 1
 #sample = 20000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.05412639] + [-0.4988049]x^1
 #Diffusion term of Levy Noise:  tensor([0.7150])
 #L2 error:  0.008681393
 #'train' took 4.619529 s
 
 #sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
 # Drift term:  [-0.03307164] + [-0.5838918]x^1
 #Diffusion term of Levy Noise:  tensor([0.7662])
 #L2 error:  0.012834115
 #'train' took 3.138693 s
 
 ###用原来的数据逼近发现一阶多项式的误差最小，阶数越高，误差越大，但是原函数-2x*e^{-x^2}是一个在[-2, 2]呈现双峰（左峰上凸，右峰下凸）
 ##原来的数据范围大概在[-6.1, 7.7], 所以在[-6.1, -2]\cup [2, 7.7]中会没法逼近，原函数是趋于0的，而现实是有数据的；
 ##所以考虑在[-2,2]随机生成数据检验拟合（逼近）情况。
 