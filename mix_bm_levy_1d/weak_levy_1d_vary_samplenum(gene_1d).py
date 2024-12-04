# -*- coding: utf-8 -*-
"""
1d: vary sample number

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
        func = (1/self.sigma/np.sqrt(2))**self.lap_alpha *sp.gamma((self.dim+self.lap_alpha)/2)*2**self.lap_alpha/sp.gamma(self.dim/2)*sp.hyp1f1((self.dim+self.lap_alpha)/2, self.dim/2, -torch.sum(x**2,dim = 2)) \
            *1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi)))
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
        print("basis_theta", basis_theta.shape)
            
            
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
                
            #for ii in range(0,self.dimension):
                #for jj in range(ii,self.dimension):
                    #Xi[:,basis_count2] = torch.mul( X[:,ii], X[:,jj])
                    #basis_count2 += 1
                    
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
        print("self.basis.shape", self.basis.shape)
        self.basis_number = self.basis1_number + self.basis2_number
    
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 
        #unsqueeze()用于增加一个维度

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2)     
        #torch.matmul(b, a) 矩阵b与a相乘

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
            gauss = self.net(mu, sigma, 3/2) #########alpha在哪赋值？？？
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
    np.random.seed(7)
    torch.manual_seed(7)

    dt = 0.0001
    # t = np.linspace(0, T, int(T/dt) + 1).astype(np.float32)
    t = torch.tensor([0.1, 0.4, 0.7, 1.0])
    # t = torch.tensor([0.2, 0.5, 1.0])
    #t = torch.linspace(0,1,10)
    #t = torch.tensor(t)

    # t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
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
    model.compile(basis_order=3, basis_xi_order=1, gauss_variance=1.0, type='LMM_2_nonequal', drift_term=drift, diffusion_term=diffusion, xi_term=xi,\
                  gauss_samp_way='lhs', lhs_ratio=1.1)
    model.train(gauss_samp_number=120, lam=0.02, STRidge_threshold=0.1)
    
#_____________________________________________________________________________-

###t = torch.tensor([0.2, 0.5, 1.0])

   ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.65, lhs_ratio=0.9
   #Drift term:  [0.] + [0.8980628]x^1 + [0.]x^2 + [-0.7899772]x^3
   #Diffusion term of Brown Motion:  tensor([1.0060])
   #Diffusion term of Levy Noise:  tensor([0.7962])
   #Maximum relative error:  0.2100228
   #'train' took 0.183668 s
   
   ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.55, lhs_ratio=0.9
   # Drift term:  [0.] + [0.87322783]x^1 + [0.]x^2 + [-0.7566744]x^3
   #Diffusion term of Brown Motion:  tensor([0.8830])
   #Diffusion term of Levy Noise:  tensor([0.8616])
   #Maximum relative error:  0.24332559
   #'train' took 0.190955 s
   
   ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.55, lhs_ratio=0.7
   # Drift term:  [0.] + [0.90203387]x^1 + [0.]x^2 + [-0.75852704]x^3
   #Diffusion term of Brown Motion:  tensor([1.0458])
   #Diffusion term of Levy Noise:  tensor([0.7497])
   #Maximum relative error:  0.25028616
   #'train' took 0.189087 s
   
   ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.5, lhs_ratio=0.7
   # Drift term:  [0.] + [0.891807]x^1 + [0.]x^2 + [-0.74730414]x^3
   #Diffusion term of Brown Motion:  tensor([0.9331])
   #Diffusion term of Levy Noise:  tensor([0.8387])
   #Maximum relative error:  0.25269586
   #'train' took 0.214308 s
   
   
   
   
   

  ###t = torch.tensor([0.1, 0.4, 0.7, 1.0]) 
   
   
   ####samples = 5000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2,gauss_variance=0.7, lhs_ratio=1.0
   #Drift term:  [0.] + [0.86529535]x^1 + [0.]x^2 + [-0.86914724]x^3
   #Diffusion term of Brown Motion:  tensor([1.1032])
   #Diffusion term of Levy Noise:  tensor([0.7964])
   #Maximum relative error:  0.20355636
   #'train' took 1.269530 s
   
   ####samples = 5000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2,gauss_variance=0.65, lhs_ratio=1.0
   #Drift term:  [0.] + [0.8606252]x^1 + [0.]x^2 + [-0.86027145]x^3
   #Diffusion term of Brown Motion:  tensor([1.0748])
   #Diffusion term of Levy Noise:  tensor([0.8126])
   #Maximum relative error:  0.18739516
   #'train' took 1.146559 s
   
   #----------------------------------------
   ####samples = 5000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2,gauss_variance=0.65, lhs_ratio=1.0
   # Drift term:  [0.] + [0.8703292]x^1 + [0.]x^2 + [-0.87966394]x^3
   #Diffusion term of Brown Motion:  tensor([1.0533])
   #Diffusion term of Levy Noise:  tensor([0.8451])
   #Maximum relative error:  0.15485364
   #'train' took 1.038647 s
   #-------------------------------------------
   
   
   
   
   
   
   
   
   
   ####samples = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=0.65, lhs_ratio=1.0
   #Drift term:  [0.] + [0.8435883]x^1 + [0.]x^2 + [-0.9032599]x^3
   #Diffusion term of Brown Motion:  tensor([0.7689])
   #Diffusion term of Levy Noise:  tensor([1.0842])
  # Maximum relative error:  0.23106933
   #'train' took 1.972280 s
   
   ####samples = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=0.75, lhs_ratio=1.0
   # Drift term:  [0.] + [0.8521251]x^1 + [0.]x^2 + [-0.91565526]x^3
   #Diffusion term of Brown Motion:  tensor([0.8245])
   #Diffusion term of Levy Noise:  tensor([1.0614])
   #Maximum relative error:  0.17549759
   #'train' took 2.315121 s
   
   ####samples = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=0.85, lhs_ratio=1.0
   # Drift term:  [0.] + [0.8499228]x^1 + [0.]x^2 + [-0.917123]x^3
   #Diffusion term of Brown Motion:  tensor([0.8992])
   #Diffusion term of Levy Noise:  tensor([1.0232])
   #Maximum relative error:  0.15007722
   #'train' took 1.854305 s
   
   ####samples = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=0.95, lhs_ratio=1.0
   # Drift term:  [0.] + [0.8343201]x^1 + [0.]x^2 + [-0.90942156]x^3
   # Diffusion term of Brown Motion:  tensor([0.9909])
   #Diffusion term of Levy Noise:  tensor([0.9701])
   #Maximum relative error:  0.16567987
   #'train' took 1.374657 s
   
   #--------------------------------
   ####samples = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=1.0, lhs_ratio=1.1
   # Drift term:  [0.] + [0.8506907]x^1 + [0.]x^2 + [-0.9168545]x^3
   #Diffusion term of Brown Motion:  tensor([0.9308])
   #Diffusion term of Levy Noise:  tensor([1.0041])
   #Maximum relative error:  0.14930928
   #'train' took 1.709831 s
   #--------------------------------
   
   
   
   
   
#____________________________________________________________________________

###t = torch.linspace(0,1,10)
    ##### samples = 1000, gauss_samp_number=50, lam=0.02, STRidge_threshold=0.0, gauss_variance=0.85, lhs_ratio=1.0
    #Drift term:  [-0.05951567] + [0.8391304]x^1 + [0.00312384]x^2 + [-0.9295226]x^3
    #Diffusion term of Brown Motion:  tensor([1.1396])
    #Diffusion term of Levy Noise:  tensor([1.0211])
    #Maximum relative error:  0.1608696
    #'train' took 0.232213 s
    
    #### samples = 1000, gauss_samp_number=30, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.7, lhs_ratio=0.65
    # Drift term:  [0.] + [1.1126698]x^1 + [0.]x^2 + [-1.0280555]x^3
    #Diffusion term of Brown Motion:  tensor([1.1587])
    #Diffusion term of Levy Noise:  tensor([0.9173])
    #Maximum relative error:  0.15867436
    #'train' took 0.248768 s
    
    ##### samples = 1000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.7, lhs_ratio=1.0
    #Drift term:  [0.] + [0.8909193]x^1 + [0.]x^2 + [-0.9546612]x^3
    #Diffusion term of Brown Motion:  tensor([0.9385])
    #Diffusion term of Levy Noise:  tensor([1.1416])
    #Maximum relative error:  0.141626
    #'train' took 0.347540 s
    
    ##### samples = 1000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.8, lhs_ratio=1.0
    # Drift term:  [0.] + [0.8423229]x^1 + [0.]x^2 + [-0.9204448]x^3
    #Diffusion term of Brown Motion:  tensor([1.0887])
    #Diffusion term of Levy Noise:  tensor([1.0355])
    #Maximum relative error:  0.15767711
    #'train' took 0.367690 s
    
    ##### samples = 1000, gauss_samp_number=60, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.8, lhs_ratio=1.0
    # Drift term:  [0.] + [0.869649]x^1 + [0.]x^2 + [-0.9362472]x^3
    #Diffusion term of Brown Motion:  tensor([1.0602])
    #Diffusion term of Levy Noise:  tensor([1.0551])
    #Maximum relative error:  0.130351
    #'train' took 0.433930 s
    
    ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.8, lhs_ratio=1.0
    # Drift term:  [0.] + [0.90196896]x^1 + [0.]x^2 + [-0.9559859]x^3
    #Diffusion term of Brown Motion:  tensor([1.0529])
    #Diffusion term of Levy Noise:  tensor([1.0597])
   # Maximum relative error:  0.098031044
    #'train' took 0.436834 s
    
    ##### samples = 1000, gauss_samp_number=80, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.8, lhs_ratio=1.0
    # Drift term:  [0.] + [0.88274264]x^1 + [0.]x^2 + [-0.9463102]x^3
    #Diffusion term of Brown Motion:  tensor([1.0507])
    #Diffusion term of Levy Noise:  tensor([1.0623])
    #Maximum relative error:  0.11725736
    #'train' took 0.382025 s
    
    #--------------------------------------------------------------
    ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.78, lhs_ratio=1.0
    # Drift term:  [0.] + [0.91075784]x^1 + [0.]x^2 + [-0.96263224]x^3
    #Diffusion term of Brown Motion:  tensor([1.0275])
    #Diffusion term of Levy Noise:  tensor([1.0783])
    #Maximum relative error:  0.08924216
    #'train' took 0.335864 s
    #-----------------------------------------------------------------
    
    ##### samples = 1000, gauss_samp_number=70, lam=0.0, STRidge_threshold=0.2, gauss_variance=0.75, lhs_ratio=1.1
    # Drift term:  [0.] + [0.97408795]x^1 + [0.]x^2 + [-0.9863201]x^3
    #Diffusion term of Brown Motion:  tensor([1.0905])
    #Diffusion term of Levy Noise:  tensor([1.0246])
    #Maximum relative error:  0.090510726
    #'train' took 0.323336 s
    
    
    
    
    
    
    
    
    ####samples = 5000, gauss_samp_number=100
    
    
    ####samples = 5000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.0,gauss_variance=0.85, lhs_ratio=1.0
    # Drift term:  [-0.05256595] + [0.9406588]x^1 + [0.00302821]x^2 + [-0.9825822]x^3
    #Diffusion term of Brown Motion:  tensor([1.0228])
    #Diffusion term of Levy Noise:  tensor([1.0030])
    #Maximum relative error:  0.059341192
    #'train' took 2.361221 s
    
    ####samples = 5000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.1,gauss_variance=0.85, lhs_ratio=1.0
    #Drift term:  [0.] + [0.9471043]x^1 + [0.]x^2 + [-0.98947424]x^3
    #Diffusion term of Brown Motion:  tensor([1.0189])
    #Diffusion term of Levy Noise:  tensor([0.9866])
    #Maximum relative error:  0.052895725
    #'train' took 3.479190 s
    
    #-----------------------------------------------------------------------
    ####samples = 5000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=0.85, lhs_ratio=1.1
    # Drift term:  [0.] + [0.95081174]x^1 + [0.]x^2 + [-0.99379426]x^3
    #Diffusion term of Brown Motion:  tensor([1.0112])
    #Diffusion term of Levy Noise:  tensor([0.9991])
    #Maximum relative error:  0.049188256
    #'train' took 4.129840 s
    #--------------------------------------------------------------------
    
    ####samples = 5000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.1,gauss_variance=0.85, lhs_ratio=1.2
    #Drift term:  [0.] + [0.9444229]x^1 + [0.]x^2 + [-0.98240316]x^3
    #Diffusion term of Brown Motion:  tensor([0.9861])
    #Diffusion term of Levy Noise:  tensor([1.0127])
    #Maximum relative error:  0.0555771
    #'train' took 4.185633 s
    
    
    
    
    
    
    
    ####samples = 10000, gauss_samp_number=100
    
    
    ####samples = 10000, gauss_samp_number=100, lam=0.02, STRidge_threshold=0.2,gauss_variance=0.85, lhs_ratio=1.0
    #Drift term:  [-0.0637259] + [0.97401935]x^1 + [0.03297497]x^2 + [-1.0103631]x^3
    #Diffusion term of Brown Motion:  tensor([1.0412])
    #Diffusion term of Levy Noise:  tensor([1.0130])
    #Maximum relative error:  0.041179776
    #'train' took 4.258664 s
    
    ####samples = 10000, gauss_samp_number=100, lam=0.02 or 0.0, STRidge_threshold=0.1,gauss_variance=0.85, lhs_ratio=1.0
    #Drift term:  [0.] + [0.9780249]x^1 + [0.]x^2 + [-1.0120925]x^3
    #Diffusion term of Brown Motion:  tensor([1.0371])
    #Diffusion term of Levy Noise:  tensor([0.9952])
    #Maximum relative error:  0.03708887
    #'train' took 6.037544 s
    
    ####samples = 10000, gauss_samp_number=100, lam=0.02 or 0.0, STRidge_threshold=0.1,gauss_variance=0.84, lhs_ratio=1.0
    # Drift term:  [0.] + [0.9783372]x^1 + [0.]x^2 + [-1.0123413]x^3
    #Diffusion term of Brown Motion:  tensor([1.0327])
    #Diffusion term of Levy Noise:  tensor([0.9984])
    #Maximum relative error:  0.03267467
    #'train' took 6.081634 s
    
    ####samples = 10000, gauss_samp_number=100, lam=0.02 or 0.0, STRidge_threshold=0.1,gauss_variance=0.84, lhs_ratio=0.9
    # Drift term:  [0.] + [0.9695422]x^1 + [0.]x^2 + [-1.0086426]x^3
    #Diffusion term of Brown Motion:  tensor([0.9957])
    #Diffusion term of Levy Noise:  tensor([1.0314])
    #Maximum relative error:  0.03144908
    #'train' took 6.465037 s
    
    #----------------------------------------------------------------------------
    ####samples = 10000, gauss_samp_number=100, lam=0.02 or 0.0, STRidge_threshold=0.1,gauss_variance=0.84, lhs_ratio=1.1
    # Drift term:  [0.] + [0.97887504]x^1 + [0.]x^2 + [-1.0136153]x^3
    #Diffusion term of Brown Motion:  tensor([1.0204])
    #Diffusion term of Levy Noise:  tensor([0.9996])
    #Maximum relative error:  0.021124959
    #'train' took 6.367489 s
    #--------------------------------------------------------------------------
    