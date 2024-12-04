# -*- coding: utf-8 -*-
"""
Add noise to data

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from generate_data_1d import DataSet
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
        
        #basis_order = 1 用1阶多项式展开
        #self.basis_number 展开有多少项 一维时，basis number = basis order;  二维时，basis order = 2, basis number = 6(1, x,y,x^2, y^2, xy)


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
            Xi = torch.ones(X.size(0),1)
            
            #assert basis_count2 == self.basis2_number
            basis2.append(Xi)
            basis_xi = torch.stack(basis2)
        print("basis_xi", basis_xi.shape)
            
                
            
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
    #t = torch.tensor([0.1, 0.3, 0.5])
    #t = torch.tensor([0.2, 0.5, 1.0])
    #t = torch.linspace(0,1,10)
    #t = torch.tensor([0.1, 0.4, 0.7, 1])

    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #t = torch.tensor([0.2, 0.5, 1])
    # data = scipy.io.loadmat('./data/data1d.mat')['bb'].astype(np.float32)
    # data = torch.tensor(data).unsqueeze(-1)
    drift = torch.tensor([0, 1, 0, -1])
    diffusion = torch.tensor([1.0])
    xi = torch.tensor([1.0])
    samples = 10000
    dataset = DataSet(t, dt=0.0001, samples_num=samples, dim=1,
                      drift_term=drift, diffusion_term=diffusion, xi_term=xi, alpha_levy = 3/2, initialization=torch.normal(0, 0.2,[samples, 1]),
                      explosion_prevention=False) #initialization=torch.randint(-1000,1000,[10000, 1])
    data = dataset.get_data(plot_hist=False)
    print("data: ", data.shape, data.max(), data.min())
    
    scale = 0.0
    
    """
    Aadditive noise: uniform or Gaussian
    """
    
    noise = (torch.rand((data.shape[0], data.shape[1], data.shape[2]))*2 -1) *scale
    #noise = torch.randn((data.shape[0], data.shape[1], data.shape[2])) *scale
    noisydata = data + noise
    
    """
    Multiplicative noise: uniform or Gaussian
    """
    #noise = (torch.rand((data.shape[0], data.shape[1], data.shape[2]))*2 -1) *scale
    #noise = torch.randn((data.shape[0], data.shape[1], data.shape[2])) *scale
    #noisydata = data *(1+ noise)
    
    """
    Learn
    """

    testFunc = Gaussian
    model = Model(t, noisydata, testFunc)
    model.compile(basis_order=3,basis_xi_order=1, gauss_variance=0.7, type='LMM_2_nonequal', drift_term=drift, diffusion_term=diffusion, xi_term=xi,
                  gauss_samp_way='lhs', lhs_ratio=1.0)
    model.train(gauss_samp_number=100, lam=0.0, STRidge_threshold=0.05)
    
    
 ################## No noise:
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.7, lhs = 1.0 , scale = 0.05, noisetype = uniform 
    # Drift term:  [0.] + [0.9680448]x^1 + [0.]x^2 + [-0.9823367]x^3
    #Diffusion term of Brown Motion:  tensor([1.0022])
    #Diffusion term of Levy Noise:  tensor([0.9904])
    #Maximum relative error:  0.031955183
    #'train' took 9.259703 s 
    
    
##################Aadditive noise:   
   
  #####################################

  #### (3)  noise scale = 0.01

  #####################################
  
  ############# (3.1) Gauss noise  
    
   ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
   # gauss_var = 0.7, lhs = 0.85 , scale = 0.01, noisetype = uniform 
   #Drift term:  [0.] + [0.97314626]x^1 + [0.]x^2 + [-1.0004219]x^3
   #Diffusion term of Brown Motion:  tensor([1.0259])
   #Diffusion term of Levy Noise:  tensor([0.9820])
   #Maximum relative error:  0.02685374
   #'train' took 7.063641 s
   
   
   ### 0.05
   ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
   # gauss_var = 0.7, lhs = 0.85 , scale = 0.05, noisetype = uniform 
   #Drift term:  [0.] + [0.9703035]x^1 + [0.]x^2 + [-0.9954403]x^3
   #Diffusion term of Brown Motion:  tensor([1.0380])
   #Diffusion term of Levy Noise:  tensor([0.9717])
   #Maximum relative error:  0.038016558
   #'train' took 7.063124 s
   
   

  ############# (3.2) Uniform noise  
    
   ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
   # gauss_var = 0.75, lhs = 0.85 , scale = 0.01, noisetype = uniform 
   # Drift term:  [0.] + [0.9744522]x^1 + [0.]x^2 + [-1.0000483]x^3
   #Diffusion term of Brown Motion:  tensor([1.0254])
   #Diffusion term of Levy Noise:  tensor([0.9810])
   #Maximum relative error:  0.025547802
   #'train' took 8.499127 s
 
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.72, lhs = 0.85 , scale = 0.01, noisetype = uniform 
    #Drift term:  [0.] + [0.9703722]x^1 + [0.]x^2 + [-0.9956715]x^3
    #Diffusion term of Brown Motion:  tensor([1.0298])
    #Diffusion term of Levy Noise:  tensor([0.9773])
    #Maximum relative error:  0.02976191
    #'train' took 7.815923 s
    
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.7, lhs = 0.85 , scale = 0.01, noisetype = uniform 
    #Drift term:  [0.] + [0.97018665]x^1 + [0.]x^2 + [-0.9956801]x^3
    #Diffusion term of Brown Motion:  tensor([1.0246])
    #Diffusion term of Levy Noise:  tensor([0.9808])
    #Maximum relative error:  0.02981335
    #'train' took 8.640704 s
    
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.7, lhs = 0.8, scale = 0.01, noisetype = uniform 
    #Drift term:  [0.] + [0.9891266]x^1 + [0.]x^2 + [-1.0019461]x^3
    #Diffusion term of Brown Motion:  tensor([1.0110])
    #Diffusion term of Levy Noise:  tensor([0.9883])
    #Maximum relative error:  0.011715472
    #'train' took 7.568281 s
    
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.7, lhs = 0.75, scale = 0.01, noisetype = uniform 
    # Drift term:  [0.] + [0.96043396]x^1 + [0.]x^2 + [-0.99255115]x^3
    #Diffusion term of Brown Motion:  tensor([1.0260])
    #Diffusion term of Levy Noise:  tensor([0.9860])
    #Maximum relative error:  0.03956604
    #'train' took 8.169860 s
    
    
  
    #####################################

    #### (4)  noise scale = 0.1

    #####################################
    
    ############# (4.1) Gauss noise  
      
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.7, lhs = 0.85 , scale = 0.1, noisetype = uniform 
     # Drift term:  [0.] + [0.90452844]x^1 + [0.]x^2 + [-0.9626875]x^3
     #Diffusion term of Brown Motion:  tensor([1.0372])
     #Diffusion term of Levy Noise:  tensor([0.9907])
     #Maximum relative error:  0.09547156
     #'train' took 8.401492 s

    ############# (4.2) Uniform noise  
      
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.7, lhs = 0.85 , scale = 0.1, noisetype = uniform 
     # Drift term:  [0.] + [0.955295]x^1 + [0.]x^2 + [-0.98406327]x^3
     #Diffusion term of Brown Motion:  tensor([1.0261])
     #Diffusion term of Levy Noise:  tensor([0.9803])
    # Maximum relative error:  0.044704974
     #'train' took 8.799814 s
     
     
     
     #####################################

     #### (5)  noise scale = 0.2, 0.3 for Uniform noise

     #####################################
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.7, lhs = 0.85 , scale = 0.2, noisetype = uniform 
     # Drift term:  [0.] + [0.8959093]x^1 + [0.]x^2 + [-0.9417649]x^3
     #Diffusion term of Brown Motion:  tensor([1.0380])
     #Diffusion term of Levy Noise:  tensor([0.9780])
     #Maximum relative error:  0.10409069
     #'train' took 22.928865 s
     
     
     ## 0.3
     
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.7, lhs = 0.85 , scale = 0.3, noisetype = uniform 
     # Drift term:  [0.] + [0.80266106]x^1 + [0.]x^2 + [-0.877913]x^3
     #Diffusion term of Brown Motion:  tensor([1.0622])
     #Diffusion term of Levy Noise:  tensor([0.9727])
     #Maximum relative error:  0.19733894
     #'train' took 7.957278 s
     
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.75, lhs = 1.0, scale = 0.3, noisetype = uniform 
     #Drift term:  [0.] + [0.8000488]x^1 + [0.]x^2 + [-0.86589456]x^3
     #Diffusion term of Brown Motion:  tensor([1.0328])
     #Diffusion term of Levy Noise:  tensor([0.9902])
     #Maximum relative error:  0.19995117
     #'train' took 7.343616 s
     
     ##########sample = 10000, gauss_samp_number=80, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.75, lhs = 1.0 , scale = 0.3, noisetype = uniform 
     # Drift term:  [0.] + [0.81431377]x^1 + [0.]x^2 + [-0.8647381]x^3
     #Diffusion term of Brown Motion:  tensor([1.0440])
     #Diffusion term of Levy Noise:  tensor([0.9694])
     #Maximum relative error:  0.18568623
     #'train' took 5.532959 s
     
     ##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.75, lhs = 1 , scale = 0.3, noisetype = uniform 
     # Drift term:  [0.] + [0.83764625]x^1 + [0.]x^2 + [-0.87756044]x^3
     #Diffusion term of Brown Motion:  tensor([1.0758])
     #Diffusion term of Levy Noise:  tensor([0.9471])
     #Maximum relative error:  0.16235375
     #'train' took 3.163567 s
     
     ##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.72, lhs = 1 , scale = 0.3, noisetype = uniform 
     #Drift term:  [0.] + [0.83715]x^1 + [0.]x^2 + [-0.8777566]x^3
     #Diffusion term of Brown Motion:  tensor([1.0612])
     #Diffusion term of Levy Noise:  tensor([0.9584])
     #Maximum relative error:  0.16285002
     #'train' took 3.413159 s
     
     
     
     
     
     
     
     
 ################## Multiplicative noise       
     
     #####################################

     #### (3)  noise scale = 0.01, 0.05

     #####################################
     
     ############# (3.1) Gauss noise  
       
      ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
      # gauss_var = 0.7, lhs = 0.85 , scale = 0.01, noisetype = uniform 
      #Drift term:  [0.] + [0.9694472]x^1 + [0.]x^2 + [-0.9985294]x^3
      #Diffusion term of Brown Motion:  tensor([1.0244])
      #Diffusion term of Levy Noise:  tensor([0.9825])
      #Maximum relative error:  0.030552804
      #'train' took 23.447188 s
      
     ############# (3.2) Uniform noise  
       
      ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
      # gauss_var = 0.7, lhs = 0.85 , scale = 0.01, noisetype = uniform 
      # Drift term:  [0.] + [0.9717453]x^1 + [0.]x^2 + [-0.99916315]x^3
      #Diffusion term of Brown Motion:  tensor([1.0258])
      #Diffusion term of Levy Noise:  tensor([0.9814])
      #Maximum relative error:  0.028254688
      #'train' took 7.769535 s
      
      ### 0.05
      
       ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
       # gauss_var = 0.7, lhs = 0.85 , scale = 0.01, noisetype = uniform 
       # Drift term:  [0.] + [0.9485816]x^1 + [0.]x^2 + [-0.9885878]x^3
       #Diffusion term of Brown Motion:  tensor([1.0345])
       #Diffusion term of Levy Noise:  tensor([0.9796])
       #Maximum relative error:  0.051418424
     
        ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
        # gauss_var = 0.7, lhs = 0.75 , scale = 0.01, noisetype = uniform 
        # Drift term:  [0.] + [0.93996173]x^1 + [0.]x^2 + [-0.9850319]x^3
        #Diffusion term of Brown Motion:  tensor([1.0347])
        #Diffusion term of Levy Noise:  tensor([0.9862])
        #Maximum relative error:  0.06003827
        #'train' took 6.539582 s
       
       ##########sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold= 0.05,
       # gauss_var = 0.7, lhs = 0.85 , scale = 0.01, noisetype = uniform 
       # Drift term:  [0.] + [0.947115]x^1 + [0.]x^2 + [-0.98300177]x^3
       #Diffusion term of Brown Motion:  tensor([1.0140])
       #Diffusion term of Levy Noise:  tensor([0.9958])
       #Maximum relative error:  0.052884996
       #'train' took 3.572421 s
       
     
       #####################################

       #### (4)  noise scale = 0.1

       #####################################
       
       ############# (4.1) Gauss noise  
         
        ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
        # gauss_var = 0.7, lhs = 0.85 , scale = 0.1, noisetype = uniform 
        # Drift term:  [0.] + [0.79119176]x^1 + [0.]x^2 + [-0.9289798]x^3
        #Diffusion term of Brown Motion:  tensor([0.9096])
        #Diffusion term of Levy Noise:  tensor([1.0970])
        #Maximum relative error:  0.20880824
        #'train' took 7.731515 s
        

       ############# (4.2) Uniform noise  
         
        ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
        # gauss_var = 0.75, lhs = 0.85 , scale = 0.1, noisetype = uniform 
        # Drift term:  [0.] + [0.89441013]x^1 + [0.]x^2 + [-0.9648336]x^3
        #Diffusion term of Brown Motion:  tensor([1.0178])
        #Diffusion term of Levy Noise:  tensor([1.0010])
        #Maximum relative error:  0.10558987
        #'train' took 7.360212 s
          
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.78, lhs = 0.85 , scale = 0.2, noisetype = uniform 
     # Drift term:  [0.] + [0.70432305]x^1 + [0.]x^2 + [-0.8786564]x^3
     #Diffusion term of Brown Motion:  tensor([0.9764])
     #Diffusion term of Levy Noise:  tensor([1.0608])
     #Maximum relative error:  0.29567695
     #'train' took 7.262264 s
     
     
    #####################################

    #### (5)  noise scale = 0.2, 0.3

    #####################################  
     
    #############  Uniform noise  
    
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.8, lhs = 0.85 , scale = 0.2, noisetype = uniform 
    # Drift term:  [0.] + [0.7008838]x^1 + [0.]x^2 + [-0.875999]x^3
    #Diffusion term of Brown Motion:  tensor([0.9894])
    #Diffusion term of Levy Noise:  tensor([1.0523])
    #Maximum relative error:  0.2991162
    #'train' took 6.277540 s
    
    ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
    #gauss_var = 0.8, lhs = 0.75 and 1.0(drift and total误差差不多，以0.75为例) , scale = 0.2, noisetype = uniform 
    # Drift term:  [0.] + [0.6914]x^1 + [0.]x^2 + [-0.86835885]x^3
    #Diffusion term of Brown Motion:  tensor([0.9865])
    #Diffusion term of Levy Noise:  tensor([1.0621])
    #Maximum relative error:  0.3086
    #'train' took 7.468413 s
    
    ##########sample = 10000, gauss_samp_number=120, lam=0.0, STRidge_threshold= 0.05,
    # gauss_var = 0.8, lhs = 0.85 , scale = 0.2, noisetype = uniform 
    #Drift term:  [0.] + [0.70166636]x^1 + [0.]x^2 + [-0.873209]x^3
    #Diffusion term of Brown Motion:  tensor([0.9842])
    #Diffusion term of Levy Noise:  tensor([1.0554])
    #Maximum relative error:  0.29833364
    #'train' took 8.517650 s
    
    
    ###0.3
    
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.75, lhs = 0.85 , scale = 0.3, noisetype = uniform 
     # 不收敛
     
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.85, lhs = 0.85 , scale = 0.3, noisetype = uniform 
     # rift term:  [-0.05092943] + [0.40183854]x^1 + [0.05283279]x^2 + [-0.7362181]x^3
     #Diffusion term of Brown Motion:  tensor([0.9963])
     #Diffusion term of Levy Noise:  tensor([1.0975])
     #Maximum relative error:  0.59816146
     #'train' took 7.080308 s
     
     
     ##########sample = 10000, gauss_samp_number=150, lam=0.0, STRidge_threshold= 0.05,
     # gauss_var = 0.85, lhs = 0.85 , scale = 0.3, noisetype = uniform 
     # Drift term:  [0.] + [0.39283964]x^1 + [0.]x^2 + [-0.7296093]x^3
     #Diffusion term of Brown Motion:  tensor([0.9754])
     #Diffusion term of Levy Noise:  tensor([1.1053])
     #Maximum relative error:  0.60716033
     #'train' took 8.633171 s
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     #####################################

     #### (1)  noise scale = 0.0001

     #####################################

     ############# (1.1) Gauss noise

     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2,
     # gauss_var = 0.75, lhs = 0.75 , scale = 0.0001, noisetype = gauss 
     # Drift term:  [0.] + [0.96014893]x^1 + [0.]x^2 + [-0.99438334]x^3
     #Diffusion term of Brown Motion:  tensor([1.0459])
     #Diffusion term of Levy Noise:  tensor([0.9734])
     #Maximum relative error:  0.04589069
     #'train' took 7.496523 s

     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2,
     # gauss_var = 0.72, lhs = 0.75 , scale = 0.0001, noisetype = gauss 
     # Drift term:  [0.] + [0.96179974]x^1 + [0.]x^2 + [-0.9961811]x^3
     #Diffusion term of Brown Motion:  tensor([1.0321])
     #Diffusion term of Levy Noise:  tensor([0.9837])
     #Maximum relative error:  0.03820026
     #'train' took 8.707478 s

     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2,
     # gauss_var = 0.72, lhs = 0.85 , scale = 0.0001, noisetype = gauss 
     # Drift term:  [0.] + [0.97456765]x^1 + [0.]x^2 + [-1.0004126]x^3
     #Diffusion term of Brown Motion:  tensor([1.0313])
     #Diffusion term of Levy Noise:  tensor([0.9775])
     #Maximum relative error:  0.031261444
     #'train' took 9.935604 s

     #--------------------------------------------
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2 or 0.05,
     # gauss_var = 0.7, lhs = 0.85 , scale = 0.0001, noisetype = gauss 
     # Drift term:  [0.] + [0.97442544]x^1 + [0.]x^2 + [-1.0004461]x^3
     #Diffusion term of Brown Motion:  tensor([1.0259])
     #Diffusion term of Levy Noise:  tensor([0.9811])
     #Maximum relative error:  0.02593863
     #'train' took 10.513682 s
     #-------------------------------------------

     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2,
     # gauss_var = 0.7, lhs = 0.9 , scale = 0.0001, noisetype = gauss 
     #Drift term:  [0.] + [0.9667335]x^1 + [0.]x^2 + [-1.0018362]x^3
     #Diffusion term of Brown Motion:  tensor([0.9904])
     #Diffusion term of Levy Noise:  tensor([1.0101])
     #Maximum relative error:  0.033266485





     ############# (1.2) Uniform noise


     #--------------------------------------------
     ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.2 or 0.05,
     # gauss_var = 0.7, lhs = 0.85 , scale = 0.0001, noisetype = uniform 
     # Drift term:  [0.] + [0.97443414]x^1 + [0.]x^2 + [-1.0004394]x^3
     #Diffusion term of Brown Motion:  tensor([1.0259])
     #Diffusion term of Levy Noise:  tensor([0.9811])
     #Maximum relative error:  0.02593422
     #'train' took 6.569951 s


#####################################

#### (2)  noise scale = 0.001

#####################################

############# (2.2) Uniform noise

##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.05,
# gauss_var = 0.7, lhs = 0.85 , scale = 0.001, noisetype = uniform 
#Drift term:  [0.] + [0.97445357]x^1 + [0.]x^2 + [-1.0004153]x^3
#Diffusion term of Brown Motion:  tensor([1.0259])
#Diffusion term of Levy Noise:  tensor([0.9811])
#Maximum relative error:  0.025884986
#'train' took 8.607564 s


###################### 乘性噪音

   #####################################

   #### (2)  noise scale = 0.001

   #####################################
   
   ############# (2.1) Gauss noise

   ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.05,
   # gauss_var = 0.7, lhs = 0.85 , scale = 0.001, noisetype = uniform 
   #
   

   ############# (2.2) Uniform noise

   ##########sample = 10000, gauss_samp_number=100, lam=0.0, STRidge_threshold=0.05,
   # gauss_var = 0.7, lhs = 0.85 , scale = 0.001, noisetype = uniform 
   #



      