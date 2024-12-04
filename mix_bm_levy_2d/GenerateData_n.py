# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:10:35 2023

@author: gly
"""

''''
Generate Data of    dX_t = drift(X_t) dt + diffusion(X_t) dB_t,  0<=t<=1, dim>=2
time_instants: E.g. torch.tensor([0, 0.2, 0.5, 1])
samples_num: E.g. 10000
dim: E.g. 3
drift_term: E.g.
    torch.tensor([[0, 1, 0, -1], [0, 1, 0, -1]), drift_independence=True -- that means drift = [x - x^3, y - y^3]
    torch.tensor([[0, 1, -1], [1, -2, -1]), drift_independence=False -- that means drift = [x - y, 1 - 2x - y] (only consider linear condition when False)
diffusion_term: E.g. torch.tensor([1, 2]) -- that means diffusion = diag{1, 2}
return data: [time, samples, dim]
'''

import numpy as np
import torch
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DataSet(object):
    def __init__(self, time_instants, true_dt, samples_num, dim, drift_term, diffusion_term, xi_term, alpha_levy,
                 initialization, drift_independence=True, explosion_prevention=False, trajectory_information=True):
        self.time_instants = time_instants
        self.true_dt = true_dt
        self.trajectory_information = trajectory_information
        if self.trajectory_information:
            self.samples_num = samples_num
        else:
            self.samples_num_true, self.samples_num = samples_num
        self.dim = dim
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.xi_term = xi_term
        self.alpha_levy = alpha_levy
        self.initialization = initialization
        self.drift_independence = drift_independence
        self.explosion_prevention = explosion_prevention

        self.shape_t = self.time_instants.shape[0]
        self.t_diff = torch.from_numpy(np.diff(self.time_instants.numpy()))

    def drift(self, x):
        y = 0
        for i in range(self.drift_term.shape[1]):
            y = y + self.drift_term[:, i] * x ** i
        return y
    
    def levy_variable(self):
        V = (torch.rand(self.samples_num,self.dim)*np.pi - np.pi/2)*0.9999
        U = torch.rand(self.samples_num,self.dim)*0.9999
        W = -torch.log(U+1e-5)
        X =  torch.sin(self.alpha_levy*V)/torch.cos(V)**(1/self.alpha_levy)*(torch.cos((1-self.alpha_levy)*V)/W)**((1-self.alpha_levy)/self.alpha_levy)
        return X
    

    def hat3d(self, x):
        # V = -5(x2+y2+z2) + (x2+y2+z2)2,   drift = - grad V
        norm = torch.sum(x ** 2, dim=1).unsqueeze(1)
        norm2 = norm.repeat(1, x.shape[1])
        return 10 * x - 4 * x * norm2

    @utils.timing
    def get_data(self, plot_hist=False):
        data = torch.zeros(self.shape_t, self.samples_num, self.dim)
        data[0, :, :] = self.initialization     # initial condition
        for i in range(self.shape_t - 1):
            if self.drift_independence:
                data[i + 1, :, :] = data[i, :, :] + self.drift(data[i, :, :]) * self.t_diff[i]
            elif self.drift_term.shape == torch.Size([self.dim, self.dim + 1]):     # linear
                data[i + 1, :, :] = data[i, :, :] + torch.mm(data[i, :, :], torch.t(self.drift_term[:, 1:])) * self.t_diff[i]
            elif self.drift_term.shape == torch.Size([3, 20]) or self.drift_term.shape == torch.Size([2, 10]):      # 3d/2d-hat
                data[i + 1, :, :] = data[i, :, :] + self.hat3d(data[i, :, :]) * self.t_diff[i]
            else:
                print("The input dimension is incorrect when drift_independence!")
                
            # 这里的if以及两个elif对应的高维下drift项的三种情形：
            
            #1、drift的每个维度只包含自己维度的变量，代码里用drift_independence这个变量设为True来表示这个情形。这种情况下，你输入的比如drift=[1,2,3; 4,5,6]表示真实的drift项是[1+2x+3x^2; 4+5y+6y^2]；
            #2、drift的每个维度还包含其他维度的变量，但还是线性关系，代码里用drift_term的shape是否为[dim, dim+1]来判断是否是这个情形。这种情况下，你输入的比如drift=[1,2,3; 4,5,6]表示真实的drift项是[1+2x+3y; 4+5x+6y]；
            #3、drift的每个维度还包含其他维度的变量，且不是线性关系，这个情况drift的输入就很复杂了，我们只考虑了三维帽子和二维帽子两种情况，二维帽子drift=[10x-4x^3-4xy^2; 10y-4x^2y-4y^3]，
               #实际输入是drift=[0,10,0,0,0,0,-4,0,-4,0; 0,0,10,0,0,0,0,-4,0,-4]，它的size是[2,10]，三维帽子同理
                
            data[i + 1, :, :] = data[i + 1, :, :] + self.diffusion_term.repeat(self.samples_num, 1) * torch.sqrt(self.t_diff[i]) * torch.randn(self.samples_num, self.dim)\
                + torch.pow(self.t_diff[i], 1/self.alpha_levy) * self.levy_variable() * \
                    torch.mm(torch.ones(self.samples_num, self.dim), torch.diag(self.xi_term))
                #+ self.xi_term.repeat(self.samples_num, 1) * torch.pow(self.t_diff[i], 1/self.alpha_levy) * torch.randn(self.samples_num, self.dim)
            if self.explosion_prevention:
                data[i + 1, :, :][data[i + 1, :, :] < 0] = 0
                
                
        if plot_hist:
            for i in range(self.dim):
                plt.figure()
                plt.hist(x=data[-1, :, i].numpy(), bins=100, range=[data.min().numpy(), data.max().numpy()], density=True)
            # plt.figure()
            # plt.hist(x=data[-1, :, 0].numpy(), bins=50, range=[-2.2, 2.2], density=True)
            # plt.figure()
            # plt.hist(x=data[-1, :, 1].numpy(), bins=50, range=[-2.2, 2.2], density=True)
            # plt.figure()
            # plt.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=50, range=[[-2.2, 2.2], [-2.2, 2.2]],
            #            density=True)
            # fig = plt.figure()
            # X, Y = torch.meshgrid(torch.linspace(-2.3, 2.3, 100), torch.linspace(-2.3, 2.3, 100))
            # Z = -5 * (X ** 2 + Y ** 2) + (X ** 2 + Y ** 2) ** 2
            # ax = Axes3D(fig)
            # ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), rstride=1, cstride=1, cmap='rainbow')
            # plt.figure()
            # x = torch.linspace(-2.3, 2.3, 100)
            # plt.plot(x, -5 * x ** 2 + x ** 4)
            plt.show()
        index0 = np.arange(0, self.shape_t, int(self.true_dt // self.t_diff[0]))
        data0 = data[index0, :, :]
        if self.trajectory_information:
            return data0
        else:
            data1 = torch.zeros(data0.shape[0], self.samples_num_true, data0.shape[2])
            for i in range(data0.shape[0]):
                index1 = np.arange(self.samples_num)
                np.random.shuffle(index1)
                data1[i, :, :] = data0[i, index1[0: self.samples_num_true], :]
            return data1


if __name__ == '__main__':
    torch.manual_seed(100)
    # drift = torch.tensor([[0, -0.5, 0, 0], [0, 0, -0.7, 0], [0, 0, 0, -1]])
    # drift = torch.tensor([[0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1]])
    # drift = torch.tensor([[0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, -4, 0, -4, 0, 0, 0, 0],
    #                       [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4, 0],
    #                       [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, -4, 0, -4]])
    drift = torch.tensor([[0, 10, 0, 0, 0, 0, -4, 0, -4, 0], [0, 0, 10, 0, 0, 0, 0, -4, 0, -4]])
    diffusion = torch.tensor([1, 1])
    xi = torch.tensor([1,1],dtype=torch.float)
    dataset = DataSet(torch.linspace(0, 10, 10001), true_dt=0.1, samples_num=2000, dim=2,
                      drift_term=drift, diffusion_term=diffusion, xi_term = xi, alpha_levy = 3/2, initialization=torch.rand(2000, 2) + 1,
                      drift_independence=False, explosion_prevention=False,trajectory_information=True)
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())
    # plt.figure()
    # plt.hist(x=data[-1, :, 0].numpy(), bins=50, range=[-2.2, 2.2],  density=True)
    # plt.figure()
    # plt.hist(x=data[-1, :, 1].numpy(), bins=50, range=[-2.2, 2.2],  density=True)
    # plt.figure()
    # plt.hist2d(x=data[-1, :, 0].numpy(), y=data[-1, :, 1].numpy(), bins=50, range=[[-2.2, 2.2], [-2.2, 2.2]], density=True)
    # fig = plt.figure()
    # X, Y = torch.meshgrid(torch.linspace(-2.3, 2.3, 100), torch.linspace(-2.3, 2.3, 100))
    # Z = -5 * (X ** 2 + Y ** 2) + (X ** 2 + Y ** 2) ** 2
    # ax = Axes3D(fig)
    # ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), rstride=1, cstride=1, cmap='rainbow')
    # plt.figure()
    # x = torch.linspace(-2.3, 2.3, 100)
    # plt.plot(x, -5 * x ** 2 + x ** 4)
    # plt.show()