# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 11:48:23 2023

@author: gly
"""

import torch
import torch.nn as nn
import numpy as np
import utils
import scipy.io
import scipy.special as sp
import math
from sympy import symbols, integrate, cos

import matplotlib.pyplot as plt


x=np.linspace(-1.5, 1.5)
y=-2*x*np.exp(-x**2)
plt.figure()
plt.plot(x,y,color = 'black')


drift_6 = [-0.01472363, -0.91135603, 0.08226614, 0.2745569, -0.00886573, -0.02043224, 0.00036193]
y6 = drift_6[0] + drift_6[1]*x + drift_6[2]*x**2 + drift_6[3]*x**3 + drift_6[4]*x**4 + drift_6[5]*x**5 + drift_6[6]*x**6
plt.plot(x,y6,linestyle='--')

# Drift term:  [-0.06518914] + [-1.1244197]x^1 + [0.07053359]x^2 + [0.36674154]x^3 + [-0.03195769]x^4 + [-0.03816029]x^5 + [0.00240541]x^6 + [0.00095004]x^7 + [-7.0805596e-05]x^8 + [-3.8871717e-06]x^9

drift_7 = [-0.0180993, -1.1519837, -0.04395188, 0.44346815, 0.01883244, -0.04915128, -0.00206866, 0.0010749]
y7 = drift_7[0] + drift_7[1]*x + drift_7[2]*x**2 + drift_7[3]*x**3 + drift_7[4]*x**4 + drift_7[5]*x**5 + drift_7[6]*x**6 + drift_7[7]*x**7
plt.plot(x,y7,linestyle='--')


drift_8 = [-0.01834646, -1.1515284, -0.03952326, 0.444074, 0.01694677, -0.04927403, -0.00185248, 0.00108577, -5.152078e-06]
y8 = drift_8[0] + drift_8[1]*x + drift_8[2]*x**2 + drift_8[3]*x**3 + drift_8[4]*x**4 + drift_8[5]*x**5 + drift_8[6]*x**6 + drift_8[7]*x**7 + drift_8[8]*x**8
plt.plot(x,y8,linestyle='--')


#drift_9 = [-0.01752794, -1.3023754, -0.05727665, 0.630604, 0.03071469, -0.10139746, -0.00496915, 0.00533971, 0.00015979, -8.415742e-05]
drift_9 = [-0.01950205, -1.3609132, -0.07663355, 0.682524, 0.03165837, -0.1186888, -0.00517874, 0.00702644, 0.00018803, -0.00012683]
y9 = drift_9[0] + drift_9[1]*x + drift_9[2]*x**2 + drift_9[3]*x**3 + drift_9[4]*x**4 + drift_9[5]*x**5 + drift_9[6]*x**6 + drift_9[7]*x**7 + drift_9[8]*x**8 + drift_9[9]*x**9
plt.plot(x,y9,linestyle='--')


drift_10 = [-0.01912476, -1.3008037, -0.02403562, 0.63799775, -0.00168368, -0.10222293, 0.00312856, 0.0054085, -0.0003479, -9.310986e-05, 8.595213e-06]
y10 = drift_10[0] + drift_10[1]*x + drift_10[2]*x**2 + drift_10[3]*x**3 + drift_10[4]*x**4 + drift_10[5]*x**5 + drift_10[6]*x**6 + drift_10[7]*x**7 + drift_10[8]*x**8 + drift_10[9]*x**9 + drift_10[10]*x**10
plt.plot(x,y10,linestyle='--')


drift_11 = [-0.01508181, -1.3371543, -0.07138138, 0.6960255, 0.03635037, -0.12574202, -0.00477249, 0.00846437, 0.00021501, -0.00023401, -2.7374867e-06, 2.1936246e-06]
y11 = drift_11[0] + drift_11[1]*x + drift_11[2]*x**2 + drift_11[3]*x**3 + drift_11[4]*x**4 + drift_11[5]*x**5 + drift_11[6]*x**6 + drift_11[7]*x**7 + drift_11[8]*x**8 + drift_11[9]*x**9 + drift_11[10]*x**10 + drift_11[11]*x**11
plt.plot(x,y11,linestyle='--')


 ################ basis_order= 12  
   ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
     #   Drift term:  [-0.02512893] + [-1.4207127]x^1 + [-0.01113998]x^2 + [0.8000165]x^3 + [-0.03089279]x^4 + [-0.17018779]x^5 + [0.01146896]x^6 + [0.01443475]x^7 + [-0.00132237]x^8 + [-0.00053922]x^9 + [5.7019322e-05]x^10 + [7.1716477e-06]x^11 + [-8.2936486e-07]x^12
     #Diffusion term of Levy Noise:  tensor([0.9186])
     #L2 error:  0.08294476
     #'train' took 4.292853 s
   

 
plt.xlabel('x', fontsize=12)
plt.ylabel('Drift', fontsize=12)
plt.tick_params(labelsize=12)
plt.legend(['True', "6-th", "7-th", "8-th", "9-th", "10-th", "11-th"], fontsize=12)
plt.show() 


 ################ basis_order=6     
   ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
     #Drift term:  [-0.01472363] + [-0.91135603]x^1 + [0.08226614]x^2 + [0.2745569]x^3 + [-0.00886573]x^4 + [-0.02043224]x^5 + [0.00036193]x^6
     #Diffusion term of Levy Noise:  tensor([0.8643])
     #L2 error:  0.22371158
     #'train' took 4.161567 s   
    
   ################ basis_order=7     
     ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
       #  Drift term:  [-0.0180993] + [-1.1519837]x^1 + [-0.04395188]x^2 + [0.44346815]x^3 + [0.01883244]x^4 + [-0.04915128]x^5 + [-0.00206866]x^6 + [0.0010749]x^7
       #Diffusion term of Levy Noise:  tensor([0.8958])
       #L2 error:  0.124843724
       #'train' took 4.283743 s
  
   ################ basis_order=8     
     ###sample = 10000, gauss_samp_number=20, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
       #  Drift term:  [-0.01834646] + [-1.1515284]x^1 + [-0.03952326]x^2 + [0.444074]x^3 + [0.01694677]x^4 + [-0.04927403]x^5 + [-0.00185248]x^6 + [0.00108577]x^7 + [-5.152078e-06]x^8
       #Diffusion term of Levy Noise:  tensor([0.8957])
       #L2 error:  0.12331158
       #'train' took 5.209281 s
      
    
  ################ basis_order=9     
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      # Drift term:  [-0.01318259] + [-1.2411803]x^1 + [-0.08882214]x^2 + [0.5512496]x^3 + [0.04585158]x^4 + [-0.07859498]x^5 + [-0.0052851]x^6 + [0.00344506]x^7 + [0.00012925]x^8 + [-4.8753765e-05]x^9
      #Diffusion term of Levy Noise:  tensor([0.9029])
      #L2 error:  0.09240185
      #'train' took 4.200607 s
    
  ################ basis_order= 10  
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
     #Drift term:  [-0.01912476] + [-1.3008037]x^1 + [-0.02403562]x^2 + [0.63799775]x^3 + [-0.00168368]x^4 + [-0.10222293]x^5 + [0.00312856]x^6 + [0.0054085]x^7 + [-0.0003479]x^8 + [-9.310986e-05]x^9 + [8.595213e-06]x^10
     #Diffusion term of Levy Noise:  tensor([0.9078])
     #L2 error:  0.06298099
     #'train' took 4.640334 s
   
    
  ################ basis_order= 11  
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.52, lhs = 1.0
      #  Drift term:  [-0.01508181] + [-1.3371543]x^1 + [-0.07138138]x^2 + [0.6960255]x^3 + [0.03635037]x^4 + [-0.12574202]x^5 + [-0.00477249]x^6 + [0.00846437]x^7 + [0.00021501]x^8 + [-0.00023401]x^9 + [-2.7374867e-06]x^10 + [2.1936246e-06]x^11
      #Diffusion term of Levy Noise:  tensor([0.9097])
      #L2 error:  0.058145747
      #'train' took 4.656719 s
    
  ################ basis_order= 12  
    ###sample = 10000, gauss_samp_number=50, lam=0.0, STRidge_threshold=0.0, gauss_var = 0.55, lhs = 1.0
      #   Drift term:  [-0.02512893] + [-1.4207127]x^1 + [-0.01113998]x^2 + [0.8000165]x^3 + [-0.03089279]x^4 + [-0.17018779]x^5 + [0.01146896]x^6 + [0.01443475]x^7 + [-0.00132237]x^8 + [-0.00053922]x^9 + [5.7019322e-05]x^10 + [7.1716477e-06]x^11 + [-8.2936486e-07]x^12
      #Diffusion term of Levy Noise:  tensor([0.9186])
      #L2 error:  0.08294476
      #'train' took 4.292853 s
    
    
 