# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:02:27 2023

@author: gly
"""

import numpy as np
from sympy import *
import math
import matplotlib.pyplot as plt
 
#n代表n-1次逼近
n = 4
 
#建立hilbert矩阵
H = np.zeros((n,n))
 
for i in range(n):
    for j in range(n):
        H[i][j] = 1/(i+j+1)
 
 
f = np.zeros((n,1))
 
for i in range(n):
    x = symbols("x")
    f[i][0] = integrate((math.e**x)*(x**i),(x,0,1))
 
#a为系数向量
a = np.linalg.inv(H).dot(f)
print(a)
 
x = np.linspace(0,1,20)
y = -2*x*math.e**(x)
 
def fun(a,x):
    return a[0]+x*a[1]+x**2*a[2]+x**3*a[3]
y_pred = fun(a,x)
 
plt.plot(x,y,'b')
plt.plot(x,y_pred,'r')