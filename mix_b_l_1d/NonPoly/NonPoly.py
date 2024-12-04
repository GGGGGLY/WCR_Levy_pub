# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:02:39 2023

@author: gly
"""

import matplotlib.pyplot as plt
import numpy as np
#import math

x=np.linspace(-3,3,50)
y = -2 * x * np.exp(-x**2)
plt.figure(num=3,figsize=(8,5))
# 绘制 y=x^2 的图像，设置 color 为 red，线宽度是 1，线的样式是 --
plt.plot(x,y,color='red',linewidth=1.0,linestyle='--')
