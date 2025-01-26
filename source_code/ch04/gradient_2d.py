# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成一个形状和 x 相同、所有元素都为 0 的数组
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1) # 生成 网格点的坐标矩阵。它主要用于在 2D 或更高维空间中生成坐标网格，这在可视化、插值和数值计算（如曲面绘制、梯度计算等）中非常常用
    
    X = X.flatten() # 用于将多维数组转换为 一维数组，即将数组“展平”。它会按照元素存储顺序（默认按行优先顺序，C 风格）依次排列数组中的所有元素。
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X, Y]) )
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

# 由全部变量的偏导数汇总 而成的向量称为梯度(gradient)

# 像这样，我们可以计算 (x0, x1) 在各点处的梯度。上例中，点 (3, 4) 处的 梯度是 (6, 8)、点 (0, 2) 处的梯度是 (0, 4)、点 (3, 0) 处的梯度是 (6, 0)。这个 梯度意味着什么呢?为了更好地理解，我们把 的梯度 画在图上。不过，这里我们画的是元素值为负梯度B 的向量
# 虽然求到的值是[6.0000000000037801, 7.9999999999991189]，但实际输出的是[6., 8.]。 这是因为在输出 NumPy 数组时，数值会被改成“易读”的形式。

# 我们发现梯度指向函数 f(x0,x1) 的“最低处”(最小值)，就像指南针 一样，所有的箭头都指向同一点。其次，我们发现离“最低处”越远，箭头越大。
# 虽然图4-9中的梯度指向了最低处，但并非任何时候都这样。实际上， 梯度会指向各点处的函数值降低的方向。更严格地讲，梯度指示的方向 是各点处的函数值减小最多的方向 A。