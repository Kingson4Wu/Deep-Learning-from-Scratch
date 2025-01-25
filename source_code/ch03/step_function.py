# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# 对 NumPy 数组进行不等号运算后，数组的各个元素都会进行不等号运算， 生成一个布尔型数组。这里，数组 x 中大于 0 的元素被转换为 True，小于等 于 0 的元素被转换为 False，从而生成一个新的数组 y。
# 数组 y 是一个布尔型数组，但是我们想要的阶跃函数是会输出 int 型的 0 或 1 的函数。

def step_function(x):
    return np.array(x > 0, dtype=int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
