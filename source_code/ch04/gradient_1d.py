# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

# 这里计算的导数是 f(x) 相对于 x 的变化量，对应函数的斜率。另外， f(x)=0.01x2 +0.1x的解析解是 。因此，在x=5和 x = 10 处，“真的导数”分别为 0.2 和 0.3。和上面的结果相比，我们发现虽然 严格意义上它们并不一致，但误差非常小。实际上，误差小到基本上可以认 为它们是相等的。
# 现在，我们用上面的数值微分的值作为斜率，画一条直线。
# x = 5、x = 10处的切线:直线的斜率使用数值微分的值
