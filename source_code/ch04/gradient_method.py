# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


# 参数 f 是要进行最优化的函数，init_x 是初始值，lr 是学习率 learning rate，step_num 是梯度法的重复次数。numerical_gradient(f,x) 会求函数的 梯度，用该梯度乘以学习率得到的值进行更新操作，由 step_num 指定重复的 次数。
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

# 用梯度法求函数的最小值
# 开始使用梯度法寻找最小值。最终的结 果是(-6.1e-10, 8.1e-10)，非常接近(0，0)。实际上，真的最小值就是(0，0)， 所以说通过梯度法我们基本得到了正确结果
# 可以发现，原点处是最低的地方，函数的取值一 点点在向其靠近。
