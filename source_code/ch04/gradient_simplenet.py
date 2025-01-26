# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# simpleNet 类只有 一个实例变量，即形状为 2×3 的权重参数。它有两个方法，一个是用于预 测的 predict(x)，另一个是用于求损失函数值的 loss(x,t)。这里参数 x 接收 输入数据，t 接收正确解标签。
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)

# 以一个简单的神经网络为例，来实现求梯度( 相当于其函数中某个点的梯度值)

# 求出神经网络的梯度后，接下来只需根据梯度法，更新权重参数即可
