# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    # 参数从头开始依次表示输入层的神经元数、隐藏层 的神经元数、输出层的神经元数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 进行识别(推理)。 参数 x 是图像数据
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    # 参数 x 是图像数据，t 是正确解标签    
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算权重参数的梯度    
    # 基于数值微分计算参数的梯度
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    # 计算权重参数的梯度。
    # numerical_gradient() 的高速版，将在下一章实现    
    # 该方法使用误差反向传播法高效地计算梯度
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
    

# 实现手写数字识别的神经网络。这里以2层神经网络(隐 藏层为 1 层的网络)为对象，使用 MNIST 数据集进行学习。

# 神经网络的学习 步骤如下所示。

# 前提
# 神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的 过程称为“学习”。神经网络的学习分成下面 4 个步骤。

# 步骤 1(mini-batch) 从训练数据中随机选出一部分数据，这部分数据称为 mini-batch。我们 的目标是减小 mini-batch 的损失函数的值。
# 步骤 2(计算梯度)
# 为了减小 mini-batch 的损失函数的值，需要求出各个权重参数的梯度。 梯度表示损失函数的值减小最多的方向。
# 步骤 3(更新参数) 将权重参数沿梯度方向进行微小更新。
# 步骤 4(重复)
# 重复步骤 1、步骤 2、步骤 3。

# params
# 保存神经网络的参数的字典型变量(实例变量)。
# params['W1'] 是第 1 层的权重，params['b1'] 是第 1 层的偏置。 
# params['W2'] 是第 2 层的权重，params['b2'] 是第 2 层的偏置

# grads
# 保存梯度的字典型变量(numerical_gradient() 方法的返回值)。 
# grads['W1'] 是第 1 层权重的梯度，grads['b1'] 是第 1 层偏置的梯度。 
# grads['W2'] 是第 2 层权重的梯度，grads['b2'] 是第 2 层偏置的梯度
