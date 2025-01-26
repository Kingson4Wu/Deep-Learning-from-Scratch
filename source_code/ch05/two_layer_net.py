# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    # 进行初始化。
    # 参数从头开始依次是输入层的神经元数、隐藏层的神经元数、输出层的神经元数、初始化权重时的高斯分布的规模
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        # 神经网络的层保存为 OrderedDict 这一点非常重要。OrderedDict 是有序字典，“有序”是指它可以 记住向字典里添加元素的顺序。因此，神经网络的正向传播只需按照添加元 素的顺序调用各层的 forward() 方法就可以完成处理，而反向传播只需要按 照相反的顺序调用各层即可。因为 Affine 层和 ReLU 层的内部会正确处理正 向传播和反向传播，所以这里要做的事情仅仅是以正确的顺序连接各层，再 按顺序(或者逆序)调用各层。
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    # 进行识别(推理)。参数 x 是图像数据    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    # 计算损失函数的值。
    # 参数 X 是图像数据、t 是正确解标签    
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 通过数值微分计算关于权重参数的梯度    
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

    # 通过误差反向传播法计算关于权重参数的梯度    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

# params
# 保存神经网络的参数的字典型变量。
# params['W1'] 是第 1 层的权重，params['b1'] 是第 1 层的偏置。 
# params['W2'] 是第 2 层的权重，params['b2'] 是第 2 层的偏置。

# layers
# 保存神经网络的层的有序字典型变量。
# 以 layers['Affine1']、layers['ReLu1']、layers['Affine2'] 的形式， 通过有序字典保存各个层

# lastLayer
# 神经网络的最后一层。 
# 本例中为 SoftmaxWithLoss 层

# 像这样通过将神经网络的组成元素以层的方式实现，可以轻松地构建神 经网络。这个用层进行模块化的实现具有很大优点。因为想另外构建一个神 经网络(比如 5 层、10 层、20 层......的大的神经网络)时，只需像组装乐高 积木那样添加必要的层就可以了。之后，通过各个层内部实现的正向传播和 反向传播，就可以正确计算进行识别处理或学习所需的梯度。


