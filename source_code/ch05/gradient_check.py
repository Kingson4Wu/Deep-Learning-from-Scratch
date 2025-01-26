# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

# 确认数值 微分求出的梯度结果和误差反向传播法求出的结果是否一致(严格地讲，是 非常相近)的操作称为梯度确认(gradient check)。    
# 这里误差的计 算方法是求各个权重参数中对应元素的差的绝对值，并计算其平均值。
# 数值微分和误差反向传播法的计算结果之间的误差为 0 是很少见的。 这 是 因 为 计 算 机 的 计 算 精 度 有 限 ( 比 如 ， 3 2 位 浮 点 数 )。 受 到 数 值 精 度的限制，刚才的误差一般不会为 0，但是如果实现正确的话，可 以期待这个误差是一个接近 0 的很小的值。如果这个值很大，就说 明误差反向传播法的实现存在错误。