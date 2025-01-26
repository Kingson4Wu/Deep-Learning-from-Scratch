# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 神经网络的学习的实现使用的是前面介绍过的 mini-batch 学习。
# 所谓 mini-batch 学习，就是从训练数据中随机选择一部分数据(称为 mini-batch)， 
# 再以这些 mini-batch 为对象，使用梯度法更新参数的过程。


# mini-batch 的大小为 100，需要每次从 60000 个训练数据中随机 取出100个数据(图像数据和正确解标签数据)。
# 然后，对这个包含100笔数 据的 mini-batch 求梯度，使用随机梯度下降法(SGD)更新参数。这里，梯 度法的更新次数(循环的次数)为 10000。
# 每更新一次，都对训练数据计算损 失函数的值，并把该值添加到数组中。用图像来表示这个损失函数的值的推移


# 随着学习的进行，损失函数的值在不断减小。这 是学习正常进行的信号，表示神经网络的权重参数在逐渐拟合数据。也就是 说，神经网络的确在学习!通过反复地向它浇灌(输入)数据，神经网络正 在逐渐向最优参数靠近。


# 神经网络学习的最初目标是掌握泛化能力，因此，要评价神经网络的泛 化能力，就必须使用不包含在训练数据中的数据。
# 下面的代码在进行学习的 过程中，会定期地对训练数据和测试数据记录识别精度。这里，每经过一个 epoch，我们都会记录下训练数据和测试数据的识别精度。

# epoch 是一个单位。一个 epoch 表示学习中所有训练数据均被使用过 一次时的更新次数。
# 比如，对于 10000 笔训练数据，用大小为 100 笔数据的 mini-batch 进行学习时，重复随机梯度下降法 100 次，所有的训练数据就都被“看过”了 A。此时，100 次就是一个 epoch。

# 随着 epoch 的前进(学习的进行)，我们发现使用训练数据和 测试数据评价的识别精度都提高了，并且，这两个识别精度基本上没有差异(两 条线基本重叠在一起)。因此，可以说这次的学习中没有发生过拟合的现象。

