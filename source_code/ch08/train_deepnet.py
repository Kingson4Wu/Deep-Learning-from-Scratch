# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")


# 虽然使用这些代码可以重现这里 进行的学习，不过深度网络的学习需要花费较多的时间(大概要半天 以上)。本书以 ch08/deep_conv_net_params.pkl 的形式给出了学习完 的权重参数。刚才的 deep_convnet.py 备有读入学习完的参数的功能， 请根据需要进行使用。

