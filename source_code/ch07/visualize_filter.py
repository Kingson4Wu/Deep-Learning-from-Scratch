# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 随机进行初始化后的权重
filter_show(network.params['W1'])

# 学习后的权重
network.load_params("params.pkl")
filter_show(network.params['W1'])

# CNN 的可视化
# 我们将卷积层(第 1 层)的滤波器显示为图像。这里，我 们来比较一下学习前和学习后的权重
# 学习前的滤波器是随机进行初始化的，所以在黑白的浓淡上 没有规律可循，但学习后的滤波器变成了有规律的图像。我们发现，通过学 习，滤波器被更新成了有规律的滤波器，比如从白到黑渐变的滤波器、含有 块状区域(称为 blob)的滤波器等。