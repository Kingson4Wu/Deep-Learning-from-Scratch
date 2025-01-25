# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# load_mnist 函数以“( 训练图像 , 训练标签 )，( 测试图像，测试标签 )”的 形式返回读入的 MNIST 数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
# reshape 是 NumPy 中的一个函数，用于改变数组的形状（shape），而不改变数组中的数据。它允许你将一维数组、二维数组或更高维数组重塑为其他维度的形状，只要数组中的元素总数不变。
# 28 * 28 = 784
print(img.shape)  # (28, 28)

img_show(img)
