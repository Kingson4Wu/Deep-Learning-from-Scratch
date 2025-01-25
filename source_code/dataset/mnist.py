# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


#import urllib.request
#import ssl

# 创建忽略 SSL 验证的上下文
#ssl_context = ssl._create_unverified_context()

# 创建一个 HTTPSHandler，使用忽略验证的上下文
#https_handler = urllib.request.HTTPSHandler(context=ssl_context)

# 创建一个自定义 opener
#opener = urllib.request.build_opener(https_handler)

# 安装 opener，使其成为全局默认
#urllib.request.install_opener(opener)

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

# 第 1 个 参 数 normalize 设置是否将输入图像正规化为 0.0~1.0 的值。如果将该参数设置 为 False，则输入图像的像素会保持原来的 0~255。第 2 个参数 flatten 设置 是 否 展 开 输 入 图 像 ( 变 成 一 维 数 组 )。 如 果 将 该 参 数 设 置 为 F a l s e ， 则 输 入 图 像为 1 × 28 × 28 的三维数组;若设置为 True，则输入图像会保存为由 784 个 元素构成的一维数组。第 3 个参数 one_hot_label 设置是否将标签保存为 one- hot 表示(one-hot representation)。one-hot 表示是仅正确解标签为 1，其余 皆为 0 的数组，就像 [0,0,1,0,0,0,0,0,0,0] 这样。当 one_hot_label 为 False 时， 只是像 7、2 这样简单保存正确解标签;当 one_hot_label 为 True 时，标签则 保存为 one-hot 表示。
# Python 有 pickle 这个便利的功能。这个功能可以将程序运行中的对 象保存为文件。如果加载保存过的 pickle 文件，可以立刻复原之前 程序运行中的对象。用于读入 MNIST 数据集的 load_mnist() 函数内 部也使用了 pickle 功能(在第 2 次及以后读入时)。利用 pickle 功能， 可以高效地完成 MNIST 数据的准备工作。
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
