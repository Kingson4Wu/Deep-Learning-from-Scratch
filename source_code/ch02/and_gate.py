# coding: utf-8
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))

# 实际上，满足图 2-2 的条件的参数的选择方法有无数多个。比如，当 (w1, w2, θ) = (0.5, 0.5, 0.7) 时，可 以 满 足 图 2-2 的 条 件。此 外，当 (w1, w2, θ) 为 (0.5, 0.5, 0.8) 或者 (1.0, 1.0, 1.0) 时，同样也满足与门的条件。设定这样的 参数后，仅当 x1 和 x2 同时为 1 时，信号的加权总和才会超过给定的阈值 θ。        
