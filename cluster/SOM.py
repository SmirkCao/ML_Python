# -*-coding:utf-8-*-
# Project: Smirk  
# Filename: SOM
# Author: 😏 <smirk dot cao at gmail dot com>


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin
from sklearn.preprocessing import normalize


class SOM(object):
    def __init__(self,
                 max_iter_=100,
                 batch_size=10,
                 k_=10,
                 d_=100,
                 eta_=0.01
                 ):
        self.max_iter_ = max_iter_
        self.batch_size_ = batch_size
        self.w_ = None
        self.d_ = d_
        self.eta_ = eta_
        self.k_ = k_

    def f_neighbor(self,
                   x_,
                   y_):
        m_, d_, _ = self.w_.shape
        theta_ = self.w_.copy()
        for i in range(m_):
            for j in range(d_):
                theta_[i, j] = np.exp(((x_ - i)**2 + (y_ - j)**2)**0.5)
        return theta_

    def norm_weight(self,):
        m_, d_, _ = self.w_.shape
        for i in range(m_):
            for j in range(d_):
                self.w_[i, j] = normalize(self.w_[i, j].reshape(1,-1))[0]
        return self.w_

    def fit(self,
            x_,
            y_=None):
        # m, n, d 是可以随意取的，这是你造的map
        # 为了和data有区分，用mm_, nn_表示
        m_ = x_.shape[0]
        n_ = x_.shape[1]
        mm_ = self.k_
        nn_ = self.d_

        n_iter_ = 0
        # 1. Randomize the node weight vectors in a map
        self.w_ = np.random.rand(mm_ * n_ * nn_).reshape(mm_, nn_, n_)
        self.norm_weight()
        while n_iter_ < self.max_iter_:
            # 2. Randomly pick an input vector $D(t)$
            idx = np.argmax(np.random.random_sample(m_))
            Dt = x_[idx]
            # 3. Traverse each node in the map
            # 3.1 calculate similarity
            w_ = np.ones((mm_, nn_))
            for i in range(mm_):
                w_[i] = pairwise_distances(self.w_[i], Dt.reshape((-1, n_))).reshape(-1)
            # 3.2 track BMU
            min_ = np.inf
            for i in range(mm_):
                for j in range(nn_):
                    # print(w_.shape)
                    if w_[i, j] < min_:
                        x, y, min_ = i, j, w_[i, j]
            # 4. Update the weight vectors of the nodes
            # in the neighborhood of the BMU (including the BMU itself) by pulling them closer to the input vector
            # 4.1 $W_{v}(s + 1) = W_{v}(s) + \theta(u, v, s) \cdot \alpha(s) \cdot (D(t) - W_{v}(s))$

            self.w_ = self.w_ + self.f_neighbor(x, y)*self.eta_*(Dt-self.w_)
            self.norm_weight()
            n_iter_ = n_iter_ + 1

    def predict(self,
                x_):
        m_ = x_.shape[0]
        n_ = x_.shape[1]
        self.w_ = self.w_.reshape(self.k_, n_)
        return pairwise_distances_argmin(x_, self.w_)


if __name__ == '__main__':
    df = pd.read_csv(".\Input\melon_4.0.csv")
    df.dropna(inplace=True, axis=1)
    df.set_index("ID", inplace=True)
    x = df.values
    # 聚成四类， 1D聚类
    # todo:维度处理适配
    # todo:学习率自适应
    som = SOM(d_=1, k_=4)
    som.fit(x)
    rst = som.predict(x)
    print(rst)
