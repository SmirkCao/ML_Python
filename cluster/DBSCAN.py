# -*-coding:utf-8-*-
# Project: Smirk  
# Filename: DBSCAN
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class DBSCAN(object):
    def __init__(self,
                 eps_=0.11,
                 min_samples_=5):
        """

        :param eps_: 样本分布紧密程度
        :param min_samples_: 最少包含样本数量
        """

        self.labels_ = None
        self.eps_ = eps_
        self.n_eps_ = dict()
        self.min_samples_ = min_samples_
        self.core_sample_indices_ = []
        self.ck = []

    def fit(self,
            x_):
        # print("x_ shape", x_.shape)
        idxs = np.array(range(x_.shape[0]))
        for idx in range(x_.shape[0]):
            # 确定x的邻域
            self.n_eps_.update({idx: idxs[(pairwise_distances([x_.iloc[idx]], x_)[0] < self.eps_)]})
            if len(self.n_eps_[idx]) > self.min_samples_:
                self.core_sample_indices_.append(idx)
            # print(self.n_eps_[idx])
        core_objects = set(self.core_sample_indices_)
        k_ = 0
        samples_ = set(idxs)
        while core_objects:
            samples_old_ = samples_.copy()
            o = core_objects.pop()
            Q = [o]
            samples_ -= set({o})
            while Q:
                q = Q[0]
                Q.remove(q)
                if len(self.n_eps_[q]) >= self.min_samples_:
                    delta = set(self.n_eps_[q]) & samples_
                    Q.extend(list(delta))
                    samples_ -= set(delta)
            k_ = k_ + 1
            self.ck.append(samples_old_-samples_)
            core_objects -= self.ck[-1]
        return self.ck

    def fit_predict(self,
                    x_):
        return self.fit(x_)


if __name__ == '__main__':
    pass
