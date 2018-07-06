# -*-coding:utf-8-*-
# Project: Smirk  
# Filename: AGNES
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


def min_xy(M):
    m = M.shape[0]
    min_ = np.inf
    for i in range(m):
        for j in range(i+1, m):
            if M[i, j] < min_:
                x, y, min_ = i, j, M[i, j]
    return x, y, min_


class AGNES(object):
    def __init__(self,
                 k_=2,
                 d_=min):
        self.k_ = k_
        self.d_ = d_
        self.ck = dict()
        self.dendrogram_ = list()

    def fit(self,
            x_):
        m = x_.shape[0]
        n = x_.shape[1]
        # init ck
        # for i in range(m):
        #     self.ck[i] = x_[i]
        # List comprehension is faster than a for loop
        [self.ck.update({i: x_[i]}) for i in range(m)]

        # init M
        M = np.ones((m, m))
        for i in range(m):
            for j in range(i+1, m):
                M[i, j] = M[j, i] = pairwise_distances(self.ck[i].reshape(-1, n),
                                                       self.ck[j].reshape(-1, n))
        q = m
        while q > self.k_:
            x, y, min_ = min_xy(M)
            # x, y = 0, 1
            self.ck[x] = np.vstack([self.ck[x], self.ck[y]])
            M = np.delete(M, y, axis=1)
            M = np.delete(M, y, axis=0)

            for j in range(y, q-1):
                self.ck[j] = self.ck[j+1]
                # print(self.ck[x].shape, self.ck[j].reshape(-1, n).shape)
                # print(pairwise_distances(self.ck[x].reshape(-1, n), self.ck[j].reshape(-1, n)).shape)
                M[x, j] = M[j, x] = self.d_(pairwise_distances(self.ck[x].reshape(-1, n),
                                                               self.ck[j].reshape(-1, n)).flatten())
            for j in range(y):
                M[x, j] = M[j, x] = self.d_(pairwise_distances(self.ck[x].reshape(-1, n),
                                                               self.ck[j].reshape(-1, n)).flatten())
            self.dendrogram_.append([min_, self.ck[x]])
            self.ck.pop(q-1)
            q = q - 1
        return self.ck


if __name__ == '__main__':
    pass
