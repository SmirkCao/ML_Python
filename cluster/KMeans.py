# -*-coding:utf-8-*-
# Project: ML_Python
# Filename: Kmeans
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.metrics.pairwise import pairwise_distances_argmin
import pandas as pd
# todo: init
# todo: distance methods


class KMeans(object):
    def __init__(self,
                 k_=2,
                 max_iter_=100,
                 tol=0.001):
        self.k_ = k_
        self.labels_ = None
        self.cluster_centers_ = None
        self.max_iter_ = max_iter_
        self.tol_ = tol

    def fit(self,
            x_):
        self.cluster_centers_ = x_.sample(self.k_)
        n_iter_ = 0
        while n_iter_ < self.max_iter_:
            # 1. clustering: x -> ck
            if "cluster" in x_.columns.tolist():
                self.labels_ = x_["cluster"] = pairwise_distances_argmin(x_.drop(["cluster"], axis=1),
                                                                         self.cluster_centers_,
                                                                         metric="euclidean")
            else:
                self.labels_ = x_["cluster"] = pairwise_distances_argmin(x_,
                                                                         self.cluster_centers_,
                                                                         metric="euclidean")

            # 2. recalculate means_
            cluster_centers_ = x_.groupby(by=["cluster"]).mean().sort_values(by=x_.columns.tolist()[0])
            # distance(centers)
            dis_ = sum(sum((self.cluster_centers_.values - cluster_centers_.values) ** 2))
            if dis_ < self.tol_:
                print("n_iter_ is %d, means_ is %s" % (n_iter_, self.cluster_centers_))
                return self.cluster_centers_
            else:
                self.cluster_centers_ = cluster_centers_
            n_iter_ += 1

        print("n_iter_ is %d cluster_centers_ is %s" % (n_iter_, self.cluster_centers_.values))
        return self.cluster_centers_

    def predict(self,
                x_):
        res = pairwise_distances_argmin(x_,
                                        self.cluster_centers_,
                                        metric="euclidean")
        return res


if __name__ == '__main__':
    df = pd.read_csv(".\Input\melon_4.0.csv")
    df.dropna(inplace=True, axis=1)
    kmeans = KMeans(k_=3)
    kmeans.fit(df.drop(["ID"], axis=1))
    rst = kmeans.predict(df.drop(["ID"], axis=1))
    print(rst)
    print(kmeans.labels_)

