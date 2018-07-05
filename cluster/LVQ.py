# -*-coding:utf-8-*-
# Project: ML_Python
# Filename: LVQ
# Author: 😏 <smirk dot cao at gmail dot com>
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import accuracy_score
import pandas as pd


class LVQ(object):
    def __init__(self,
                 vectors_=None,
                 eta_=0.0001,
                 q_=5,
                 max_iter_=500,
                 tol_=0.0001
                 ):
        self.vectors_ = vectors_
        self.eta_ = eta_
        self.q_ = q_
        self.max_iter_ = max_iter_
        self.tol_ = tol_
        self.labels_ = None

    def fit(self,
            x_,):
        all_label = set(x_["label"].values)
        vectors_ = x_.sample(self.q_)
        # cover every class of samples
        while set(vectors_["label"].values) != all_label:
            vectors_ = x_.sample(self.q_)

        self.vectors_ = vectors_.drop(["label"], axis=1).values
        self.labels_ = vectors_["label"].values

        n_iter_ = 0
        while n_iter_ < self.max_iter_:
            #
            sample_ = x_.sample(1).values[0]
            label_ = self.labels_[pairwise_distances_argmin([sample_[:-1]],
                                                            self.vectors_)[0]]
            if label_ == sample_[-1]:
                vectors_ = self.vectors_ + self.eta_*(self.vectors_ - sample_[:-1])
            else:
                vectors_ = self.vectors_ - self.eta_*(self.vectors_ - sample_[:-1])

            if sum(sum(self.vectors_ - sample_[:-1])**2) < self.tol_:
                print(n_iter_, self.vectors_)
                return self.vectors_

            self.vectors_ = vectors_

            n_iter_ += 1
        print(n_iter_, self.vectors_)
        return self.vectors_

    def predict(self,
                x_):
        res = pairwise_distances_argmin(x_,
                                        self.vectors_,
                                        metric="euclidean")
        return self.labels_[res]


if __name__ == '__main__':
    df = pd.read_csv(".\Input\mmelon_4.0.csv")
    # print(df.head())
    # print(df.drop(["ID", "label"], axis=1))
    lvq = LVQ(max_iter_=400)
    rst = lvq.fit(df.drop(["ID"], axis=1))
    y_pred = lvq.predict(df.drop(["ID", "label"], axis=1))
    y = df.label.values
    print(accuracy_score(y, y_pred))