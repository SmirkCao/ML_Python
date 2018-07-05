# -*-coding:utf-8-*-
# Project: ML_Python
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>

import pandas as pd
from SOM import *
from LVQ import *
from KMeans import *


def test_load_data(path_=".\Input\melon_4.0.csv"):
    test_df = pd.read_csv(path_)
    test_df.dropna(inplace=True, axis=1)
    return test_df


def test_smo_fit():
    df = test_load_data()
    df.set_index("ID", inplace=True)

    som = SOM()
    som.fit(df)
    return df


def test_smo_predict():
    df = test_load_data()
    return df


def test_kmeans_fit():
    df = test_load_data()
    kmeans = KMeans(k_=3)
    kmeans.fit(df.drop(["ID"], axis=1))

    return kmeans


def test_kmeans_predict():
    df = test_load_data()
    kmeans = test_kmeans_fit()
    rst = kmeans.predict(df.drop(["ID"], axis=1))
    print(rst)
    print(kmeans.labels_)


def test_lvq_fit():
    df = test_load_data(path_=".\Input\mmelon_4.0.csv")
    lvq = LVQ(max_iter_=400)
    rst = lvq.fit(df.drop(["ID"], axis=1))
    return rst


if __name__ == '__main__':
    # 1
    # df = test_load_data()
    # print(df.head())
    # 2
    # rst = test_kmeans_fit()
    # 3
    # rst = test_kmeans_predict()
    # 4
    rst = test_lvq_fit()