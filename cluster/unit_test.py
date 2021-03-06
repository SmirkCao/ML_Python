# -*-coding:utf-8-*-
# Project: ML_Python
# Filename: unit_test
# Author: 😏 <smirk dot cao at gmail dot com>

import pandas as pd
from SOM import *
from LVQ import *
from KMeans import *
from DBSCAN import *
from AGNES import *


def test_load_data(path_=".\Input\melon_4.0.csv"):
    test_df = pd.read_csv(path_)
    test_df.dropna(inplace=True, axis=1)
    return test_df


def test_smo_fit():
    df = test_load_data()
    df.set_index("ID", inplace=True)

    som = SOM(d_=1)
    som.fit(df.values)

    return som, df


def test_smo_predict():
    df = test_load_data()
    df.set_index("ID", inplace=True)
    x = df.values
    som = SOM(d_=1, k_=4)
    som.fit(x)
    rst = som.predict(x)
    print(rst)
    return som, df


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


def test_dbscan_fit():
    df = test_load_data(path_=".\Input\mmelon_4.0.csv")
    db = DBSCAN(eps_=0.1, min_samples_=4)
    db.fit(df.drop(["ID", "label"], axis=1))
    print(db.ck)
    return db


def test_agnes_fit():
    df = test_load_data(path_=".\Input\mmelon_4.0.csv")
    agnes = AGNES(k_=4, d_=max)
    agnes.fit(df.drop(["ID", "label"], axis=1).values)
    print(agnes.ck)
    return agnes


if __name__ == '__main__':
    # 1
    # df = test_load_data()
    # print(df.head())
    # 2
    # rst = test_kmeans_fit()
    # 3
    # rst = test_kmeans_predict()
    # 4
    # rst = test_lvq_fit()
    # 5
    # rst = test_dbscan_fit()
    # 6
    # rest = test_agnes_fit()
    # 7
    # rst = test_smo_fit()
    # 8
    smo, df = test_smo_predict()