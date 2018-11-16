#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: nn_capacity
# Date: 11/17/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

from dataflow.datasets.dummy import *
import tensorflow as tf
import matplotlib.pyplot as plt


def dense(inputs, in_size, out_size, activation_function=None):
    w = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_b = tf.matmul(inputs, w) + b
    if activation_function is None:
        outputs = wx_b
    else:
        outputs = activation_function(wx_b)
    return outputs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    x_data, y_data = load_curve1()
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = dense(xs, 1, 50, activation_function=tf.nn.sigmoid)
    # DL P_123
    # 一个前馈神经网络如果具有线性输出层和至少一层具有任何一种"挤压"性质的激活函数的隐藏层, 只要给予网络足够数量的隐藏单元, 它可以任意精度
    # 近似任何从一个有限维空间到另一个有限维空间的Borel可测函数.
    # 通过增加网络容量可以提升拟合结果.
    # 1. 增加宽度(每层神经元数量)
    # 2. 增加深度(网络层数)
    # 考虑激活函数的形状, 对拟合会有好处, curve数据集不适合用relu
    # l2 = dense(l1, 10, 100, activation_function=tf.nn.sigmoid)
    # l3 = dense(l2, 100, 50, activation_function=tf.nn.sigmoid)

    prediction = dense(l1, 50, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    with tf.Session() as sess:
        sess.run(init)
        for idx in range(2000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if idx % 50 == 0:
                # leave one result along
                # try:
                #     ax.lines.remove(lines[0])
                # except NameError:
                #     pass

                logger.info(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                pred = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, pred, "-r", alpha=0.2)
                plt.pause(0.1)

    plt.waitforbuttonpress()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
