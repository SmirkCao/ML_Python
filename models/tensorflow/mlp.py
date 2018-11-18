#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: mlp
# Date: 11/16/18
# Author: 😏 <smirk dot cao at gmail dot com>
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)
import logging
import tensorflow as tf
from dataflow.datasets.dummy import load_mnist
import numpy as np


def dense(inputs, in_size, out_size, activation_function=None):
    """
    dense layer
    :param inputs:
    :param in_size:
    :param out_size:
    :param activation_function:
    :return:
    """
    w = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    tf.summary.histogram("weights", w)
    tf.summary.histogram("bias", b)
    wx_b = tf.matmul(inputs, w) + b
    if activation_function is None:
        outputs = wx_b
    else:
        outputs = activation_function(wx_b)
    return outputs


def data_generator(x, y, batch_size=100):
    """
    todo: add shuffle process.
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    global p
    global p_
    while True:
        if p + batch_size > len(x):
            p_ = p
            p = (p + batch_size) % len(x)
            indexes = np.append(np.arange(len(x))[p_:],
                                np.arange(len(x))[:p])
        else:
            p_ = p
            p += batch_size
            indexes = np.arange(len(x))[p:p + batch_size]
        yield x[indexes], y[indexes]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    x_data, y_data = load_mnist()

    # one_hot encoding the label
    # 1. by tf.one_hot: need to call eval in session.
    # # on_value 1.0 and 1 are different
    # # tf.one_hot return a tensor
    # y_data = tf.one_hot(y_data.reshape(-1), 10, on_value=1.0, off_value=0.0, axis=-1)

    # 2. by numpy.eye
    y_data = np.eye(10)[y_data.reshape(-1)]

    # inputs
    with tf.name_scope("inputs"):
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])

    # module
    with tf.name_scope("module"):
        # 这里softmax和上面的one-hot对应.
        pred = dense(xs, 784, 10, activation_function=tf.nn.softmax)
    # loss
    with tf.name_scope("loss"):
        # DL: 交叉熵损失大大提高了具有Sigmoid和Softmax输出的模型的性能.
        # ref to: DL P_{137}
        # + 1e-10 or with clip tf.clip_by_value(pred,1e-10,1.0)
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(pred + 1e-10), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(ys * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)), reduction_indices=[1]))

    # optimizer
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_gen = data_generator(x_data, y_data, batch_size=100)
        p = 0
        p_ = 0
        for idx in range(10000):
            # batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs, batch_ys = next(data_gen)
            # print(batch_ys.shape, batch_xs.shape, type(batch_xs[0, 0]))
            # logger.info(sess.run(pred, feed_dict={xs: batch_xs, ys: batch_ys}))
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            if idx % 50 == 0:
                # print(batch_xs, batch_ys)
                # 感觉这里用不到xs, 但是也要传.
                logger.info(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
