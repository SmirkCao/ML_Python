# -*-coding:utf-8-*-
# Project:  nn
# Filename: alexnet
# Date: 7/27/18
# Author: 😏 <smirk dot cao at gmail dot com>

# coding=utf-8
# imagenet classification with deep convolutional neural networks
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# Total params: 62,378,344
# 650,000 neurons
# 5 CONV + 3 FC

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np

# Total params: 62,378,344


class AlexNet(object):
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # #INPUT
        # input_size = (227, 227,3)
        # (input_size - filter_size)/stride+1 = output_size

        # #CONV 1
        # output_size = (227-11)/4+1 = 55 -> (55, 55, 96)
        # neurons_num = 55*55*96 = 290400
        # parameters_size = filter_size*filter_num+bias_num -> wx+b: ([w,1])dot([x,b])
        # 11*11*3*96+96 = 34944
        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        # overlapping pooling: "This is what we use throughout our network, with s = 2 and z = 3"
        # output_size = (55-3)/2+1 = 27 -> (27, 27, 96)
        # parameters_size = 0
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # #CONV 2
        # note: padding -> "valid" 表示「不填充」。 "same" 表示填充输入以使输出具有与原始输入相同的长度。
        # output_size = (27-5+2*2)/1+1 = 27  2*2 from "same" -> (27, 27, 256)
        # neurons_num = 27*27*256 = 186624
        # parameters_size = (5*5*96+1)*256=614656
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # output_size = (27-3)/2+1 = 13, (13, 13, 256)
        # parameters_size = 0
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # #CONV 3
        # output_size = (13-3+1*2)/1+1 = 13, (13, 13, 384)
        # parameters_size = (3*3*256+1)*384 = 885120
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

        # #CONV 4
        # output_size = (13-3+1*2)/1+1 = 13, (13, 13, 384)
        # parameters_size = (3*3*384+1)*384 = 1327488
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

        # #CONV 5
        # output_size = (13-3+1*2)/1+1 = 13, (13, 13, 384)
        # parameters_size = (3*3*384+1)*256 = 884992
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # output_size = (13-3)/2+1 = 6, (6, 6, 256)
        # parameters_size = 0
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # output_size = 6*6*256 = 9216
        model.add(Flatten())

        # #FC 6
        # parameters_size = (9216+1)*4096 = 37752832
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # #FC 7
        # parameters_size = (4096+1)*4096 = 16781312
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # #OUTPUT
        # parameters_size = (4096+1)*1000 = 409700
        model.add(Dense(classes, activation='softmax'))
        return model


if __name__ == '__main__':
    # set fix random seed
    seed = 2018
    np.random.seed(seed)

    model = AlexNet.build((227, 227, 3), 1000)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
