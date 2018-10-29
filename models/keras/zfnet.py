# -*-coding:utf-8-*-
# Project:  nn
# Filename: zfnet
# Date: 7/27/18
# Author: 😏 <smirk dot cao at gmail dot com>
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np

# note：
# 1. input size
# 1. padding parameter in following layers


class ZFNet(object):
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # #INPUT
        # input_size = (225, 225,3)

        # #CONV 1
        model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=input_shape, padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # #CONV 2
        model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # #CONV 3
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

        # #CONV 4
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

        # #CONV 5
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # #FC 6
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # #FC 7
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # #OUTPUT
        model.add(Dense(classes, activation='softmax'))
        return model


if __name__ == '__main__':
    # set fix random seed
    seed = 2018
    np.random.seed(seed)

    model = ZFNet.build((225, 225, 3), 1000)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
