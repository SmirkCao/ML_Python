# -*-coding:utf-8-*-
# Project:  nn
# Filename: lenet_5
# Date: 7/29/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential


class LeNet5(object):
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # #Conv1
        model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='valid', input_shape=input_shape, activation='relu',
                         kernel_initializer='uniform'))
        # pooling
        model.add(MaxPool2D((2, 2), strides=(2, 2), padding='same'))

        # #Conv2
        model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
        # pooling
        model.add(MaxPool2D((2, 2), strides=(2, 2), padding='same'))
        model.add(Flatten())

        # #FC1
        model.add(Dense(120))

        # #FC2
        model.add(Dense(84))

        # #output
        model.add(Dense(classes, activation="softmax"))
        # model.add(Activation('softmax'))
        return model


if __name__ == '__main__':

    # set fix random seed
    seed = 2018
    np.random.seed(seed)

    model = LeNet5.build((32, 32, 1), 10)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
