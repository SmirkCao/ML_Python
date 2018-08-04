# -*-coding:utf-8-*-
# Project:  nn
# Filename: GoogLeNet
# Date: 8/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input, concatenate
from keras.models import Model

# Going deeper with convolutions 2014
# Global Constants
NB_CLASS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
WEIGHT_DECAY = 0.0005
USE_BN = True
IM_WIDTH = 224
IM_HEIGHT = 224
EPOCH = 50

IM_WIDTH = 224
IM_HEIGHT = 224
batch_size = 32


def inception_module(x, params, concat_axis, padding='same'):
    (branch1, branch2, branch3, branch4) = params
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding)(x)

    # 1x1->3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding)(x)
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding)(pathway2)

    # 1x1->5x5
    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding)(x)
    pathway3 = Conv2D(filters=branch3[1], kernel_size=(5, 5), strides=1, padding=padding)(pathway3)

    # 3x3->1x1
    pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding)(pathway4)

    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)


class GoogLeNet(object):
    @staticmethod
    def build(input_shape, classes):
        img_input = Input(shape=input_shape)
        CONCAT_AXIS = 3
        # convolution
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), input_shape=input_shape, padding='same')(img_input)
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # convolution
        x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # inception(3a)
        x = inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS)  # 3a
        # inception(3b)
        x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3b
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # inception(4a)
        x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS)  # 4a
        # inception(4b)
        x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4b
        # inception(4c)
        x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4c
        # inception(4d)
        x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4d
        # inception(4e)
        x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)  # 4e
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # inception(5a)
        x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)  # 5a
        # inception(5b)
        x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS)  # 5b
        # ave pool
        x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(x)
        x = Flatten()(x)
        x = Dropout(0.4, name="dropout_0.4")(x)
        x = Dense(units=classes, activation='linear', name="linear")(x)
        x = Dense(units=classes, activation='softmax', name="softmax")(x)
        model = Model(inputs=img_input, outputs=x)
        return model


if __name__ == '__main__':
    model= GoogLeNet.build((224, 224, 3), NB_CLASS)
    model.summary()
