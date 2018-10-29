# -*-coding:utf-8-*-
# Project:  nn
# Filename: GoogLeNet
# Date: 8/4/18
# Author: 😏 <smirk dot cao at gmail dot com>
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input, concatenate
from keras.models import Model
from keras.optimizers import SGD

# Going deeper with convolutions 2014
# Total params: 10,442,960

# Global Constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
WEIGHT_DECAY = 0.0005
EPOCH = 50

batch_size = 32
CONCAT_AXIS = 3


class GoogLeNet(object):
    def inception_module(self, x_, params, concat_axis, padding='same'):
        (branch1, branch2, branch3, branch4) = params
        # 1x1
        pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, activation="relu")(x_)

        # 1x1->3x3
        pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, activation="relu")(x_)
        pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding, activation="relu")(pathway2)

        # 1x1->5x5
        pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding, activation="relu")(x_)
        pathway3 = Conv2D(filters=branch3[1], kernel_size=(5, 5), strides=1, padding=padding, activation="relu")(pathway3)

        # 3x3->1x1
        pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding)(x_)
        pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, activation="relu")(pathway4)

        return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)

    # all the convolutions, including those inside the Inception modules, use relu.
    def build(self, input_shape, classes):
        img_input = Input(shape=input_shape)
        # convolution
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), input_shape=input_shape,
                   padding='same', activation="relu")(img_input)
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # convolution
        x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu")(x)
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # inception(3a)
        x = self.inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS)
        # inception(3b)
        x = self.inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS)
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # inception(4a)
        x = self.inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS)
        x0 = self.aux_clf_module(x, classes)
        # inception(4b)
        x = self.inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS)
        # inception(4c)
        x = self.inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS)
        # inception(4d)
        x = self.inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS)
        x1 = self.aux_clf_module(x, classes)
        # inception(4e)
        x = self.inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # inception(5a)
        x = self.inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)
        # inception(5b)
        x = self.inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS)
        # ave pool
        x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(x)
        x = Flatten()(x)
        x = Dropout(0.4, name="dropout_0.4")(x)
        x = Dense(units=classes, activation='linear', name="linear")(x)
        x2 = Dense(units=classes, activation='softmax', name="softmax")(x)
        model = Model(inputs=img_input, outputs=[x0, x1, x2])
        return model

    def aux_clf_module(self, x_, classes):
        x_ = AveragePooling2D((5, 5), strides=(3, 3))(x_)
        x_ = Conv2D(filters=128, kernel_size=(1, 1), activation="relu")(x_)
        x_ = Dense(units=1024, activation="relu")(x_)
        x_ = Dropout(0.7)(x_)
        x_ = Dense(units=classes, activation="softmax")(x_)
        return x_


if __name__ == '__main__':
    inception_v1 = GoogLeNet()
    model= inception_v1.build((224, 224, 3), 1000)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(momentum=0.9))
    model.summary()
