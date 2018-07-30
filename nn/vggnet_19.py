# -*-coding:utf-8-*-
# Project:  nn
# Filename: vggnet_19
# Date: 7/27/18
# Author: 😏 <smirk dot cao at gmail dot com>


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np


# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
# MODEL E


class VGG19(object):
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # #INPUT
        # input_size = (224, 224,3)
        # we use very small 3 × 3 receptive fields throughout the whole net,
        # which are convolved with the input at every pixel (with stride 1).
        # #CONV Layer Group 1
        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=input_shape, padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        # Max-pooling is performed over a 2 × 2 pixel window, with stride 2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        # #CONV Layer Group 2
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        # #CONV Layer Group 3
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

        # #CONV Layer Group 4
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        # #CONV Layer Group 5
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        # #
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation="softmax"))
        return model


if __name__ == '__main__':
    # set fix random seed
    seed = 2018
    np.random.seed(seed)

    model = VGG19.build((224, 224, 3), 1000)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
