from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, MaxPool2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input, concatenate
from keras.models import Model
from keras import regularizers

# Going deeper with convolutions 2014
# Global Constants
NB_CLASS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.4
WEIGHT_DECAY = 0.0005
LRN2D_NORM = True
DATA_FORMAT = 'channels_last'  # tensorflow
USE_BN = True
IM_WIDTH = 224
IM_HEIGHT = 224
EPOCH = 50

IM_WIDTH = 224
IM_HEIGHT = 224
batch_size = 32


# normalization
def conv2D_lrn2d(x, filters, kernel_size, strides=(1, 1), padding='same', data_format=DATA_FORMAT, dilation_rate=(1, 1),
                 activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, lrn2d_norm=LRN2D_NORM, weight_decay=WEIGHT_DECAY):
    # l2 normalization
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
               dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
               bias_constraint=bias_constraint)(x)

    if lrn2d_norm:
        # batch normalization
        x = BatchNormalization()(x)

    return x


def inception_module(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, dilation_rate=(1, 1),
                     activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                     bias_constraint=None, lrn2d_norm=LRN2D_NORM, weight_decay=None):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)

    # 1x1->3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)

    # 1x1->5x5
    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway3 = Conv2D(filters=branch3[1], kernel_size=(5, 5), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway3)

    # 3x3->1x1
    pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)


def create_model():
    INP_SHAPE = (224, 224, 3)
    img_input = Input(shape=INP_SHAPE)
    CONCAT_AXIS = 3
    # convolution
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), input_shape=INP_SHAPE, padding='same')(img_input)
    # max pool
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)
    # convolution
    x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # max pool
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)
    # inception(3a)
    x = inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS)  # 3a
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3b
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS)  # 4a
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4b
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4c
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS)  # 4d
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)  # 4e
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)  # 5a
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS)  # 5b
    x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', data_format=DATA_FORMAT)(x)

    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(output_dim=NB_CLASS, activation='linear')(x)
    x = Dense(output_dim=NB_CLASS, activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE, DATA_FORMAT


if __name__ == '__main__':
    x, img_input, CONCAT_AXIS, INP_SHAPE, DATA_FORMAT = create_model()

    model = Model(input=img_input, output=[x])
    model.summary()
