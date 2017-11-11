from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, \
    Flatten, Activation
import keras.regularizers as regularizers


def construct_cnn_L3_orig_vision_model():
    """
    Constructs a model that replicates the vision subnetwork  used in Look,
    Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    weight_decay = 1e-5
    l2_weight_decay = regularizers.l2(weight_decay)
    ####
    # Image subnetwork
    ####
    # INPUT
    x_i = Input(shape=(224, 224, 3), dtype='float32')

    # CONV BLOCK 1
    n_filter_i_1 = 64
    filt_size_i_1 = (3, 3)
    pool_size_i_1 = (2, 2)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_regularizer=l2_weight_decay)(x_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = Activation('relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_1, strides=2, padding='same')(y_i)

    # CONV BLOCK 2
    n_filter_i_2 = 128
    filt_size_i_2 = (3, 3)
    pool_size_i_2 = (2, 2)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_2, strides=2, padding='same')(y_i)

    # CONV BLOCK 3
    n_filter_i_3 = 256
    filt_size_i_3 = (3, 3)
    pool_size_i_3 = (2, 2)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_3, strides=2, padding='same')(y_i)

    # CONV BLOCK 4
    n_filter_i_4 = 512
    filt_size_i_4 = (3, 3)
    pool_size_i_4 = (28, 28)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 kernel_regularizer=l2_weight_decay)(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_4, padding='same')(y_i)
    y_i = Flatten(name='vision_embedding')(y_i)

    m = Model(inputs=x_i, outputs=y_i)

    return m, x_i, y_i
