
from keras.layers import Input, Convolution2D, BatchNormalization

def construct_cnn_L3_orig():

    # INPUT
    x = Input(shape=(n_freq_cnn, n_frames_cnn, 1), dtype='float32')

    # CONV 1
    y = Convolution2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(x)
    y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # CONV 2
    y = Convolution2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # CONV 3
    y = Convolution2D(n_filters_cnn, filter_size_cnn, padding='valid',
               activation='relu')(y)
    # y = MaxPooling2D(pool_size=pool_size_cnn, strides=None, padding='valid')(y)
    y = BatchNormalization()(y)

    # Flatten for dense layers
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(n_dense_cnn, activation='relu')(y)
    if large_cnn:
        y = Dropout(0.5)(y)
        y = Dense(n_dense_cnn, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(n_classes, activation='sigmoid')(y)

    m = Model(inputs=x, outputs=y)
    return m



MODELS = {'cnn_L3_orig': construct_cnn_L3_orig}