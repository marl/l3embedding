from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D,\
                         Flatten, Concatenate, Dense
from kapre.time_frequency import Spectrogram


def construct_cnn_L3_orig():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    """
    ####
    # Image subnetwork
    ####
    # INPUT
    x_i = Input(shape=(224, 224, 3), dtype='float32')

    # CONV BLOCK 1
    n_filter_i_1 = 64
    filt_size_i_1 = (3, 3)
    pool_size_i_1 = (2,2)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 activation='relu')(x_i)
    y_i = BatchNormalization()(y_i)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_1, strides=2, padding='same')(y_i)

    # CONV BLOCK 2
    n_filter_i_2 = 128
    filt_size_i_2 = (3, 3)
    pool_size_i_2 = (2,2)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_2, strides=2, padding='same')(y_i)

    # CONV BLOCK 3
    n_filter_i_3 = 256
    filt_size_i_3 = (3, 3)
    pool_size_i_3 = (2,2)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_3, strides=2, padding='same')(y_i)

    # CONV BLOCK 4
    n_filter_i_4 = 512
    filt_size_i_4 = (3, 3)
    pool_size_i_4 = (28, 28)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 activation='relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_4, strides=2, padding='same')(y_i)
    y_i = Flatten()(y_i)


    ####
    # Audio subnetwork
    ####
    n_dft = 512
    n_hop = 16
    asr = 48000
    audio_window_dur = 1
    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # SPECTROGRAM PREPROCESSING
    # 257 x 199 x 1
    y_a = Spectrogram(n_dft=n_dft, n_hop=n_hop,
                      return_decibel_spectrogram=True)(x_a)
    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2,2)
    y_a= Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
               activation='relu')(y_a)
    y_a= BatchNormalization()(y_a)
    y_a= Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                       activation='relu')(y_a)
    y_a= BatchNormalization()(y_a)
    y_a= MaxPooling2D(pool_size=pool_size_a_1, strides=2, padding='same')(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2,2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                activation='relu')(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                activation='relu')(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2, padding='same')(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2,2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                activation='relu')(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                activation='relu')(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2, padding='same')(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                activation='relu')(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                activation='relu')(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4, strides=2, padding='same')(y_a)

    y_a = Flatten()(y_a)



    # Merge the subnetworks
    y = Concatenate()([y_i, y_a])
    y = Dense(128, activation='relu')(y)
    y = Dense(2, activation='softmax')(y)

    m = Model(inputs=[x_i, x_a], outputs=y)
    return m, [x_i, x_a], y


MODELS = {'cnn_L3_orig': construct_cnn_L3_orig}