from keras.models import Model
from keras.layers import concatenate, Dense
import keras.regularizers as regularizers
from .vision_model import construct_cnn_L3_orig_vision_model
from .audio_model import construct_cnn_L3_orig_audio_model


def L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a):
    """
    Merges the audio and vision subnetworks and adds additional fully connected
    layers in the fashion of the model used in Look, Listen and Learn

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
    # Merge the subnetworks
    weight_decay = 1e-5
    l2_weight_decay = regularizers.l2(weight_decay)
    y = concatenate([vision_model(x_i), audio_model(x_a)])
    y = Dense(128, activation='relu', kernel_regularizer=l2_weight_decay)(y)
    y = Dense(2, activation='softmax', kernel_regularizer=l2_weight_decay)(y)
    m = Model(inputs=[x_i, x_a], outputs=y)

    return m, [x_i, x_a], y


def construct_cnn_L3_orig():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

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
    vision_model, x_i, y_i = construct_cnn_L3_orig_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_orig_audio_model()

    return L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a)


MODELS = {
    'cnn_L3_orig': construct_cnn_L3_orig,
}
