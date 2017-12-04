from keras.models import Model
from keras.layers import concatenate, Dense
import keras.regularizers as regularizers
from .vision_model import construct_cnn_L3_orig_vision_model
from .audio_model import construct_cnn_L3_orig_audio_model


def L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, model_name):
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
    m.name = model_name

    return m, [x_i, x_a], y


def load_model(weights_path, model_type):
    """
    Loads an audio-visual correspondence model

    Args:
        weights_path:  Path to Keras weights file
                       (Type: str)
        model_type:    Name of model type
                       (Type: str)

    Returns:
        model:  Loaded model object
                (Type: keras.engine.training.Model)
    """
    if model_type not in MODELS:
        raise ValueError('Invalid model type: "{}"'.format(model_type))

    m, inputs, output = MODELS[model_type]()
    m.load_weights(weights_path)
    return m


def load_embedding(weights_path, model_type, embedding_type):
    """
    Loads an embedding model

    Args:
        weights_path:    Path to Keras weights file
                         (Type: str)
        model_type:      Name of model type
                         (Type: str)
        embedding_type:  Type of embedding to load ('audio' or 'vision')
                         (Type: str)

    Returns:
        model:  Loaded model object
                (Type: keras.engine.training.Model)
    """
    m = load_model(weights_path, model_type)
    if embedding_type == 'vision':
        m_emb = m.get_layer('vision_embedding')
    elif embedding_type == 'audio':
        m_emb = m.get_layer('audio_embedding')
    else:
        raise ValueError('Invalid embedding type: "{}"'.format(embedding_type))

    return m_emb


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

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_orig')
    return m


MODELS = {
    'cnn_L3_orig': construct_cnn_L3_orig,
}
