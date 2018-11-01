from keras.layers import concatenate, Dense
from .vision_model import *
from .audio_model import *
from .training_utils import multi_gpu_model


def L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, model_name, layer_size=128):
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
    y = concatenate([vision_model(x_i), audio_model(x_a)])
    y = Dense(layer_size, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Dense(2, activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    m = Model(inputs=[x_i, x_a], outputs=y)
    m.name = model_name

    return m, [x_i, x_a], y


def convert_num_gpus(model, inputs, outputs, model_type, src_num_gpus, tgt_num_gpus):
    """
    Converts a multi-GPU model to a model that uses a different number of GPUs

    If the model is single-GPU/CPU, the given model is returned

    Args:
        model:  Keras model
                (Type: keras.models.Model)

        inputs: Input Tensor.
                (Type: keras.layers.Input)

        outputs: Embedding output Tensor/Layer.
                 (Type: keras.layers.Layer)

        model_type: Name of model type
                    (Type: str)

        src_num_gpus: Number of GPUs the source model uses
                      (Type: int)

        tgt_num_gpus: Number of GPUs the converted model will use
                      (Type: int)

    Returns:
        model_cvt:  Embedding model object
                    (Type: keras.engine.training.Model)

        inputs_cvt: Input Tensor. Not returned if return_io is False.
                    (Type: keras.layers.Input)

        ouputs_cvt: Embedding output Tensor/Layer. Not returned if return_io is False.
                    (Type: keras.layers.Layer)
    """
    if src_num_gpus <= 1 and tgt_num_gpus <= 1:
        return model, inputs, outputs

    m_new, inputs_new, output_new = MODELS[model_type]()
    m_new.set_weights(model.layers[-2].get_weights())

    if tgt_num_gpus > 1:
        m_new = multi_gpu_model(m_new, gpus=tgt_num_gpus)

    return m_new, inputs_new, output_new


def load_model(weights_path, model_type, src_num_gpus=0, tgt_num_gpus=None, return_io=False):
    """
    Loads an audio-visual correspondence model

    Args:
        weights_path:  Path to Keras weights file
                       (Type: str)
        model_type:    Name of model type
                       (Type: str)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)

    Returns:
        model:  Loaded model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """
    if model_type not in MODELS:
        raise ValueError('Invalid model type: "{}"'.format(model_type))

    m, inputs, output = MODELS[model_type]()
    if src_num_gpus > 1:
        m = multi_gpu_model(m, gpus=src_num_gpus)
    m.load_weights(weights_path)

    if tgt_num_gpus is not None and src_num_gpus != tgt_num_gpus:
        m, inputs, output = convert_num_gpus(m, inputs, output, model_type,
                                             src_num_gpus, tgt_num_gpus)

    if return_io:
        return m, inputs, output
    else:
        return m


def load_embedding(weights_path, model_type, embedding_type, pooling_type,
                   src_num_gpus=0, tgt_num_gpus=None, return_io=False):
    """
    Loads an embedding model

    Args:
        weights_path:    Path to Keras weights file
                         (Type: str)
        model_type:      Name of model type
                         (Type: str)
        embedding_type:  Type of embedding to load ('audio' or 'vision')
                         (Type: str)
        pooling_type:    Type of pooling applied to final convolutional layer
                         (Type: str)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)

    Returns:
        model:  Embedding model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """
    m, inputs, output = load_model(weights_path, model_type, src_num_gpus=src_num_gpus,
                                   tgt_num_gpus=tgt_num_gpus, return_io=True)
    x_i, x_a = inputs
    if embedding_type == 'vision':
        m_embed_model = m.get_layer('vision_model')
        m_embed, x_embed, y_embed = construct_cnn_l3_orig_vision_embedding_model(m_embed_model, x_i)

    elif embedding_type == 'audio':
        m_embed_model = m.get_layer('audio_model')
        # m_embed, x_embed, y_embed = AUDIO_EMBEDDING_MODELS[model_type](m_embed_model, x_a)
        m_embed, x_embed, y_embed = convert_audio_model_to_embedding(m_embed_model, x_a, model_type, pooling_type)
    else:
        raise ValueError('Invalid embedding type: "{}"'.format(embedding_type))

    if return_io:
        return m_embed, x_embed, y_embed
    else:
        return m_embed


def gpu_wrapper(model_f):
    """
    Decorator for creating multi-gpu models
    """
    def wrapped(num_gpus=0, *args, **kwargs):
        m, inp, out = model_f(*args, **kwargs)
        if num_gpus > 1:
            m = multi_gpu_model(m, gpus=num_gpus)

        return m, inp, out

    return wrapped


@gpu_wrapper
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

@gpu_wrapper
def construct_cnn_L3_kapredbinputbn():
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
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_kapredbinputbn_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn')
    return m

@gpu_wrapper
def construct_cnn_L3_melspec1():
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
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec1_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_melspec1')
    return m

@gpu_wrapper
def construct_cnn_L3_melspec2():
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
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec2_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_melspec2')
    return m

@gpu_wrapper
def construct_tiny_L3():
    """
    Constructs a model that implements a small L3 model for validation purposes

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    vision_model, x_i, y_i = construct_tiny_L3_vision_model()
    audio_model, x_a, y_a = construct_tiny_L3_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'tiny_L3', layer_size=64)
    return m


MODELS = {
    'cnn_L3_orig': construct_cnn_L3_orig,
    'tiny_L3': construct_tiny_L3,
    'cnn_L3_kapredbinputbn': construct_cnn_L3_kapredbinputbn,
    'cnn_L3_melspec1': construct_cnn_L3_melspec1,
    'cnn_L3_melspec2': construct_cnn_L3_melspec2
}
