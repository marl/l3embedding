import logging
import os
import warnings
import librosa
import numpy as np
import scipy as sp
import soundfile as sf
import resampy
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .vggish import vggish_input
from .vggish import vggish_postprocess
from .vggish import vggish_slim

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


def load_audio(path, sr):
    """
    Load audio file
    """
    data, sr_orig = sf.read(path, dtype='float32', always_2d=True)
    data = data.mean(axis=-1)

    if sr_orig != sr:
        data = resampy.resample(data, sr_orig, sr)

    return data


def one_hot(idx, n_classes=10):
    """
    Creates a one hot encoding vector

    Args:
        idx:  Class index
              (Type: int)

    Keyword Args:
        n_classes: Number of classes
                   (Type: int)

    Returns:
        one_hot_vector: One hot encoded vector
                        (Type: np.ndarray)
    """
    y = np.zeros((n_classes,))
    y[idx] = 1
    return y


def framewise_to_stats(data):
    X = []
    for start_idx, end_idx in data['file_idxs']:
        X.append(compute_stats_features(data['features'][start_idx:end_idx]))

    data['features'] = np.vstack(X)

    idxs = np.arange(data['features'].shape[0])
    data['file_idx'] = np.column_stack((idxs, idxs + 1))


def expand_framewise_labels(data):
    labels = []
    for y, (start_idx, end_idx) in zip(data['labels'], data['file_idxs']):
        num_frames = end_idx - start_idx
        labels.append(np.tile(y, num_frames))

    data['labels'] = np.concatenate(labels)


def preprocess_split_data(train_data, valid_data, test_data,
                          feature_mode='framewise'):
    # NOTE: This function mutates data so there aren't extra copies

    # Apply min max scaling to data
    min_max_scaler = MinMaxScaler()
    train_data['features'] = min_max_scaler.fit_transform(
        train_data['features'])
    valid_data['features'] = min_max_scaler.transform(valid_data['features'])
    test_data['features'] = min_max_scaler.transform(test_data['features'])

    if feature_mode == 'framewise':
        # Expand training and validation labels to apply to each frame
        expand_framewise_labels(train_data)
        expand_framewise_labels(valid_data)
    elif feature_mode == 'stats':
        # Summarize frames in each file using summary statistics
        framewise_to_stats(train_data)
        framewise_to_stats(valid_data)
        framewise_to_stats(test_data)
    else:
        raise ValueError('Invalid feature mode: {}'.format(feature_mode))

    # Standardize features
    stdizer = StandardScaler()
    train_data['features'] = stdizer.fit_transform(train_data['features'])
    valid_data['features'] = stdizer.transform(valid_data['features'])
    test_data['features'] = stdizer.transform(test_data['features'])

    return min_max_scaler, stdizer


def preprocess_features(data, min_max_scaler, stdizer,
                        feature_mode='framewise'):
    data['features'] = min_max_scaler.fit_transform(data['features'])
    if feature_mode == 'framewise':
        # Expand training and validation labels to apply to each frame
        expand_framewise_labels(data)
    elif feature_mode == 'stats':
        # Summarize frames in each file using summary statistics
        framewise_to_stats(data)
    else:
        raise ValueError('Invalid feature mode: {}'.format(feature_mode))
    data['features'] = stdizer.transform(data['features'])


def extract_vggish_embedding(audio_path, input_op_name='vggish/input_features',
                             output_op_name='vggish/embedding',
                             resources_dir=None, **params):
    # TODO: Make more efficient so we're not loading model every time we extract features
    fs = params.get('target_sample_rate', 16000)
    frame_win_sec = params.get('frame_win_sec', 0.96)
    audio_data = load_audio(audio_path, fs)

    # For some reason 0.96 doesn't work, padding to 0.975 empirically works
    frame_samples = int(np.ceil(fs * max(frame_win_sec, 0.975)))
    if audio_data.shape[0] < frame_samples:
        pad_length = frame_samples - audio_data.shape[0]
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')

    if not resources_dir:
        resources_dir = os.path.join(os.path.dirname(__file__), '../../resources/vggish')

    pca_params_path = os.path.join(resources_dir, 'vggish_pca_params.npz')
    model_path = os.path.join(resources_dir, 'vggish_model.ckpt')


    examples_batch = vggish_input.waveform_to_examples(audio_data, fs, **params)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params_path, **params)

    # If needed, prepare a record writer to store the postprocessed embeddings.
    #writer = tf.python_io.TFRecordWriter(
    #    FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

    input_tensor_name = input_op_name + ':0'
    output_tensor_name = output_op_name + ':0'

    with tf.Graph().as_default(), tf.Session() as sess:
      # Define the model in inference mode, load the checkpoint, and
      # locate input and output tensors.
      vggish_slim.define_vggish_slim(training=False, **params)
      vggish_slim.load_vggish_slim_checkpoint(sess, model_path, **params)
      features_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
      embedding_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

      # Run inference and postprocessing.
      [embedding_batch] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: examples_batch})
      postprocessed_batch = pproc.postprocess(embedding_batch, **params)

      # Write the postprocessed embeddings as a SequenceExample, in a similar
      # format as the features released in AudioSet. Each row of the batch of
      # embeddings corresponds to roughly a second of audio (96 10ms frames), and
      # the rows are written as a sequence of bytes-valued features, where each
      # feature value contains the 128 bytes of the whitened quantized embedding.

    return postprocessed_batch.astype(np.float32)


def get_vggish_frames_uniform(audio_path, hop_size=0.1):
    """
    Get vggish embedding features for each frame in the given audio file

    Args:
        audio: Audio data or path to audio file
               (Type: np.ndarray or str)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Array of embedding vectors
                   (Type: np.ndarray)
    """
    return extract_vggish_embedding(audio_path, frame_hop_sec=hop_size)


def get_vggish_stats(audio_path, hop_size=0.1):
    """
    Get vggish embedding features summary statistics for all frames in the
    audio file

    Args:
        audio: Audio data or path to audio file
               (Type: np.ndarray or str)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Array of embedding vectors
                   (Type: np.ndarray)
    """
    vggish_embedding = extract_vggish_embedding(audio_path, frame_hop_sec=hop_size)
    return compute_stats_features(vggish_embedding)


def get_l3_stack_features(audio_path, l3embedding_model, hop_size=0.1):
    """
    Get stacked L3 embedding features, i.e. stack embedding features for each
    1 second (overlapping) window of the given audio

    Computes a _single_ feature for the given audio file

    Args:
        audio_path: Path to audio file
                    (Type: str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Feature vector
                   (Type: np.ndarray)
    """
    sr = 48000
    audio = load_audio(audio_path, sr)
    audio_length = len(audio)
    frame_length = sr
    hop_length = int(sr * hop_size)

    # Zero pad to 4 seconds
    target_len = 48000 * 4
    if audio_length < target_len:
        pad_length = target_len - audio_length
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio = np.pad(audio, (left_pad, right_pad), mode='constant')
    elif audio_length > target_len:
        # Take center of audio (ASSUMES NOT MUCH GREATER THAN TARGET LENGTH)
        center_sample = audio_length // 2
        half_len = target_len // 2
        audio = audio[center_sample-half_len:center_sample+half_len]



    # Divide into overlapping 1 second frames
    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    # Get the L3 embedding for each frame
    l3embedding = l3embedding_model.predict(x)

    # Return a flattened vector of the embeddings
    return l3embedding.flatten()


def compute_stats_features(embeddings):
    # Compute statistics on the time series of embeddings
    minimum = np.min(embeddings, axis=0)
    maximum = np.max(embeddings, axis=0)
    median = np.median(embeddings, axis=0)
    mean = np.mean(embeddings, axis=0)
    var = np.var(embeddings, axis=0)
    skewness = sp.stats.skew(embeddings, axis=0)
    kurtosis = sp.stats.kurtosis(embeddings, axis=0)

    return np.concatenate((minimum, maximum, median, mean, var, skewness, kurtosis))


def get_l3_stats_features(audio_path, l3embedding_model, hop_size=0.1):
    """
    Get L3 embedding stats features, i.e. compute statistics for each of the
    embedding features across 1 second (overlapping) window of the given audio

    Computes a _single_ feature for the given audio file


    Args:
        audio_path: Path to audio file
                    (Type: str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Feature vector
                   (Type: np.ndarray)
    """
    sr = 48000
    audio = load_audio(audio_path, sr)

    hop_length = int(hop_size * sr)
    frame_length = 48000 * 1

    audio_length = len(audio)
    if audio_length < frame_length:
        # Make sure we can have at least three frames so that we can compute
        # all of the stats.
        pad_length = frame_length - audio_length
    else:
        # Zero pad so we compute embedding on all samples
        pad_length = int(np.ceil(audio_length - frame_length)/hop_length) * hop_length \
                     - (audio_length - frame_length)

    if pad_length > 0:
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio = np.pad(audio, (left_pad, right_pad), mode='constant')


    # Divide into overlapping 1 second frames
    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    # Get the L3 embedding for each frame
    l3embedding = l3embedding_model.predict(x)

    return compute_stats_features(l3embedding)


def get_l3_frames_uniform(audio, l3embedding_model, hop_size=0.1, sr=48000):
    """
    Get L3 embedding for each frame in the given audio file

    Args:
        audio: Audio data or path to audio file
               (Type: np.ndarray or str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Keyword Args:
        hop_size: Hop size in seconds
                  (Type: float)

    Returns:
        features:  Array of embedding vectors
                   (Type: np.ndarray)
    """
    if type(audio) == str:
        audio = load_audio(audio, sr)

    hop_size = 0.25 # REVISIT
    hop_length = int(hop_size * sr)
    frame_length = sr * 1

    audio_length = len(audio)
    if audio_length < frame_length:
        # Make sure we can have at least one frame of audio
        pad_length = frame_length - audio_length
    else:
        # Zero pad so we compute embedding on all samples
        pad_length = int(np.ceil(audio_length - frame_length)/hop_length) * hop_length \
                     - (audio_length - frame_length)

    if pad_length > 0:
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio = np.pad(audio, (left_pad, right_pad), mode='constant')

    # Divide into overlapping 1 second frames
    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    # Get the L3 embedding for each frame
    l3embedding = l3embedding_model.predict(x)

    return l3embedding

def get_l3_frames_random(audio, l3embedding_model, num_samples, sr=48000):
    """
    Get L3 embedding for random frames in the given audio file

    Args:
        audio: Numpy array or path to audio file
               (Type: np.ndarray or str)

        l3embedding_model:  Audio embedding model
                            (Type: keras.engine.training.Model)

        num_samples: Number of samples
                     (Type: int)

    Returns:
        features:  Array of embedding vectors
                   (Type: np.ndarray)
    """
    if type(audio) == str:
        audio = load_audio(audio, sr)

    frame_length = sr * 1

    audio_length = len(audio)
    pad_length = frame_length - audio_length

    if pad_length > 0:
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio = np.pad(audio, (left_pad, right_pad), mode='constant')

    if audio_length != frame_length:
        sample_start_idxs = np.random.randint(low=0,
                                              high=audio_length - frame_length,
                                              size=num_samples)

        x = []
        for start_idx in sample_start_idxs:
            end_idx = start_idx + frame_length
            x.append(audio[start_idx:end_idx])


        x = np.array(x)
        x = x.reshape((x.shape[0], 1, x.shape[-1]))
        l3embedding = l3embedding_model.predict(x).T

    else:
        warn_msg = 'Replicating samples'
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)

        x = audio.reshape((1,1,audio.shape[0]))
        x = x.reshape((x.shape[0], 1, x.shape[-1]))
        frame_l3embedding = l3embedding_model.predict(x).flatten()

        l3embedding = np.tile(frame_l3embedding, (num_samples, 1))

    return l3embedding


def compute_file_features(path, feature_type, l3embedding_model=None, **feature_args):
    if feature_type.startswith('l3') and not l3embedding_model:
        err_msg = 'Must provide L3 embedding model to use {} features'
        raise ValueError(err_msg.format(feature_type))

    if feature_type == 'l3_stack':
        hop_size = feature_args.get('hop_size', 0.1)
        file_features = get_l3_stack_features(path, l3embedding_model,
                                              hop_size=hop_size)
    elif feature_type == 'l3_stats':
        hop_size = feature_args.get('hop_size', 0.1)
        file_features = get_l3_stats_features(path, l3embedding_model,
                                              hop_size=hop_size)
    elif feature_type == 'l3_frames_uniform':
        hop_size = feature_args.get('hop_size', 0.1)
        file_features = get_l3_frames_uniform(path, l3embedding_model,
                                              hop_size=hop_size)
    elif feature_type == 'l3_frames_random':
        num_samples = feature_args.get('num_random_samples')
        if not num_samples:
            raise ValueError('Must specify "num_samples" for "l3_frame_random" features')
        file_features = get_l3_frames_random(path, l3embedding_model,
                                             num_samples)
    elif feature_type == 'vggish_stats':
        hop_size = feature_args.get('hop_size', 0.1)
        file_features = get_vggish_stats(path, hop_size=hop_size)
    elif feature_type == 'vggish_frames_uniform':
        hop_size = feature_args.get('hop_size', 0.1)
        file_features = get_vggish_frames_uniform(path, hop_size=hop_size)
    else:
        raise ValueError('Invalid feature type: {}'.format(feature_type))

    return file_features


def flatten_file_frames(X, y):
    """
    For data organized by file, flattens the frame features for training data
    and duplicates the file labels for each frame

    Args:
        X: Framewise feature data, which consists of lists of frame features
           for each file
                 (Type: np.ndarray[list[np.ndarray]])
        y: Label data for each file
                 (Type: np.ndarray)

    Returns:
        X_flattened: Flatten matrix of frame features
                     (Type: np.ndarray)
        y_flattened: Label data for each frame
                     (Type: np.ndarray)

    """
    if X.ndim == 1:
        # In this case the number of frames per file varies, which makes the
        # numpy array store lists for each file
        num_frames_per_file = []
        X_flatten = []
        for X_file in X:
            num_frames_per_file.append(len(X_file))
            X_flatten += X_file
        X_flatten = np.array(X_flatten)
    else:
        # In this case the number of frames per file is the same, which means
        # all of the data is in a unified numpy array
        X_shape = X.shape
        num_files, num_frames_per_file = X_shape[0], X_shape[1]
        new_shape = (num_files * num_frames_per_file,) + X_shape[2:]
        X_flatten = X.reshape(new_shape)

    # Repeat the labels for each frame
    y_flatten = np.repeat(y, num_frames_per_file)

    return X_flatten, y_flatten
