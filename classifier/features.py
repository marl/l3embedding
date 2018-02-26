import logging
import warnings
import librosa
import numpy as np
import scipy as sp

LOGGER = logging.getLogger('classifier-features')
LOGGER.setLevel(logging.DEBUG)


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


def get_l3_stack_features(audio_path, l3embedding_model, hop_size=0.25):
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
        hop_size: Hop size as a fraction of the window
                  (Type: float)

    Returns:
        features:  Feature vector
                   (Type: np.ndarray)
    """
    sr = 48000
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
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


def get_l3_stats_features(audio_path, l3embedding_model, hop_size=0.25):
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
        hop_size: Hop size as a fraction of the window
                  (Type: float)

    Returns:
        features:  Feature vector
                   (Type: np.ndarray)
    """
    sr = 48000
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    hop_length = int(hop_size * sr)
    frame_length = 48000 * 1

    audio_length = len(audio)
    if audio_length < (frame_length + 2*hop_length):
        # Make sure we can have at least three frames so that we can compute
        # all of the stats.
        pad_length = frame_length + 2*hop_length - audio_length
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

    # Compute statistics on the time series of embeddings
    minimum = np.min(l3embedding, axis=0)
    maximum = np.max(l3embedding, axis=0)
    median = np.median(l3embedding, axis=0)
    mean = np.mean(l3embedding, axis=0)
    var = np.var(l3embedding, axis=0)
    skewness = sp.stats.skew(l3embedding, axis=0)
    kurtosis = sp.stats.kurtosis(l3embedding, axis=0)

    # Compute statistics on the first and second derivatives of time series of embeddings

    # Use finite differences to approximate the derivatives
    d1 = np.gradient(l3embedding, 1/sr, edge_order=1, axis=0)
    d2 = np.gradient(l3embedding, 1/sr, edge_order=2, axis=0)

    d1_mean = np.mean(d1, axis=0)
    d1_var = np.var(d1, axis=0)

    d2_mean = np.mean(d2, axis=0)
    d2_var = np.var(d2, axis=0)

    return np.concatenate((minimum, maximum, median, mean, var, skewness, kurtosis,
                           d1_mean, d1_var, d2_mean, d2_var))


def get_l3_frames_uniform(audio_path, l3embedding_model, hop_size=0.25):
    """
    Get L3 embedding stats features, i.e. compute statistics for each of the
    embedding features across 1 second (overlapping) window of the given audio

    Args:
        audio_path: Path to audio file
                    (Type: str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Keyword Args:
        hop_size: Hop size as a fraction of the window
                  (Type: float)

    Returns:
        features:  List of embedding vectors
                   (Type: list[np.ndarray])
    """
    sr = 48000
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    hop_size = 0.25 # REVISIT
    hop_length = int(hop_size * sr)
    frame_length = 48000 * 1

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
    l3embedding = l3embedding_model.predict(x).T

    return list(l3embedding)

def get_l3_frames_random(audio, l3embedding_model, num_samples):
    """
    Get L3 embedding stats features, i.e. compute statistics for each of the
    embedding features across 1 second (overlapping) window of the given audio

    Args:
        audio_path: numpy array or ath to audio file
                    (Type: np.ndarray or str)

        l3embedding_model:  Audio embedding model
                            (Type: keras.engine.training.Model)

        num_samples: Number of samples
                     (Type: int)

    Returns:
        features:  List of embedding vectors
                   (Type: list[np.ndarray])
    """
    sr = 48000

    if type(audio) == str:
        audio, _ = librosa.load(audio, sr=sr, mono=True)

    frame_length = 48000 * 1

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
        frame_l3embedding = l3embedding_model.predict(x).T.flatten()

        l3embedding = np.tile(frame_l3embedding, (num_samples, 1))

    return list(l3embedding)


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
