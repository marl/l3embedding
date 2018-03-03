import csv
import logging
import os
import random
import pescador

import numpy as np

import data.usc.features as cls_features
from log import LogTimer

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


def load_us8k_metadata(path):
    """
    Load UrbanSound8K metadata

    Args:
        path: Path to metadata csv file
              (Type: str)

    Returns:
        metadata: List of metadata dictionaries
                  (Type: list[dict[str, *]])
    """
    metadata = [{} for _ in range(10)]
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fname = row['slice_file_name']
            row['start'] = float(row['start'])
            row['end'] = float(row['end'])
            row['salience'] = float(row['salience'])
            fold_num = row['fold'] = int(row['fold'])
            row['classID'] = int(row['classID'])
            metadata[fold_num-1][fname] = row

    return metadata


def get_us8k_folds(metadata_path, data_dir, l3embedding_model=None,
                   features='l3_stack', label_format='int', **feature_args):
    """
    Load all of the data for each fold

    Args:
        metadata_path: Path to metadata file
                       (Type: str)

        data_dir: Path to data directory
                  (Type: str)

    Keyword Args:
        l3embedding_model: L3 embedding model, used if L3 features are used
                           (Type: keras.engine.training.Model or None)

        features: Type of features to be computed
                  (Type: str)

        label_format: Type of format used for encoding outputs
                      (Type: str)

    Returns:
        fold_data: List of data for each fold
                   (Type: list[tuple[np.ndarray, np.ndarray]])

    """
    raise NotImplementedError()


def get_us8k_fold_data(metadata, data_dir, fold_idx, l3embedding_model=None,
                       features='l3_stack', label_format='int', **feature_args):
    """
    Load all of the data for a specific fold

    Args:
        metadata: List of metadata dictionaries
                  (Type: list[dict[str,*]])

        data_dir: Path to data directory
                  (Type: str)

        fold_idx: Index of fold to load
                  (Type: int)

    Keyword Args:
        l3embedding_model: L3 embedding model, used if L3 features are used
                           (Type: keras.engine.training.Model or None)

        features: Type of features to be computed
                  (Type: str)

        label_format: Type of format used for encoding outputs
                      (Type: str)

    Returns:
        X: Feature data
           (Type: np.ndarray)
        y: Label data
           (Type: np.ndarray)

    """
    X = []
    y = []

    raise NotImplementedError()


def get_us8k_fold_split(fold_data, fold_idx, frame_features, shuffle=True):
    """
    Given the fold to use as held out, return the data split between training
    and testing

    Args:
        fold_data: List of data for each fold
                   (Type: list[tuple[np.ndarray, np.ndarray]])

        fold_idx: Fold to use as held out data
                  (Type: int)

        frame_features: If True, training frame features organized by file are
                        flattened and labels are replicated accordingly
                        (Type: bool)

    Returns:
        X_train: Training feature data
                 (Type: np.ndarray)
        y_train: Training label data
                 (Type: np.ndarray)
        X_test: Testing feature data
                (Type: np.ndarray)
        y_test: Testing label data
                (Type: np.ndarray)
    """
    X_test, y_test = fold_data[fold_idx]
    train = fold_data[:fold_idx] + fold_data[fold_idx+1:]

    X_train, y_train = zip(*train)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # If features are computed framewise flatten the audio file dimension
    if frame_features:
        X_train, y_train = cls_features.flatten_file_frames(X_train, y_train)

    if shuffle:
        train_idxs = np.arange(X_train.shape[0])
        np.random.shuffle(train_idxs)
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]

    return X_train, y_train, X_test, y_test


def generate_us8k_folds(metadata_path, data_dir, output_dir, l3embedding_model=None,
                        features='l3_stack', label_format='int',
                        random_state=12345678, **feature_args):
    """
    Generate all of the data for each fold

    Args:
        metadata_path: Path to metadata file
                       (Type: str)

        data_dir: Path to data directory
                  (Type: str)

        output_dir: Path to output directory where fold data will be stored
                    (Type: str)

    Keyword Args:
        l3embedding_model: L3 embedding model, used if L3 features are used
                           (Type: keras.engine.training.Model or None)

        features: Type of features to be computed
                  (Type: str)

        label_format: Type of format used for encoding outputs
                      (Type: str)

    """
    LOGGER.info('Generating all folds.')
    metadata = load_us8k_metadata(metadata_path)

    for fold_idx in range(10):
        generate_us8k_fold_data(metadata, data_dir, fold_idx, output_dir,
                                l3embedding_model=l3embedding_model,
                                features=features, label_format=label_format,
                                random_state=random_state, **feature_args)


def generate_us8k_fold_data(metadata, data_dir, fold_idx, output_dir, l3embedding_model=None,
                            features='l3_stack', label_format='int',
                            random_state=12345678, **feature_args):
    """
    Generate all of the data for a specific fold

    Args:
        metadata: List of metadata dictionaries, or a path to a metadata file to be loaded
                  (Type: list[dict[str,*]] or str)

        data_dir: Path to data directory
                  (Type: str)

        fold_idx: Index of fold to load
                  (Type: int)

        output_dir: Path to output directory where fold data will be stored
                    (Type: str)

    Keyword Args:
        l3embedding_model: L3 embedding model, used if L3 features are used
                           (Type: keras.engine.training.Model or None)

        features: Type of features to be computed
                  (Type: str)

        label_format: Type of format used for encoding outputs
                      (Type: str)

    """

    if type(metadata) == str:
        metadata = load_us8k_metadata(metadata)

    # Set random seed
    random_state = random_state + fold_idx
    random.seed(random_state)
    np.random.seed(random_state)

    audio_fold_dir = os.path.join(data_dir, "fold{}".format(fold_idx+1))

    # Create fold directory if it does not exist
    output_fold_dir = os.path.join(output_dir, "fold{}".format(fold_idx+1))
    if not os.path.isdir(output_fold_dir):
        os.makedirs(output_fold_dir)

    LOGGER.info('Generating fold {} in {}'.format(fold_idx+1, output_fold_dir))

    num_files = len(metadata[fold_idx])

    for idx, (fname, example_metadata) in enumerate(metadata[fold_idx].items()):
        desc = '({}/{}) Processed {} -'.format(idx+1, num_files, fname)
        with LogTimer(LOGGER, desc, log_level=logging.DEBUG):
            generate_us8k_file_data(fname, example_metadata, audio_fold_dir,
                                    output_fold_dir, features, label_format,
                                    l3embedding_model, **feature_args)


def generate_us8k_file_data(fname, example_metadata, audio_fold_dir,
                            output_fold_dir, features, label_format,
                            l3embedding_model, **feature_args):
    audio_path = os.path.join(audio_fold_dir, fname)

    basename, _ = os.path.splitext(fname)
    output_path = os.path.join(output_fold_dir, basename + '.npz')

    if os.path.exists(output_path):
        LOGGER.info('File {} already exists'.format(output_path))
        return

    X = cls_features.compute_file_features(audio_path, features, l3embedding_model=l3embedding_model, **feature_args)

    # If we were not able to compute the features, skip this file
    if X is None:
        LOGGER.error('Could not generate data for {}'.format(audio_path))
        return

    class_label = example_metadata['classID']
    if label_format == 'int':
        y = class_label
    elif label_format == 'one_hot':
        y = cls_features.one_hot(class_label)
    else:
        raise ValueError('Invalid label format: {}'.format(label_format))

    np.savez_compressed(output_path, X=X, y=y)

    return output_path, 'success'


