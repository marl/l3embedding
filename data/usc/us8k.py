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



def load_feature_file(feature_filepath):
    data = np.load(feature_filepath)
    X, y = data['X'], data['y']
    if type(y) == np.ndarray and y.ndim == 0:
        y = int(y)
    return X, y


def us8k_file_sampler(feature_filepath, shuffle=True):
    X, y = load_feature_file(feature_filepath)

    num_frames = X.shape[0]

    if shuffle:
        frame_idxs = np.random.permutation(num_frames)
    else:
        frame_idxs = range(num_frames)

    for idx in frame_idxs:
        yield {
            'features': X[idx],
            'label': y,
            'filepath': feature_filepath,
            'frame_idx': idx
        }



def get_us8k_batch_generator(features_dir, test_fold_idx, valid=False, num_streamers=None,
                             batch_size=64, random_state=20171021,
                             rate=None, cycle=True):

    random.seed(random_state)
    np.random.seed(random_state)


    LOGGER.info("Loading subset list")

    LOGGER.info("Creating streamers...")
    seeds = []

    valid_fold_idx = (test_fold_idx - 1) % 10

    for fold_dirname in os.listdir(features_dir):
        fold_dir = os.path.join(features_dir, fold_dirname)

        if not os.path.isdir(fold_dir):
            continue

        fold_idx = int(fold_dirname.replace('fold', '')) - 1
        if fold_idx == test_fold_idx:
            continue

        if valid and fold_idx != valid_fold_idx:
            continue

        fold_dir = os.path.join(features_dir, fold_dirname)
        for feature_filename in os.listdir(fold_dir):
            feature_filepath = os.path.join(fold_dir, feature_filename)
            streamer = pescador.Streamer(us8k_file_sampler, feature_filepath)
            seeds.append(streamer)

        if valid:
            break

    # Randomly shuffle the seeds
    random.shuffle(seeds)

    if num_streamers is None:
        num_streamers = len(seeds)

    mux = pescador.Mux(seeds, num_streamers, rate=rate, random_state=random_state)
    if cycle:
        mux = mux.cycle()

    if batch_size == 1:
        return mux
    else:
        return pescador.maps.buffer_stream(mux, batch_size)


def get_us8k_batch(features_dir, test_fold_idx, valid=False, num_streamers=None,
                   batch_size=64, random_state=20171021,
                   rate=None, cycle=True):
    gen = iter(get_us8k_batch_generator(features_dir, test_fold_idx, valid=valid,
                                   num_streamers=num_streamers, batch_size=batch_size,
                                   random_state=random_state, rate=rate, cycle=cycle))
    return next(gen)


def load_test_fold(feature_dir, fold_idx):
    X = []
    y = []
    file_idxs = []
    fold_dir = os.path.join(feature_dir, 'fold{}'.format(fold_idx + 1))

    filenames = os.listdir(fold_dir)

    start_idx = 0
    for feature_filename in filenames:
        feature_filepath = os.path.join(fold_dir, feature_filename)
        file_X, file_y = load_feature_file(feature_filepath)

        if file_X.ndim > 1:
            end_idx = start_idx + file_X.shape[0]
        else:
            end_idx = start_idx + 1

        X.append(X)
        y.append(file_y)
        file_idxs.append((start_idx, end_idx))

        start_idx = end_idx

    X = np.vstack(X)

    if y[0].ndim == 0:
        y = np.array(y)
    else:
        y = np.concatenate(y)

    return {'features': X, 'labels': y, 'file_idxs': file_idxs, 'filenames': filenames}


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


