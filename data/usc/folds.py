import os
import random
import pickle as pk
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .us8k import NUM_FOLDS as NUM_FOLDS_US8K
from .esc50 import NUM_FOLDS as NUM_FOLDS_ESC50
from .dcase2013 import NUM_FOLDS as NUM_FOLDS_DCASE2013
from .features import expand_framewise_labels


DATASET_NUM_FOLDS = {
    'us8k': NUM_FOLDS_US8K,
    'esc50': NUM_FOLDS_ESC50,
    'dcase2013': NUM_FOLDS_DCASE2013
}


def load_feature_file(feature_filepath):
    data = np.load(feature_filepath)
    X, y = data['X'], data['y']
    if type(y) == np.ndarray and y.ndim == 0:
        y = int(y)
    return X, y


def get_fold(feature_dir, fold_idx, augment=False, train_filenames_to_idxs=None):
    X = []
    y = []
    file_idxs = []
    fold_dir = os.path.join(feature_dir, 'fold{}'.format(fold_idx + 1))

    filenames_ = os.listdir(fold_dir)

    if train_filenames_to_idxs:
        nonaugmented_files = set([os.path.basename(x)
                                  for x in train_filenames_to_idxs.keys()])
    filenames = []

    start_idx = 0
    for feature_filename in filenames_:
        # Hack for skipping augmented files for US8K
        if 'us8k' in fold_dir and '_' in feature_filename and not augment:
            continue

        feature_filepath = os.path.join(fold_dir, feature_filename)
        file_X, file_y = load_feature_file(feature_filepath)

        if train_filenames_to_idxs and feature_filename in nonaugmented_files:
            train_idxs = np.array(train_filenames_to_idxs[feature_filename])
            file_X = file_X[train_idxs]

        if file_X.ndim > 1:
            end_idx = start_idx + file_X.shape[0]
        else:
            end_idx = start_idx + 1

        X.append(file_X)
        y.append(file_y)
        file_idxs.append([start_idx, end_idx])


        start_idx = end_idx
        filenames.append(feature_filename)

    X = np.vstack(X)

    if type(y[0]) == int or y[0].ndim == 0:
        y = np.array(y)
    else:
        y = np.concatenate(y)

    file_idxs = np.array(file_idxs)

    return {'features': X, 'labels': y, 'file_idxs': file_idxs, 'filenames': filenames}


def get_split(feature_dir, test_fold_idx, dataset_name, valid=True):
    if dataset_name not in DATASET_NUM_FOLDS:
        raise ValueError('Invalid dataset: {}'.format(dataset_name))
    num_folds = DATASET_NUM_FOLDS[dataset_name]
    #train_data = get_train_folds(feature_dir, test_fold_idx, num_folds, valid=valid)
    train_data = None
    """
    if valid:
        valid_data = get_fold(feature_dir, get_valid_fold_idx(test_fold_idx, num_folds))
    else:
        valid_data = None
    """
    valid_data = None
    test_data = get_fold(feature_dir, test_fold_idx)

    return train_data, valid_data, test_data


def get_valid_fold_idx(test_fold_idx, num_folds):
    return (test_fold_idx - 1) % num_folds


def get_train_folds(feature_dir, test_fold_idx, num_folds, valid=True):
    X = []
    y = []
    file_idxs = []
    filenames = []

    valid_fold_idx = get_valid_fold_idx(test_fold_idx, num_folds)

    for fold_idx in range(num_folds):
        if fold_idx == test_fold_idx or (valid and fold_idx == valid_fold_idx):
            continue

        fold_data = get_fold(feature_dir, fold_idx, augment=True)

        X.append(fold_data['features'])
        y.append(fold_data['labels'])
        idxs = fold_data['file_idxs']
        if len(file_idxs) > 0:
            # Since we're appending all of the file indices together, increment
            # the current fold indices by the current global index
            idxs = idxs + file_idxs[-1][-1, -1]
        file_idxs.append(idxs)

        filenames += fold_data['filenames']

    X = np.vstack(X)
    y = np.concatenate(y)
    file_idxs = np.vstack(file_idxs)

    return {'features': X, 'labels': y, 'file_idxs': file_idxs,
            'filenames': filenames}


def fold_sampler(feature_dir, standardizer_dir, fold_idx):
    stdizer_path = os.path.join(standardizer_dir, 'fold{}'.format(fold_idx+1))
    with open(stdizer_path, 'rb') as f:
        stdizer = pk.load(f)

    fold_dir = os.path.join(feature_dir, "fold{}".format(fold_idx+1))
    batches = os.listdir(fold_dir)
    while True:
        random.shuffle(batches)
        for batch_fname in batches:
            batch_path = os.path.join(fold_dir, batch_fname)
            data = np.load(batch_path)
            X = data['X']
            y = data['y']

            shuffle_idxs = np.random.permutation(X.shape[0])
            X = X[shuffle_idxs]
            y = y[shuffle_idxs]

            for X_pt, y_pt in zip(X, y):
                y_pt_oh = np.zeros(10,)
                y_pt_oh[y_pt] = 1.0
                yield {'features': X_pt, 'labels': y_pt_oh}


def get_train_folds_generator(feature_dir, standardizer_dir, test_fold_idx, num_folds, valid=True, batch_size=32):
    import pescador

    valid_fold_idx = get_valid_fold_idx(test_fold_idx, num_folds)

    seeds = []
    for fold_idx in range(num_folds):
        if fold_idx == test_fold_idx or (valid and fold_idx == valid_fold_idx):
            continue
        streamer = pescador.Streamer(fold_sampler, feature_dir,
                                     standardizer_dir, fold_idx)
        seeds.append(streamer)

    # Randomly shuffle the seeds
    random.shuffle(seeds)

    mux = pescador.ShuffledMux(seeds)

    return pescador.maps.keras_tuples(pescador.maps.buffer_stream(mux, batch_size), inputs='features', outputs='labels')

def get_train_validation_nonaugmented_split(feature_dir, fold_idx, valid_ratio=0.15):
    data = get_fold(feature_dir, fold_idx)
    # Create mapping from indices to filenames
    idx_to_filename_and_idx = []
    for filename, (start_idx, end_idx) in zip(data['filenames'], data['file_idxs']):
        for idx in range(end_idx - start_idx):
            idx_to_filename_and_idx.append({
                'filename': filename,
                'idx': idx
            })

    idx_to_filename_and_idx = np.array(idx_to_filename_and_idx)

    expand_framewise_labels(data)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio)
    train_idxs, valid_idxs = next(splitter.split(data['features'],
                                                 data['labels']))

    X_valid = data['features'][valid_idxs]
    y_valid = data['labels'][valid_idxs]

    train_filenames_and_idxs = idx_to_filename_and_idx[train_idxs]
    train_filename_idxs = {}
    for filename_item in train_filenames_and_idxs:
        filename = filename_item['filename']
        idx = filename_item['idx']

        if filename not in train_filename_idxs:
            train_filename_idxs[filename] = []

        train_filename_idxs[filename].append(idx)


    return train_filename_idxs, {'features': X_valid, 'labels': y_valid}


def load_nonaugmented_validation_data(valid_data_dir, test_fold_idx):
    X_valid = []
    y_valid = []
    for fold_idx in range(10):
        if fold_idx == test_fold_idx:
            continue

        valid_fold_data_path = os.path.join(valid_data_dir, "fold{}.npz".format(fold_idx+1))
        valid_fold_data = np.load(valid_fold_data_path)

        X_valid.append(valid_fold_data['features'])
        y_valid.append(valid_fold_data['labels'])

    X_valid = np.vstack(X_valid)
    y_valid = np.concatenate(y_valid)

    return {'features': X_valid, 'labels': y_valid}
