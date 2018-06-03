import os
import numpy as np

from .us8k import NUM_FOLDS as NUM_FOLDS_US8K
from .esc50 import NUM_FOLDS as NUM_FOLDS_ESC50
from .dcase2013 import NUM_FOLDS as NUM_FOLDS_DCASE2013


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


def get_fold(feature_dir, fold_idx, augment=False):
    X = []
    y = []
    file_idxs = []
    fold_dir = os.path.join(feature_dir, 'fold{}'.format(fold_idx + 1))

    filenames = os.listdir(fold_dir)

    start_idx = 0
    for feature_filename in filenames:
        # Hack for skipping augmented files for US8K
        if 'us8k' in fold_dir and '_' in feature_filename and not augment:
            continue

        feature_filepath = os.path.join(fold_dir, feature_filename)
        file_X, file_y = load_feature_file(feature_filepath)

        if file_X.ndim > 1:
            end_idx = start_idx + file_X.shape[0]
        else:
            end_idx = start_idx + 1

        X.append(file_X)
        y.append(file_y)
        file_idxs.append([start_idx, end_idx])

        start_idx = end_idx

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
    train_data = get_train_folds(feature_dir, test_fold_idx, num_folds, valid=valid)
    if valid:
        valid_data = get_fold(feature_dir, get_valid_fold_idx(test_fold_idx, num_folds))
    else:
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
