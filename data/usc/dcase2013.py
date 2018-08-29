import glob
import logging
import os
import random

import numpy as np

from data.usc.features import compute_file_features
from log import LogTimer

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


NUM_FOLDS = 2

CLASS_TO_INT = {
    'bus': 0,
    'busystreet': 1,
    'office': 2,
    'openairmarket': 3,
    'park': 4,
    'quietstreet': 5,
    'restaurant': 6,
    'supermarket': 7,
    'tube': 8,
    'tubestation': 9
}


def generate_dcase2013_folds(data_dir, output_dir, l3embedding_model=None,
                             features='l3', random_state=12345678, **feature_args):
    for fold_idx in range(NUM_FOLDS):
        generate_dcase2013_fold_data(data_dir, fold_idx, output_dir,
                                     l3embedding_model=l3embedding_model,
                                     features=features, random_state=random_state,
                                     **feature_args)


def generate_dcase2013_fold_data(data_dir, fold_idx, output_dir, l3embedding_model=None,
                                 features='l3', random_state=12345678, **feature_args):
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

    files = glob.glob(audio_fold_dir + '/*')
    num_files = len(files)

    for idx, f in enumerate(files):
        fname = f.split('/')[-1]
        desc = '({}/{}) Processed {} -'.format(idx+1, num_files, fname)
        with LogTimer(LOGGER, desc, log_level=logging.DEBUG):
            generate_dcase2013_file_data(fname, audio_fold_dir, output_fold_dir,
                                     features, l3embedding_model, **feature_args)


def generate_dcase2013_file_data(fname, audio_fold_dir, output_fold_dir,
                                 features, l3embedding_model, **feature_args):
    audio_path = os.path.join(audio_fold_dir, fname)

    basename, _ = os.path.splitext(fname)
    output_path = os.path.join(output_fold_dir, basename + '.npz')

    if os.path.exists(output_path):
        LOGGER.info('File {} already exists'.format(output_path))
        return

    X = compute_file_features(audio_path, features, l3embedding_model=l3embedding_model, **feature_args)

    # If we were not able to compute the features, skip this file
    if X is None:
        LOGGER.error('Could not generate data for {}'.format(audio_path))
        return

    class_label = CLASS_TO_INT[basename[:-2]]
    y = class_label

    np.savez_compressed(output_path, X=X, y=y)

    return output_path, 'success'
