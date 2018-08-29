import csv
import logging
import os
import glob
import random
import numpy as np

import data.usc.features as cls_features
from log import LogTimer

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


NUM_FOLDS = 10

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
    metadata = [{} for _ in range(NUM_FOLDS)]
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



def generate_us8k_folds(metadata_path, data_dir, output_dir, l3embedding_model=None,
                        features='l3', random_state=12345678, **feature_args):
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

    """
    LOGGER.info('Generating all folds.')
    metadata = load_us8k_metadata(metadata_path)

    for fold_idx in range(NUM_FOLDS):
        generate_us8k_fold_data(metadata, data_dir, fold_idx, output_dir,
                                l3embedding_model=l3embedding_model,
                                features=features, random_state=random_state,
                                **feature_args)


def generate_us8k_fold_data(metadata, data_dir, fold_idx, output_dir, l3embedding_model=None,
                            features='l3', random_state=12345678, **feature_args):
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
            # TODO: Make sure glob doesn't catch things with numbers afterwards
            variants = [x for x in glob.glob(os.path.join(audio_fold_dir,
                '**', os.path.splitext(fname)[0] + '[!0-9]*[wm][ap][v3]'), recursive=True)
                if os.path.isfile(x) and not x.endswith('.jams')]
            num_variants = len(variants)
            for var_idx, var_path in enumerate(variants):
                audio_dir = os.path.dirname(var_path)
                var_fname = os.path.basename(var_path)
                desc = '\t({}/{}) Variants {} -'.format(var_idx+1, num_variants, var_fname)
                with LogTimer(LOGGER, desc, log_level=logging.DEBUG):
                    generate_us8k_file_data(var_fname, example_metadata, audio_dir,
                                            output_fold_dir, features,
                                            l3embedding_model, **feature_args)


def generate_us8k_file_data(fname, example_metadata, audio_fold_dir,
                            output_fold_dir, features,
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
    y = class_label

    np.savez_compressed(output_path, X=X, y=y)

    return output_path, 'success'


