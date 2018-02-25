import csv
import logging
import os
import numpy as np
import classifier.features as cls_features

LOGGER = logging.getLogger('us8k')
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
    metadata = load_us8k_metadata(metadata_path)

    fold_data = []
    for fold_idx in range(10):
        LOGGER.info("Loading fold {}...".format(fold_idx+1))
        fold_data.append(get_us8k_fold_data(metadata, data_dir, fold_idx,
                                            l3embedding_model=l3embedding_model,
                                            features=features, label_format=label_format,
                                            **feature_args))

    return fold_data


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

    for idx, (fname, example_metadata) in enumerate(metadata[fold_idx].items()):
        path = os.path.join(data_dir, "fold{}".format(fold_idx+1), fname)

        if features.startswith('l3') and not l3embedding_model:
            raise ValueError('Must provide L3 embedding model to use {} features'.format(features))

        if features == 'l3_stack':
            hop_size = feature_args.get('hop_size', 0.25)
            file_features = cls_features.get_l3_stack_features(path, l3embedding_model,
                                                          hop_size=hop_size)
        elif features == 'l3_stats':
            hop_size = feature_args.get('hop_size', 0.25)
            file_features = cls_features.get_l3_stats_features(path, l3embedding_model,
                                                          hop_size=hop_size)
        elif features == 'l3_frames_uniform':
            hop_size = feature_args.get('hop_size', 0.25)
            file_features = cls_features.get_l3_frames_uniform(path, l3embedding_model,
                                                          hop_size=hop_size)
        elif features == 'l3_frames_random':
            num_samples = feature_args.get('num_random_samples')
            if not num_samples:
                raise ValueError('Must specify "num_samples" for "l3_frame_random" features')
            file_features = cls_features.get_l3_frames_random(path, l3embedding_model,
                                                         num_samples)
        else:
            raise ValueError('Invalid feature type: {}'.format(features))

        # If we were not able to compute the features, skip this file
        if file_features is None:
            continue

        X.append(file_features)

        class_label = example_metadata['classID']
        if label_format == 'int':
            y.append(class_label)
        elif label_format == 'one_hot':
            y.append(cls_features.one_hot(class_label))
        else:
            raise ValueError('Invalid label format: {}'.format(label_format))

    return np.array(X), np.array(y)


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
