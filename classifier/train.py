import os
import scipy as sp
import numpy as np
import soundfile as sf
import json
import csv
import logging
import librosa
import keras
from keras.optimizers import Adam
import keras.regularizers as regularizers
from keras.models import Model
from keras.layers import Input, Dense, Activation
from itertools import islice
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from l3embedding.train import sample_one_second, LossHistory
from l3embedding.model import load_embedding
from log import *

LOGGER = logging.getLogger('classifier')
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


def get_l3_stack_features(audio_path, l3embedding_model):
    """
    Get stacked L3 embedding features, i.e. stack embedding features for each
    1 second (overlapping) window of the given audio


    Args:
        audio_path: Path to audio file
                    (Type: str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Returns:
        features:  Feature vector
                   (Type: np.ndarray)
    """
    sr = 48000
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio_length = len(audio)

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
    x = librosa.util.utils.frame(audio, frame_length=sr * 1, hop_length=sr // 2).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    # Get the L3 embedding for each frame
    l3embedding = l3embedding_model.predict(x)

    # Return a flattened vector of the embeddings
    return l3embedding.flatten()


def get_l3_stats_features(audio_path, l3embedding_model):
    """
    Get L3 embedding stats features, i.e. compute statistics for each of the
    embedding features across 1 second (overlapping) window of the given audio

    Args:
        audio_path: Path to audio file
                    (Type: str)

        l3embedding_model:  Audio embedding model
                            (keras.engine.training.Model)

    Returns:
        features:  Feature vector
                   (Type: np.ndarray)
    """
    sr = 48000
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    hop_size = 0.25 # REVISIT
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


def get_us8k_folds(metadata, data_dir, l3embedding_model=None,
                   features='l3_stack', label_format='int'):
    """
    Load all of the data for each fold

    Args:
        metadata: List of metadata dictionaries
                  (Type: dict[str,*])

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
    fold_data = []
    for fold_idx in range(10):
        LOGGER.info("Loading fold {}...".format(fold_idx+1))
        fold_data.append(get_fold_data(metadata, data_dir, fold_idx,
                                       l3embedding_model=l3embedding_model,
                                       features=features, label_format=label_format))

    return fold_data


def get_fold_data(metadata, data_dir, fold_idx, l3embedding_model=None, features='l3_stack', label_format='int'):
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
            feature_vector = get_l3_stack_features(path, l3embedding_model)
        elif features == 'l3_stats':
            feature_vector = get_l3_stats_features(path, l3embedding_model)
        else:
            raise ValueError('Invalid feature type: {}'.format(features))

        # If we were not able to compute the features, skip this file
        if feature_vector is None:
            continue

        X.append(feature_vector)


        class_label = example_metadata['classID']
        if label_format == 'int':
            y.append(class_label)
        elif label_format == 'one_hot':
            y.append(one_hot(class_label))
        else:
            raise ValueError('Invalid label format: {}'.format(label_format))

    return np.array(X), np.array(y)


def get_fold_split(fold_data, fold_idx):
    """
    Given the fold to use as held out, return the data split between training
    and testing

    Args:
        fold_data: List of data for each fold
                   (Type: list[tuple[np.ndarray, np.ndarray]])

        fold_idx: Fold to use as held out data
                  (Type: int)

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

    return X_train, y_train, X_test, y_test


def train_svm(X_train, y_train, X_test, y_test, C=1e-4, verbose=False, **kwargs):
    """
    Train a Support Vector Machine model on the given data

    Args:
        X_train: Training feature data
                 (Type: np.ndarray)
        y_train: Training label data
                 (Type: np.ndarray)
        X_test: Testing feature data
                (Type: np.ndarray)
        y_test: Testing label data
                (Type: np.ndarray)

    Keyword Args:
        C: SVM regularization hyperparameter
           (Type: float)

        verbose:  If True, print verbose messages
                  (Type: bool)

    Returns:
        clf: Classifier object
             (Type: sklearn.svm.SVC)

        y_train_pred: Predicted train output of classifier
                     (Type: np.ndarray)

        y_test_pred: Predicted test output of classifier
                     (Type: np.ndarray)
    """
    # Standardize
    LOGGER.debug('Standardizing data...')
    stdizer = StandardScaler()
    X_train = stdizer.fit_transform(X_train)

    clf = SVC(C=C, verbose=verbose)
    LOGGER.debug('Fitting model to data...')
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)

    X_test = stdizer.transform(X_test)
    y_test_pred = clf.predict(X_test)

    return clf, y_train_pred, y_test_pred

def construct_mlp_model(input_shape, weight_decay=1e-5):
    weight_decay = 1e-5
    l2_weight_decay = regularizers.l2(weight_decay)
    inp = Input(shape=input_shape, dtype='float32')
    y = Dense(512, activation='relu', kernel_regularizer=l2_weight_decay)(inp)
    y = Dense(128, activation='relu', kernel_regularizer=l2_weight_decay)(y)
    y = Dense(10, activation='softmax', kernel_regularizer=l2_weight_decay)(y)
    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m, inp, y


def train_mlp(X_train, y_train, X_test, y_test, model_dir,
              batch_size=64, num_epochs=100, train_epoch_size=None,
              validation_epoch_size=None, validation_split=0.1,
              learning_rate=1e-4, weight_decay=1e-5,
              verbose=False, **kwargs):
    """
    Train a Multi-layer perceptron model on the given data

    Args:
        X_train: Training feature data
                 (Type: np.ndarray)
        y_train: Training label data
                 (Type: np.ndarray)
        X_test: Testing feature data
                (Type: np.ndarray)
        y_test: Testing label data
                (Type: np.ndarray)
        model_dir: Path to model directory
                   (Type: str)

    Keyword Args:
        verbose:  If True, print verbose messages
                  (Type: bool)
    """
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    monitor = 'val_loss'

    m, inp, out = construct_mlp_model(X_train.shape[1:], weight_decay=weight_decay)
    weight_path = os.path.join(model_dir, 'model.h5')

    cb = []
    cb.append(keras.callbacks.ModelCheckpoint(weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor=monitor))

    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                        separator=','))

    LOGGER.debug('Compiling model...')

    m.compile(Adam(lr=learning_rate), loss=loss, metrics=metrics)

    LOGGER.debug('Fitting model to data...')

    history = m.fit(x=X_train, y=y_train, batch_size=batch_size,
                    epochs=num_epochs,
                    steps_per_epoch=train_epoch_size,
                    validation_split=validation_split,
                    validation_steps=validation_epoch_size,
                    callbacks=cb,
                    verbose=10 if verbose else 1)

    y_train_pred = m.predict(X_train)
    y_test_pred = m.predict(X_test)

    return m, y_train_pred, y_test_pred


def compute_metrics(y, pred):
    """
    Compute perfomance metrics given the predicted labels and the true labels

    Args:
        y: True label vector
           (Type: np.ndarray)

        pred: Predicted label vector
              (Type: np.ndarray)

    Returns:
        metrics: Metrics dictionary
                 (Type: dict[str, *])
    """
    # Convert from one-hot to integer encoding if necessary
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    if pred.ndim == 2:
        pred = np.argmax(pred, axis=1)

    acc = (y == pred).mean()

    sum_class_acc = 0.0
    for class_idx in range(10):
        idxs = (y == class_idx)
        sum_class_acc += (y[idxs] == pred[idxs]).mean()

    ave_class_acc = sum_class_acc / 10

    return {
        'accuracy': acc,
        'average_class_accuracy': ave_class_acc
    }


def aggregate_metrics(fold_metrics):
    """
    Aggregate fold metrics using different stats

    Args:
        fold_metrics: List of fold metrics dictionaries
                      (Type: list[dict[str, *]])
    """
    metric_keys = list(fold_metrics[0].keys())
    fold_metrics_list = {k: [fold[k] for fold in fold_metrics]
                         for k in metric_keys}
    aggr_metrics = {}

    for metric in metric_keys:
        metric_list = fold_metrics_list[metric]
        aggr_metrics[metric] = {
            'mean': np.mean(metric_list),
            'var': np.var(metric_list),
            'min': np.min(metric_list),
            '25_%ile': np.percentile(metric_list, 25),
            '75_%ile': np.percentile(metric_list, 75),
            'median': np.median(metric_list),
            'max': np.max(metric_list)
        }

    return aggr_metrics


def print_metrics(metrics, subset_name):
    """
    Print classifier metrics

    Args:
        metrics: Metrics dictionary
                 (Type: dict[str, *])
    """
    LOGGER.info("Results metrics for {}".format(subset_name))
    LOGGER.info("=====================================================")
    for metric, metric_stats in metrics.items():
        LOGGER.info("* " + metric)
        for stat_name, stat_val in metric_stats.items():
            LOGGER.info("\t- {}: {}".format(stat_name, stat_val))
        LOGGER.info("\n")


def train(metadata_path, data_dir, model_id, output_dir,
          model_type='svm', features='l3_stack', label_format='int',
          l3embedding_model_path=None, l3embedding_model_type='cnn_L3_orig',
          random_state=20171021, verbose=False, log_path=None,
          disable_logging=False, **model_args):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    # Make sure the directories we need exist
    model_dir = os.path.join(output_dir, model_id)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if features.startswith('l3'):
        LOGGER.info('Loading embedding model...')
        l3embedding_model = load_embedding(l3embedding_model_path,
                                           l3embedding_model_type, 'audio')
    else:
        l3embedding_model = None


    LOGGER.info('Loading data...')
    # Load metadata
    metadata = load_us8k_metadata(metadata_path)

    # Get fold data
    fold_data = get_us8k_folds(metadata, data_dir, l3embedding_model=l3embedding_model,
                               features=features, label_format=label_format)

    model_output_path = os.path.join(model_dir, "model_fold{}.{}")

    results = {
        'folds': []
    }
    # Fit the model
    LOGGER.info('Preparing to fit models...')
    for fold_idx in range(10):
        LOGGER.info('\t* Training with fold {} held out'.format(fold_idx+1))
        X_train, y_train, X_test, y_test = get_fold_split(fold_data, fold_idx)
        if model_type == 'svm':
            model, y_train_pred, y_test_pred = train_svm(
                    X_train, y_train, X_test, y_test,
                    verbose=verbose, **model_args)

            LOGGER.info('Saving model...')
            # Save the model for this fold
            joblib.dump(model, model_output_path.format(fold_idx+1, 'pkl'))

        elif model_type == 'mlp':
            model, y_train_pred, y_test_pred = train_mlp(
                    X_train, y_train, X_test, y_test, model_dir,
                    verbose=verbose, **model_args)

        else:
            raise ValueError('Invalid model type: {}'.format(model_type))

        # Compute metrics for this fold
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        results['folds'].append({
            'train': {
                'metrics': train_metrics,
                'target': y_train.tolist(),
                'prediction': y_train_pred.tolist()
            },
            'test': {
                'metrics': test_metrics,
                'target': y_test.tolist(),
                'prediction': y_test_pred.tolist()
            }
        })

    train_metrics = aggregate_metrics([fold['train']['metrics']
                                       for fold in results['folds']])
    print_metrics(train_metrics, 'train')

    test_metrics = aggregate_metrics([fold['test']['metrics']
                                      for fold in results['folds']])
    print_metrics(test_metrics, 'test')

    results['summary'] = {
        'train': train_metrics,
        'test': test_metrics
    }

    LOGGER.info('Done training. Saving results to disk...')

    # Evaluate model
    # print('Evaluate model...')
    # Load best params
    # m.load_weights(weight_path)
    # with open(os.path.join(output_dir, 'index_test.json'), 'r') as fp:
    #     test_idx = json.load(fp)['id']

    # Compute eval scores
    # results = score_model(output_dir, pump, model, test_idx, working,
    #                       strong_label_file, duration, modelid,
    #                       use_orig_duration=True)

    # Save results to disk
    results_file = os.path.join(model_dir, 'results.json')
    with open(results_file, 'w') as fp:
        json.dump(results, fp, indent=2)

    LOGGER.info('Done!')
