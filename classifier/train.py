import datetime
import getpass
import json
import os
import pickle as pk
import random
import git

import keras
import keras.regularizers as regularizers
from tensorflow import set_random_seed
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import hinge_loss
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from classifier.metrics import compute_metrics
from data.usc.features import preprocess_split_data
from data.usc.folds import get_split
from l3embedding.train import LossHistory
from log import *

from gsheets import get_credentials, append_row, update_experiment
from googleapiclient import discovery

LOGGER = logging.getLogger('classifier')
LOGGER.setLevel(logging.DEBUG)


DATASET_NUM_CLASSES = {
    'us8k': 10,
    'esc50': 50,
    'dcase2013': 10,
}


class MetricCallback(keras.callbacks.Callback):

    def __init__(self, valid_data, verbose=False):
        super(MetricCallback).__init__()
        self.valid_data = valid_data
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.train_loss.append(logs.get('loss'))
        self.valid_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.valid_acc.append(logs.get('val_acc'))

        if self.verbose:
            train_msg = 'Train - loss: {}, acc: {}'
            valid_msg = 'Valid - loss: {}, acc: {}'
            LOGGER.info('Epoch {}'.format(epoch))
            LOGGER.info(train_msg.format(self.train_loss[-1],
                                         self.train_acc[-1]))
            LOGGER.info(valid_msg.format(self.valid_loss[-1],
                                         self.valid_acc[-1]))


def train_svm(train_data, valid_data, test_data, model_dir, C=1.0, kernel='rbf',
              num_classes=10, tol=0.001, max_iterations=-1, verbose=False,
              random_state=12345678, **kwargs):
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
    np.random.seed(random_state)
    random.seed(random_state)

    X_train = train_data['features']
    y_train = train_data['labels']

    model_output_path = os.path.join(model_dir, "model.pkl")

    # Create classifier
    clf = SVC(C=C, probability=True, kernel=kernel, max_iter=max_iterations,
              tol=tol, random_state=random_state, verbose=verbose)

    # Fit data and get output for train and valid batches
    LOGGER.debug('Fitting model to data...')
    clf.fit(X_train, y_train)

    LOGGER.info('Saving model...')
    joblib.dump(clf, model_output_path)

    y_train_pred = clf.predict(X_train)
    # Compute new metrics
    classes = np.arange(num_classes)
    train_loss = hinge_loss(y_train, clf.decision_function(X_train), labels=classes)
    train_metrics = compute_metrics(y_train, y_train_pred)
    train_metrics['loss'] = train_loss
    train_msg = 'Train - hinge loss: {}, acc: {}'
    LOGGER.info(train_msg.format(train_loss, train_metrics['accuracy']))

    if valid_data:
        X_valid = valid_data['features']
        y_valid = valid_data['labels']
        y_valid_pred = clf.predict(X_valid)
        valid_loss = hinge_loss(y_valid, clf.decision_function(X_valid), labels=classes)
        valid_metrics = compute_metrics(y_valid, y_valid_pred)
        valid_metrics['loss'] = valid_loss
        valid_msg = 'Valid - hinge loss: {}, acc: {}'
        LOGGER.info(valid_msg.format(valid_loss, valid_metrics['accuracy']))
    else:
        valid_metrics = {}

    # Evaluate model on test data
    if test_data:
        X_test = test_data['features']
        y_test_pred_frame = clf.predict_proba(X_test)
        y_test_pred = []
        for start_idx, end_idx in test_data['file_idxs']:
            class_pred = y_test_pred_frame[start_idx:end_idx].mean(axis=0).argmax()
            y_test_pred.append(class_pred)

        y_test_pred = np.array(y_test_pred)
        test_metrics = compute_metrics(test_data['labels'], y_test_pred)
    else:
        test_metrics = {}

    return clf, train_metrics, valid_metrics, test_metrics


def construct_mlp_model(input_shape, weight_decay=1e-5, num_classes=10):
    """
    Constructs a multi-layer perceptron model

    Args:
        input_shape: Shape of input data
                     (Type: tuple[int])
        weight_decay: L2 regularization factor
                      (Type: float)

    Returns:
        model: L3 CNN model
               (Type: keras.models.Model)
        input: Model input
               (Type: list[keras.layers.Input])
        output:Model output
                (Type: keras.layers.Layer)
    """
    l2_weight_decay = regularizers.l2(weight_decay)
    LOGGER.info(str(input_shape))
    inp = Input(shape=input_shape, dtype='float32')
    y = Dense(512, activation='relu', kernel_regularizer=l2_weight_decay)(inp)
    y = Dense(128, activation='relu', kernel_regularizer=l2_weight_decay)(y)
    y = Dense(num_classes, activation='softmax', kernel_regularizer=l2_weight_decay)(y)
    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m, inp, y


def train_mlp(train_data, valid_data, test_data, model_dir,
              batch_size=64, num_epochs=100, valid_split=0.15, patience=20,
              learning_rate=1e-4, weight_decay=1e-5, num_classes=10,
              random_state=12345678, verbose=False, **kwargs):
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
        frame_features: If True, test data will be handled as a tensor, where
                        the first dimension is the audio file, and the second
                        dimension is the frame in the audio file. Evaluation
                        will additionally be done at the audio file level
                        (Type: bool)

    Keyword Args:
        verbose:  If True, print verbose messages
                  (Type: bool)
        batch_size: Number of smamples per batch
                    (Type: int)
        num_epochs: Number of training epochs
                    (Type: int)
        learning_rate: Learning rate value
                       (Type: float)
        weight_decay: L2 regularization factor
                      (Type: float)
    """
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    monitor = 'val_loss'
    set_random_seed(random_state)

    # Set up data inputs
    enc = OneHotEncoder(n_values=num_classes, sparse=False)

    X_train = train_data['features']
    y_train = enc.fit_transform(train_data['labels'].reshape(-1, 1))

    if valid_data:
        validation_data = (valid_data['features'],
                           enc.fit_transform(valid_data['labels'].reshape(-1, 1)))
        valid_split = 0.0
    else:
        validation_data = None

    # Set up model
    m, inp, out = construct_mlp_model(X_train.shape[1:],
                                      weight_decay=weight_decay,
                                      num_classes=num_classes)

    # Set up callbacks
    cb = []
    weight_path = os.path.join(model_dir, 'model.h5')
    cb.append(keras.callbacks.ModelCheckpoint(weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor=monitor))
    cb.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience))
    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))
    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                        separator=','))
    metric_cb = MetricCallback(valid_data, verbose=verbose)
    cb.append(metric_cb)

    # Fit model
    LOGGER.debug('Compiling model...')
    m.compile(Adam(lr=learning_rate), loss=loss, metrics=metrics)
    LOGGER.debug('Fitting model to data...')
    m.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=validation_data, validation_split=valid_split, callbacks=cb)

    # Compute metrics for train and valid
    train_pred = m.predict(X_train)
    train_metrics = compute_metrics(y_train, train_pred)
    # Set up train and validation metrics
    train_metrics = {
        'loss': metric_cb.train_loss[-1],
        'accuracy': metric_cb.train_acc[-1],
        'class_accuracy': train_metrics['class_accuracy'],
        'average_class_accuracy': train_metrics['average_class_accuracy']
    }

    valid_metrics = {
        'loss': metric_cb.valid_loss[-1],
        'accuracy': metric_cb.valid_acc[-1],
    }

    if valid_data:
        valid_pred = m.predict(validation_data[0])
        valid_metrics = compute_metrics(validation_data[1], valid_pred)
        valid_metrics.update({
            'class_accuracy': valid_metrics['class_accuracy'],
            'average_class_accuracy': valid_metrics['average_class_accuracy']
        })

    if test_data:
        # Evaluate model on test data
        X_test = test_data['features']
        y_test_pred_frame = m.predict(X_test)
        y_test_pred = []
        for start_idx, end_idx in test_data['file_idxs']:
            class_pred = y_test_pred_frame[start_idx:end_idx].mean(axis=0).argmax()
            y_test_pred.append(class_pred)
        y_test_pred = np.array(y_test_pred)
        test_metrics = compute_metrics(test_data['labels'], y_test_pred)
    else:
        test_metrics = {}

    return m, train_metrics, valid_metrics, test_metrics


def train_param_search(train_data, valid_data, test_data, model_dir, train_func,
                   search_param, search_space=None, valid_ratio=0.15, **kwargs):

    search_train_metrics = {}
    search_valid_metrics = {}

    if not search_space:
        if search_param == 'learning_rate':
            search_space = [1e-5, 1e-4, 1e-3, 1e-2]
        elif search_param == 'C':
            search_space = [0.1, 1, 10, 100, 1000]
        else:
            raise ValueError('Defaults for parameter {} not implemented'.format(search_param))


    LOGGER.info('Starting hyperparameter search on {}.'.format(search_param))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio)
    train_idxs, valid_idxs = splitter.split(train_data['features'], train_data['labels'])

    best_valid_loss = float('inf')
    best_param = None

    for param in search_space:
        LOGGER.info('Evaluating {} = {}'.format(search_param, param))
        train_data_skf = {
            'features': train_data['features'][train_idxs],
            'labels': train_data['labels'][train_idxs]
        }
        valid_data_skf = {
            'features': train_data['features'][valid_idxs],
            'labels': train_data['labels'][valid_idxs]
        }

        kwargs[search_param] = param

        LOGGER.info('')
        _, train_metrics, valid_metrics, _ \
            = train_func(train_data_skf, valid_data_skf, None, model_dir, **kwargs)

        if valid_metrics['accuracy'] < best_valid_loss:
            best_valid_loss = valid_metrics['accuracy']
            best_param = param

        search_train_metrics[param] = train_metrics
        search_valid_metrics[param] = valid_metrics

    LOGGER.info('Best {} = {}, ave valid loss = {}'.format(search_param, best_param,
                                                           best_valid_loss))

    kwargs[search_param] = best_param
    clf, train_metrics, _, test_metrics, \
        = train_func(train_data, None, test_data, model_dir, **kwargs)

    train_metrics['search'] = search_train_metrics
    train_metrics['search_param'] = search_param
    train_metrics['search_param_best_value'] = best_param

    valid_metrics = {
        'search': search_valid_metrics,
        'search_param': search_param,
        'search_param_best_value': best_param
    }
    valid_metrics.update(search_valid_metrics[best_param])

    return clf, train_metrics, valid_metrics, test_metrics


def train(features_dir, output_dir, fold_num,
          model_type='svm', feature_mode='framewise',
          train_batch_size=64, random_state=20171021, search_param=None,
          search_space=None, gsheet_id=None, google_dev_app_name=None,
          verbose=False, non_overlap=False, non_overlap_chunk_size=10,
          use_min_max=False, **model_args):
    init_console_logger(LOGGER, verbose=verbose)
    LOGGER.debug('Initialized logging.')

    # Set random state
    np.random.seed(random_state)
    random.seed(random_state)

    datasets = ['us8k', 'esc50', 'dcase2013']

    features_desc_str = features_dir[features_dir.rindex('features')+9:]
    dataset_name = features_desc_str.split('/')[0]

    if dataset_name not in datasets:
        err_msg = 'Feature directory must contain name of dataset ({})'
        raise ValueError(err_msg.format(str(datasets)))

    model_id = os.path.join(features_desc_str, feature_mode,
                            "non-overlap" if non_overlap else "overlap",
                            "min-max" if use_min_max else "no-min-max",
                            model_type)

    # Make sure the directories we need exist
    model_dir = os.path.join(output_dir, 'classifier', model_id,
                             'fold{}'.format(fold_num),
                             datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    # Make sure model directory exists
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    config = {
        'username': getpass.getuser(),
        'features_dir': features_dir,
        'output_dir': output_dir,
        'model_dir': model_dir,
        'model_id': model_id,
        'fold_num': fold_num,
        'model_type': model_type,
        'feature_mode': feature_mode,
        'train_batch_size': train_batch_size,
        'non_overlap': non_overlap,
        'non_overlap_chunk_size': non_overlap_chunk_size,
        'random_state': random_state,
        'verbose': verbose,
        'git_commit': git.Repo(os.path.dirname(os.path.abspath(__file__)),
                               search_parent_directories=True).head.object.hexsha,
        'gsheet_id': gsheet_id,
        'google_dev_app_name': google_dev_app_name
    }
    config.update(model_args)

    # Save configs
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as fp:
        json.dump(config, fp)
    LOGGER.info('Saved configurations to {}'.format(config_path))

    if gsheet_id:
        # Add a new entry in the Google Sheets spreadsheet
        LOGGER.info('Creating new spreadsheet entry...')
        config.update({
              'train_loss': '-',
              'valid_loss': '-',
              'train_acc': '-',
              'train_avg_class_acc': '-',
              'train_class_acc': '-',
              'valid_acc': '-',
              'valid_avg_class_acc': '-',
              'valid_class_acc': '-',
              'test_acc': '-',
              'test_avg_class_acc': '-',
              'test_class_acc': '-'
        })
        credentials = get_credentials(google_dev_app_name)
        service = discovery.build('sheets', 'v4', credentials=credentials)
        append_row(service, gsheet_id, config, 'classifier')
    else:
        LOGGER.error(gsheet_id)

    fold_idx = fold_num - 1

    LOGGER.info('Loading data for configuration with test fold {}...'.format(fold_num))
    train_data, valid_data, test_data = get_split(features_dir, fold_idx, dataset_name)

    LOGGER.info('Preprocessing data...')
    min_max_scaler, stdizer = preprocess_split_data(train_data, valid_data, test_data,
                                                feature_mode=feature_mode,
                                                non_overlap=non_overlap,
                                                non_overlap_chunk_size=non_overlap_chunk_size,
                                                use_min_max=use_min_max)

    min_max_scaler_output_path = os.path.join(model_dir, "min_max_scaler.pkl")
    joblib.dump(min_max_scaler, min_max_scaler_output_path)
    stdizer_output_path = os.path.join(model_dir, "stdizer.pkl")
    joblib.dump(stdizer, stdizer_output_path)

    LOGGER.info('Training {} with fold {} held out'.format(model_type, fold_num))
    # Fit the model
    if model_type == 'svm':
        if search_param:
            model, train_metrics, valid_metrics, test_metrics \
                = train_param_search(train_data, valid_data, test_data, model_dir,
                    train_func=train_svm, search_param=search_param,
                    search_space=search_space,
                    num_classes=DATASET_NUM_CLASSES[dataset_name],
                    random_state=random_state, verbose=verbose, **model_args)
        else:
            model, train_metrics, valid_metrics, test_metrics \
                = train_svm(train_data, valid_data, test_data, model_dir,
                    num_classes=DATASET_NUM_CLASSES[dataset_name],
                    random_state=random_state, verbose=verbose, **model_args)

    elif model_type == 'mlp':
        if search_param:
            model, train_metrics, valid_metrics, test_metrics \
                = train_param_search(train_data, valid_data, test_data, model_dir,
                                 train_func=train_mlp, search_param=search_param,
                                 search_space=search_space,
                                 batch_size=train_batch_size, random_state=random_state,
                                 num_classes=DATASET_NUM_CLASSES[dataset_name],
                                 verbose=verbose, **model_args)
        else:
            model, train_metrics, valid_metrics, test_metrics \
                    = train_mlp(train_data, valid_data, test_data, model_dir,
                        batch_size=train_batch_size, random_state=random_state,
                        num_classes=DATASET_NUM_CLASSES[dataset_name],
                        verbose=verbose, **model_args)

    else:
        raise ValueError('Invalid model type: {}'.format(model_type))

    # Assemble metrics for this training run
    results = {
        'train': train_metrics,
        'valid': valid_metrics,
        'test': test_metrics
    }

    LOGGER.info('Done training. Saving results to disk...')

    # Save results to disk
    results_file = os.path.join(model_dir, 'results.pkl')
    with open(results_file, 'wb') as fp:
        pk.dump(results, fp, protocol=pk.HIGHEST_PROTOCOL)


    # TODO: Update spreadsheet for cross validation!
    if gsheet_id:
        # Update spreadsheet with results
        LOGGER.info('Updating spreadsheet...')
        update_values = [
              train_metrics['loss'],
              valid_metrics['loss'],
              train_metrics['accuracy'],
              valid_metrics['accuracy'],
              train_metrics['average_class_accuracy'],
              valid_metrics['average_class_accuracy'],
              ', '.join(map(str, train_metrics['class_accuracy'])),
              ', '.join(map(str, valid_metrics['class_accuracy'])),
              test_metrics['accuracy'],
              test_metrics['average_class_accuracy'],
              ', '.join(map(str, test_metrics['class_accuracy']))
        ]
        update_experiment(service, gsheet_id, config, 'R', 'AB',
                          update_values, 'classifier')

    LOGGER.info('Done!')
