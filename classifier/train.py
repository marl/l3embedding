import datetime
import getpass
import json
import os
import pickle as pk
import random
import git

import keras
import keras.regularizers as regularizers
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from scipy.stats import mode
from sklearn.metrics import hinge_loss
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
import pescador

from classifier.metrics import compute_metrics, collapse_metrics
from data.usc.us8k import get_us8k_batch_generator, get_us8k_batch, load_test_fold
from l3embedding.train import LossHistory
from log import *

from gsheets import get_credentials, append_row, update_experiment
from googleapiclient import discovery

LOGGER = logging.getLogger('classifier')
LOGGER.setLevel(logging.DEBUG)


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
        self.valid_class_acc = []
        self.valid_avg_class_acc = []

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.train_loss.append(logs.get('loss'))
        self.valid_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.valid_acc.append(logs.get('val_acc'))

        valid_pred = self.model.predict(self.valid_data['features'])
        valid_metrics = compute_metrics(self.valid_data['label'], valid_pred)
        self.valid_class_acc.append(valid_metrics['class_accuracy'])
        self.valid_avg_class_acc.append(valid_metrics['average_class_accuracy'])

        if self.verbose:
            train_msg = 'Train - loss: {}, acc: {}'
            valid_msg = 'Valid - loss: {}, acc: {}'
            LOGGER.info('Epoch {}'.format(epoch))
            LOGGER.info(train_msg.format(self.train_loss[-1],
                                         self.train_acc[-1]))
            LOGGER.info(valid_msg.format(self.valid_loss[-1],
                                         self.valid_acc[-1]))


def train_svm(train_gen, valid_data, test_data, model_dir, C=1e-4, reg_penalty='l2',
              num_classes=10, tol=1e-3, max_iterations=1000000, verbose=False, **kwargs):
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
    # Set up standardizer
    stdizer = StandardScaler()

    X_valid = valid_data['features']
    y_valid = valid_data['label']

    train_loss_history = []
    valid_loss_history = []
    train_metric_history = []
    valid_metric_history = []

    model_output_path = os.path.join(model_dir, "model.pkl")
    stdizer_output_path = os.path.join(model_dir, "stdizer.pkl")

    # Create classifier
    clf = SGDClassifier(alpha=C, penalty=reg_penalty, n_jobs=-1, verbose=verbose)

    classes = np.arange(num_classes)

    LOGGER.debug('Fitting model to data...')
    for iter_idx, train_data in enumerate(train_gen):
        X_train = train_data['features']
        y_train = train_data['label']
        stdizer.partial_fit(X_train)

        # Fit data and get output for train and valid batches
        clf.partial_fit(X_train, y_train, classes=classes)
        X_train_std = stdizer.transform(X_train)
        X_valid_std = stdizer.transform(X_valid)
        y_train_pred = clf.predict(X_train_std)
        y_valid_pred = clf.predict(X_valid_std)

        # Compute new metrics
        valid_loss_history.append(list(y_valid_pred))
        train_loss_history.append(hinge_loss(y_train, clf.decision_function(X_train_std),
                                             labels=classes))
        valid_loss_history.append(hinge_loss(y_valid, clf.decision_function(X_valid_std),
                                             labels=classes))
        train_metric_history.append(compute_metrics(y_train, y_train_pred))
        valid_metric_history.append(compute_metrics(y_valid, y_valid_pred))

        # Save the model for this iteration
        LOGGER.info('Saving model...')
        joblib.dump(clf, model_output_path)
        joblib.dump(stdizer, stdizer_output_path)

        if verbose:
            train_msg = 'Train - loss: {}, acc: {}'
            valid_msg = 'Valid - loss: {}, acc: {}'
            LOGGER.info('Epoch {}'.format(iter_idx + 1))
            LOGGER.info(train_msg.format(train_loss_history[-1],
                                         train_metric_history[-1]['accuracy']))
            LOGGER.info(valid_msg.format(valid_loss_history[-1],
                                         valid_metric_history[-1]['accuracy']))

        # Finish training if the loss doesn't change much
        if len(train_loss_history) > 1 and abs(train_loss_history[-2] - train_loss_history[-1]) < tol:
            break

        # Break if we reach the maximum number of iterations
        if iter_idx >= max_iterations:
            break

    # Post process metrics
    train_metrics = collapse_metrics(train_metric_history)
    valid_metrics = collapse_metrics(valid_metric_history)
    train_metrics['loss'] = train_loss_history
    valid_metrics['loss'] = valid_loss_history

    # Evaluate model on test data
    X_test = stdizer.transform(test_data['features'])
    y_test_pred_frame = clf.predict(X_test)
    y_test_pred = []
    for start_idx, end_idx in test_data['file_idxs']:
        class_pred = mode(y_test_pred_frame[start_idx:end_idx])[0][0]
        y_test_pred.append(class_pred)

    y_test_pred = np.array(y_test_pred)
    test_metrics = compute_metrics(test_data['labels'], y_test_pred)

    return (stdizer, clf), train_metrics, valid_metrics, test_metrics


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


def train_mlp(train_gen, valid_data, test_data, model_dir,
              batch_size=64, num_epochs=100, train_epoch_size=None,
              learning_rate=1e-4, weight_decay=1e-5, num_classes=10,
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
        train_epoch_size: Number of training batches per training epoch
                          (Type: int)
        learning_rate: Learning rate value
                       (Type: float)
        weight_decay: L2 regularization factor
                      (Type: float)
    """
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    monitor = 'val_loss'

    # Set up data inputs
    train_gen = pescador.maps.keras_tuples(train_gen, 'features', 'label')
    enc = OneHotEncoder(n_values=num_classes, sparse=False)
    # Transform int targets to produce one hot targets
    train_gen = ((X, enc.fit_transform(y.reshape(-1, 1))) for X, y in train_gen)
    valid_data_keras = (valid_data['features'],
                        enc.fit_transform(valid_data['label'].reshape(-1,1)))

    train_iter = iter(train_gen)
    train_batch = next(train_iter)

    # Set up model
    m, inp, out = construct_mlp_model(train_batch[0].shape[1:],
                                      weight_decay=weight_decay,
                                      num_classes=num_classes)

    # Set up callbacks
    cb = []
    weight_path = os.path.join(model_dir, 'model.h5')
    cb.append(keras.callbacks.ModelCheckpoint(weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor=monitor))
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
    m.fit_generator(train_gen,
          epochs=num_epochs, steps_per_epoch=train_epoch_size,
          validation_data=valid_data_keras, callbacks=cb)

    # Set up train and validation metrics
    train_metrics = {
        'loss': metric_cb.train_loss,
        'accuracy': metric_cb.train_acc
    }

    valid_metrics = {
        'loss': metric_cb.valid_loss,
        'accuracy': metric_cb.valid_acc,
        'class_accuracy': metric_cb.valid_class_acc,
        'average_class_accuracy': metric_cb.valid_avg_class_acc
    }

    # Evaluate model on test data
    X_test = test_data['features']
    y_test_pred_frame = m.predict(X_test)
    y_test_pred = []
    for start_idx, end_idx in test_data['file_idxs']:
        class_pred = y_test_pred_frame[start_idx:end_idx].mean(axis=0).argmax()
        y_test_pred.append(class_pred)
    y_test_pred = np.array(y_test_pred)
    test_metrics = compute_metrics(test_data['labels'], y_test_pred)

    return m, train_metrics, valid_metrics, test_metrics


def train(features_dir, output_dir, model_id, fold_num, model_type='svm',
          train_num_streamers=None, train_batch_size=64, train_mux_rate=None,
          valid_num_streamers=None, valid_batch_size=64, valid_mux_rate=None,
          random_state=20171021, gsheet_id=None, google_dev_app_name=None,
          verbose=False, **model_args):
    init_console_logger(LOGGER, verbose=verbose)
    LOGGER.debug('Initialized logging.')

    # Set random state
    np.random.seed(random_state)
    random.seed(random_state)

    # Make sure the directories we need exist
    model_dir = os.path.join(output_dir, model_id,
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
        'train_num_streamers': train_num_streamers,
        'train_batch_size': train_batch_size,
        'train_mux_rate': train_mux_rate,
        'valid_num_streamers': valid_num_streamers,
        'valid_batch_size': valid_batch_size,
        'valid_mux_rate': valid_mux_rate,
        'random_state': random_state,
        'verbose': verbose,
        'git_commit': git.Repo(os.path.dirname(os.path.abspath(__file__)),
                               search_parent_directories=True).head.object.hexsha,
        'gsheet_id': gsheet_id,
        'google_dev_app_name': google_dev_app_name
    }
    config.update(model_args)

    # Save configs
    with open(os.path.join(model_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp)


    if gsheet_id:
        # Add a new entry in the Google Sheets spreadsheet
        LOGGER.info('Creating new spreadsheet entry...')
        config.update({
              'train_loss': '-',
              'valid_loss': '-',
              'train_acc': '-',
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

    LOGGER.info('Loading data...')

    fold_idx = fold_num - 1
    LOGGER.info('Preparing training data for fold {}'.format(fold_num))
    train_gen = get_us8k_batch_generator(features_dir, fold_idx,
                         valid=False, num_streamers=train_num_streamers,
                         batch_size=train_batch_size, random_state=random_state,
                         rate=train_mux_rate)
    LOGGER.info('Preparing validation data for fold {}'.format(fold_num))
    valid_data = get_us8k_batch(features_dir, fold_idx,
                         valid=True, num_streamers=valid_num_streamers,
                         batch_size=valid_batch_size, random_state=random_state,
                         rate=valid_mux_rate)
    LOGGER.info('Preparing test data for fold {}'.format(fold_num))
    test_data = load_test_fold(features_dir, fold_idx)

    LOGGER.info('Training {} with fold {} held out'.format(model_type, fold_num))
    # Fit the model
    if model_type == 'svm':
        model, train_metrics, valid_metrics, test_metrics \
            = train_svm(train_gen, valid_data, test_data, model_dir,
                verbose=verbose, **model_args)

    elif model_type == 'mlp':
        model, train_metrics, valid_metrics, test_metrics \
                = train_mlp(train_gen, valid_data, test_data, model_dir,
                batch_size=train_batch_size, verbose=verbose, **model_args)

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


    if gsheet_id:
        # Update spreadsheet with results
        LOGGER.info('Updating spreadsheet...')
        update_values = [
              train_metrics['loss'][-1],
              valid_metrics['loss'][-1],
              train_metrics['accuracy'][-1],
              valid_metrics['accuracy'][-1],
              valid_metrics['average_class_accuracy'][-1],
              ', '.join(map(str, valid_metrics['class_accuracy'][-1])),
              test_metrics['accuracy'],
              test_metrics['average_class_accuracy'],
              ', '.join(map(str, test_metrics['class_accuracy']))
        ]
        update_experiment(service, gsheet_id, config, 'V', 'AE',
                          update_values, 'classifier')

    LOGGER.info('Done!')
