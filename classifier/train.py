import json
import os

import keras
import keras.regularizers as regularizers
import numpy as np
import random
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import mode

from classifier.features import flatten_file_frames
from classifier.metrics import compute_metrics, aggregate_metrics, print_metrics
from classifier.us8k import load_us8k_metadata, get_us8k_folds, \
    get_us8k_fold_split
from l3embedding.model import load_embedding
from l3embedding.train import LossHistory
from log import *

LOGGER = logging.getLogger('classifier')
LOGGER.setLevel(logging.DEBUG)


def train_svm(X_train, y_train, X_test, y_test, frame_features, C=1e-4, verbose=False, **kwargs):
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

    if frame_features:
        y_test_pred = []

        for X_test_file in X_test:
            output = clf.predict(X_test_file)
            class_pred = mode(output.flatten())[0][0]
            y_test_pred.append(class_pred)

        y_test_pred = np.array(y_test_pred)
    else:
        y_test_pred = clf.predict(X_test)

    return clf, y_train_pred, y_test_pred


def construct_mlp_model(input_shape, weight_decay=1e-5):
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
    weight_decay = 1e-5
    l2_weight_decay = regularizers.l2(weight_decay)
    inp = Input(shape=input_shape, dtype='float32')
    y = Dense(512, activation='relu', kernel_regularizer=l2_weight_decay)(inp)
    y = Dense(128, activation='relu', kernel_regularizer=l2_weight_decay)(y)
    y = Dense(10, activation='softmax', kernel_regularizer=l2_weight_decay)(y)
    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m, inp, y


def train_mlp(X_train, y_train, X_test, y_test, model_dir, frame_features,
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
        validation_epoch_size: Number of validation batches per validation epoch
                               (Type: int)
        validation_split: Percentage of training data used for validation
                          (Type: float)
        learning_rate: Learning rate value
                       (Type: float)
        weight_decay: L2 regularization factor
                      (Type: float)
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
    if frame_features:
        y_test_pred = []

        for X_test_file in X_test:
            output = m.predict(X_test_file).flatten()
            y_test_pred.append(output.mean(axis=0))

        y_test_pred = np.array(y_test_pred)

    else:
        y_test_pred = m.predict(X_test)

    return m, y_train_pred, y_test_pred


def train(metadata_path, dataset_name, data_dir, model_id, output_dir,
          model_type='svm', features='l3_stack', label_format='int',
          l3embedding_model_path=None, l3embedding_model_type='cnn_L3_orig',
          random_state=20171021, verbose=False, log_path=None,
          disable_logging=False, num_random_samples=None, hop_size=0.25, **model_args):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    # Set random state
    np.random.seed(random_state)
    random.seed(random_state)

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

    # Get fold data
    if dataset_name == 'us8k':
        fold_data = get_us8k_folds(metadata_path, data_dir, l3embedding_model=l3embedding_model,
                                   features=features, label_format=label_format,
                                   num_random_samples=num_random_samples, hop_size=hop_size)
    else:
        err_msg = 'Unsupported dataset "{}"'.format(dataset_name)
        LOGGER.error(err_msg)
        raise ValueError(err_msg)

    frame_features = 'frames' in features

    model_output_path = os.path.join(model_dir, "model_fold{}.{}")

    results = {
        'folds': []
    }
    # Fit the model
    LOGGER.info('Preparing to fit models...')
    for fold_idx in range(10):
        LOGGER.info('\t* Training with fold {} held out'.format(fold_idx+1))
        X_train, y_train, X_test, y_test = get_us8k_fold_split(fold_data, fold_idx,
                                                               frame_features=frame_features)
        if model_type == 'svm':
            model, y_train_pred, y_test_pred = train_svm(
                    X_train, y_train, X_test, y_test, frame_features,
                    verbose=verbose, **model_args)

            LOGGER.info('Saving model...')
            # Save the model for this fold
            joblib.dump(model, model_output_path.format(fold_idx+1, 'pkl'))

        elif model_type == 'mlp':
            model, y_train_pred, y_test_pred = train_mlp(
                    X_train, y_train, X_test, y_test, model_dir, frame_features,
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
