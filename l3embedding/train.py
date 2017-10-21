from .model import construct_cnn_L3_orig
import json
import os
import pickle
import pescador
import pandas as pd
from tqdm import tqdm
import keras
from keras.optimizers import Adam


def sampler(filename, file_list):
    '''
    Sample audio/video fromthe filename, with 50% change of using pairs from
    the same file and 50% chance of mixing audio/video from another file
    chosen at random from the file_list.

    Parameters
    ----------
    filename
    file_list

    Returns
    -------

    '''


def data_generator(csv_file, batch_size=64):
    '''
    - Load up CSV file
    - Iterate over all training files
    - Create a streamer per traning file

    Parameters
    ----------
    csv_file

    Returns
    -------

    '''

    seeds = []

    for track in tqdm(tracks):
        fname = os.path.join(working,
                             os.path.extsep.join([str(track), 'h5']))
        seeds.append(pescador.Streamer(sampler, fname, file_list))

    # Send it all to a mux
    mux = pescador.Mux(seeds, k, **kwargs)

    if batch_size == 1:
        return mux
    else:
        return pescador.BufferedStreamer(mux, batch_size)


class LossHistory(keras.callbacks.Callback):

    def __init__(self, outfile):
        super().__init__()
        self.outfile = outfile

    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        loss_dict = {'loss': self.loss, 'val_loss': self.val_loss}
        with open(self.outfile, 'wb') as fp:
            pickle.dump(loss_dict, fp)


def train(csv_file, model_id, output_dir, epochs=150, epoch_size=512,
          batch_size=64, validation_size=1024, rate=16,
          seed=20171011, verbose=False):
    m, inputs, outputs = construct_cnn_L3_orig()
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    #monitor = 'val_loss'

    # Make sure the directories we need exist
    model_dir = os.path.join(output_dir, model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    print('Compile model...')
    m.compile(Adam(),
              loss=loss,
              metrics=metrics)

    # Save the model
    model_spec_path = os.path.join(model_dir, 'model_spec.pkl')
    model_spec = keras.utils.serialize_keras_object(m)
    with open(model_spec_path, 'wb') as fd:
        pickle.dump(model_spec, fd)
    model_json_path = os.path.join(model_dir, 'model.json')
    model_json = m.to_json()
    with open(model_json_path, 'w') as fd:
        json.dump(model_json, fd, indent=2)

    weight_path = os.path.join(model_dir, 'model.h5')

    cb = []
    cb.append(keras.callbacks.ModelCheckpoint(weight_path,
                                              save_best_only=True,
                                              verbose=1,))
                                              #monitor=monitor))

    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                        separator=','))


    train_gen = data_generator(
        csv_file,
        batch_size=batch_size,
        random_state=seed).tuples('video', 'audio', 'label')

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    # Fit the model
    print('Fit model...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2
    history = m.fit_generator(train_gen, epoch_size, epochs,
    #                          validation_data=gen_val,
    #                          validation_steps=validation_size,
                              callbacks=cb,
                              verbose=verbosity)

    print('Done training. Saving results to disk...')
    # Save history
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)

    # Evaluate model
    print('Evaluate model...')
    # Load best params
    m.load_weights(weight_path)
    with open(os.path.join(output_dir, 'index_test.json'), 'r') as fp:
        test_idx = json.load(fp)['id']

    # Compute eval scores
    #results = score_model(output_dir, pump, model, test_idx, working,
    #                      strong_label_file, duration, modelid,
    #                      use_orig_duration=True)

    # Save results to disk
    #results_file = os.path.join(model_dir, 'results.json')
    #with open(results_file, 'w') as fp:
    #    json.dump(results, fp, indent=2)

    #print('Done!')