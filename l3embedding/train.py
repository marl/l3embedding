import argparse
import json
import os
import pickle
import glob
import random
import pescador
import skvideo.io
import soundfile as sf
from tqdm import tqdm
import keras
from keras.optimizers import Adam
from .model import construct_cnn_L3_orig


#TODO: Consider putting the sampling functionality into another file

def get_file_list(data_dir):
    """Return audio and video file list.

    Args:
        data_dir: input directory that contains audio/ and video/

    Returns:
        audio_files: list of audio files
        video_files: list of video files

    """

    audio_files = glob.glob('{}/audio/*'.format(data_dir))
    video_files = glob.glob('{}/video/*'.format(data_dir))
    return audio_files, video_files


def video_to_audio(video_file):
    """Return corresponding audio_file.

    Args:
        video_file: video_file

    Returns:
        audio_file

    """

    *path, _, name = video_file.split('/')
    name = name.split('.')[0] + '.flac'
    return '/'.join(path + ['audio', name])


def sample_one_second(audio_file, start, label):
    """Return one second audio samples randomly if start is not specified,
       otherwise, return one second audio samples including start (seconds).

    Args:
        audio_file: audio_file to sample from
        start: starting time to fetch one second samples

    Returns:
        One second samples

    """

    audio_data, sampling_frequency = sf.read(audio_file)
    if label:
        start = min(0, int(start * sampling_frequency) - random.randint(0, sampling_frequency))
    else:
        start = random.randrange(len(audio_data) - sampling_frequency)
    return audio_data[start:start+sampling_frequency], start / sampling_frequency


def sample_one_frame(video_data, fps=30):
    """Return one frame randomly and time (seconds).

    Args:
        video_data: video data to sample from
        fps: frame per second

    Returns:
        One frame sampled randomly and time in seconds

    """

    num_frames = video_data.shape[0]
    frame = random.randrange(num_frames - fps)
    return video_data[frame, :, :, :], frame / fps


def sampler(video_file, audio_files):
    """Sample one frame from video_file, with 50% chance sample one second from corresponding audio_file,
       50% chance sample one second from another audio_file in the list of audio_files.

    Args:
        video_file: video_file to sample from
        audio_files: candidate audio_files to sample from

    Returns:
        A generator that yields dictionary of video sample, audio sample,
        and label (0: not from corresponding files, 1: from corresponding files)

    """

    video_data = skvideo.io.vread(video_file)
    audio_file = video_to_audio(video_file)

    if random.random() < 0.5:
        audio_file = random.choice([af for af in audio_files if af != audio_file])
        label = 0
    else:
        label = 1

    while True:
        sample_video_data, video_start = sample_one_frame(video_data)
        sample_audio_data, audio_start = sample_one_second(audio_file, video_start, label)

        yield {
            'video': sample_video_data,
            'audio': sample_audio_data[:,0],
            'label': label,
            'audio_file': audio_file,
            'video_file': video_file,
            'audio_start': audio_start,
            'video_start': video_start
        }


def data_generator(data_dir, k=32, batch_size=64, random_seed=20171021):
    """Sample video and audio from data_dir, returns a streamer that yield samples infinitely.

    Args:
        data_dir: directory to sample video and audio from
        k: number of concurrent open streamer
        batch_size: batch size

    Returns:
        A generator that yield infinite video and audio samples from data_dir

    """

    random.seed(random_seed)

    audio_files, video_files = get_file_list(data_dir)
    seeds = []
    for video_file in tqdm(random.sample(video_files, k)):
        seeds.append(pescador.Streamer(sampler, video_file, audio_files))

    mux = pescador.Mux(seeds, k)
    if batch_size == 1:
        return mux
    else:
        return pescador.BufferedStreamer(mux, batch_size)


class LossHistory(keras.callbacks.Callback):

    def __init__(self, outfile):
        super().__init__()
        self.outfile = outfile

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.loss = []
        self.val_loss = []

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        loss_dict = {'loss': self.loss, 'val_loss': self.val_loss}
        with open(self.outfile, 'wb') as fp:
            pickle.dump(loss_dict, fp)


#def train(train_csv_path, model_id, output_dir, num_epochs=150, epoch_size=512,
def train(train_data_dir, model_id, output_dir, num_epochs=150, epoch_size=512,
          batch_size=64, validation_size=1024, num_streamers=16,
          random_seed=20171021, verbose=False):
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
        #train_csv_path,
        train_data_dir,
        batch_size=batch_size,
        random_seed=random_seed,
        k=num_streamers)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    # Fit the model
    print('Fit model...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2
    history = m.fit_generator(train_gen, epoch_size, num_epochs,
    #                          validation_data=gen_val,
    #                          validation_steps=validation_size,
                              callbacks=cb,
                              verbose=verbosity)

    print('Done training. Saving results to disk...')
    # Save history
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)

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
    # results_file = os.path.join(model_dir, 'results.json')
    # with open(results_file, 'w') as fp:
    #     json.dump(results, fp, indent=2)

    print('Done!')


def parse_arguments():
    """
    Parse arguments from the command line


    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an L3-like audio-visual correspondence model')

    parser.add_argument('-e',
                        '--num-epochs',
                        dest='num_epochs',
                        action='store',
                        type=int,
                        default=150,
                        help='Maximum number of training epochs')

    parser.add_argument('-es',
                        '--epoch-size',
                        dest='epoch_size',
                        action='store',
                        type=int,
                        default=512,
                        help='Number of training batches per epoch')

    parser.add_argument('-b',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of training examples per batch')

    parser.add_argument('-v',
                        '--validation-size',
                        dest='validation_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of trianing examples in the validation set')

    parser.add_argument('-s',
                        '--num-streamers',
                        dest='num_streamers',
                        action='store',
                        type=int,
                        default=32,
                        help='Number of pescador streamers that can be open concurrently')

    parser.add_argument('-r',
                        '--random-seed',
                        dest='random_seed',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    """
    parser.add_argument('train_csv_path',
                        action='store',
                        type=str,
                        help='Path to training csv file')
    """
    parser.add_argument('train_data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where training subset files are stored')

    parser.add_argument('model_id',
                        action='store',
                        type=str,
                        help='Identifier for this model')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')


    return vars(parser.parse_args())


