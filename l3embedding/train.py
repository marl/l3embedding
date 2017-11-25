import glob
import json
import math
import os
import pickle
import random
import warnings

import keras
from keras.optimizers import Adam
import numpy as np
import pescador
import scipy.misc
from skvideo.io import vread
import soundfile as sf
from tqdm import tqdm

from .image import *
from .model import construct_cnn_L3_orig
from .training_utils import multi_gpu_model


#TODO: Consider putting the sampling functionality into another file

def get_filename(path):
    """Return the filename of a path

    Args: path: path to file

    Returns:
        filename: name of file (without extension)
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_file_list(data_dir):
    """Return audio and video file list.

    Args:
        data_dir: input directory that contains audio/ and video/

    Returns:
        audio_files: list of audio files
        video_files: list of video files

    """
    data_dir_contents = set(os.listdir(data_dir))
    if 'audio' in data_dir_contents and 'video' in data_dir_contents:
        audio_files = glob.glob('{}/audio/*'.format(data_dir))
        video_files = glob.glob('{}/video/*'.format(data_dir))
    else:
        audio_files = glob.glob('{}/**/audio/*'.format(data_dir))
        video_files = glob.glob('{}/**/video/*'.format(data_dir))

    # Make sure that audio files and video files correspond to each other
    audio_filenames = set([get_filename(path) for path in audio_files])
    video_filenames = set([get_filename(path) for path in video_files])

    valid_filenames = audio_filenames & video_filenames
    audio_files = [path for path in audio_files if get_filename(path) in valid_filenames]
    video_files = [path for path in video_files if get_filename(path) in valid_filenames]

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


def sample_one_second(audio_data, sampling_frequency, augment=False):
    """Return one second audio samples randomly

    Args:
        audio_data: audio data to sample from
        sampling_frequency: audio sample rate
        augment: if True, perturb the data in some fashion

    Returns:
        One second samples, start time, and augmentation parameters

    """
    sampling_frequency = int(sampling_frequency)
    if len(audio_data) > sampling_frequency:
        start = random.randrange(len(audio_data) - sampling_frequency)
    else:
        start = 0

    audio_data = audio_data[start:start+sampling_frequency]
    if audio_data.shape[0] != sampling_frequency:
        # Pad audio that isn't one second
        warnings.warn('Got audio that is less than one second', UserWarning)
        audio_data = np.pad(audio_data,
                            ((0, sampling_frequency - audio_data.shape[0]), (0,0)),
                            mode='constant')
    if augment:
        # Make sure we don't clip
        if np.abs(audio_data).max():
            max_gain = min(0.1, 1.0/np.abs(audio_data).max() - 1)
        else:
            # TODO: Handle audio with all zeros
            warnings.warn('Got audio sample with all zeros', UserWarning)
            max_gain = 0.1
        gain = 1 + random.uniform(-0.1, max_gain)
        audio_data *= gain
        audio_aug_params = {'gain': gain}
    else:
        audio_aug_params = {}

    return audio_data, start / sampling_frequency, audio_aug_params


def l3_frame_scaling(frame_data):
    """
    Scale and crop an video frame, using the method from Look, Listen and Learn


    Args:
        frame_data: video frame data array

    Returns:
        scaled_frame_data: scaled and cropped frame data
        bbox: bounding box for the cropped image
    """
    nx, ny, nc = frame_data.shape
    scaling = 256.0 / min(nx, ny)

    new_nx, new_ny = math.ceil(scaling * nx), math.ceil(scaling * ny)
    assert 256 in (new_nx, new_ny), str((new_nx, new_ny))


    resized_frame_data = scipy.misc.imresize(frame_data, (new_nx, new_ny, nc))

    start_x, start_y = random.randrange(new_nx - 224), random.randrange(new_ny - 224)
    end_x, end_y = start_x + 224, start_y + 224

    bbox = {
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x,
        'end_y': end_y
    }

    return resized_frame_data[start_x:end_x, start_y:end_y, :], bbox


def sample_one_frame(video_data, start=None, fps=30, scaling_func=None, augment=False):
    """Return one frame randomly and time (seconds).

    Args:
        video_data: video data to sample from
        start: start time of a one second window from which to sample
        fps: frame per second
        scaling_func: function that rescales the sampled video frame
        augment: if True, perturb the data in some fashion

    Returns:
        One frame sampled randomly, start time in seconds, and augmentation parameters

    """
    if not scaling_func:
        scaling_func = l3_frame_scaling

    num_frames = video_data.shape[0]
    if start is not None:
        start_frame = int(start * fps)
        # Sample frame from a one second window, or until the end of the video
        # if the video is less than a second for some reason
        # Audio should always be sampled one second from the end of the audio,
        # so video frames we're sampling from should also be a second. If it's
        # not, then our video is probably less than a second
        duration = min(fps, num_frames - start_frame)
        if duration != fps:
            warnings.warn('Got video that is less than one second', UserWarning)

        if duration > 0:
            frame = start_frame + random.randrange(duration)
        else:
            warnings.warn('Got video with only a single frame', UserWarning)
            # For robustness, use the last frame if the start_frame goes past
            # the end of video frame
            frame = min(start_frame, num_frames - 1)
    else:
        frame = random.randrange(num_frames)

    frame_data = video_data[frame, :, :, :]
    frame_data, bbox = scaling_func(frame_data)

    video_aug_params = {'bounding_box': bbox}

    if augment:
        # Randomly horizontally flip the image
        horizontal_flip = False
        if random.random() < 0.5:
            frame_data = horiz_flip(frame_data)
            horizontal_flip = True

        # Ranges taken from https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

        # Randomize the order of saturation jitter and brightness jitter
        if random.random() < 0.5:
            # Add saturation jitter
            saturation_factor = random.random() + 0.5
            frame_data = adjust_saturation(frame_data, saturation_factor)

            # Add brightness jitter
            max_delta = 32. / 255.
            brightness_delta = (2*random.random() - 1) * max_delta
            frame_data = adjust_brightness(frame_data, brightness_delta)
        else:
            # Add brightness jitter
            max_delta = 32. / 255.
            brightness_delta = (2*random.random() - 1) * max_delta
            frame_data = adjust_brightness(frame_data, brightness_delta)

            # Add saturation jitter
            saturation_factor = random.random() + 0.5
            frame_data = adjust_saturation(frame_data, saturation_factor)

        video_aug_params.update({
            'horizontal_flip': horizontal_flip,
            'saturation_factor': saturation_factor,
            'brightness_delta': brightness_delta
        })


    return frame_data, frame / fps, video_aug_params


def sampler(video_files, audio_files, augment=False):
    while True:
        video_file = random.choice(video_files)
        video_data = vread(video_file)
        audio_file = video_to_audio(video_file)

        if random.random() < 0.5:
            audio_file = random.choice([af for af in audio_files if af != audio_file])
            label = 0
        else:
            label = 1

        audio_data, sampling_frequency = sf.read(audio_file, always_2d=True)

        sample_audio_data, audio_start, audio_aug_params = sample_one_second(audio_data,
                                                                             sampling_frequency,
                                                                             augment=augment)
        sample_video_data, video_start, video_aug_params = sample_one_frame(video_data,
                                                                            start=audio_start,
                                                                            augment=augment)
        sample_audio_data = sample_audio_data.mean(axis=-1).reshape((1, sample_audio_data.shape[0]))

        sample = {
            'video': sample_video_data,
            'audio': sample_audio_data,
            'label': np.array([label, 1 - label]),
            'audio_file': audio_file,
            'video_file': video_file,
            'audio_start': audio_start,
            'video_start': video_start,
            'audio_augment_params': audio_aug_params,
            'video_augment_params': video_aug_params
        }
        yield sample


def data_generator(data_dir, k=32, batch_size=64, random_state=20171021,
                   augment=False):
    """Sample video and audio from data_dir, returns a streamer that yield samples infinitely.

    Args:
        data_dir: directory to sample video and audio from
        k: number of concurrent open streamer
        batch_size: batch size

    Returns:
        A generator that yield infinite video and audio samples from data_dir

    """

    random.seed(random_state)

    audio_files, video_files = get_file_list(data_dir)

    # Randomly shuffle the files
    ordering = list(range(len(audio_files)))
    random.shuffle(ordering)
    audio_files = [audio_files[idx] for idx in ordering]
    video_files = [video_files[idx] for idx in ordering]

    seeds = [pescador.Streamer(sampler,
                               video_files,
                               audio_files,
                               augment=augment) for i in range(batch_size)]

    mux = pescador.Mux(seeds, k, rate=16)
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
def train(train_data_dir, validation_data_dir, model_id, output_dir,
          num_epochs=150, epoch_size=512, batch_size=64, validation_size=1024,
          num_streamers=16, learning_rate=1e-4, random_state=20171021,
          verbose=False, checkpoint_interval=10, augment=False, gpus=1):
    m, inputs, outputs = construct_cnn_L3_orig()
    if gpus > 1:
        m = multi_gpu_model(m, gpus=gpus)
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    monitor = 'val_loss'

    # Make sure the directories we need exist
    model_dir = os.path.join(output_dir, model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    print('Compile model...')
    m.compile(Adam(lr=learning_rate),
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
    checkpoint_weight_path = os.path.join(model_dir, 'model.{epoch:02d}.h5')

    cb = []
    cb.append(keras.callbacks.ModelCheckpoint(weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1,
                                              monitor=monitor))

    cb.append(keras.callbacks.ModelCheckpoint(checkpoint_weight_path,
                                              save_weights_only=True,
                                              monitor=monitor,
                                              period=checkpoint_interval))

    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                        separator=','))


    print('Setting up train data generator...')
    train_gen = data_generator(
        #train_csv_path,
        train_data_dir,
        batch_size=batch_size,
        random_state=random_state,
        k=num_streamers,
        augment=augment)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    print('Setting up validation data generator...')
    val_gen = data_generator(
        validation_data_dir,
        batch_size=batch_size,
        random_state=random_state,
        k=num_streamers)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                           ['video', 'audio'],
                                           'label')



    # Fit the model
    print('Fit model...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2
    history = m.fit_generator(train_gen, epoch_size, num_epochs,
                              validation_data=val_gen,
                              validation_steps=validation_size,
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
