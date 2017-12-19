import glob
import json
import math
import datetime
import time
import os
import pickle
import random
import warnings
import csv

import keras
from keras.optimizers import Adam
import pescador
import scipy.misc
from skvideo.io import FFmpegReader, ffprobe
import soundfile as sf
from tqdm import tqdm

from .image import *
from .model import MODELS
from .training_utils import multi_gpu_model
from audioset.ontology import ASOntology
from log import *

LOGGER = logging.getLogger('l3embedding')
LOGGER.setLevel(logging.DEBUG)


#TODO: Consider putting the sampling functionality into another file

def get_filename(path):
    """Return the filename of a path

    Args: path: path to file

    Returns:
        filename: name of file (without extension)
    """
    return os.path.splitext(os.path.basename(path))[0]


def load_metadata(metadata_path):
    metadata = {}
    for path in glob.glob(metadata_path):
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx in (0, 1):
                    continue
                elif idx == 2:
                    fields = [field.strip() for field in line.lstrip('# ').rstrip().split(', ')]
                else:
                    row = [val.strip() for val in line.strip().split(', ')]
                    ytid = row[0]

                    entry = {field: val
                            for field, val in zip(fields[1:], row[1:])}

                    entry['positive_labels'] = entry['positive_labels'].strip('"').split(',')
                    entry['start_seconds'] = float(entry['start_seconds'])
                    entry['end_seconds'] = float(entry['end_seconds'])

                    metadata[ytid] = entry

    return metadata


def load_filters(filter_path):
    filters = []

    with open(filter_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filters.append(row)

    return filters

def get_ytid_from_filename(filename):
    first_us_idx = filename.rindex('_')
    second_us_idx = filename.rindex('_', 0, first_us_idx)
    return filename[:second_us_idx]


def get_file_list(data_dir, metadata_path=None, filter_path=None, ontology_path=None):
    """Return audio and video file list.

    Args:
        data_dir: input directory that contains audio/ and video/

    Keyword Args:
        metadata_path: Path to audioset metadata file
        filter_path: Path to filter specification file
        ontology_path: Path to AudioSet ontology file

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

    if metadata_path and filter_path:
        LOGGER.info('Filtering examples...')
        if not ontology_path:
            err_msg = 'Must provide ontology path to filter'
            LOGGER.error(err_msg)
            raise ValueError(err_msg)

        ontology = ASOntology(ontology_path)

        metadata = load_metadata(metadata_path)
        filters = load_filters(filter_path)

        filtered_filenames = []

        for filename in valid_filenames:
            ytid = get_ytid_from_filename(filename)
            video_metadata = metadata[ytid]

            video_labels = [ontology.get_node(label_id).name.lower()
                            for label_id in video_metadata['positive_labels']]

            accept = True
            for _filter in filters:
                filter_type = _filter['filter_type']
                filter_accept = _filter['accept_reject'] == 'accept'
                string = _filter['string']

                if filter_type == 'ytid':
                    match = ytid == string

                elif filter_type == 'label':
                    match = string.lower() in video_labels

                # TODO: check this logic
                if match == filter_accept:
                    accept = False

            if accept:
                #LOGGER.debug('Using video: "{}"'.format(filename))
                filtered_filenames.append(filename)

        valid_filenames = set(filtered_filenames)

    LOGGER.info('Total videos used: {}'.format(len(valid_filenames)))
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

    with LogTimer(LOGGER, 'Slicing audio'):
        audio_data = audio_data[start:start+sampling_frequency]

    if audio_data.shape[0] != sampling_frequency:
        # Pad audio that isn't one second
        warnings.warn('Got audio that is less than one second', UserWarning)
        with LogTimer(LOGGER, 'Slicing audio'):
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
        with LogTimer(LOGGER, 'Applying gain to audio'):
            audio_data *= gain
        audio_aug_params = {'gain': gain}
    else:
        audio_aug_params = {}

    return audio_data, start / sampling_frequency, audio_aug_params


def sample_cropped_frame(frame_data):
    """
    Randomly crop a video frame, using the method from Look, Listen and Learn


    Args:
        frame_data: video frame data array

    Returns:
        scaled_frame_data: scaled and cropped frame data
        bbox: bounding box for the cropped image
    """
    nx, ny, nc = frame_data.shape
    start_x, start_y = random.randrange(nx - 224), random.randrange(ny - 224)
    end_x, end_y = start_x + 224, start_y + 224

    bbox = {
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x,
        'end_y': end_y
    }

    with LogTimer(LOGGER, 'Cropping frame'):
        frame_data = frame_data[start_x:end_x, start_y:end_y, :]

    return frame_data, bbox


def sample_one_frame(video_data, start=None, fps=30, augment=False):
    """Return one frame randomly and time (seconds).

    Args:
        video_data: video data to sample from

    Keyword Args:
        start: start time of a one second window from which to sample
        fps: frame per second
        augment: if True, perturb the data in some fashion

    Returns:
        One frame sampled randomly, start time in seconds, and augmentation parameters

    """

    num_frames = len(video_data)
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

    frame_data = video_data[frame]
    frame_data, bbox = sample_cropped_frame(frame_data)

    video_aug_params = {'bounding_box': bbox}

    if augment:
        # Randomly horizontally flip the image
        horizontal_flip = False
        if random.random() < 0.5:
            with LogTimer(LOGGER, 'Flipping frame'):
                frame_data = horiz_flip(frame_data)
            horizontal_flip = True


        # Ranges taken from https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

        # Randomize the order of saturation jitter and brightness jitter
        if random.random() < 0.5:
            # Add saturation jitter
            saturation_factor = np.float32(random.random() + 0.5)
            with LogTimer(LOGGER, 'Adjusting saturation'):
                frame_data = adjust_saturation(frame_data, saturation_factor)

            # Add brightness jitter
            max_delta = 32. / 255.
            brightness_delta = np.float32((2*random.random() - 1) * max_delta)
            with LogTimer(LOGGER, 'Adjusting brightness'):
                frame_data = adjust_brightness(frame_data, brightness_delta)
        else:
            # Add brightness jitter
            max_delta = 32. / 255.
            brightness_delta = np.float32((2*random.random() - 1) * max_delta)
            with LogTimer(LOGGER, 'Adjusting brightness'):
                frame_data = adjust_brightness(frame_data, brightness_delta)

            # Add saturation jitter
            saturation_factor = np.float32(random.random() + 0.5)
            with LogTimer(LOGGER, 'Adjusting saturation'):
                frame_data = adjust_saturation(frame_data, saturation_factor)

        video_aug_params.update({
            'horizontal_flip': horizontal_flip,
            'saturation_factor': saturation_factor,
            'brightness_delta': brightness_delta
        })

    frame_data = skimage.img_as_float32(frame_data)


    return frame_data, frame / fps, video_aug_params


def read_video(video_path):
    """
    Read a video file as a numpy array

    Resizes frames so that the minimum side is 256 pixels

    Args:
        video_path: Path to video file

    Returns:
        video: Numpy data array

    """
    vinfo = ffprobe(video_path)['video']
    width = int(vinfo['@width'])
    height = int(vinfo['@height'])

    scaling = 256.0 / min(width, height)
    new_width = math.ceil(scaling * width)
    new_height = math.ceil(scaling * height)

    # Resize frames
    reader = FFmpegReader(video_path,
                          outputdict={'-s': "{}x{}".format(new_width,
                                                           new_height) })

    frames = []
    for frame in reader.nextFrame():
        frames.append(frame)

    return frames


def generate_sample(audio_file_1, audio_data_1, audio_file_2, audio_data_2,
                    video_file_1, video_data_1, video_file_2, video_data_2,
                    audio_sampling_frequency, augment=False):
    """
    Generate a sample from the given audio and video files

    The video from which the audio come from is chosen with a fair coin, as is
    the video frame. Thus, there is a 50% chance of producing a positive or
    negative example.

    Args:
        audio_file_1: audio filename
        audio_data_1: audio data array
        audio_file_2: audio filename
        audio_data_2: audio data array
        video_file_1: video filename
        video_data_1: video data array
        video_file_2: video filename
        video_data_2: video data array
        audio_sampling_frequency: audio sample rate

    Keyword Args
        augment: If True, perform data augmention

    Returns:
        sample: sample dictionary
    """
    video_choice = random.random() < 0.5
    audio_choice = random.random() < 0.5

    if audio_choice:
        audio_file = audio_file_1
        audio_data = audio_data_1
    else:
        audio_file = audio_file_2
        audio_data = audio_data_2

    if video_choice:
        video_file = video_file_1
        video_data = video_data_1
    else:
        video_file = video_file_2
        video_data = video_data_2

    label = int(video_choice != audio_choice)

    sample_audio_data, audio_start, audio_aug_params \
        = sample_one_second(audio_data, audio_sampling_frequency, augment=augment)

    sample_video_data, video_start, video_aug_params \
        = sample_one_frame(video_data, start=audio_start, augment=augment)

    sample_audio_data = sample_audio_data.mean(axis=-1).reshape((1, sample_audio_data.shape[0]))

    sample = {
        'video': np.ascontiguousarray(sample_video_data),
        'audio': np.ascontiguousarray(sample_audio_data),
        'label': np.ascontiguousarray(np.array([label, 1 - label])),
#       'audio_file': audio_file,
#       'video_file': video_file,
#       'audio_start': audio_start,
#       'video_start': video_start,
#       'audio_augment_params': audio_aug_params,
#       'video_augment_params': video_aug_params
    }

    return sample


def sampler(video_file_1, video_file_2, rate=32, augment=False, precompute=False):
    """Sample one frame from video_file, with 50% chance sample one second from corresponding audio_file,
       50% chance sample one second from another audio_file in the list of audio_files.

    Args:
        video_file_1: video_file to sample from
        video_file_2: candidate audio_files to sample from

    Keyword Args:
        rate: Poisson rate parameter. Used for precomputing samples
        augment: If True, perform data augmention
        precompute: If True, precompute samples during initialization so that
                    memory can be discarded

    Returns:
        A generator that yields dictionary of video sample, audio sample,
        and label (0: not from corresponding files, 1: from corresponding files)

    """
    debug_msg = 'Initializing streamer with videos "{}" and "{}"'
    LOGGER.debug(debug_msg.format(video_file_1, video_file_2))
    audio_file_1 = video_to_audio(video_file_1)
    audio_file_2 = video_to_audio(video_file_2)

    # Hack: choose a number of samples such that we with high probability, we
    #       won't run out of samples, but is also less than the entire length of
    #       the video so we don't have to resize all of the frames
    num_samples = int(scipy.stats.poisson.ppf(0.999, rate))


    try:
        with LogTimer(LOGGER, 'Reading video'):
            video_data_1 = read_video(video_file_1)
    except Exception as e:
        warn_msg = 'Could not open video file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(video_file_1, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    try:
        with LogTimer(LOGGER, 'Reading video'):
            video_data_2 = read_video(video_file_2)
    except Exception as e:
        warn_msg = 'Could not open video file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(video_file_2, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    try:
        with LogTimer(LOGGER, 'Reading audio'):
            audio_data_1, sampling_frequency = sf.read(audio_file_1,
                                                       dtype='float32',
                                                       always_2d=True)
    except Exception as e:
        warn_msg = 'Could not open audio file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(audio_file_1, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    try:
        with LogTimer(LOGGER, 'Reading audio'):
            audio_data_2, sampling_frequency = sf.read(audio_file_2,
                                                       dtype='float32',
                                                       always_2d=True)
    except Exception as e:
        warn_msg = 'Could not open audio file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(audio_file_2, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    if precompute:
        samples = []
        for _ in range(num_samples):
            sample = generate_sample(
                audio_file_1, audio_data_1, audio_file_2, audio_data_2,
                video_file_1, video_data_1, video_file_2, video_data_2,
                sampling_frequency, augment=augment)

            samples.append(sample)

        # Clear the data from memory
        video_data_1 = None
        video_data_2 = None
        audio_data_1 = None
        audio_data_2 = None
        video_data = None
        audio_data = None
        del video_data_1
        del video_data_2
        del audio_data_1
        del audio_data_2
        del video_data
        del audio_data

        while samples:
            # Yield the sample, and remove from the list to free up some memory
            yield samples.pop()
    else:
        while True:
            yield generate_sample(
                audio_file_1, audio_data_1, audio_file_2, audio_data_2,
                video_file_1, video_data_1, video_file_2, video_data_2,
                sampling_frequency, augment=augment)

    raise StopIteration()



def data_generator(data_dir, metadata_path=None, filter_path=None, ontology_path=None,
                   k=32, batch_size=64, random_state=20171021, precompute=False,
                   num_distractors=1, augment=False, rate=32, max_videos=None):
    """Sample video and audio from data_dir, returns a streamer that yield samples infinitely.

    Args:
        data_dir: directory to sample video and audio from
        metadata_path: Path to audioset metadata file
        filter_path: Path to filter specification file
        ontology_path: Path to AudioSet ontology file
        k: number of concurrent open streamer
        batch_size: batch size
        random_state: Value used to initialize state of RNG
        num_distractors: Number of pairs to generate a stream for each video

    Returns:
        A generator that yield infinite video and audio samples from data_dir

    """

    random.seed(random_state)

    LOGGER.info("Getting file list...")
    _, video_files = get_file_list(data_dir, metadata_path=metadata_path,
                                             filter_path=filter_path, ontology_path=ontology_path)

    LOGGER.info("Creating streamers...")
    if max_videos is not None and max_videos < len(video_files):
        LOGGER.info("Using a subset of {} videos".format(max_videos))
        random.shuffle(video_files)
        video_files = video_files[:max_videos]

    seeds = []
    for video_file_1 in tqdm(video_files):
        for _ in range(num_distractors):
            video_file_2 = video_file_1
            # Make sure we sample a different file
            while video_file_2 == video_file_1:
                video_file_2 = random.choice(video_files)

            #debug_msg = 'Created streamer for videos "{}" and "{}'
            #LOGGER.debug(debug_msg.format(video_file_1, video_file_2))
            streamer = pescador.Streamer(sampler, video_file_1, video_file_2,
                                         rate=rate, augment=augment,
                                         precompute=precompute)
            # ZMQ streamers make the script stall inexplicably
            #streamer = pescador.ZMQStreamer(streamer)
            seeds.append(streamer)

    # Randomly shuffle the seeds
    random.shuffle(seeds)

    mux = pescador.Mux(seeds, k, rate=rate)
    if batch_size == 1:
        return mux
    else:
        return pescador.maps.buffer_stream(mux, batch_size)


class LossHistory(keras.callbacks.Callback):
    """
    Keras callback to record loss history
    """

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


class TimeHistory(keras.callbacks.Callback):
    """
    Keras callback to log epoch and batch running time
    """
    # Copied from https://stackoverflow.com/a/43186440/1260544
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.batch_times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        t = time.time() - self.epoch_time_start
        LOGGER.info('Epoch took {} seconds'.format(t))
        self.epoch_times.append(t)

    def on_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs=None):
        t = time.time() - self.batch_time_start
        LOGGER.info('Batch took {} seconds'.format(t))
        self.batch_times.append(t)


#def train(train_csv_path, model_id, output_dir, num_epochs=150, epoch_size=512,
def train(train_data_dir, validation_data_dir, model_id, output_dir,
          num_epochs=150,
          train_metadata_path=None, validation_metadata_path=None,
          train_filter_path=None, validation_filter_path=None,
          train_epoch_size=512, validation_epoch_size=1024,
          train_batch_size=64, validation_batch_size=64,
          train_num_streamers=16, validation_num_streamers=16,
          train_num_distractors=1, validation_num_distractors=2,
          model_type='cnn_L3_orig', train_mux_rate=16, validation_mux_rate=16,
          learning_rate=1e-4, random_state=20171021, train_max_videos=None,
          validation_max_videos=None, verbose=False, checkpoint_interval=10,
          ontology_path=None, log_path=None, disable_logging=False,
          precompute=False, augment=False, gpus=1):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    m, inputs, outputs = MODELS[model_type]()
    if gpus > 1:
        m = multi_gpu_model(m, gpus=gpus)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    monitor = 'val_loss'

    # Make sure the directories we need exist
    model_dir = os.path.join(output_dir, model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    LOGGER.info('Compiling model...')
    m.compile(Adam(lr=learning_rate),
              loss=loss,
              metrics=metrics)

    LOGGER.info('Model files can be found in "{}"'.format(model_dir))

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

    timer_cb = TimeHistory()
    cb.append(timer_cb)

    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                        separator=','))

    LOGGER.info('Setting up train data generator...')
    train_gen = data_generator(
        train_data_dir,
        metadata_path=train_metadata_path,
        ontology_path=ontology_path,
        filter_path=train_filter_path,
        batch_size=train_batch_size,
        random_state=random_state,
        k=train_num_streamers,
        augment=augment,
        num_distractors=train_num_distractors,
        max_videos=train_max_videos,
        precompute=precompute,
        rate=train_mux_rate)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    LOGGER.info('Setting up validation data generator...')
    val_gen = data_generator(
        validation_data_dir,
        metadata_path=validation_metadata_path,
        ontology_path=ontology_path,
        filter_path=validation_filter_path,
        batch_size=validation_batch_size,
        random_state=random_state,
        k=validation_num_streamers,
        num_distractors=validation_num_distractors,
        max_videos=validation_max_videos,
        precompute=precompute,
        rate=validation_mux_rate)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                           ['video', 'audio'],
                                           'label')



    # Fit the model
    LOGGER.info('Fitting model...')
    if verbose:
        verbosity = 1
    else:
        verbosity = 2
    history = m.fit_generator(train_gen, train_epoch_size, num_epochs,
                              validation_data=val_gen,
                              validation_steps=validation_epoch_size,
                              #use_multiprocessing=True,
                              callbacks=cb,
                              verbose=verbosity)

    LOGGER.info('Done training. Saving results to disk...')
    # Save history
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)

    LOGGER.info('Done!')
